"""Shared DSV4 mixed INT8 KV-cache layout.

Per-token byte layout (DeepSeek-V4-Flash: ``qk_nope_head_dim=448``,
``qk_rope_head_dim=64``, group size 64)::

    nope   qk_nope_head_dim                    int8    448 B
    rope   qk_rope_head_dim * 2                bf16    128 B
    scale  qk_nope_head_dim // group_size * 4  fp32     28 B
                                               total   604 B

This is not the fp8 layout upstream's ``DeepSeekV4SingleKVPool`` assumes
(448 fp8 nope + 128 bf16 rope + 7 B ue8m0 scale + 1 B pad = 584 B/token), and
it is not ``kv_lora_rank + qk_rope_head_dim`` either. The write kernels and the
attention backend's ``reshape(-1, 604)`` both depend on this exact width, so
every site derives it from here instead of hard-coding its own number.

The write kernels take the nope/rope dims as explicit arguments and cannot
recover them from the tensors, so they are pinned here next to the stride they
imply.
"""

from __future__ import annotations

import torch

# DeepSeek-V4-Flash int8 mixed tiling.
NOPE_DIM = 448
ROPE_DIM = 64
GROUP_SIZE = 64
SCALE_BYTES = NOPE_DIM // GROUP_SIZE * 4  # one fp32 scale per 64 nope elements


def mixed_int8_bytes_per_token(
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    group_size: int = GROUP_SIZE,
) -> int:
    """Return the byte-packed mixed INT8 footprint for one token."""
    if qk_nope_head_dim % group_size:
        raise ValueError(
            f"qk_nope_head_dim={qk_nope_head_dim} is not divisible by "
            f"group_size={group_size}"
        )
    return (
        qk_nope_head_dim
        + qk_rope_head_dim * 2
        + qk_nope_head_dim // group_size * 4
    )


# Byte width of one cached token, and the offset where its trailing fp32 scales
# begin (the layout is [nope][rope][scale]).
STRIDE = mixed_int8_bytes_per_token(NOPE_DIM, ROPE_DIM)
ROPE_END = STRIDE - SCALE_BYTES
ROPE_BYTES = ROPE_END - NOPE_DIM


def unpack_mixed_int8(cache: torch.Tensor) -> torch.Tensor:
    """Dequantize a packed mixed int8 cache back to dense bf16 ``[N, 512]``.

    Inverse of what ``quantize_mla_kv_cache_split`` produces (nope is stored as
    per-group symmetric int8 with an fp32 step, rope is kept in bf16). Used to
    run the dense torch reference over an int8 cache, which is how a packed
    write can be validated independently of the mixed-cache attention kernel.
    """
    if cache.ndim != 2 or cache.shape[-1] != STRIDE:
        raise ValueError(f"expected [N, {STRIDE}] int8 cache, got {tuple(cache.shape)}")
    nope_q = cache[:, :NOPE_DIM].float().reshape(-1, NOPE_DIM // GROUP_SIZE, GROUP_SIZE)
    rope = cache[:, NOPE_DIM:ROPE_END].contiguous().view(torch.bfloat16)
    scale = cache[:, ROPE_END:STRIDE].contiguous().view(torch.float32).reshape(
        -1, NOPE_DIM // GROUP_SIZE
    )
    nope = (nope_q * scale.unsqueeze(-1)).reshape(-1, NOPE_DIM)
    return torch.cat([nope, rope.float()], dim=-1).to(torch.bfloat16)
