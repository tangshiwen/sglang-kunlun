ARG BASE_IMAGE=iregistry.baidu-int.com/xmlir/xpytorch:torch29-py312-ubuntu20.04-v0.45
FROM ${BASE_IMAGE}

RUN rm -rf /workspace/*
RUN mkdir -p /workspace/

COPY sglang-kunlun /workspace/sglang-kunlun
COPY sglang /workspace/sglang

RUN bash /workspace/sglang-kunlun/dockerfile/install.sh

WORKDIR /workspace

ENV LD_LIBRARY_PATH=/root/miniconda/envs/python312_torch29_cuda/lib:/root/miniconda/envs/python312_torch29_cuda/xcudart/lib:/root/miniconda/envs/python312_torch29_cuda/lib/python3.12/site-packages/torch_xmlir:/usr/local/xre/so:/usr/local/xccl/so:/usr/local/lib/:${LD_LIBRARY_PATH}
