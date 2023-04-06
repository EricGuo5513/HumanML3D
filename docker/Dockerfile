FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt update && apt install -y git wget python3-pip software-properties-common

ENV OS=ubuntu2204
ENV cudnn_version=8.8.1.3
ENV cuda_version=cuda11.8

RUN apt-get install -y libcudnn8=${cudnn_version}-1+${cuda_version} \
                       libcudnn8-dev=${cudnn_version}-1+${cuda_version}

RUN pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install tensorflow-datasets
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
