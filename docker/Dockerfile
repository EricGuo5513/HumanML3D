FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt update && apt install -y git wget python3-pip software-properties-common

ENV OS=ubuntu2204
ENV cudnn_version=8.8.1.3
ENV cuda_version=cuda11.8

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin 

RUN mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/3bf863cc.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /"
RUN apt-get update

RUN apt-get install -y libcudnn8=${cudnn_version}-1+${cuda_version} \
                       libcudnn8-dev=${cudnn_version}-1+${cuda_version}

RUN pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install tensorflow-datasets
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
