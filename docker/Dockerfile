# see latest azureml base images tags here
# - https://github.com/Azure/AzureML-Containers/tree/master/base/gpu
# - https://hub.docker.com/_/microsoft-azureml?tab=description

ARG BASE_IMAGE=openmpi4.1.0-cuda11.3-cudnn8-ubuntu20.04:latest

FROM mcr.microsoft.com/azureml/${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-7 \
    git \
    gpg \
    curl \
    vim \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    openssh-client \
    openssh-server \
    libsm6 \
    libxext6 \
    ffmpeg \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN conda update -n base -c defaults conda

RUN gpg --version && apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC && \
    apt update -y && apt install -y libgl1-mesa-glx libosmesa6 libosmesa6-dev libglfw3-dev libgles2-mesa-dev freeglut3-dev

ADD environment.yml /tmp/environment.yml
RUN conda env update -n base -f /tmp/environment.yml

RUN apt-get install patchelf