FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu14.04

VOLUME /storage/logs

ARG DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/*.list \
    && apt-get clean && apt-get update \
    && apt-get install build-essential \
        apt-transport-https \
        tzdata \
        software-properties-common \
        curl -y\
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install python3.6 python3.6-dev -y \
    && curl https://bootstrap.pypa.io/get-pip.py | python3.6 \
    && pip3 install \
        tqdm \
        pillow \
        scipy \
        sklearn \
        scikit-image \
        pydensecrf \
        tensorflow==1.4.0

WORKDIR /src
