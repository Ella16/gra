FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

RUN apt update \
    && echo $TZ > /etc/timezone \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && apt-get install -y --no-install-recommends tzdata git \
    && apt install -y --no-install-recommends python3.10 python3.10-dev python3.10-distutils python3-pip wget build-essential libgl1-mesa-glx libglib2.0-0 \
    && pip3 install -U pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
