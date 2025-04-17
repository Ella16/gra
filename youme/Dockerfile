# Base image
FROM ubuntu:20.04

LABEL email="youme-poc@growdle.com"
LABEL name="team-r2d2-youme-poc"
LABEL version="1.0"
LABEL description="Growdle Team R2D2 Youme PoC Dockerfile"

#Set default shell to /bin/bash
# SHELL ["/bin/bash", "-cu"]
SHELL ["/bin/bash", "-c"]

# WORKDIR /work

ARG DEBIAN_FRONTEND=noninteractive
ARG python=3.11
ARG pip_version=20.0.2
ARG black_version=23.7.0
ARG isort_version=5.12.0
ENV PYTHON_VERSION=${python} \
    PATH="/usr/bin:$PATH" \
    PYTHONPATH="/usr/bin/${PYTHON_VERSION}:${PYTHONPATH}" \
    JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    software-properties-common

RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:git-core/ppa && \
    add-apt-repository -y universe && \
    apt-get update

RUN apt-get install -y --no-install-recommends \
          vim \
          poppler-utils \
          apt-utils && \
          rm -rf /var/lib/apt/lists/*

# Basic
RUN apt-get update && \
         apt-get install -y \
         python3-setuptools \
         python3-wheel \
         gcc \
         wget \
         unzip \
         locales \
         locales-all \
         python3-pip \
         python${PYTHON_VERSION} \
         python${PYTHON_VERSION}-dev \
         curl \
         git \
         git-lfs \
         openssh-server \
         rsync \
         python${PYTHON_VERSION}-distutils \
         python${PYTHON_VERSION}-venv

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# python link
RUN rm /usr/bin/python3 && \
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# pip install
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&\
    python${PYTHON_VERSION} get-pip.py &&\
    rm get-pip.py &&\
    python${PYTHON_VERSION} -m pip install pip==${pip_version} &&\
    python${PYTHON_VERSION} -m pip install black==${black_version} isort==${isort_version}


# command
WORKDIR /work/youme_invoice
RUN mkdir -p ../data
# directories. files. clear
COPY . .
RUN rm -rf /work/youme_invoice/.env
RUN rm -rf /work/youme_invoice/.git
RUN rm -rf /work/youme_invoice/.vscode
RUN rm -rf /work/youme_invoice/main.py
RUN rm -rf /work/youme_invoice/.gitignore

# requirements
RUN python -m pip install -r /work/youme_invoice/requirements.txt

# CMD python youme_poc_server.py
CMD uvicorn youme_poc_server:app

# CMD ["nohup","sh","-c","python main.py --run --async_mode","&"]
# CMD ["nohup","sh","-c","python main.py --run","&"]
# CMD ["nohup","sh","-c","python title_generation_server.py","&"]
