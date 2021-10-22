FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION="3.6.0"

# Install core packages
RUN apt-get update

RUN apt-get install -y build-essential checkinstall software-properties-common llvm cmake wget git vim nasm yasm zip unzip pkg-config \
    wget \
    nginx \
    ca-certificates \
    apt-transport-https \
    libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev mysql-client default-libmysqlclient-dev

RUN apt-get autoclean

# Install python 3.6.0
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz \
    && tar xvf Python-${PYTHON_VERSION}.tar.xz \
    && rm Python-${PYTHON_VERSION}.tar.xz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure \
    && make altinstall \
    && cd / \
    && rm -rf Python-${PYTHON_VERSION}

RUN python3 -V

ENV PYTHONPATH "${PYTHONPATH}:/usr/bin/python3"

RUN apt-get -y install python3-pip python-setuptools

# setup file system
RUN mkdir argus
ENV HOME=/argus
ENV SHELL=/bin/bash
VOLUME /argus
WORKDIR /argus
ADD . /argus

RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r requirements.txt
