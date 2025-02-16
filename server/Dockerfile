# Change CUDA and cuDNN version here
FROM nvidia/cuda:12.4.1-base-ubuntu22.04
ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
        git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python$PYTHON_VERSION \
        python$PYTHON_VERSION-dev \
        python$PYTHON_VERSION-venv \
    && wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py \
    && python$PYTHON_VERSION get-pip.py \
    && rm get-pip.py \
    && ln -sf /usr/bin/python$PYTHON_VERSION /usr/bin/python \
    && ln -sf /usr/local/bin/pip$PYTHON_VERSION /usr/local/bin/pip \
    && python --version \
    && pip --version \
    && apt-get purge -y --auto-remove software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

####### Add your own installation commands here #######
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    && pip3 install numpy==1.26.4 \
    git+https://github.com/dungscout96/mne-python.git \
    braindecode \
    tensorboard \
    lightning==2.5.0 \
    "lightning[pytorch-extra]" \
    torchmetrics==1.6.1 \
    wandb 
# RUN wget https://path/to/some/data/or/weights
# RUN apt-get update && apt-get install -y <package-name>
WORKDIR /app
COPY . /app
COPY server /app
COPY libs /app/libs
# Install litserve and requirements
RUN pip install --no-cache-dir litserve==0.2.6 
EXPOSE 8000
CMD ["python", "/app/server.py"]
