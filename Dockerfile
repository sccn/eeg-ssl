# Base image
FROM vault.habana.ai/gaudi-docker/1.15.1/ubuntu22.04/habanalabs/pytorch-installer-2.2.0:latest

# Install dependencies
RUN pip install numpy==1.26.4 \
    && pip install git+https://github.com/dungscout96/mne-python.git \
    && pip install braindecode \
    && pip install tensorboard \
    && pip install lightning==2.5.0 \
    && pip install "lightning[pytorch-extra]" \
    && pip install torchmetrics==1.6.1

# Run hl-smi as an initial command and specify the script
CMD ["sh", "-c"]
