# Base image
FROM vault.habana.ai/gaudi-docker/1.15.1/ubuntu22.04/habanalabs/pytorch-installer-2.2.0:latest

# Install dependencies
RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    git+https://github.com/dungscout96/mne-python.git \
    braindecode \
    tensorboard \
    lightning==2.5.0 \
    "lightning[pytorch-extra]" \
    torchmetrics==1.6.1 \
    wandb \
    litserve

# Run hl-smi as an initial command and specify the script
CMD ["sh", "-c"]
