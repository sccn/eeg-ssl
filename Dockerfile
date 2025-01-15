FROM python:3.11
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN pip install git+https://github.com/dungscout96/mne-python.git
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install braindecode
RUN pip install tensorboard
RUN pip install lightning
CMD ["python"]
