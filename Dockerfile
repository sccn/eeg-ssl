FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "-m", "main"]
