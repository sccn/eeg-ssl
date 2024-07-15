FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
COPY . /app
WORKDIR /app
RUN pip install -r /app/requirements.txt
CMD ["python", "-m", "main"]
