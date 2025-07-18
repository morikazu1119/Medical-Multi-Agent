FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

RUN ["echo", "CUDA touch image built successfully"]