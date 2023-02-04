FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

WORKDIR /opt/ml

ADD . .

RUN pip install -r requirements.txt