FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN pip install -r requirements.txt

COPY . .

ARG HOST
ARG PORT

ENV SERVER_HOST=${HOST}
ENV SERVER_PORT=${PORT}

CMD ["sh", "-c", "echo \"SERVER_HOST is ${SERVER_HOST}\" && echo \"SERVER_PORT is ${SERVER_PORT}\" && python server.py --host ${SERVER_HOST} --port ${SERVER_PORT}"]