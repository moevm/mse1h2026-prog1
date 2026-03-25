#FROM trampgeek/jobeinabox@sha256:a454ee1ea7a495268fe203f88b801bd797fbca1571320c999a4095c4c5157f07
FROM python:3.13.2-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
#RUN apt install -y software-properties-common
RUN apt-get install --reinstall ca-certificates
#RUN add-apt-repository -y 'ppa:deadsnakes/ppa'
RUN apt update -y && \
    apt install -y wget \
                   libglib2.0-dev \
                   libmpc-dev \
                   gcc \
                   nano \
                   file
#RUN apt install -y python3.9
#RUN python3 -v
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 9

WORKDIR /app
ENV PYTHONPATH=/app
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY main.py .
ENTRYPOINT ["python3", "main.py"]

COPY ./src ./src
COPY ./build.py ./build.py
RUN python3 build.py && rm -r requirements.txt build.py
