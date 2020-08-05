# Build an image that can do training and inference in SageMaker
FROM ubuntu:18.04

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         python3 \
         python3-pip \
         python3-dev \
         python3-setuptools \
         build-essential \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch \
        transformers \
        requests \
        mxnet \
        pytorch-crf \
        konlpy \
        gluonnlp \
        sentencepiece \
        pandas \
        flask \
        gevent \
        gunicorn && rm -rf /root/.cache

RUN pip3 install sagemaker
RUN pip3 install tensorflow

# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

ENV AWS_DEFAULT_REGION="ap-northeast-2"
ENV AWS_ACCESS_KEY_ID="AKIA5SIZZPEDDP5NHPWG"
ENV AWS_SECRET_ACCESS_KEY="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"

# Set up the program in the image
COPY product-tags /opt/program
WORKDIR /opt/program

RUN chmod +x /opt/program/serve