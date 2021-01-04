FROM mcr.microsoft.com/vscode/devcontainers/base:focal

RUN set -eux; \
    apt-get update; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get --no-install-recommends -y install build-essential cmake gdb git qttools5-dev

RUN git clone -b v2.4.0 --depth 1 https://github.com/tensorflow/tensorflow.git /opt/tensorflow

RUN set -eux; \
    wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip -O /tmp/model.zip; \
    mkdir -p /opt/model; \
    unzip /tmp/model.zip labels_mobilenet_quant_v1_224.txt mobilenet_v1_1.0_224_quant.tflite -d /opt/model/
