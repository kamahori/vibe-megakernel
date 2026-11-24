# MegaBench Trainium image: native PyTorch backend (github.com/aws-neuron/torch-neuronx),
# installed as its README describes for Ubuntu 26.04.
# - Ubuntu 26.04: the torch-neuronx 2.14 wheel needs glibc 2.43 (Amazon Linux 2023 has 2.34).
# - The latest Neuron runtime from the apt repo (2.35): torch-neuronx 2.14.3 links symbols
#   missing from 2.34. The driver (dkms) is the host's.
# - libarchive13t64: neuronx-cc's walrus_driver links libarchive.so.13.
FROM ubuntu:26.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg git build-essential libarchive13t64 \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | gpg --dearmor -o /usr/share/keyrings/neuron.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/neuron.gpg] https://apt.repos.neuron.amazonaws.com noble main" > /etc/apt/sources.list.d/neuron.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends aws-neuronx-collectives aws-neuronx-runtime-lib aws-neuronx-tools \
 && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
 && UV_PYTHON_INSTALL_DIR=/opt/python uv venv --python 3.12 /opt/venv \
 && uv pip install -q --python /opt/venv/bin/python torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && uv pip install -q --python /opt/venv/bin/python --index-strategy unsafe-best-match \
      --extra-index-url https://pip.repos.neuron.amazonaws.com \
      "torch-neuronx==2.14.*" "neuronx-cc==2.*" "nki>=0.6" numpy
ENV PATH=/opt/venv/bin:/opt/aws/neuron/bin:$PATH \
    LD_LIBRARY_PATH=/opt/aws/neuron/lib
