# Image for agent sessions on MegaBench's Trainium cases (Claude Code), and for grading
# their submissions. Built by
#   python -m megabench.integrations.container_run build-image --platform neuron
# on top of the Neuron image (megabench/docker/neuron/Dockerfile: torch-neuronx, neuronx-cc,
# nki and the Neuron runtime and tools in /opt/venv), with the host's `claude` staged in the
# build context, so the image carries the same CLI version as the host. Nothing from the
# repository is copied in: the trusted benchmark and the workspace are mounted at run time.

ARG BASE_IMAGE=megabench-neuron:native
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN set -eux; \
    for attempt in 1 2 3 4 5; do \
        if apt-get update -qq && apt-get install -y -qq --no-install-recommends \
            git ripgrep tmux curl ca-certificates jq less procps file unzip python3; then \
            break; \
        fi; \
        [ "${attempt}" -lt 5 ]; sleep $((attempt * 5)); \
    done; \
    rm -rf /var/lib/apt/lists/*; \
    ln -sf /opt/venv/bin/python /usr/local/bin/python3.12

# The Neuron profiler (neuron-profile / neuron-explorer) comes with aws-neuronx-tools in
# the base image; the plugin's profiling skills use it.
COPY claude /usr/local/bin/claude
COPY egress_proxy.py /opt/megabench/egress_proxy.py
RUN chmod 755 /usr/local/bin/claude && claude --version && \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 /opt/venv/bin/python -c "import torch, torch_neuronx, nki; print(torch.__version__, torch_neuronx.__version__, nki.__version__)" && \
    ls /opt/aws/neuron/bin

ENV PATH=/opt/venv/bin:/opt/aws/neuron/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/aws/neuron/lib \
    DISABLE_AUTOUPDATER=1 \
    LANG=C.UTF-8

ENTRYPOINT []
