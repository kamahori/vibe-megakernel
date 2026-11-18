# Image for agent sessions on MegaBench cases (Claude Code or Codex), and for
# grading their submissions. Built by
#   python -m megabench.integrations.container_run build-image
# which stages this file, requirements.lock, egress_proxy.py and the host's
# `claude` and `codex` binaries in one build context, so the image carries the
# same CLI versions as the host. Nothing from the repository is copied in: the
# trusted benchmark and the workspace are mounted at run time.

ARG BASE_IMAGE=docker.io/nvidia/cuda:13.1.2-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN set -eux; \
    for attempt in 1 2 3 4 5; do \
        if apt-get update -qq && apt-get install -y -qq --no-install-recommends \
            python3.12 python3.12-dev python3.12-venv build-essential ninja-build \
            git ripgrep tmux curl ca-certificates jq less procps file unzip; then \
            break; \
        fi; \
        [ "${attempt}" -lt 5 ]; sleep $((attempt * 5)); \
    done; \
    rm -rf /var/lib/apt/lists/*

# Nsight Compute: the devel image may not carry it; ./profile_candidate needs it.
RUN set -eux; \
    if ! ls /usr/local/cuda/bin/ncu >/dev/null 2>&1; then \
        apt-get update -qq; \
        apt-get install -y -qq --no-install-recommends "$(apt-cache search --names-only '^cuda-nsight-compute-13-' | awk '{print $1}' | sort -V | tail -1)"; \
        rm -rf /var/lib/apt/lists/*; \
        ln -sf "$(ls -d /opt/nvidia/nsight-compute/*/ncu | sort -V | tail -1)" /usr/local/cuda/bin/ncu; \
    fi; \
    /usr/local/cuda/bin/ncu --version

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
COPY requirements.lock /opt/requirements.lock
RUN set -eux; \
    uv venv --python /usr/bin/python3.12 /opt/venv; \
    UV_HTTP_TIMEOUT=300 uv pip install --python /opt/venv/bin/python --no-cache \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match -r /opt/requirements.lock; \
    /opt/venv/bin/python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)"

# The host's CLIs (staged by build-image): same versions as the host. Codex
# keeps its vendor layout (its binary, helper binaries and bundled rg).
COPY claude /usr/local/bin/claude
COPY codex-vendor /opt/codex
COPY egress_proxy.py /opt/megabench/egress_proxy.py
RUN chmod -R a+rX /opt/codex && chmod 755 /usr/local/bin/claude && \
    ln -s /opt/codex/bin/codex /usr/local/bin/codex && \
    claude --version && codex --version

ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:/opt/codex/codex-path:${PATH} \
    CUDA_HOME=/usr/local/cuda \
    DISABLE_AUTOUPDATER=1 \
    LANG=C.UTF-8

# No banner entrypoint (the CUDA base prints one to stdout, which would land in the
# result files); container_round.sh also passes --entrypoint= for older builds.
ENTRYPOINT []
