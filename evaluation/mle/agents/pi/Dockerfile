# The pi agent for MLE-bench.
#
# Base image is mle-bench's own `mlebench-env`, so the Python/CUDA environment and the
# competition mounts are EXACTLY the official benchmark environment. This layer adds only:
#   - Node 22 + the pi coding agent CLI
#   - the harness (harness/run_pi.py) and the task skill library
#
# The build context is this directory alone -- everything the agent needs (harness, skills,
# task metadata) lives here, so the image builds identically from this repo and from a copy
# dropped into <mle-bench>/agents/pi/:
#   docker build --platform=linux/amd64 -t pi agents/pi/
# (scripts/build_image.sh does this for you.)
ARG BASE_IMAGE=mlebench-env
FROM ${BASE_IMAGE}

ARG SUBMISSION_DIR=/home/submission
ARG LOGS_DIR=/home/logs
ARG CODE_DIR=/home/code
ARG AGENT_DIR=/home/agent
ENV SUBMISSION_DIR=${SUBMISSION_DIR} LOGS_DIR=${LOGS_DIR} CODE_DIR=${CODE_DIR} AGENT_DIR=${AGENT_DIR}

RUN mkdir -p ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR} ${SUBMISSION_DIR}

# --- Node 22 + pi -----------------------------------------------------------------
# pi needs Node >= 22 (it uses the RegExp `v` flag). Installed system-wide so the
# non-root agent user can run it.
ARG NODE_VERSION=22.18.0
ARG PI_VERSION=latest
RUN set -eux; \
    arch="$(uname -m)"; case "$arch" in \
      x86_64) narch=x64 ;; aarch64|arm64) narch=arm64 ;; \
      *) echo "unsupported arch $arch" >&2; exit 1 ;; esac; \
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${narch}.tar.xz" \
      -o /tmp/node.tar.xz; \
    mkdir -p /opt/node; \
    tar -xf /tmp/node.tar.xz -C /opt/node --strip-components=1; \
    rm /tmp/node.tar.xz; \
    /opt/node/bin/npm install -g "@earendil-works/pi-coding-agent@${PI_VERSION}"; \
    ln -s /opt/node/bin/pi /usr/local/bin/pi; \
    /opt/node/bin/node --version; pi --version || true
ENV PATH="/opt/node/bin:${PATH}"

# --- extra python packages (normally none: see requirements.txt) -------------------
COPY requirements.txt ${AGENT_DIR}/requirements.txt
RUN conda run -n agent pip install --no-cache-dir -r ${AGENT_DIR}/requirements.txt && conda clean -afy

# --- the harness itself ------------------------------------------------------------
COPY harness ${AGENT_DIR}/harness
COPY skills ${AGENT_DIR}/skills
COPY tasks.json ${AGENT_DIR}/tasks.json
COPY start.sh ${AGENT_DIR}/start.sh
COPY models.json.template ${AGENT_DIR}/models.json.template
RUN chmod +x ${AGENT_DIR}/start.sh && chmod -R a+rX ${AGENT_DIR}

# IMPORTANT: do not change USER -- mle-bench's entrypoint must start as root and it execs the
# agent as `nonroot` itself.
