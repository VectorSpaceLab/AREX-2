#!/bin/bash
# Entry point executed inside the benchmark container (as the `nonroot` user).
# mle-bench's own entrypoint has already started the submission-validation server on :5000
# and mounted the competition data read-only at /home/data.
set -x

export AGENT_DIR="${AGENT_DIR:-/home/agent}"
export DATA_DIR="${DATA_DIR:-/home/data}"
export SUBMISSION_DIR="${SUBMISSION_DIR:-/home/submission}"
export LOGS_DIR="${LOGS_DIR:-/home/logs}"
export CODE_DIR="${CODE_DIR:-/home/code}"
mkdir -p "$LOGS_DIR" "$CODE_DIR" "$SUBMISSION_DIR"

eval "$(conda shell.bash hook)"
conda activate agent

# Report the hardware into the prompt so the agent sizes its training to the real machine.
if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed 's/^[ \t]*//; s/[ \t]*$//' | sort | uniq -c \
    | sed 's/^ *\([0-9]*\) *\(.*\)$/\1 \2/' | paste -sd ', ' -)
else
  HARDWARE="a CPU (no GPU visible)"
fi
export HARDWARE
python -c "import torch; print('torch cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || true

# Teach pi about any custom / self-hosted model endpoints. The template keeps $VAR placeholders;
# pi expands them from its own environment at request time, so no key is ever written to disk.
if [ -f "${AGENT_DIR}/models.json.template" ]; then
  mkdir -p "$HOME/.pi/agent"
  cp "${AGENT_DIR}/models.json.template" "$HOME/.pi/agent/models.json"
fi

# The harness enforces its own wall clock per round and always writes result.json, so it is run
# without an outer `timeout`; the +600s guard only catches a hung harness itself.
timeout $((${TIME_LIMIT_SECS:-14400} + 600)) \
  python "${AGENT_DIR}/harness/run_pi.py" "$@" 2>&1 | tee -a "${LOGS_DIR}/start.log"
