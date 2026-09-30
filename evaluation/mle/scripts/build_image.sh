#!/usr/bin/env bash
# Build the `pi` agent image. Requires mle-bench's base image `mlebench-env` to exist:
#   cd <mle-bench> && docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile .
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${TAG:-pi}"
BASE_IMAGE="${BASE_IMAGE:-mlebench-env}"

if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "error: base image '$BASE_IMAGE' not found." >&2
  echo "Build it first inside your mle-bench checkout:" >&2
  echo "  docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile ." >&2
  exit 1
fi

docker build --platform=linux/amd64 -t "$TAG" agents/pi/ \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg SUBMISSION_DIR=/home/submission \
  --build-arg LOGS_DIR=/home/logs \
  --build-arg CODE_DIR=/home/code \
  --build-arg AGENT_DIR=/home/agent \
  "$@"
echo "built image: $TAG"
