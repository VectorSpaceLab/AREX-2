#!/usr/bin/env bash
# Shared path and configuration helpers for the MLE-bench Lite wrappers.

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Source this file from one of the MLE wrappers in this directory.\n' >&2
    exit 2
fi

MLE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREX_REPO_ROOT="$(cd "$MLE_SCRIPT_DIR/../.." && pwd)"
MLE_LITE_ROOT="$AREX_REPO_ROOT/evaluation/mle"
MLE_LITE_SCRIPTS="$MLE_LITE_ROOT/scripts"

export DATA_DIR="${DATA_DIR:-$HOME/.cache/mle-bench/data}"
export RUNS_DIR="${RUNS_DIR:-$AREX_REPO_ROOT/runs/mle}"
export TIME_LIMIT_SECS="${TIME_LIMIT_SECS:-14400}"
export GPUS="${GPUS:-all}"
export TAG="${TAG:-pi}"

print_mle_config() {
    local competition="$1"
    printf 'competition: %s\n' "$competition"
    printf 'time limit: %s seconds\n' "$TIME_LIMIT_SECS"
    printf 'data dir: %s\n' "$DATA_DIR"
    printf 'runs dir: %s\n' "$RUNS_DIR"
    printf 'docker image: %s\n' "$TAG"
    printf 'GPUS: %s\n' "$GPUS"
    printf 'provider/model/thinking/criterion: %s / %s / %s / %s\n' \
        "${PI_PROVIDER:-default}" "${PI_MODEL:-default}" \
        "${PI_THINKING:-default}" "${PI_CRITERION:-default}"
    printf 'refine rounds/provider retries: %s / %s\n' \
        "${REFINE_ROUNDS:-0}" "${PI_PROVIDER_RETRIES:-3}"
}
