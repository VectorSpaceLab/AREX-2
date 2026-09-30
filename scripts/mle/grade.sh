#!/usr/bin/env bash
# Grade a finished MLE-bench Lite run on the host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

run_dir="${1:?Usage: scripts/mle/grade.sh RUN_DIR COMPETITION}"
competition="${2:?Usage: scripts/mle/grade.sh RUN_DIR COMPETITION}"
shift 2
exec bash "$MLE_LITE_SCRIPTS/grade.sh" "$run_dir" "$competition" "$@"
