#!/usr/bin/env bash
# Summarize finished MLE-bench Lite runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
exec python3 "$MLE_LITE_SCRIPTS/summarize.py" "$@"
