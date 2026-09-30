#!/usr/bin/env bash
# Prepare one competition, or all 22 MLE-bench Lite competitions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

competition="${1:-all}"
shift || true
dry_run=0
extra=()
for option in "$@"; do
    if [[ "$option" == "--dry-run" ]]; then
        dry_run=1
    else
        extra+=("$option")
    fi
done

if [[ "$competition" == "all" ]]; then
    if (( dry_run )); then
        printf '$ MLE_BENCH=%q DATA_DIR=%q bash %q\n' \
            "${MLE_BENCH:-}" "$DATA_DIR" "$MLE_LITE_SCRIPTS/prepare_data.sh"
        exit 0
    fi
    exec bash "$MLE_LITE_SCRIPTS/prepare_data.sh" "${extra[@]}"
fi

command=(python3 "$AREX_REPO_ROOT/evaluate.py" mle "$competition" --prepare)
(( dry_run )) && command+=(--dry-run)
command+=("${extra[@]}")
(cd "$AREX_REPO_ROOT" && "${command[@]}")
