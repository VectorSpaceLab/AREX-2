#!/usr/bin/env bash
# Run one MLE-bench Lite competition with the checked-in pi harness.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

competition="${1:?Usage: scripts/mle/run.sh COMPETITION [--dry-run]}"
shift
dry_run=0
extra=()
for option in "$@"; do
    if [[ "$option" == "--dry-run" ]]; then
        dry_run=1
    else
        extra+=("$option")
    fi
done

if (( dry_run )); then
    print_mle_config "$competition"
fi

command=(python3 "$AREX_REPO_ROOT/evaluate.py" mle "$competition" --time-limit "$TIME_LIMIT_SECS")
(( dry_run )) && command+=(--dry-run)
command+=("${extra[@]}")
(cd "$AREX_REPO_ROOT" && "${command[@]}")
