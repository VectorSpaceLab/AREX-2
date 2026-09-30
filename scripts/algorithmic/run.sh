#!/usr/bin/env bash
# Run one Frontier-CS algorithmic problem through the checkout CLI.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREX_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

problem="${1:?Usage: scripts/algorithmic/run.sh PROBLEM SOLUTION [--backend docker|skypilot] [--judge-url URL] [--dry-run]}"
solution="${2:?Usage: scripts/algorithmic/run.sh PROBLEM SOLUTION [--backend docker|skypilot] [--judge-url URL] [--dry-run]}"
shift 2

backend="${FRONTIER_BACKEND:-docker}"
judge_url="${FRONTIER_JUDGE_URL:-}"
dry_run=0
extra=()
for option in "$@"; do
    if [[ "$option" == "--dry-run" ]]; then
        dry_run=1
    else
        extra+=("$option")
    fi
done

printf 'problem: %s\n' "$problem"
printf 'solution: %s\n' "$solution"
printf 'backend: %s\n' "$backend"
if [[ -n "$judge_url" ]]; then
    printf 'judge endpoint: %s\n' "$judge_url"
else
    printf 'judge endpoint: local backend default\n'
fi

command=(python3 "$AREX_REPO_ROOT/evaluate.py" algorithmic "$problem" "$solution" --backend "$backend")
[[ -n "$judge_url" ]] && command+=(--judge-url "$judge_url")
(( dry_run )) && command+=(--dry-run)
command+=("${extra[@]}")
(cd "$AREX_REPO_ROOT" && "${command[@]}")
