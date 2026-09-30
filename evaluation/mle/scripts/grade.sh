#!/usr/bin/env bash
# Grade one finished run with mle-bench's own grader (this is what touches the private labels).
#
#   MLE_BENCH=~/mle-bench scripts/grade.sh runs/2026-09-28T…_leaf-classification leaf-classification
set -euo pipefail
RUN="${1:?usage: scripts/grade.sh <run-dir> <competition-id>}"
COMP="${2:?usage: scripts/grade.sh <run-dir> <competition-id>}"
MLE_BENCH="${MLE_BENCH:?set MLE_BENCH to your mle-bench checkout}"
DATA_DIR="${DATA_DIR:-$HOME/.cache/mle-bench/data}"

SUB="$RUN/submission/submission.csv"
[ -f "$SUB" ] || { echo "no submission at $SUB" >&2; exit 1; }

cd "$MLE_BENCH"
mlebench grade-sample "$(cd "$(dirname "$SUB")" && pwd)/$(basename "$SUB")" "$COMP" \
  --data-dir "$DATA_DIR" | tee "$RUN/grade.log"
