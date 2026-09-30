#!/usr/bin/env bash
# Prepare the 22 MLE-bench Lite competitions with mle-bench's own tooling.
# Needs the Kaggle CLI authenticated (~/.kaggle/kaggle.json) and ~400 GB of disk.
#
#   MLE_BENCH=~/mle-bench scripts/prepare_data.sh                       # all 22
#   MLE_BENCH=~/mle-bench ONLY=leaf-classification scripts/prepare_data.sh   # just one
set -euo pipefail
cd "$(dirname "$0")/.."
SPLIT="$(pwd)/splits/lite.txt"

MLE_BENCH="${MLE_BENCH:?set MLE_BENCH to your mle-bench checkout}"
DATA_DIR="${DATA_DIR:-$HOME/.cache/mle-bench/data}"

cd "$MLE_BENCH"
# `--lite` is mle-bench's own name for exactly these 22 low-complexity competitions;
# splits/lite.txt in this repo is the same list, kept so you can prepare a subset.
if [ "${ONLY:-}" = "" ]; then
  mlebench prepare --lite --data-dir "$DATA_DIR"
else
  printf "%s\n" $ONLY > /tmp/mle-lite-subset.txt
  mlebench prepare --list /tmp/mle-lite-subset.txt --data-dir "$DATA_DIR"
fi
echo "prepared into $DATA_DIR"
