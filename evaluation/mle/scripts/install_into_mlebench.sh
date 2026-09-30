#!/usr/bin/env bash
# Register this agent with an mle-bench checkout so `run_agent.py --agent-id pi` finds it.
# Copies agents/pi/ (the whole self-contained agent, ~2 MB of skills + harness) into
# <mle-bench>/agents/pi/. Re-run after changing a skill or the harness.
#
#   MLE_BENCH=~/mle-bench scripts/install_into_mlebench.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MLE_BENCH="${MLE_BENCH:?set MLE_BENCH to your mle-bench checkout}"
[ -f "$MLE_BENCH/run_agent.py" ] || { echo "not an mle-bench checkout: $MLE_BENCH" >&2; exit 1; }

rm -rf "$MLE_BENCH/agents/pi"
cp -r agents/pi "$MLE_BENCH/agents/pi"
cp splits/lite.txt "$MLE_BENCH/experiments/splits/lite.txt"
echo "installed -> $MLE_BENCH/agents/pi  (+ experiments/splits/lite.txt)"
echo
echo "next:"
echo "  cd $MLE_BENCH"
echo "  docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile ."
echo "  docker build --platform=linux/amd64 -t pi agents/pi/ \\"
echo "    --build-arg SUBMISSION_DIR=/home/submission --build-arg LOGS_DIR=/home/logs \\"
echo "    --build-arg CODE_DIR=/home/code --build-arg AGENT_DIR=/home/agent"
echo "  DEEPSEEK_API_KEY=... python run_agent.py --agent-id pi \\"
echo "    --competition-set experiments/splits/lite.txt"
