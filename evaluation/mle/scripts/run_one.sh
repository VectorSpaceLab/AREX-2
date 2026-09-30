#!/usr/bin/env bash
# Run ONE competition in a plain Docker container -- no sysbox, no mle-bench run_agent.py.
# This is the quickest way to try the harness; grading happens afterwards on the host with
# scripts/grade.sh, so the container never sees the private test labels.
#
#   DEEPSEEK_API_KEY=sk-... scripts/run_one.sh leaf-classification
#   TIME_LIMIT_SECS=1800 PI_CRITERION=none scripts/run_one.sh leaf-classification
#
# Env:
#   DATA_DIR         prepared mle-bench data root (default ~/.cache/mle-bench/data)
#   RUNS_DIR         where to write the run (default ./runs)
#   TIME_LIMIT_SECS  wall clock for the agent (default 14400)
#   GPUS             docker --gpus value, or "none" (default all)
#   plus any PI_* / REFINE_ROUNDS / *_API_KEY variable the harness understands
set -euo pipefail
cd "$(dirname "$0")/.."

COMP="${1:?usage: scripts/run_one.sh <competition-id>}"
TAG="${TAG:-pi}"
DATA_DIR="${DATA_DIR:-$HOME/.cache/mle-bench/data}"
RUNS_DIR="${RUNS_DIR:-$(pwd)/runs}"
TIME_LIMIT_SECS="${TIME_LIMIT_SECS:-14400}"
GPUS="${GPUS:-all}"

PUBLIC="$DATA_DIR/$COMP/prepared/public"
[ -d "$PUBLIC" ] || { echo "no prepared data at $PUBLIC -- run scripts/prepare_data.sh first" >&2; exit 1; }

RUN="$RUNS_DIR/$(date +%Y-%m-%dT%H-%M-%S)_${COMP}"
mkdir -p "$RUN"/{logs,code,submission}
echo "run dir: $RUN"

gpu_args=()
[ "$GPUS" != "none" ] && gpu_args=(--gpus "$GPUS")

env_args=()
for v in TIME_LIMIT_SECS PI_PROVIDER PI_MODEL PI_THINKING PI_CRITERION PI_NO_SKILL \
         PI_GPU_NOTE PI_PROVIDER_RETRIES REFINE_ROUNDS SKILLS_DIR \
         DEEPSEEK_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY KIMI_API_KEY \
         CUSTOM_BASE_URL CUSTOM_API_KEY CUSTOM_MODEL_ID; do
  [ -n "${!v:-}" ] && env_args+=(-e "$v=${!v}")
done

# --entrypoint bash bypasses mle-bench's entrypoint (which would start the grading server and
# wait for a private-data mount). Consequence: the in-container /validate endpoint is absent,
# so result.json records validation as unavailable. Grading is unaffected -- it runs on the host.
docker run --rm "${gpu_args[@]}" --shm-size=16g \
  -v "$(cd "$PUBLIC" && pwd):/home/data:ro" \
  -v "$RUN/logs:/home/logs" \
  -v "$RUN/code:/home/code" \
  -v "$RUN/submission:/home/submission" \
  -e "COMPETITION_ID=$COMP" "${env_args[@]}" \
  --entrypoint bash "$TAG" -c 'bash /home/agent/start.sh'

echo
echo "submission: $RUN/submission/submission.csv"
echo "trajectory: $RUN/logs/traj/"
echo "grade it:   MLE_BENCH=... scripts/grade.sh $RUN $COMP"
