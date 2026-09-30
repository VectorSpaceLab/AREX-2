# MLE-bench Lite runners

These wrappers keep MLE-bench's four lifecycle steps visible at the checkout
level while delegating the actual harness and grader to
`evaluation/mle_lite/scripts/`:

| Script | Purpose |
| --- | --- |
| `run.sh COMPETITION` | Run one competition in the `pi` Docker image |
| `prepare.sh COMPETITION` | Prepare one competition; use `all` for the Lite split |
| `grade.sh RUN_DIR COMPETITION` | Grade a submission on the host with private labels |
| `summarize.sh RUNS_DIR` | Collect finished runs into a readable table |

The runner defaults are:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MLE_BENCH` | required by prepare/grade | Path to the mle-bench checkout |
| `DATA_DIR` | `$HOME/.cache/mle-bench/data` | Prepared competition data |
| `RUNS_DIR` | `<checkout>/runs/mle` | Output directory for agent runs |
| `TIME_LIMIT_SECS` | `14400` | Per-competition wall-clock limit |
| `GPUS` | `all` | Docker GPU selection; set `none` for CPU-only runs |
| `TAG` | `pi` | Docker image containing the agent |
| `PI_PROVIDER` / `PI_MODEL` | harness defaults | Model provider and model identifier |
| `PI_THINKING` / `PI_CRITERION` | harness defaults | Reasoning mode and criterion |
| `PI_PROVIDER_RETRIES` | `3` | Provider retry count |
| `REFINE_ROUNDS` | `0` | Additional refinement rounds |

The provider-specific key and endpoint variables are passed through to the
container by the maintained `run_one.sh` script. They include
`DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `KIMI_API_KEY`,
`CUSTOM_BASE_URL`, `CUSTOM_API_KEY`, and `CUSTOM_MODEL_ID`.

Examples:

```bash
export MLE_BENCH="$HOME/mle-bench"
export PI_PROVIDER=custom
export CUSTOM_BASE_URL='https://model.example/v1'
export CUSTOM_API_KEY='...'
export CUSTOM_MODEL_ID='your-model'

# Prepare only the selected competition.
scripts/mle/prepare.sh leaf-classification

# Inspect the resolved run settings and CLI command.
TIME_LIMIT_SECS=1800 scripts/mle/run.sh leaf-classification --dry-run

# Run and then grade the generated submission.
TIME_LIMIT_SECS=1800 scripts/mle/run.sh leaf-classification
scripts/mle/grade.sh runs/mle/<timestamp>_leaf-classification leaf-classification

# Summarize one run tree.
scripts/mle/summarize.sh runs/mle
```
