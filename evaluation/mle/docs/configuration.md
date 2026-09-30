# Configuration reference

Everything is an environment variable. In mle-bench mode they come from the `env_vars:` block of
`agents/pi/config.yaml`; in standalone mode `scripts/run_one.sh` forwards them from your shell.

## Fixed by the container (do not override)

| Variable | Value | Meaning |
|---|---|---|
| `COMPETITION_ID` | e.g. `leaf-classification` | which competition this container solves |
| `DATA_DIR` | `/home/data` | competition data, read-only mount |
| `SUBMISSION_DIR` | `/home/submission` | `submission.csv` here is what gets graded |
| `LOGS_DIR` | `/home/logs` | trajectory, `pi.log`, `result.json` |
| `CODE_DIR` | `/home/code` | the agent's source is exported here at the end |
| `AGENT_DIR` | `/home/agent` | harness + skill library |

## Run shape

| Variable | Default | Meaning |
|---|---|---|
| `TIME_LIMIT_SECS` | `14400` | total wall clock for the agent, split evenly across rounds |
| `REFINE_ROUNDS` | `0` | extra rounds after the first; each inherits the previous workspace, cache, checkpoints, run log and self-diagnosis |
| `PI_PROVIDER` | `deepseek` | pi provider |
| `PI_MODEL` | `deepseek-v4-flash` | pi model id |
| `PI_THINKING` | `high` | pi reasoning effort |
| `PI_PROVIDER_RETRIES` | `3` | restarts when a round dies in seconds having emitted zero tokens (endpoint 5xx) |

## What the agent is told

| Variable | Default | Meaning |
|---|---|---|
| `PI_NO_SKILL` | unset | `1` drops the task skill: the agent designs its own approach (ablation baseline) |
| `SKILLS_DIR` | `skills` | directory under `AGENT_DIR` to read `SKILL.md` from |
| `PI_CRITERION` | `medal` | how much of the grading scale is disclosed — `medal` (all thresholds, bronze is the pass line), `median` (median line only), `none` (nothing). See [differences.md](differences.md#the-pass-line-is-disclosed-by-default) |
| `PI_GPU_NOTE` | a sizing hint | replaces the GPU/fold-budget guidance in the prompt |
| `HARDWARE` | detected via `nvidia-smi` | reported to the agent so it sizes training to the real machine |

## Credentials

Whatever your provider needs — `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`KIMI_API_KEY`. For a self-hosted endpoint (a vLLM/SGLang server serving your own fine-tune),
set `CUSTOM_BASE_URL`, `CUSTOM_API_KEY`, `CUSTOM_MODEL_ID` and run with
`PI_PROVIDER=custom-openai PI_MODEL=<your model id>`; `agents/pi/models.json.template` wires
those into pi's model registry. Keys are passed as `$VAR` references and never written to disk.
