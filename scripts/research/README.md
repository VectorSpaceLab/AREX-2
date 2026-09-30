# Research runners

Each script in this directory evaluates one research benchmark through the
repository-level `evaluate.py` entrypoint. The wrappers select the dataset;
the evaluator keeps the prompts, tools, scoring contracts, and run metadata.

The four scripts use the same `refine-equal` profile. Its fixed settings are:

| Setting | Value |
| --- | ---: |
| Concurrent cases | `1` |
| Outer rounds | `10` |
| Calls per outer round | `300` |
| Total calls per case | `1500` |
| Confidence review | tiered, thresholds `95 / 90` |
| Context / response tokens | `240000 / 16384` |
| Temperature | `1.0` |
| Top-p / top-k / min-p | `0.95 / 20 / 0.0` |
| Presence / repetition penalty | `1.5 / 1.0` |
| Tool-call retries | `20` |
| LLM retries | `5` |
| General attempts / case attempts | `10 / 1` |

The inference model, endpoint, judge, task range, data root, and output path
are supplied by environment variables. The defaults make `--dry-run` useful
without credentials:

| Variable | Default | Meaning |
| --- | --- | --- |
| `AREX_MODEL_NAME` | `YOUR_MODEL_NAME` | Inference model identifier |
| `AREX_BASE_URL` | empty | OpenAI-compatible inference endpoint |
| `AREX_API_KEY_ENV` | `MODEL_API_KEY` | Name of the environment variable holding the model key |
| `AREX_TOKENIZER_PATH` | empty | Inference model tokenizer path (required for non-HLE runs) |
| `AREX_JUDGE_MODEL` | `YOUR_JUDGE_MODEL` | External judge model identifier |
| `AREX_JUDGE_BASE_URL` | `http://judge.example/v1` | External judge endpoint |
| `AREX_JUDGE_API_KEY_ENV` | `JUDGE_API_KEY` | Name of the environment variable holding the judge key |
| `NUM_TASKS` | `10` | Number of rows to evaluate |
| `START_INDEX` | `0` | First zero-based row |
| `SAVE_PATH` | `runs/<dataset>-<start>-<count>` | Run output directory |
| `AREX_DATA_ROOT` | evaluator default | Prepared benchmark data root |
| `CONCURRENCY` | `1` | Maximum concurrent cases; normally leave unchanged |

For a real run, export the model and judge key variables named above. Keys
are read by name and are never placed in command-line arguments or output
metadata.

Examples:

```bash
export AREX_MODEL_NAME='your-model'
export AREX_BASE_URL='https://model.example/v1'
export AREX_TOKENIZER_PATH='/models/your-model'
export MODEL_API_KEY='...'
export AREX_JUDGE_MODEL='your-judge'
export AREX_JUDGE_BASE_URL='https://judge.example/v1'
export JUDGE_API_KEY='...'

# Inspect the fully expanded evaluator command first.
scripts/research/browsecomp.sh --dry-run

# Evaluate the first 10 BrowseComp rows.
scripts/research/browsecomp.sh

# Select another benchmark; the same settings apply.
NUM_TASKS=50 START_INDEX=100 scripts/research/hle.sh
```

Use repeated `--extra VALUE` options only for evaluator tokens that are
intentionally outside the shared profile. The scripts do not duplicate
evaluator logic, so changes to the maintained profile automatically reach all
four benchmarks.
