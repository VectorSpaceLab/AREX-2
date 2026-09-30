# Configuration

The repository wrapper keeps the common options small. Dataset-specific options
remain in `evaluation/research/eval_unified.py` and can be passed with repeated
`--extra` arguments.

## Common options

```text
evaluate DATASET [DATASET ...]
  --n / --num-tasks N       number of rows
  --start-index I           zero-based first row (default 0)
  --mode direct|refine_summary|return
  --model NAME              provider model identifier
  --api-key-env ENV_NAME    environment variable containing the key
  --base-url URL             OpenAI-compatible model endpoint
  --data-path PATH           override one selected dataset's input
  --data-root PATH           prepared data root (default: ./data/files)
  --save-path PATH           result root (default: runs/<timestamp>)
  --concurrency N            maximum concurrent cases (profile default: 1; otherwise 4)
  --profile auto|default|refine-equal
  --extra FLAG               evaluator-specific flag; repeat it
```

For example, the headline research datasets use the shared `refine-equal`
profile automatically:

```bash
python3 evaluate.py BrowseComp --start-index 100 --n 20 \
  --model YOUR_MODEL --base-url http://model.example/v1 \
  --tokenizer-path /path/to/tokenizer \
  --judge-model YOUR_JUDGE --judge-base-url http://judge.example/v1 \
  --judge-api-key-env JUDGE_API_KEY
```

`--api-key-env` is a variable name, not the secret. The adapter copies that
variable to the evaluator's private environment and never puts its value in the
command preview. `--model` and `--base-url` are safe to show.

Use `--dry-run` to inspect the exact subprocess command. It does not call a
model, search service, page reader, judge, or Docker. A real invocation checks
that the selected data exists before launching the evaluator.

## Environment variables

| Variable | Used by | Required |
| --- | --- | --- |
| `MODEL_API_KEY` (or the variable named by `--api-key-env`) | model SDK | every real research run |
| `AREX_MODEL_NAME` | default model | every real research run |
| `AREX_BASE_URL` | default model endpoint | when using a custom endpoint |
| `AREX_TOKENIZER_PATH` | local token counting | unified research datasets |
| `JUDGE_API_KEY` (or `--judge-api-key-env`) | external judge SDK | `refine-equal` profile |
| `SERPER_API_KEY` | `search`, `google_scholar` | research tools |
| `JINA_API_KEY` | `visit` | private or rate-limited Jina |
| `HF_TOKEN` | Hugging Face downloader | HLE and GAIA preparation |
| `MLE_BENCH` | MLE-bench Lite | MLE prepare/run |

Optional endpoint variables are `SERPER_API_URL`,
`SERPER_SCHOLAR_API_URL`, and `JINA_API_URL`. Keep real values in the local
ignored `.env` or a secret manager.

## Data paths

Downloadable datasets use `data/files/<dataset>/...`:

| Dataset | Default input |
| --- | --- |
| BrowseComp | `data/files/BrowseComp/browse_comp_test_set.csv` |
| DeepSearch-QA | `data/files/DeepSearch-QA/DSQA-full.csv` |
| HLE | `data/files/HLE/text_items.jsonl` |
| GAIA-2023-validation-text-103 | `data/files/GAIA-2023-validation-text-103/standardized_data.jsonl` |

When no `--data-root` is supplied, a matching legacy file under
`data/files/legacy/` is still accepted. Other evaluator datasets use the path
in `data/research/*/config.json`; inspect them with
`python3 evaluate.py download --list`.

`--data-path` is rejected for a multi-dataset command rather than applying one
file to every dataset.

## Shared research profile

`auto` selects `refine-equal` for BrowseComp, GAIA-2023-validation-text-103,
HLE, and DeepSearch-QA. The profile uses one concurrent case, ten outer rounds,
300 calls per round, a 1,500-call total cap, confidence-tiered review, shared
thinking/sampling/token/retry settings, and an external judge. Summary requests
use the inference model in the unified backend. HLE's dedicated solver also uses
the inference model, while its adapter owns context and review calls. Supply the judge explicitly with
`--judge-model`, `--judge-base-url`, and `--judge-api-key-env`; use
`--profile default` to opt out. HLE uses its dedicated adapter for context
truncation and judge calls while keeping the same shared generation and retry
values.

| Setting | Default |
| --- | --- |
| Concurrent cases / outer rounds | 1 / at most 10 |
| Model-call budget per round / per case | 300 / 1,500, including confidence review |
| Confidence thresholds | accept ≥ 95; middle tier ≥ 90 |
| Thinking / preserve thinking / summary thinking | enabled |
| Temperature / top-p / top-k / min-p | 1.0 / 0.95 / 20 / 0.0 |
| Presence / repetition penalty | 1.5 / 1.0 |
| Context / response / review tokens | 240,000 / 16,384 / 4,096 |
| Tool-call regeneration / logical-call attempts / request attempts | 20 / 5 / 10 |
| Whole-case attempts / timeout | 1 / 86,400 seconds |
| Context refresh trigger / maximum updates | 128,000 tokens / 24 (unified backend) |

BrowseComp, GAIA, and DeepSearch-QA run in `refine_summary` mode. HLE keeps its
dedicated solver, reviews the previous answer before each later round, and counts
that review against the same budget. Its truncation response limit is also
16,384; the adapter retains a 262,144-token per-request usage guard. Judge
protocols and their scoring-specific retries remain dataset-specific. HLE
intermediate rounds live in `_hle_outer/`; only the final case is copied into
`HLE/`, so an interrupted run can continue with the same `--save-path`.

## Advanced options and reruns

The underlying evaluator accepts options such as:

```bash
python3 evaluate.py HLE --n 20 \
  --extra=--hle-max-completion-tokens=32768 \
  --extra=--disable-visit-fallback
```

Use `python3 evaluation/research/eval_unified.py --help` for the full list. Set a
unique `--save-path` for each model, mode, and task range. The shared profile
skips all saved final cases; `--profile default` skips correct cases. To start
a fresh evaluation, use a new result directory. `--extra` overrides individual
settings and therefore changes the reported evaluation configuration.
