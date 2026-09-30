# Evaluation and scoring

Select a dataset at the repository boundary and let the wrapper choose its
bundled evaluator:

```bash
python3 evaluate.py list
python3 evaluate.py BrowseComp --n 1 --dry-run
python3 evaluate.py BrowseComp --n 10 --save-path runs/browsecomp-10
```

`--start-index I --n N` evaluates rows in `[I, I+N)`. The wrapper uses one
result directory per selected dataset. It checks prepared paths before starting
a real run; dry-run only prints the constructed subprocess command.

## Research datasets

Every case writes a `result.json`. Read `status`, `official_scorer`,
`judge_raw`, and error fields before aggregating `score_result.score`.
Missing credentials or missing data are configuration errors and should not be
counted as model failures.

| Dataset | Input and scorer |
| --- | --- |
| BrowseComp | Encrypted CSV; BrowseComp official judge |
| DeepSearch-QA | CSV; official Gemini-autorater style scorer |
| HLE | Text-only HLE JSONL; HLE judge, with `metrics.full_credit` |
| GAIA-2023-validation-text-103 | Text-only GAIA JSONL; WebAgent-style text judge |

The four datasets with a repository download recipe are prepared with:

```bash
python3 evaluate.py download BrowseComp
python3 evaluate.py download DeepSearch-QA
python3 evaluate.py download HLE                 # requires HF_TOKEN and access
python3 evaluate.py download GAIA-2023-validation-text-103  # requires HF_TOKEN
python3 evaluate.py download --list
```

## Frontier-CS algorithmic

Download the public problem archive and evaluate a C++17 solution:

```bash
python3 scripts/download_data.py --dataset algorithmic
python3 evaluate.py algorithmic 1 path/to/solution.cpp --backend docker
```

The checker reports case scores such as `scoreRatio` and
`scoreRatioUnbounded`. Keep the complete checker and Docker log with the
result. A solution passes only when all required cases pass.

## MLE-bench Lite

Prepare and run one competition, then use the host grader:

```bash
MLE_BENCH=$HOME/mle-bench python3 evaluate.py mle leaf-classification --prepare
MLE_BENCH=$HOME/mle-bench python3 evaluate.py mle leaf-classification
MLE_BENCH=$HOME/mle-bench \
  bash scripts/mle/grade.sh runs/<run-dir> leaf-classification
```

Use `score`, `valid_submission`, and medal/threshold fields from
`grade.log` as the final metric. Container `/validate` output is diagnostic.

## Reproducibility

Record the Git commit, model, base URL, task range, and evaluator mode, plus the
data checksum beside each result root. Research runs write `run_metadata.json`
under each dataset result root and include the same record in every case JSON.
The record includes the explicit `git_commit`, `model`, `endpoint`,
`task_range`, `evaluator_mode`, and `data_checksum` fields. Never record API key
values. Upstream commits and licenses are listed in `SNAPSHOT.txt` and
`THIRD_PARTY_NOTICES.md`.
