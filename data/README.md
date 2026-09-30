# Dataset evaluation guide

This directory is the contract for every selectable benchmark. The outer command is always:

```bash
python3 evaluate.py DATASET --n N
```

The wrapper resolves the matching directory below, checks the prepared input, and starts the unified research evaluator. A dataset directory is deliberately self-contained: `config.json` names the fields and scorer, `prompt.py` contains the model-facing prompt and tools, and `judge_local.py` / `judge_offical.py` contain optional judge implementations.

`--dry-run` prints the exact subprocess command without requiring data, a model key, or a tokenizer. A real run writes one result JSON per task. Check `status` and the raw judge fields before computing an average; missing data and credentials are setup errors, not zero scores.

## Research datasets

| Dataset | Loader | Input | Scoring protocol | Dataset files |
| --- | --- | --- | --- | --- |
| [BrowseComp](research/BrowseComp/) | `csv` | CSV encrypted question/answer rows | BrowseComp official judge after canary/decryption checks | `prompt.py, judge_local.py, judge_offical.py` |
| [DeepSearch-QA](research/DeepSearch-QA/) | `csv` | CSV question/answer rows | DeepSearchQA official Gemini-autorater style judge | `prompt.py, judge_local.py, judge_offical.py` |
| [GAIA-2023-validation-text-103](research/GAIA-2023-validation-text-103/) | `jsonl` | JSONL text-only GAIA rows | GAIA validation text rubric / WebAgent LLM judge | `prompt.py, judge_local.py, judge_offical.py` |
| [HLE](research/HLE/) | `jsonl` | JSONL text-only HLE rows (image rows excluded) | HLE judge adapter | `prompt.py, judge_local.py, judge_offical.py` |

## Data locations

The default data root is `data/files/`. Downloadable files are placed in a named subdirectory there. Benchmarks without a publisher recipe use `data/files/external/…`; obtain them under their own license and pass `--data-path` when the location differs. The catalog in [catalog.json](catalog.json) owns the four download recipes and is also used by `python3 evaluate.py download --list`.

```bash
python3 evaluate.py download --list
python3 evaluate.py download BrowseComp
python3 evaluate.py download DeepSearch-QA
python3 evaluate.py download HLE
python3 evaluate.py download GAIA-2023-validation-text-103
```

## Other tracks

- [Frontier-CS algorithmic](algorithmic/README.md): C++17 submissions are sent to the checker for every test case. The result is the checker score (`scoreRatio`, with the unbounded value retained).
- MLE-bench Lite: `python3 evaluate.py mle COMPETITION`; the host grader in `scripts/mle/grade.sh` supplies the final competition score.

The research evaluator implementation is in `evaluation/research/`; it is kept separate from these dataset contracts so changing a runner does not hide how a benchmark is scored.
