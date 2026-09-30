# Research evaluator

`eval_unified.py` is the shared runner for BrowseComp, GAIA, HLE, and
DeepSearch-QA. The `unified_eval/` package contains common loaders, tools, and
scorers; `hle_vendor/` contains the HLE-specific adapter and its pinned helper
code.

Dataset prompts and judge contracts are intentionally kept in
`data/research/<dataset>/`. The root command injects the selected data paths
and profile settings before starting this runner. Run the public command from
the repository root:

```bash
python3 evaluate.py BrowseComp --n 10 --dry-run
```

Direct invocation is useful for inspecting advanced evaluator flags, but it is
an implementation detail:

```bash
python3 evaluation/research/eval_unified.py --help
```
