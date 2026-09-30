# Experiments and reproducibility

This directory holds commands that prepare data and run auxiliary experiments. The evaluator itself stays in `evaluation/`; dataset contracts stay in `data/`.

```text
configs/       safe local environment template
doctor.py      repository diagnostics
download_*.py  data preparation entrypoints
algorithmic/   Frontier-CS evaluation, solution-generation, and Pi runners
research/      one runner per research benchmark
mle/           MLE-bench Lite prepare, run, grade, and summary wrappers
```

The four headline research datasets have dedicated wrappers in
`research/`. Each wrapper selects one dataset and the shared `refine-equal`
profile, so the command exposes the benchmark choice while keeping the
evaluation settings in one maintained place. See
[`research/README.md`](research/README.md) for the fixed settings and the
model, endpoint, judge, task-range, and output variables.

The normal research path is still short:

```bash
python3 evaluate.py DATASET --n 10 --save-path runs/example
```

Use `--dry-run` first. A research run writes the model name, endpoint, Git
commit, task range, evaluator mode, and data checksum to `run_metadata.json`
and every case result. API keys belong in the shell environment and never in
command-line arguments or result files.

Compile the maintained Python entrypoints from the root:

```bash
python3 -m compileall -q evaluation/arex_eval evaluation/research/unified_eval scripts
```
