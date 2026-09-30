# Evaluation internals

The public entrypoint is the root-level `evaluate.py`. It resolves a dataset,
checks the prepared input, and dispatches to the matching evaluator. The
directories below separate checkout glue, benchmark implementations, and
third-party snapshots.

```text
evaluation/
├── arex_eval/            root CLI, dataset catalog, profiles, and dispatch
├── research/             unified research evaluator and HLE adapter
├── frontier/
│   ├── source/            Frontier-CS Python evaluator
│   ├── judge/             local Docker/SkyPilot judge service
│   └── profiles/          Harbor/Pi experiment profiles
├── mle/                  MLE-bench Lite agent, harness, and host grader
└── vendor/harbor/        pinned Harbor source used by Frontier experiments
```

Dataset contracts stay in `data/`: prompts, loaders, judge selection, and
input preparation notes are kept next to each dataset. User-facing launchers
stay in `scripts/`, while `evaluation/` contains the implementation they call.

The normal commands are deliberately short:

```bash
python3 evaluate.py BrowseComp --n 10
python3 evaluate.py algorithmic 1 path/to/solution.cpp --backend docker
python3 evaluate.py mle leaf-classification
```

Use [data/README.md](../data/README.md) for dataset-specific contracts and
[the evaluation guide](../docs/evaluation/evaluation.md) for result recording.
Pinned upstream versions and licenses are listed in the repository-level
`SNAPSHOT.txt` and `THIRD_PARTY_NOTICES.md`.
