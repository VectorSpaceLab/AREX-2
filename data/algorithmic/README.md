# Frontier-CS algorithmic track

This is the data contract for the algorithmic benchmark. The judge service code
lives in `evaluation/frontier/judge/`; the downloaded problems and run artifacts stay in
this directory so they are easy to inspect and can remain ignored by Git.

## Evaluation protocol

1. A problem is selected by its numeric ID.
2. The model (or a user) supplies one C++17 source file.
3. Frontier-CS starts the local Docker judge, mounts `problems/<id>`, and submits
   the source.
4. The checker runs every required test case and returns per-case status and a
   score ratio. The aggregate fields are `scoreRatio` and
   `scoreRatioUnbounded`; keep both in the result record.

A submission that fails compilation, exceeds a limit, or fails a required case
is represented in the checker result. It must not be silently converted to a
research-style pass rate.

## Files

A downloaded problem normally looks like this:

```text
problems/<id>/
├── statement.txt
├── config.yaml          # time/memory limits and test count
├── testdata/            # .in/.ans cases
└── chk.cc or interactor.cc
```

`solutions/`, `submissions/`, and `data/` are runtime directories. They are
ignored by Git and are mounted into the judge container.

## Run

```bash
python3 evaluate.py download algorithmic
python3 evaluate.py algorithmic 1 path/to/solution.cpp --backend docker
```

The same command can use a running remote judge with `--judge-url`, or launch
the SkyPilot configuration at `evaluation/frontier/judge/sky-judge.yaml` with
`--backend skypilot`. The preparation script accepts a pinned archive URL and
checksum:

```bash
python3 scripts/download_algorithmic.py \
  --url https://mirror.example/frontier-cs.tar.gz \
  --sha256 SHA256_HEX
```
