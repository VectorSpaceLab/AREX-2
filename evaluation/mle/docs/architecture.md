# How a run works

One container solves one competition. Nothing about the run depends on the host beyond the data
mount and an API key.

```
mle-bench container (image `pi`, built FROM mlebench-env)
├── /home/data                     competition data, read-only
├── /home/agent
│   ├── harness/run_pi.py          the harness (everything below is it)
│   ├── skills/<comp>/SKILL.md     22 task skills
│   ├── tasks.json                 leaderboard thresholds per competition
│   └── workspace/                 the agent's cwd
│       ├── input -> /home/data    the only path the agent is told about
│       ├── CONTEXT.md             assembled task brief
│       ├── criterion.json         the pass line
│       ├── skills/<comp>/SKILL.md the skill, also on disk for the agent to re-read
│       ├── cache/                 agent-managed, survives into refine rounds
│       └── submission.csv         synced out every 60s
├── /home/submission/submission.csv   ← graded
├── /home/logs/traj/round<N>/*.jsonl  ← full trajectory
├── /home/logs/{pi.log,result.json}
└── /home/code                     the agent's source, exported at the end
```

## 1. Assemble `CONTEXT.md`

Four sections, built from the mount and the bundled metadata:

1. **Objective & constraints** — where the data is (`./input`, read-only), the exact
   sample-submission filename to match, the interpreter, the hardware, and a pointer to
   `./criterion.json`.
2. **How to solve** — the task skill verbatim, framed as "reproduce this recipe, do not invent a
   different approach". Omitted under `PI_NO_SKILL=1`.
3. **Data card** — headers and one example row of `train.csv` / `test.csv` / the sample submission,
   plus the first 6 KB of the competition's `description.md`.
4. **Previous attempt** — only in a refine round: a pointer to `./prev_run.log` and the previous
   round's closing self-diagnosis.

`criterion.json` carries the pass rule, the direction of the metric (inferred from the thresholds,
so it is correct for both "higher is better" and "lower is better" competitions), and the target
value. `PI_CRITERION` controls how much of it is real — see
[differences.md](differences.md#the-pass-line-is-disclosed-by-default).

## 2. Run `pi`

```
pi --provider $PI_PROVIDER --model $PI_MODEL --thinking $PI_THINKING \
   --session-dir /home/logs/traj/round<N> -nc -p "<prompt>"
```

The agent has `read` / `bash` / `edit` / `write` and **no web tool** — it cannot look up a
solution, which is what keeps a reproduction honest. `bash` has no per-command timeout, so
long foreground training is never killed mid-epoch; the wall clock is enforced from outside by
the harness, which SIGKILLs the process group at the budget.

The prompt is a fixed five-point contract, and each point exists because of a failure we hit:

1. **Baseline first** — a complete, valid `submission.csv` from the cheapest possible model within
   ~15 minutes, before any heavy training. Without this, a run that overruns on a long training
   step scores zero instead of scoring badly.
2. **Cache under `./cache/`** — features, tokenized data, checkpoints; explicitly *not*
   predictions or OOF arrays. This is what makes a refine round cheap.
3. **No label lookup** — every prediction must come from a model trained on `./input`. Joining
   test ids against an embedded label field is void.
4. **Size training to the budget** — cut folds and epochs rather than forfeit the submission.
5. **Close with a `[Diagnosis]`** — method, self-estimated score and gap to the pass line,
   bottleneck, next change. This is what a refine round reads.

A round that dies within 10 minutes having produced **zero** assistant tokens is treated as a
provider outage, not a result: the harness backs off (60s, 120s, 240s) and reopens the round with
the remaining wall clock.

## 3. Refine rounds (optional)

With `REFINE_ROUNDS=N` the wall clock is split into `N+1` equal rounds in the same workspace. Each
subsequent round gets the previous round's code, `./cache/`, checkpoints, a rendered transcript at
`./prev_run.log`, and its closing `[Diagnosis]`, plus a prompt that forbids producing a submission
by perturbing the previous one — it has to retrain.

## 4. Results

`submission.csv` is copied to `/home/submission/` every 60 seconds and once at the end, so a run
killed at the wall still submits its best-so-far file.

`/home/logs/result.json` records the model, criterion mode, per-round exit codes and token counts,
wall time, whether a submission exists, and the trajectory filenames. **No score** — the harness
never grades, so the container is never given the private labels. Grading is
`mlebench grade-sample` on the host (`scripts/grade.sh`).

## The trajectory

`/home/logs/traj/round<N>/*.jsonl` is pi's session log: one JSON object per line, the interesting
ones being `{"type":"message","message":{...}}` with content blocks of type `thinking` (the
chain of thought), `toolCall` (`name` + `arguments`), and `text` under `role: "toolResult"`, plus
per-turn `usage`. It is the complete, replayable record of the attempt — and it is the reason this
harness exists: the trajectories are training data as well as an audit trail.
