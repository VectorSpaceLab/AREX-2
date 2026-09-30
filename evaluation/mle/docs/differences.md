# What this repo is, and how it differs from how we actually run it

This repo is the **evaluation-shaped** version of our harness: one container, one competition,
fixed environment, fixed input, a submission and a trajectory out. That is deliberately *not*
how we run it internally, where the same agent+skills stack is a data-generation fleet. If you
are comparing numbers with us, these are the differences that matter.

## Aligned with official MLE-bench

* The image is built `FROM mlebench-env`, so the Python/CUDA/package environment is MLE-bench's
  own — unmodified: `agents/pi/requirements.txt` is deliberately empty, because every package the
  bundled skills import is already pinned in the base image.
* Data is the official `mlebench prepare` output, mounted read-only at `/home/data`.
* The deliverable is `/home/submission/submission.csv`, graded by `mlebench grade-sample`.
* The container never sees the private test labels; grading happens on the host.
* The competition set is MLE-bench **Lite** — the 22 low-complexity competitions, identical to
  `mlebench prepare --lite`.

## Deliberate deviations

### A task skill is injected

The headline configuration hands the agent a `SKILL.md` distilled from a strong public solution to
that exact competition. That is the point of the harness — it measures *"can the agent execute a
known-good recipe under a wall clock"*, not *"can the agent invent a solution"*. It is **not**
comparable to a published MLE-bench number for a bare agent. For the bare-agent comparison run
`--agent-id pi/no-skill` (or `PI_NO_SKILL=1`).

Skills shipped here are the **de-bound** variant: wording that names the source solution's
standing ("the silver solution did…", "gold threshold ≤ …") has been rewritten, and every
reference to a score line points at `criterion.json` instead. Our internal copy is not de-bound.

### The pass line is disclosed by default

`PI_CRITERION=medal` writes the competition's gold/silver/bronze/median leaderboard thresholds
into `criterion.json` and tells the agent bronze is the pass line. Official MLE-bench tells the
agent nothing about the grading scale. We disclose it because our runs are searching for a
medal-grade trajectory and the agent needs a stopping rule.

**If you want a number comparable with published MLE-bench results, run with
`PI_CRITERION=none`** (or `--agent-id pi/blind`). Keep `medal` only when reproducing our setup.

### No web access

pi is configured without a web tool. The agent cannot look up the competition's solutions. This is
stricter than most published agent setups.

### What we run internally and left out

| Internal | Why it is not here |
|---|---|
| Multi-round `campaign` loop with skill rewriting between rounds | The round-to-round part is here as `REFINE_ROUNDS`; the part that edits the *skill* based on failures is data generation, not evaluation |
| Data-reduction task variants (train subsampled to a fraction, graded on the unchanged test set) | A task-augmentation device for making more training tasks out of 22 competitions — meaningless as a benchmark |
| Cluster dispatch, relay queue, a run database, dashboards, node self-healing | Fleet plumbing, all of it specific to our machines |
| Trajectory scrubbing (rewriting absolute host paths out of the captured trajectory) | Only needed because our runs happen on heterogeneous bare-metal workers. Inside this container every path is already `/home/...` and identical for everyone, so there is nothing to scrub |
| Trajectory → SFT cleaning (process gates, leakage audit, tail trimming) | Downstream of this repo; the trajectories it produces are the input to that |
| Our internal model endpoints | Stripped. `agents/pi/models.json.template` has a generic OpenAI-compatible slot for your own endpoint instead |

### Smaller things

* We run many competitions concurrently on shared GPUs, so our internal prompt forbids the agent
  from touching the shared Python environment. Here each competition gets its own container, so
  the agent is told it may install what it likes.
* Our internal runs use several seeds per competition and keep the best; `run_agent.py --n-seeds`
  does the same thing here, but the default is one.
* Our wall clock is usually 1–2 hours per round because we are buying trajectories in bulk. The
  default here is 4 hours, closer to what an evaluation run should give an agent.
