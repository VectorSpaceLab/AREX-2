# mle-lite-pi-harness

An MLE-bench **Lite** harness for the [`pi`](https://www.npmjs.com/package/@earendil-works/pi-coding-agent)
coding agent, with a library of per-competition **task skills**.

One container solves one competition: fixed environment, read-only data at `/home/data`, a
`submission.csv` at `/home/submission/`, and the agent's **full trajectory** (chain of thought,
tool calls, tool results) captured on the way out. The environment is MLE-bench's own
`mlebench-env` image and the data is MLE-bench's own `prepare` output, so a score from here is a
score on the benchmark — with the deviations listed in [docs/differences.md](docs/differences.md),
which you should read before comparing numbers with anyone.

What makes it interesting is section [2] of the task brief: for each of the 22 Lite competitions
the agent is handed a `SKILL.md` distilled from a strong public solution, and told to reproduce
that recipe rather than invent one. The harness measures execution under a wall clock, not
invention. `PI_NO_SKILL=1` turns the skill off for the bare-agent baseline.

```
agents/pi/                  a drop-in mle-bench agent, self-contained
├── Dockerfile              FROM mlebench-env + Node 22 + pi + the harness
├── start.sh                container entry point
├── config.yaml             mle-bench registry entries (pi, pi/blind, pi/no-skill, pi/refine, pi/dev)
├── requirements.txt        extra python packages (empty by default: the base env has them)
├── models.json.template    slot for your own OpenAI-compatible endpoint
├── harness/run_pi.py       the whole harness: brief → pi → refine → submission
├── skills/<comp>/SKILL.md  22 task skills
└── tasks.json              leaderboard thresholds per competition
splits/lite.txt             the 22 competition ids
scripts/                    build / prepare / run / grade / summarize
docs/                       architecture, configuration, differences, skills
```

## Prerequisites

* Docker, and an NVIDIA GPU with the [container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
  Most Lite competitions want ~16 GB of VRAM; a few are CPU-only.
* An [mle-bench](https://github.com/openai/mle-bench) checkout, for the base image, the data
  preparation and the grader.
* Kaggle CLI credentials at `~/.kaggle/kaggle.json` (data preparation downloads from Kaggle) and
  ~400 GB of free disk for all 22 competitions.
* An API key for whichever model you point `pi` at.

## Quick start — one competition, no sysbox

```bash
export MLE_BENCH=~/mle-bench

# 1. base image (once)
( cd $MLE_BENCH && docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile . )

# 2. agent image
scripts/build_image.sh

# 3. data for one competition (~50 MB; `scripts/prepare_data.sh` with no ONLY does all 22)
ONLY=leaf-classification scripts/prepare_data.sh

# 4. run it
DEEPSEEK_API_KEY=sk-... TIME_LIMIT_SECS=3600 scripts/run_one.sh leaf-classification

# 5. grade + summarize
scripts/grade.sh runs/<the-run-dir> leaf-classification
scripts/summarize.py runs/
```

Each run leaves `runs/<timestamp>_<comp>/` with `submission/submission.csv`, `logs/result.json`,
`logs/traj/round1/*.jsonl` (the trajectory), `logs/pi.log`, `grade.log`, and the agent's source
under `code/`.

## Full benchmark — through mle-bench's runner

`run_agent.py` is the path for the whole split, several seeds, and mle-bench's own grading
server inside the container. It needs [Sysbox](https://github.com/nestybox/sysbox) (mle-bench's
default container runtime — or edit `environment/config/container_configs/default.json` to drop it).

```bash
MLE_BENCH=~/mle-bench scripts/install_into_mlebench.sh   # copies agents/pi/ in, prints next steps
cd $MLE_BENCH
docker build --platform=linux/amd64 -t pi agents/pi/ \
  --build-arg SUBMISSION_DIR=/home/submission --build-arg LOGS_DIR=/home/logs \
  --build-arg CODE_DIR=/home/code --build-arg AGENT_DIR=/home/agent

scripts/prepare_data.sh          # all 22 competitions, from this repo
DEEPSEEK_API_KEY=sk-... python run_agent.py --agent-id pi \
  --competition-set experiments/splits/lite.txt --n-seeds 1

python experiments/make_submission.py --metadata runs/<group>/metadata.json \
  --output runs/<group>/submission.jsonl
mlebench grade --submission runs/<group>/submission.jsonl --output-dir runs/<group>
```

Registered agent ids:

| id | what it measures |
|---|---|
| `pi` | skill-guided, medal thresholds disclosed — our setup |
| `pi/blind` | skill-guided, nothing disclosed about the grading scale — **comparable with published MLE-bench numbers** |
| `pi/no-skill` | no skill, agent designs its own approach — bare-agent baseline |
| `pi/refine` | skill-guided, 3 rounds inside the same wall clock |
| `pi/dev` | 20-minute smoke test |

## Pointing it at your own model

pi's built-in providers cover the hosted APIs (`PI_PROVIDER=deepseek|anthropic|openai|…`). For a
self-hosted endpoint — a vLLM/SGLang server serving a fine-tune, which is the case this harness
was built for — set `CUSTOM_BASE_URL`, `CUSTOM_API_KEY`, `CUSTOM_MODEL_ID` and run with
`PI_PROVIDER=custom-openai PI_MODEL=<your id>`. See [docs/configuration.md](docs/configuration.md).

## Docs

* [docs/architecture.md](docs/architecture.md) — what a run does, step by step; the prompt
  contract and why each clause is there; the trajectory format.
* [docs/configuration.md](docs/configuration.md) — every environment variable.
* [docs/differences.md](docs/differences.md) — where this deviates from official MLE-bench, and
  what our internal pipeline does that this does not. **Read before comparing scores.**
* [docs/skills.md](docs/skills.md) — what a task skill is, its conventions, how to add one.
