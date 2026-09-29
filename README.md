<div align="center">
  <img src="assets/arex-official.png" alt="AREX-2" width="430">
  <h1>AREX-2: Advancing Self-Improving Agents through Long-Horizon Reflective Tasks</h1>
  <p><strong>AREX Team · Beijing Academy of Artificial Intelligence (BAAI)</strong></p>
  <p>
    <a href="https://huggingface.co/BAAI/AREX-2"><img src="https://img.shields.io/badge/Model-Hugging%20Face-FFD21E?logo=huggingface&logoColor=111827" alt="Model on Hugging Face"></a>
    <a href="https://arxiv.org/abs/2607.21461"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv&logoColor=white" alt="Paper on arXiv"></a>
  </p>
</div>

AREX-2 studies whether an agent can turn more test-time rounds into a better solution. The paper treats this as two connected abilities: **reflection**, which uses feedback to decide what to change, and **long-horizon execution**, which keeps the improvement loop useful over many rounds. We synthesize these trajectories from machine learning engineering and algorithmic programming, where progress can be checked directly, and study how the resulting capability transfers to deep research.

This repository contains the evaluation runners and dataset definitions for the six reported tracks. It keeps benchmark files, model outputs, credentials, and run artifacts outside Git while making the evaluation command simple: choose a dataset, then record the model, endpoint, task range, evaluator mode, commit, and data checksum with every number.

<!-- - 🌐 [Project site](https://vectorspacelab.github.io/AREX-2/) — Research overview and benchmark results. -->
- 📚 [Evaluation guide](data/README.md) — Data preparation, prompts, and scoring for each dataset.
- 🧪 [Experiment notes](scripts/README.md) — Run configurations and experiment notes.
- 🇨🇳 [中文说明](README.zh-CN.md) — 中文安装与评测指南。

## Layout

```text
evaluation/   CLI, evaluation adapters, and the pinned evaluator snapshots
data/         dataset configs, preparation catalog, and per-dataset notes
assets/       benchmark PDF/SVG, logo files, and the static project site
scripts/      download scripts, run configs, and algorithmic experiments
evaluate.py   short root-level wrapper for selecting research datasets
```

The important rule is that dataset-specific behavior lives in `data/<track>/<dataset>/`: `config.json` describes the input and output fields, `prompt.py` describes the model prompt, and `judge_local.py` or `judge_offical.py` defines the scorer when the benchmark has one. The complete matrix is in [data/README.md](data/README.md).

## Install

Python 3.10+ is required. Install only the dependencies for the track you need:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[research]'   # research benchmarks
# pip install -e '.[frontier]' # Frontier-CS
# pip install -e '.[all]'      # all Python dependencies
python3 evaluate.py doctor
```

## Research evaluation

Copy [the environment template](scripts/configs/model.env.example) to `.env`,
fill in your model, tokenizer, judge, and tool credentials, then load it once:

```bash
cp scripts/configs/model.env.example .env
# Edit .env before loading it.
set -a; source .env; set +a
```

List the registered datasets first:

```bash
python3 evaluate.py list
```

The outer interface only asks for a dataset. The wrapper resolves its config, data path, prompt, loader, and scorer before launching the evaluator:

```bash
# inspect the exact subprocess command, without credentials or network calls
python3 evaluate.py BrowseComp --n 1 --dry-run

# evaluate rows [0, 10)
python3 evaluate.py BrowseComp --n 10 --save-path runs/browsecomp-10

# evaluate rows [100, 120)
python3 evaluate.py HLE --start-index 100 --n 20

# run the same range for two datasets
python3 evaluate.py BrowseComp HLE --n 5 --save-path runs/smoke
```

BrowseComp, GAIA, HLE, and DeepSearch-QA use the `refine-equal` profile when
selected directly. It defaults to one concurrent case, a 300-call per-round
budget with a 1,500-call total cap, confidence-tiered review, and the same
thinking, sampling, token, and retry settings across the four benchmarks. The
judge remains an explicit external service:

```bash
export MODEL_API_KEY=...
export JUDGE_API_KEY=...
python3 evaluate.py BrowseComp --n 10 \
  --model YOUR_MODEL --base-url http://model.example/v1 \
  --tokenizer-path /path/to/tokenizer \
  --judge-model YOUR_JUDGE --judge-base-url http://judge.example/v1 \
  --judge-api-key-env JUDGE_API_KEY \
  --save-path runs/browsecomp-10
```

See [the shared parameter table](evaluation/docs/configuration.md#shared-research-profile)
for the exact values. `--profile default` opts out of this profile;
`--dry-run` shows the expanded command without calling a service.

Prepare the four datasets with a built-in download recipe:

```bash
python3 evaluate.py download BrowseComp
python3 evaluate.py download DeepSearch-QA
python3 evaluate.py download HLE                 # requires HF_TOKEN and access
python3 evaluate.py download GAIA-2023-validation-text-103  # requires HF_TOKEN
python3 evaluate.py download --list
```

By default files are stored under `data/files/`. Use `--data-root PATH` to keep them elsewhere, or `--data-path PATH` for one manually prepared dataset. The selected path is checked before a real run starts. Results are written to `runs/<timestamp>/<dataset>/`; inspect each case's `status`, `judge_raw`, and `score_result` before aggregating scores.

The per-dataset input format, prompt, judge, metric, and preparation command are documented in [data/README.md](data/README.md). Advanced CLI and environment options are in [evaluation/docs/configuration.md](evaluation/docs/configuration.md).

## Frontier-CS algorithmic evaluation

The source for the judge is in `evaluation/algorithmic/`; the downloaded problem set and run artifacts live in `data/algorithmic/`:

```bash
python3 evaluate.py download algorithmic
python3 evaluate.py algorithmic 1 path/to/solution.cpp --backend docker
```

Solutions are C++17 files. The checker reports per-case results and `scoreRatio`/`scoreRatioUnbounded`; see [data/algorithmic/README.md](data/algorithmic/README.md) for the input layout and judge lifecycle.

## MLE-bench Lite

```bash
MLE_BENCH=$HOME/mle-bench python3 evaluate.py mle leaf-classification --prepare
MLE_BENCH=$HOME/mle-bench python3 evaluate.py mle leaf-classification
MLE_BENCH=$HOME/mle-bench \
  bash evaluation/mle_lite/scripts/grade.sh runs/<run-dir> leaf-classification
```

The final number comes from the host grader (`grade.log`), including `valid_submission` and the competition score. The runner and its upstream notes are kept in `evaluation/mle_lite/`.

## Results

![AREX benchmark results](assets/performance/arex-v2-benchmark-results.svg)

## Citation

If AREX is useful in your work, please cite:

```bibtex
@article{lu2026arex,
  title   = {AREX: Towards a Recursively Self-Improving Agent for Deep Research},
  author  = {Lu, Shuqi and Li, Chaofan and Luo, Kun and Zhang, Zhang and Wang, Hui and Xiao, Hongwang and Xiong, Lei and Wang, Jiahao and Wang, Sen and Jiang, Xiyan and Li, Wanli and Hu, Yuyang and Qian, Hongjin and Yan, Bingyu and Chen, Jianlyu and Xia, Ziyi and Shao, Yingxia and Liu, Kang and Dou, Zhicheng and He, Di and Li, Chaozhuo and Ye, Qiwei and Wang, Zhongyuan and Liu, Zheng},
  journal = {arXiv preprint arXiv:2607.21461},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.21461}
}
```
