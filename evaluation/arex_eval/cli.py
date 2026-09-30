from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from .backends import frontier, mle, research

ROOT = Path(__file__).resolve().parents[2]

from .datasets import CATALOG, canonical_name, research_names


def dataset_name(value: str) -> str:
    try:
        return canonical_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number



def _run(args: list[str], env: dict[str, str], dry_run: bool) -> int:
    print("$", " ".join(_quote(x) for x in args))
    if dry_run:
        return 0
    return subprocess.run(args, env=env, cwd=ROOT).returncode


def _quote(value: str) -> str:
    return value if value and all(c.isalnum() or c in "-._/:=@" for c in value) else repr(value)


def doctor(_: argparse.Namespace) -> int:
    checks = {
        "research evaluator": ROOT / "evaluation/research/eval_unified.py",
        "Frontier source": ROOT / "evaluation/frontier/source/frontier_cs/cli.py",
        "algorithmic launcher": ROOT / "data/algorithmic/README.md",
        "MLE Lite harness": ROOT / "evaluation/mle/scripts/run_one.sh",
    }
    for name, path in checks.items():
        print(f"{'OK' if path.exists() else 'MISSING':7} {name:24} {path}")
    for tool in ("python3", "docker", "uv", "node"):
        print(f"{'OK' if shutil.which(tool) else 'optional':7} command {tool}")
    return 0 if all(p.exists() for p in checks.values()) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluate.py", description="Unified AREX-2 evaluation CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check the assembled repository")
    sub.add_parser("list", help="list available backends")
    p = sub.add_parser("download", help="download and prepare benchmark data")
    p.add_argument("dataset", nargs="*", help="core research datasets or algorithmic; default: all core research datasets")
    p.add_argument("--dataset", dest="dataset_option", help="compatibility alias for one dataset")
    p.add_argument("--all", action="store_true", help="include the large algorithmic archive")
    p.add_argument("--data-root", default="")
    p.add_argument("--revision", default="main", help="Hugging Face dataset revision")
    p.add_argument("--force", action="store_true", help="refresh existing prepared research data")
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser(
        "research",
        aliases=["eval", "evaluate"],
        help="evaluate one or more research datasets (dataset selection is the only positional input)",
    )
    p.add_argument("dataset", nargs="+", type=dataset_name, metavar="DATASET")
    p.add_argument(
        "--mode",
        default=None,
        choices=["direct", "refine_summary", "return"],
        help="evaluator mode; profile defaults to the dataset's reference mode",
    )
    p.add_argument(
        "--profile",
        choices=["auto", "default", "refine-equal"],
        default="auto",
        help="named benchmark configuration (auto enables refine-equal for the four headline datasets)",
    )
    p.add_argument("--model", "--model-name", dest="model", default=os.environ.get("AREX_MODEL_NAME", ""))
    p.add_argument(
        "--api-key-env",
        default=os.environ.get("AREX_API_KEY_ENV", "MODEL_API_KEY"),
        help="environment variable containing the model API key (never printed or passed as a CLI value)",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("AREX_BASE_URL", os.environ.get("BASE_URL", "")),
        help="OpenAI-compatible base URL for the model endpoint",
    )
    p.add_argument("--data-path", default="")
    p.add_argument("--save-path", default="", help="default: runs/<timestamp>")
    p.add_argument("--data-root", default="")
    p.add_argument("--tokenizer-path", default=os.environ.get("AREX_TOKENIZER_PATH", ""))
    p.add_argument(
        "--concurrency",
        type=positive,
        default=None,
        help="maximum concurrent cases (profile default: 1; otherwise 4)",
    )
    p.add_argument("--shuffle", action="store_true", help="use the legacy benchmark sampling order")
    for role in ("judge", "summary"):
        p.add_argument(f"--{role}-model", default=os.environ.get(f"AREX_{role.upper()}_MODEL", ""))
        p.add_argument(f"--{role}-base-url", default=os.environ.get(f"AREX_{role.upper()}_BASE_URL", ""))
        default_key_env = ""
        p.add_argument(
            f"--{role}-api-key-env",
            default=os.environ.get(f"AREX_{role.upper()}_API_KEY_ENV", default_key_env),
        )
    p.add_argument(
        "--n",
        "--num-tasks",
        dest="num_tasks",
        type=positive,
        default=None,
        help="evaluate the first N rows (or N rows after --start-index)",
    )
    p.add_argument("--start-index", type=nonnegative, default=0, help="zero-based first row to evaluate")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra", action="append", default=[], help="pass an extra evaluator flag (repeatable)")
    p = sub.add_parser("algorithmic", help="Frontier-CS C++ solution evaluation")
    p.add_argument("problem", help="numeric problem id")
    p.add_argument("solution", help="path to a C++17 solution")
    p.add_argument("--backend", choices=["docker", "skypilot"], default="")
    p.add_argument("--judge-url", default="")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("mle", help="MLE-bench Lite one-competition runner")
    p.add_argument("competition", help="competition id, e.g. leaf-classification")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--time-limit", type=int, default=14400)
    p.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(argv)
    if ns.command == "doctor":
        raise SystemExit(doctor(ns))
    if ns.command == "download":
        from .prepare import run
        if ns.dataset_option:
            ns.dataset.append(ns.dataset_option)
        try:
            ns.dataset = ["algorithmic" if v.lower() == "algorithmic" else canonical_name(v) for v in ns.dataset]
            for name in ns.dataset:
                if name not in CATALOG and name != "algorithmic":
                    raise ValueError(f"{name} uses manual data preparation; see data/README.md")
        except ValueError as exc:
            parser.error(str(exc))
        raise SystemExit(run(ns))
    if ns.command == "list":
        print("research: " + ", ".join(research_names()))
        print("algorithmic: Frontier-CS algorithmic problems (C++17)")
        print("mle: MLE-bench Lite competitions through evaluation/mle")
        return
    if ns.command in ("research", "eval", "evaluate"):
        try:
            args, env = research.command(
                ROOT, ns.dataset, mode=ns.mode, profile=ns.profile, model=ns.model,
                api_key_env=ns.api_key_env, base_url=ns.base_url,
                data_path=ns.data_path, save_path=ns.save_path,
                data_root=ns.data_root, tokenizer_path=ns.tokenizer_path,
                num_tasks=ns.num_tasks, start_index=ns.start_index,
                concurrency=ns.concurrency, shuffle=ns.shuffle,
                judge_model=ns.judge_model, judge_base_url=ns.judge_base_url,
                judge_api_key_env=ns.judge_api_key_env,
                summary_model=ns.summary_model, summary_base_url=ns.summary_base_url,
                summary_api_key_env=ns.summary_api_key_env,
                dry_run=ns.dry_run, extra=ns.extra,
            )
        except ValueError as exc:
            parser.error(str(exc))
        raise SystemExit(_run(args, env, ns.dry_run))
    if ns.command == "algorithmic":
        args, env = frontier.command(ROOT, ns.problem, ns.solution, backend=ns.backend, judge_url=ns.judge_url, dry_run=ns.dry_run)
        raise SystemExit(_run(args, env, ns.dry_run))
    if ns.command == "mle":
        args, env = mle.command(ROOT, ns.competition, prepare=ns.prepare, time_limit=ns.time_limit, dry_run=ns.dry_run)
        if ns.dry_run:
            print("(the MLE script still needs MLE_BENCH and prepared data for a real run)")
        raise SystemExit(_run(args, env, ns.dry_run))
