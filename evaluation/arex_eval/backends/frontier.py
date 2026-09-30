from __future__ import annotations

import os
import sys
from pathlib import Path


def command(repo_root: Path, problem: str, solution: str, *, track: str = "algorithmic", backend: str = "", judge_url: str = "", dry_run: bool = False) -> tuple[list[str], dict[str, str]]:
    source_root = repo_root / "evaluation" / "frontier" / "source"
    if not (source_root / "frontier_cs").is_dir():
        raise FileNotFoundError(source_root / "frontier_cs")
    args = [sys.executable, "-c", "from frontier_cs.cli import main; main()", "eval", track, problem, solution]
    if backend:
        args += ["--backend", backend]
    if judge_url:
        args += ["--judge-url", judge_url]
    if dry_run:
        args += ["--dry-run"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["FRONTIER_CS_ALGORITHMIC_PATH"] = str(repo_root / "data" / "algorithmic")
    env["FRONTIER_CS_ALGORITHMIC_SOURCE"] = str(repo_root / "evaluation" / "frontier" / "judge")
    env["FRONTIER_CS_REPO_ROOT"] = str(repo_root)
    return args, env
