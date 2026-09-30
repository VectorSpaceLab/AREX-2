from __future__ import annotations

import os
from pathlib import Path


def command(repo_root: Path, competition: str, *, prepare: bool = False, time_limit: int = 14400, dry_run: bool = False) -> tuple[list[str], dict[str, str]]:
    scripts = repo_root / "evaluation" / "mle" / "scripts"
    script = scripts / ("prepare_data.sh" if prepare else "run_one.sh")
    if not script.is_file():
        raise FileNotFoundError(script)
    args = ["bash", str(script)]
    if not prepare:
        args.append(competition)
    env = os.environ.copy()
    env["TIME_LIMIT_SECS"] = str(time_limit)
    if prepare:
        env["ONLY"] = competition
    if dry_run:
        env["AREX_DRY_RUN"] = "1"
    return args, env
