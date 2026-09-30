"""Dataset catalog shared by downloading and evaluation."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
RESEARCH_CONFIG_ROOT = DATA_ROOT / "research"
CATALOG = json.loads((DATA_ROOT / "catalog.json").read_text(encoding="utf-8"))


def research_names() -> list[str]:
    return sorted(p.parent.name for p in RESEARCH_CONFIG_ROOT.glob("*/config.json"))


def canonical_name(value: str) -> str:
    normalized = value.strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", normalized)
    for name in research_names():
        if normalized == name.lower() or compact == re.sub(r"[^a-z0-9]", "", name.lower()):
            return name
    for name, spec in CATALOG.items():
        aliases = [name, *spec.get("aliases", [])]
        if normalized in {str(alias).lower() for alias in aliases} or compact in {
            re.sub(r"[^a-z0-9]", "", str(alias).lower()) for alias in aliases
        }:
            return name
    raise ValueError(f"Unknown dataset {value!r}. Run: python3 evaluate.py list")


def data_root(value: str = "") -> Path:
    return Path(value or os.environ.get("AREX_DATA_ROOT") or DATA_ROOT / "files").expanduser().resolve()


def _bundled_config_path(name: str, root: Path | None = None) -> Path | None:
    """Return the default path from a bundled evaluator config.

    The downloadable datasets have entries in ``data/catalog.json``. Every
    selectable research dataset also carries its input path in its bundled
    evaluator config, so the wrapper can preflight it before starting a run.
    """
    config_path = RESEARCH_CONFIG_ROOT / name / "config.json"
    if not config_path.is_file():
        return None
    raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    data_root_path = root or data_root()
    value = str(raw.get("data_path") or "")
    value = value.replace("${UNIFY_EVAL_ROOT}", str(ROOT / "evaluation/research"))
    value = value.replace("${AREX_DATA_ROOT}", str(data_root_path))
    value = value.replace("${AREX_DATASETS_ROOT}", str(DATA_ROOT))
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / "evaluation/research" / path)


def dataset_paths(root: Path, names: list[str], *, use_legacy: bool = True) -> dict[str, str]:
    paths = {}
    for name in names:
        if name in CATALOG:
            spec = CATALOG[name]
            path = root / spec["path"]
            legacy = ROOT / spec["legacy_path"]
            if use_legacy and not path.exists() and legacy.exists():
                path = legacy
            paths[name] = str(path)
            continue
        bundled = _bundled_config_path(name, root)
        if bundled is not None:
            paths[name] = str(bundled)
    return paths


def path_is_ready(path: str | Path) -> bool:
    """Whether a prepared dataset path can be consumed by the evaluator."""
    candidate = Path(path)
    try:
        if candidate.is_file():
            return candidate.stat().st_size > 0
        # Some vendor datasets are directories of traces or artifacts.
        return candidate.is_dir() and any(candidate.iterdir())
    except OSError:
        return False
