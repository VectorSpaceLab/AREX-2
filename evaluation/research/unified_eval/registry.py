import json
import os
import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .types import DatasetSpec


EVALUATOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVALUATOR_ROOT.parents[1]
UNIFY_EVAL_ROOT = Path(os.environ.get("UNIFY_EVAL_ROOT", EVALUATOR_ROOT)).expanduser().resolve()
DATA_ROOT = Path(
    os.environ.get("AREX_DATA_ROOT", REPO_ROOT / "data" / "files")
).expanduser().resolve()
CONFIG_ROOT = Path(
    os.environ.get("AREX_DATASET_CONFIG_ROOT", REPO_ROOT / "data" / "research")
).expanduser().resolve()


def _resolve_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    expanded = value
    for variable, replacement in (
        ("${UNIFY_EVAL_ROOT}", UNIFY_EVAL_ROOT),
        ("${AREX_DATA_ROOT}", DATA_ROOT),
        ("${AREX_DATASETS_ROOT}", REPO_ROOT / "data"),
    ):
        expanded = expanded.replace(variable, str(replacement))
    expanded = os.path.expandvars(expanded)
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    return str((UNIFY_EVAL_ROOT / path).resolve())


def available_datasets(config_root: Path = CONFIG_ROOT) -> List[str]:
    if not config_root.exists():
        return []
    return sorted(p.name for p in config_root.iterdir() if (p / "config.json").is_file())


def _load_json_file(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_python_module(path: Path, module_prefix: str) -> Any:
    module_name = f"{module_prefix}_{path.parent.name.replace('-', '_')}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import dataset module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_python_prompt_file(path: Path) -> Dict[str, Any]:
    module = _load_python_module(path, "unified_eval_dataset_prompt")
    loaded = {}
    for attr, key in (
        ("PROMPT_BUNDLE", "prompts"),
        ("TOOLS", "tools"),
        ("TOOL_DEFINITIONS", "tool_definitions"),
        ("GENERATION", "generation"),
    ):
        if hasattr(module, attr):
            loaded[key] = getattr(module, attr)
    return loaded


def _load_python_judge_file(path: Path) -> Dict[str, Any]:
    module = _load_python_module(path, "unified_eval_dataset_judge")
    if not hasattr(module, "SCORER"):
        raise ValueError(f"{path} must define SCORER")
    scorer = getattr(module, "SCORER")
    if not isinstance(scorer, dict):
        raise TypeError(f"{path} SCORER must be a dict")
    scorer = dict(scorer)
    scorer.setdefault("judge_file", str(path))
    return {"scorer": scorer}


def normalize_judge_mode(judge_mode: Optional[str]) -> str:
    mode = (judge_mode or "local").strip().lower()
    if mode == "official":
        return "offical"
    if mode not in ("local", "offical"):
        raise ValueError("--judge-mode must be one of: local, offical")
    return mode


def _merge_dataset_modules(config_dir: Path, raw: Dict[str, Any], judge_mode: str) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    prompt_dirs = []
    if raw.get("prompt_from"):
        prompt_from = Path(str(raw["prompt_from"]))
        prompt_dirs.append(prompt_from if prompt_from.is_absolute() else config_dir.parent / prompt_from)
    prompt_dirs.append(config_dir)
    for prompt_dir in prompt_dirs:
        prompt_py_path = prompt_dir / "prompt.py"
        if prompt_py_path.is_file():
            merged.update(_load_python_prompt_file(prompt_py_path))
            merged["prompt_source"] = str(prompt_py_path)
    merged.update(raw)
    judge_py_path = config_dir / f"judge_{judge_mode}.py"
    if judge_py_path.is_file():
        merged.update(_load_python_judge_file(judge_py_path))
    return merged


def load_dataset_spec(
    name: str,
    config_root: Path = CONFIG_ROOT,
    data_path_override: Optional[str] = None,
    judge_mode: str = "local",
) -> DatasetSpec:
    config_dir = config_root / name
    config_path = config_dir / "config.json"
    if not config_path.is_file():
        known = ", ".join(available_datasets(config_root))
        raise ValueError(f"Unknown dataset {name!r}. Available datasets: {known}")
    judge_mode = normalize_judge_mode(judge_mode)
    raw = _merge_dataset_modules(config_dir, _load_json_file(config_path), judge_mode)

    raw.setdefault("name", name)
    prepared_paths = json.loads(os.environ.get("AREX_DATA_PATHS", "{}"))
    prepared_path = data_path_override or prepared_paths.get(name)
    raw["data_path"] = _resolve_path(prepared_path or raw["data_path"])
    if prepared_path and raw.get("attachments_root"):
        raw["attachments_root"] = str(Path(raw["data_path"]).parent)
    for key in ("gold_root", "criteria_path", "reference_path", "attachments_root", "include_ids_path"):
        if raw.get(key):
            raw[key] = _resolve_path(raw[key])
    leak_filter = raw.get("leak_filter")
    if leak_filter is None and str(raw["name"]).lower().startswith("gaia"):
        leak_filter = "gaia"
    return DatasetSpec(
        name=raw["name"],
        data_path=raw["data_path"],
        data_format=raw["data_format"],
        task_type=raw["task_type"],
        question_field=raw["question_field"],
        answer_field=raw["answer_field"],
        id_field=raw.get("id_field"),
        encrypted=bool(raw.get("encrypted", False)),
        encryption_scheme=raw.get("encryption_scheme", "browsecomp_sha256_xor"),
        canary_field=raw.get("canary_field", "canary"),
        metadata_fields=list(raw.get("metadata_fields", [])),
        tools=list(raw.get("tools", ["search", "google_scholar", "visit", "finish"])),
        tool_definitions=list(raw.get("tool_definitions", [])),
        scorer=dict(raw.get("scorer", {})),
        prompts=dict(raw.get("prompts", {})),
        prompt_source=raw.get("prompt_source"),
        generation=dict(raw.get("generation", {})),
        gold_root=raw.get("gold_root"),
        criteria_path=raw.get("criteria_path"),
        reference_path=raw.get("reference_path"),
        attachments_root=raw.get("attachments_root"),
        include_ids_path=raw.get("include_ids_path"),
        row_filters=dict(raw.get("row_filters", {})),
        exclude_image_rows=bool(raw.get("exclude_image_rows", False)),
        leak_filter=leak_filter,
        evaluation_backend=str(raw.get("evaluation_backend", "unified")),
    )


def load_dataset_specs(
    names: Iterable[str],
    data_path_override: Optional[str] = None,
    judge_mode: str = "local",
) -> Dict[str, DatasetSpec]:
    names = [name.strip() for name in names if name.strip()]
    if data_path_override and len(names) != 1:
        raise ValueError("--data_path override is only supported when exactly one dataset is selected")
    return {name: load_dataset_spec(name, data_path_override=data_path_override, judge_mode=judge_mode) for name in names}
