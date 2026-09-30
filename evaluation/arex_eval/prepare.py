"""Fetch benchmark data from publisher sources; keep data out of Git."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from .data import download, sha256
from .datasets import CATALOG, ROOT, data_root, dataset_paths, path_is_ready, research_names


def convert_rows(kind: str, rows: list[dict]) -> list[dict]:
    if kind == "hle":
        # HLE in this repository evaluates the text-only subset.
        return [
            {k: v for k, v in row.items() if k not in ("image_preview", "rationale_image")}
            for row in rows if not row.get("image")
        ]
    if kind == "gaia":
        return [dict(task_id=row["task_id"], task_question=row["Question"],
                     ground_truth=row["Final answer"], Level=row.get("Level"),
                     metadata=row.get("Annotator Metadata", {}), file_name="", file_path="")
                for row in rows if not row.get("file_name") and not row.get("file_path")]
    raise ValueError(f"No converter for {kind}")


def prepare_one(name: str, root: Path, *, revision: str = "main", force: bool = False) -> Path:
    spec = CATALOG[name]
    target = root / spec["path"]
    if target.is_file() and target.stat().st_size and not force:
        print(f"READY {name}: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    # Publish only validated files, so an interrupted download is never READY.
    stage = target.with_name(target.name + ".preparing")
    try:
        if spec["kind"] == "csv":
            download(spec["url"], stage)
            with stage.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not set(spec["columns"]).issubset(reader.fieldnames or []):
                    raise ValueError(f"Unexpected CSV columns for {name}")
                count = sum(1 for _ in reader)
        else:
            try:
                import pyarrow.parquet as pq
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ValueError("Install data dependencies: pip install -e '.[research]'") from exc
            if not os.environ.get("HF_TOKEN"):
                raise ValueError(f"Accept access conditions at {spec['source']} and set HF_TOKEN, then retry")
            try:
                source = hf_hub_download(
                    repo_id=spec["hf_repo"], filename=spec["hf_file"], repo_type="dataset",
                    revision=revision, token=os.environ["HF_TOKEN"],
                    cache_dir=str(root / ".cache/huggingface"),
                )
            except Exception as exc:
                # Avoid logging a token, signed download URL, or request headers.
                raise ValueError(f"{name} download failed ({type(exc).__name__}); check HF_TOKEN and access at {spec['source']}") from None
            rows = convert_rows(spec["kind"], pq.read_table(source).to_pylist())
            count = len(rows)
            if spec["kind"] == "gaia" and count != 103:
                raise ValueError(f"Expected 103 GAIA tasks without attachments, got {count}; check the source revision")
            with stage.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if count == 0:
            raise ValueError(f"{name} contains no tasks")
        stage.replace(target)
        metadata = {"dataset": name, "source": spec["source"], "rows": count,
                    "sha256": sha256(target), "hf_revision": revision if "hf_repo" in spec else None}
        target.with_suffix(target.suffix + ".source.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"READY {name}: {count} tasks -> {target}")
        return target
    finally:
        stage.unlink(missing_ok=True)


def run(ns) -> int:
    root = data_root(ns.data_root)
    names = ns.dataset or list(CATALOG)
    if ns.all:
        names = [*CATALOG, "algorithmic"]
    if ns.list:
        names = research_names()
        existing = dataset_paths(root, names, use_legacy=not (ns.data_root or os.environ.get("AREX_DATA_ROOT")))
        for name in names:
            path = Path(existing.get(name, "")) if existing.get(name) else None
            ready = path is not None and path_is_ready(path)
            print(f"{'READY' if ready else 'MISSING':7} {name:32} {path or '(no path in config)'}")
        print(f"Frontier-CS: {ROOT / 'data/algorithmic/problems'} (download algorithmic)")
        return 0
    failed = []
    for name in dict.fromkeys(names):
        if ns.dry_run:
            print(f"Would prepare {name}: {CATALOG[name]['source'] if name in CATALOG else 'Frontier-CS public archive'}")
            continue
        try:
            if name == "algorithmic":
                subprocess.run([sys.executable, str(ROOT / "scripts/download_algorithmic.py")], cwd=ROOT, check=True)
            else:
                prepare_one(name, root, revision=ns.revision, force=ns.force)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            failed.append(name)
            print(f"FAILED {name}: {exc}", file=sys.stderr)
    if failed:
        print("Incomplete preparation: " + ", ".join(failed), file=sys.stderr)
    return 1 if failed else 0
