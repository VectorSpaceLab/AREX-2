"""Small, dependency-free data download and extraction helpers."""
from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str = "") -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "arex-v2/0.1"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    if expected_sha256 and sha256(destination) != expected_sha256.lower():
        destination.unlink(missing_ok=True)
        raise ValueError(f"sha256 mismatch for {destination}")
    return destination


def _safe_target(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    root = root.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"archive path escapes destination: {name}")
    return target


def extract_tar(path: Path, destination: Path, prefix: str = "") -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            name = member.name
            if prefix and not name.startswith(prefix):
                continue
            rel = name[len(prefix):] if prefix else name
            if not rel or rel.endswith("/"):
                continue
            target = _safe_target(destination, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            with target.open("wb") as handle:
                shutil.copyfileobj(source, handle)


def extract_zip(path: Path, destination: Path, prefix: str = "") -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = info.filename
            if prefix and not name.startswith(prefix):
                continue
            rel = name[len(prefix):] if prefix else name
            if not rel or rel.endswith("/"):
                continue
            target = _safe_target(destination, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
