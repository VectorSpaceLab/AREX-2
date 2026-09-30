#!/usr/bin/env python3
"""Download Frontier-CS algorithmic problems on demand.

The repository intentionally does not carry the multi-gigabyte problem set. This
script downloads a GitHub source archive and extracts only algorithmic/problems.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from arex_eval.data import download, extract_tar, extract_zip  # noqa: E402


def download_algorithmic(
    *,
    repo: str = "weiwch/Frontier-CS-Algo",
    ref: str = "main",
    url: str = "",
    archive: Path = ROOT / ".cache/frontier-cs-algo.tar.gz",
    sha256: str = "",
    destination: Path = ROOT / "data" / "algorithmic",
) -> Path:
    url = url or f"https://github.com/{repo}/archive/refs/heads/{ref}.tar.gz"
    if not archive.exists():
        print(f"downloading {url}")
        download(url, archive, sha256)
    prefix = f"{repo.split('/')[-1]}-{ref}/algorithmic/problems/"
    destination = destination / "problems"
    try:
        extract_tar(archive, destination, prefix=prefix)
    except tarfile.ReadError:
        with zipfile.ZipFile(archive) as z:
            names = [n for n in z.namelist() if "/algorithmic/problems/" in n]
            if not names:
                raise ValueError("archive does not contain algorithmic/problems")
            first = names[0].split("/algorithmic/problems/")[0] + "/algorithmic/problems/"
            extract_zip(archive, destination, prefix=first)
    print(f"algorithmic problems ready at {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="weiwch/Frontier-CS-Algo")
    parser.add_argument("--ref", default="main")
    parser.add_argument("--url", default="")
    parser.add_argument("--archive", type=Path, default=ROOT / ".cache/frontier-cs-algo.tar.gz")
    parser.add_argument("--sha256", default="")
    parser.add_argument("--destination", type=Path, default=ROOT / "data" / "algorithmic")
    ns = parser.parse_args()
    download_algorithmic(
        repo=ns.repo,
        ref=ns.ref,
        url=ns.url,
        archive=ns.archive,
        sha256=ns.sha256,
        destination=ns.destination,
    )


if __name__ == "__main__":
    main()
