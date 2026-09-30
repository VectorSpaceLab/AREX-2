"""Credential-free configuration for the web tools used by research tasks.

The adapters default to the public providers. URL variables support custom
endpoints: ``https://google.serper.dev/search`` (or
``/scholar``) and ``https://r.jina.ai``. Secrets are read only from the environment.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

SERPER_API_URL = os.environ.get(
    "SERPER_API_URL", "https://google.serper.dev/search"
).strip()
SERPER_SCHOLAR_API_URL = os.environ.get(
    "SERPER_SCHOLAR_API_URL", "https://google.serper.dev/scholar"
).strip()
JINA_API_URL = os.environ.get(
    "JINA_API_URL", "https://r.jina.ai"
).strip()
SERPER_API_KEY = (
    os.environ.get("SERPER_API_KEY") or os.environ.get("SERPER_KEY") or ""
).strip()
JINA_API_KEY = (
    os.environ.get("JINA_API_KEY") or os.environ.get("JINA_TOKEN") or ""
).strip()


def is_official_serper(url: str) -> bool:
    return urlparse(url).hostname == "google.serper.dev"


def is_official_jina(url: str) -> bool:
    return urlparse(url).hostname == "r.jina.ai"


def serper_url(use_scholar: bool = False) -> str:
    if use_scholar and SERPER_API_URL.rstrip("/").endswith("/search"):
        return SERPER_SCHOLAR_API_URL
    return SERPER_API_URL


def jina_reader_url(base_url: str, url: str) -> str:
    if urlparse(base_url).hostname == "r.jina.ai":
        return base_url.rstrip("/") + "/" + url
    return url
