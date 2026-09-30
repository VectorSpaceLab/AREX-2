"""Built-in web blocklist shared by the core research evaluators.

The benchmark runners keep this policy in code so a run cannot accidentally
expose a benchmark mirror by changing a local environment variable.  The
blocklist applies to tool URLs and search-result URLs; it does not affect the
separate Hugging Face downloads used to prepare local benchmark files.
"""

from __future__ import annotations

import copy
import re
from typing import Any
from urllib.parse import unquote, urlparse


CORE_RESEARCH_DATASETS = frozenset(
    {"BrowseComp", "GAIA-2023-validation-text-103", "HLE", "DeepSearch-QA"}
)

# Keep the common policy deliberately small and stable.  Dataset-specific HLE
# mirrors remain in the HLE adapter's own blocklist below this module.
DEFAULT_BLOCKED_URL_PATTERNS = (
    "huggingface.co",
    "hf.co",
    "hf-mirror.com",
    "datasets-server.huggingface.co",
)
DATASET_BLOCKED_URL_PATTERNS = {
    dataset: DEFAULT_BLOCKED_URL_PATTERNS
    for dataset in CORE_RESEARCH_DATASETS
}

WEB_BLOCK_NOTICE = (
    "[Tool result blocked: this webpage is on the built-in benchmark blocklist. "
    "Use the original question, allowed local attachments, and other sources.]"
)

_URL_RE = re.compile(r"https?://[^\s)>\]\\\"']+", re.IGNORECASE)


def _normalize_url(value: Any) -> str:
    return unquote(str(value or "")).strip().lower().replace("/", "")


def is_common_blocked_url(value: Any) -> bool:
    """Return whether a URL or URL-bearing string matches the common policy."""
    normalized = _normalize_url(value)
    return bool(normalized) and any(pattern in normalized for pattern in DEFAULT_BLOCKED_URL_PATTERNS)


def is_blocked_web_url(value: Any, dataset_name: str | None = None) -> bool:
    """Apply the common web policy only to the four core research datasets."""
    if dataset_name not in DATASET_BLOCKED_URL_PATTERNS:
        return False
    normalized = _normalize_url(value)
    return bool(normalized) and any(
        pattern in normalized for pattern in DATASET_BLOCKED_URL_PATTERNS[dataset_name]
    )


def _contains_blocked_url(value: Any, dataset_name: str | None) -> bool:
    if dataset_name not in DATASET_BLOCKED_URL_PATTERNS or value is None:
        return False
    if isinstance(value, dict):
        return any(_contains_blocked_url(item, dataset_name) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_blocked_url(item, dataset_name) for item in value)
    text = str(value)
    if is_blocked_web_url(text, dataset_name):
        return True
    return any(is_blocked_web_url(match.group(0), dataset_name) for match in _URL_RE.finditer(text))


def filter_search_results(results: Any, dataset_name: str | None) -> Any:
    """Remove blocked-domain items while preserving the search API shape."""
    if dataset_name not in DATASET_BLOCKED_URL_PATTERNS or not isinstance(results, dict):
        return results
    filtered = copy.deepcopy(results)
    organic = filtered.get("organic")
    if isinstance(organic, list):
        filtered["organic"] = [
            item for item in organic if not _contains_blocked_url(item, dataset_name)
        ]
    return filtered


def blocked_hosts_for(dataset_name: str | None) -> tuple[str, ...]:
    """Expose a stable, display-safe list for run metadata and diagnostics."""
    if dataset_name not in DATASET_BLOCKED_URL_PATTERNS:
        return ()
    return tuple(DATASET_BLOCKED_URL_PATTERNS[dataset_name])
