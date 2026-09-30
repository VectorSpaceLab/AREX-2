import copy
import json
import re
from typing import Any, Optional
from urllib.parse import unquote, urlparse


LEAK_BLOCK_NOTICE = (
    "[Tool result blocked: benchmark answer leakage risk. "
    "Use the original question, allowed local attachments, and non-benchmark sources instead.]"
)

_SUPPORTED_FILTERS = {"gaia"}
_URL_RE = re.compile(r"https?://[^\s)>\]\"']+|[A-Za-z0-9.-]*huggingface\.co/[^\s)>\]\"']+")


def normalize_filter_name(filter_name: Optional[str]) -> Optional[str]:
    name = str(filter_name or "").strip().lower()
    return name if name in _SUPPORTED_FILTERS else None


def leak_filter_enabled(filter_name: Optional[str]) -> bool:
    return normalize_filter_name(filter_name) is not None


def _is_huggingface_gaia_url(text: str) -> bool:
    decoded = unquote(str(text or "")).strip().lower()
    if not decoded:
        return False

    parsed = urlparse(decoded if "://" in decoded else f"https://{decoded}")
    host = parsed.netloc.lower()
    if not (host == "huggingface.co" or host.endswith(".huggingface.co")):
        return False
    reference = f"{parsed.netloc}{parsed.path}?{parsed.query}".lower()
    return "gaia" in reference


def _contains_huggingface_gaia_reference(value: Any) -> bool:
    """Only block HuggingFace URL/reference fields that contain 'gaia'."""
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_contains_huggingface_gaia_reference(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_huggingface_gaia_reference(item) for item in value)

    text = unquote(str(value or ""))
    if not text:
        return False

    if not any(char.isspace() for char in text) and _is_huggingface_gaia_url(text):
        return True
    return any(_is_huggingface_gaia_url(match.group(0)) for match in _URL_RE.finditer(text))


def is_blocked_reference(reference: Any, filter_name: Optional[str]) -> bool:
    """Return True only for HuggingFace URLs/strings containing 'gaia'."""
    name = normalize_filter_name(filter_name)
    if name != "gaia":
        return False
    return _contains_huggingface_gaia_reference(reference)


def looks_like_leak(text: Any, filter_name: Optional[str]) -> bool:
    """Detect only HuggingFace GAIA references in tool output."""
    name = normalize_filter_name(filter_name)
    if name != "gaia":
        return False

    if text is None:
        return False
    return _contains_huggingface_gaia_reference(text)


def _filter_json_like_value(value: Any, filter_name: Optional[str]) -> Any:
    if isinstance(value, list):
        return [
            item
            for item in value
            if not looks_like_leak(item, filter_name)
        ]
    if isinstance(value, dict):
        filtered = {}
        for key, item in value.items():
            if isinstance(item, list):
                filtered[key] = _filter_json_like_value(item, filter_name)
            elif looks_like_leak({key: item}, filter_name):
                continue
            else:
                filtered[key] = item
        return filtered
    return value


def strip_leaks(text: str, filter_name: Optional[str]) -> str:
    """Remove HuggingFace GAIA tool output before it is shown to the answer model."""
    if not leak_filter_enabled(filter_name) or not text or not looks_like_leak(text, filter_name):
        return text

    try:
        data = json.loads(text)
        filtered = _filter_json_like_value(data, filter_name)
        if filtered != data:
            return json.dumps(filtered, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return LEAK_BLOCK_NOTICE


def filter_search_results(results: Any, filter_name: Optional[str]) -> Any:
    """Filter SERP result items while preserving the search API response shape."""
    if not leak_filter_enabled(filter_name):
        return results
    if not isinstance(results, dict):
        return results

    filtered = copy.deepcopy(results)
    organic = filtered.get("organic")
    if isinstance(organic, list):
        filtered["organic"] = [
            item
            for item in organic
            if not looks_like_leak(item, filter_name)
        ]
    return filtered
