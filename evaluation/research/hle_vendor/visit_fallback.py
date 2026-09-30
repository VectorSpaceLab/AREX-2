"""Resilient URL retrieval shared by BrowseComp and the HLE tool harness."""

from __future__ import annotations

import ipaddress
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse

import requests

from web_provider import JINA_API_URL, is_official_jina, jina_reader_url


DEFAULT_VISIT_API_URL = JINA_API_URL
DEFAULT_MAX_DOWNLOAD_BYTES = int(os.environ.get("VISIT_MAX_DOWNLOAD_BYTES", str(50 * 1024 * 1024)))
DEFAULT_MAX_EXTRACTED_CHARS = int(os.environ.get("VISIT_MAX_EXTRACTED_CHARS", "380000"))
DEFAULT_MAX_PDF_PAGES = int(os.environ.get("VISIT_MAX_PDF_PAGES", "200"))
DEFAULT_OCR_MAX_PAGES = int(os.environ.get("VISIT_OCR_MAX_PAGES", "20"))
DEFAULT_LOCAL_CACHE_SIZE = int(os.environ.get("VISIT_LOCAL_CACHE_SIZE", "64"))
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_SUCCESS_CACHE: OrderedDict[str, "VisitFetchResult"] = OrderedDict()
_SUCCESS_CACHE_LOCK = threading.Lock()


@dataclass
class VisitFetchResult:
    ok: bool
    content: str = ""
    source: str = ""
    error_type: str = ""
    detail: str = ""
    attempts: list[str] = field(default_factory=list)
    status_code: Optional[int] = None
    final_url: str = ""


def _copy_result(result: VisitFetchResult) -> VisitFetchResult:
    return VisitFetchResult(
        ok=result.ok,
        content=result.content,
        source=result.source,
        error_type=result.error_type,
        detail=result.detail,
        attempts=list(result.attempts),
        status_code=result.status_code,
        final_url=result.final_url,
    )


def _get_cached_success(url: str) -> Optional[VisitFetchResult]:
    if DEFAULT_LOCAL_CACHE_SIZE <= 0:
        return None
    with _SUCCESS_CACHE_LOCK:
        result = _SUCCESS_CACHE.get(url)
        if result is None:
            return None
        _SUCCESS_CACHE.move_to_end(url)
        copied = _copy_result(result)
    copied.attempts = ["local_success_cache:ok"]
    return copied


def _cache_success(url: str, result: VisitFetchResult) -> None:
    if DEFAULT_LOCAL_CACHE_SIZE <= 0 or not result.ok:
        return
    with _SUCCESS_CACHE_LOCK:
        _SUCCESS_CACHE[url] = _copy_result(result)
        _SUCCESS_CACHE.move_to_end(url)
        while len(_SUCCESS_CACHE) > DEFAULT_LOCAL_CACHE_SIZE:
            _SUCCESS_CACHE.popitem(last=False)


def visit_fallback_default_enabled() -> bool:
    value = os.environ.get("ENABLE_VISIT_FALLBACK", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def classify_visit_content(content: Optional[str]) -> str:
    """Return an error category for a reader response, or an empty string."""
    if not content or not content.strip():
        return "empty_content"
    text = content.strip()
    lower = text.lower()
    if "insufficientbalanceerror" in lower or '"status":40203' in lower:
        return "insufficient_balance"
    if "connection error occurred" in lower:
        return "connection_error"
    if lower.startswith("[visit] failed to read page"):
        return "visit_api_error"
    if "rejected by validator" in lower:
        return "validator_error"
    match = re.search(r"warning:\s*target url returned error\s+(\d{3})", lower)
    if match:
        status = int(match.group(1))
        if status in {401, 403, 418, 468}:
            return f"target_access_denied_{status}"
        if status == 404:
            return "target_not_found_404"
        if status == 429:
            return "target_rate_limited_429"
        if 500 <= status <= 599:
            return f"target_server_error_{status}"
        return f"target_http_error_{status}"
    if re.search(r"^title:\s*(just a moment|attention required|.*security check)", text, re.I | re.M):
        return "target_captcha"
    if "requiring captcha" in lower or "cf-mitigated: challenge" in lower:
        return "target_captcha"
    if re.search(r"^title:\s*(404|page not found|not found)", text, re.I | re.M):
        return "target_not_found_404"
    if (
        re.search(r"^title:\s*(log[ -]?in|sign[ -]?in|facebook)", text, re.I | re.M)
        and re.search(r"\b(password|create (?:a )?new account|forgot password)\b", lower)
    ):
        return "target_login_required"
    if re.search(r"markdown content:\s*\Z", text, re.I):
        return "empty_content"
    if lower.startswith(("error:", "<p>error:", "{\"error\":")):
        return "reader_error"
    if len(text) < 80 and re.search(r"\b(error|failed|unavailable)\b", lower):
        return "reader_error"
    return ""


def is_pdf_candidate(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    return path.endswith(".pdf") or ".pdf/" in path


def _error_result(
    error_type: str,
    detail: str,
    source: str,
    *,
    status_code: Optional[int] = None,
    final_url: str = "",
) -> VisitFetchResult:
    return VisitFetchResult(
        ok=False,
        source=source,
        error_type=error_type,
        detail=detail[:1000],
        status_code=status_code,
        final_url=final_url,
    )


def _target_http_error_type(status: int) -> str:
    if status in {401, 403, 418, 468}:
        return f"target_access_denied_{status}"
    if status == 404:
        return "target_not_found_404"
    if status == 429:
        return "target_rate_limited_429"
    if 500 <= status <= 599:
        return f"target_server_error_{status}"
    return f"target_http_error_{status}"


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("only absolute HTTP(S) URLs are supported")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in {"localhost", "localhost.localdomain"}:
        raise ValueError("localhost URLs are not allowed")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, parsed.port)}
    except socket.gaierror as exc:
        raise ValueError(f"DNS lookup failed: {exc}") from exc
    if not any(ipaddress.ip_address(address).is_global for address in addresses):
        joined = ", ".join(sorted(addresses))
        raise ValueError(f"target has no public address: {joined}")


def _request_with_safe_redirects(
    session: requests.Session,
    url: str,
    *,
    timeout: tuple[int, int] = (15, 60),
    max_redirects: int = 8,
) -> requests.Response:
    current = url
    for _ in range(max_redirects + 1):
        _validate_public_url(current)
        response = session.get(
            current,
            headers={"User-Agent": _USER_AGENT, "Accept": "*/*"},
            timeout=timeout,
            stream=True,
            allow_redirects=False,
        )
        if response.status_code not in {301, 302, 303, 307, 308}:
            return response
        location = response.headers.get("Location")
        response.close()
        if not location:
            raise requests.TooManyRedirects("redirect response had no Location header")
        current = urljoin(current, location)
    raise requests.TooManyRedirects(f"more than {max_redirects} redirects")


def _read_limited_response(response: requests.Response, max_bytes: int) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            if int(content_length) > max_bytes:
                raise ValueError(f"response is larger than {max_bytes} bytes")
        except ValueError as exc:
            if "larger" in str(exc):
                raise
    chunks = bytearray()
    for chunk in response.iter_content(chunk_size=256 * 1024):
        if not chunk:
            continue
        chunks.extend(chunk)
        if len(chunks) > max_bytes:
            raise ValueError(f"response exceeded {max_bytes} bytes")
    return bytes(chunks)


def _extract_pdf_with_fitz(data: bytes, max_pages: int, max_chars: int) -> tuple[str, int]:
    import fitz  # type: ignore

    document = fitz.open(stream=data, filetype="pdf")
    page_count = len(document)
    chunks = []
    total = 0
    for page_number, page in enumerate(document):
        if page_number >= max_pages or total >= max_chars:
            break
        page_text = page.get_text("text") or ""
        chunk = f"\n\n## PDF page {page_number + 1}\n\n{page_text}"
        chunks.append(chunk)
        total += len(chunk)
    return "".join(chunks)[:max_chars].strip(), page_count


def _extract_pdf_with_pdftotext(data: bytes, max_pages: int, max_chars: int) -> tuple[str, int]:
    binary = shutil.which("pdftotext")
    if not binary:
        raise RuntimeError("pdftotext is unavailable")
    with tempfile.TemporaryDirectory(prefix="visit_pdf_") as temp_dir:
        input_path = os.path.join(temp_dir, "input.pdf")
        output_path = os.path.join(temp_dir, "output.txt")
        with open(input_path, "wb") as handle:
            handle.write(data)
        completed = subprocess.run(
            [binary, "-layout", "-f", "1", "-l", str(max_pages), input_path, output_path],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"pdftotext exited {completed.returncode}")
        with open(output_path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read(max_chars).strip(), 0


def _extract_pdf_with_ocr(data: bytes, max_pages: int, max_chars: int) -> tuple[str, int]:
    binary = shutil.which("tesseract")
    if not binary:
        raise RuntimeError("tesseract is unavailable")
    import fitz  # type: ignore

    document = fitz.open(stream=data, filetype="pdf")
    page_limit = min(len(document), max_pages, DEFAULT_OCR_MAX_PAGES)
    chunks = []
    total = 0
    with tempfile.TemporaryDirectory(prefix="visit_ocr_") as temp_dir:
        for page_number in range(page_limit):
            if total >= max_chars:
                break
            image_path = os.path.join(temp_dir, f"page_{page_number + 1}.png")
            pixmap = document[page_number].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pixmap.save(image_path)
            completed = subprocess.run(
                [binary, image_path, "stdout"],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if completed.returncode != 0:
                continue
            chunk = f"\n\n## OCR PDF page {page_number + 1}\n\n{completed.stdout}"
            chunks.append(chunk)
            total += len(chunk)
    return "".join(chunks)[:max_chars].strip(), len(document)


def _text_looks_readable(text: str) -> bool:
    if len(text.strip()) < 80:
        return False
    replacement_ratio = text.count("\ufffd") / max(1, len(text))
    latin1_noise = sum(1 for char in text if "\u00c0" <= char <= "\u00ff")
    latin1_ratio = latin1_noise / max(1, len(text))
    alphanumeric_ratio = sum(char.isalnum() for char in text) / max(1, len(text))
    return replacement_ratio < 0.01 and latin1_ratio < 0.20 and alphanumeric_ratio >= 0.10


def extract_pdf_text(
    data: bytes,
    *,
    max_pages: int = DEFAULT_MAX_PDF_PAGES,
    max_chars: int = DEFAULT_MAX_EXTRACTED_CHARS,
) -> VisitFetchResult:
    errors = []
    try:
        text, page_count = _extract_pdf_with_fitz(data, max_pages, max_chars)
        if _text_looks_readable(text):
            return VisitFetchResult(
                ok=True,
                content=text,
                source="direct_pdf_pymupdf",
                detail=f"pages={page_count}",
            )
        errors.append(f"PyMuPDF produced unreadable text ({len(text)} characters, {page_count} pages)")
    except Exception as exc:
        errors.append(f"PyMuPDF: {type(exc).__name__}: {exc}")
    try:
        text, _ = _extract_pdf_with_pdftotext(data, max_pages, max_chars)
        if _text_looks_readable(text):
            return VisitFetchResult(ok=True, content=text, source="direct_pdf_pdftotext")
        errors.append(f"pdftotext produced unreadable text ({len(text)} characters)")
    except Exception as exc:
        errors.append(f"pdftotext: {type(exc).__name__}: {exc}")
    try:
        text, page_count = _extract_pdf_with_ocr(data, max_pages, max_chars)
        if _text_looks_readable(text):
            return VisitFetchResult(
                ok=True,
                content=text,
                source="direct_pdf_ocr",
                detail=f"pages={page_count}",
            )
        errors.append(f"OCR produced unreadable text ({len(text)} characters, {page_count} pages)")
    except Exception as exc:
        errors.append(f"OCR: {type(exc).__name__}: {exc}")
    return _error_result("pdf_no_readable_text", "; ".join(errors), "direct_pdf")


def _html_to_text(data: bytes, encoding: Optional[str], max_chars: int) -> str:
    text = data.decode(encoding or "utf-8", errors="replace")
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(text, "lxml")
        for node in soup(["script", "style", "noscript", "svg"]):
            node.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]
        body = "\n".join(lines)
        return (f"Title: {title}\n\n{body}" if title else body)[:max_chars]
    except Exception:
        text = re.sub(r"(?is)<(script|style|noscript|svg).*?>.*?</\1>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", "\n", text)
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())[:max_chars]


def direct_fetch(
    url: str,
    *,
    expect_pdf: bool = False,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    max_chars: int = DEFAULT_MAX_EXTRACTED_CHARS,
) -> VisitFetchResult:
    last_error = None
    for attempt in range(2):
        response = None
        try:
            with requests.Session() as session:
                response = _request_with_safe_redirects(session, url)
                status = response.status_code
                final_url = response.url
                content_type = (response.headers.get("Content-Type") or "").lower()
                if status != 200:
                    return _error_result(
                        _target_http_error_type(status),
                        f"direct fetch returned HTTP {status}",
                        "direct_http",
                        status_code=status,
                        final_url=final_url,
                    )
                data = _read_limited_response(response, max_bytes)
            if data.startswith(b"%PDF-"):
                result = extract_pdf_text(data, max_chars=max_chars)
                result.final_url = final_url
                return result
            if expect_pdf:
                preview = _html_to_text(data, response.encoding if response else None, min(max_chars, 4000))
                error_type = classify_visit_content(preview) or "pdf_url_returned_non_pdf"
                return _error_result(error_type, preview[:1000], "direct_pdf", final_url=final_url)
            if "json" in content_type:
                content = data.decode(response.encoding or "utf-8", errors="replace")[:max_chars]
            else:
                content = _html_to_text(data, response.encoding if response else None, max_chars)
            error_type = classify_visit_content(content)
            if error_type:
                return _error_result(error_type, content[:1000], "direct_http", final_url=final_url)
            if len(content.strip()) < 80:
                return _error_result("empty_content", content, "direct_http", final_url=final_url)
            return VisitFetchResult(
                ok=True,
                content=content,
                source="direct_http",
                final_url=final_url,
            )
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                continue
        finally:
            if response is not None:
                response.close()
    return _error_result(
        "direct_fetch_error",
        f"{type(last_error).__name__}: {last_error}",
        "direct_http",
    )


def _jina_fetch(
    url: str,
    token: str,
    cache_type: str,
    api_url: str,
) -> VisitFetchResult:
    try:
        if is_official_jina(api_url):
            headers = {"Accept": "text/plain"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            response = requests.get(
                jina_reader_url(api_url, url), headers=headers, timeout=(15, 90)
            )
            response.raise_for_status()
            content = response.text
        else:
            if not token:
                return _error_result(
                    "missing_jina_api_key", "JINA_API_KEY is not set", "jina_config", final_url=url
                )
            response = requests.post(
                api_url,
                json={"urls": [url], "token": token, "cache_type": cache_type},
                headers={"Content-Type": "application/json"},
                timeout=(15, 90),
            )
            response.raise_for_status()
            data = response.json()
            content = (data.get("results") or {}).get(url, "")
        error_type = classify_visit_content(content)
        source = f"jina_{cache_type}"
        if error_type:
            return _error_result(error_type, str(content), source, final_url=url)
        return VisitFetchResult(ok=True, content=content, source=source, final_url=url)
    except Exception as exc:
        return _error_result(
            "visit_api_error",
            f"{type(exc).__name__}: {exc}",
            f"jina_{cache_type}",
            final_url=url,
        )


def _reader_wrapper_url(url: str) -> str:
    if urlparse(url).hostname == "r.jina.ai":
        return ""
    return f"https://r.jina.ai/{url}"


def _retry_without_cache(error_type: str) -> bool:
    return error_type in {
        "connection_error",
        "empty_content",
        "reader_error",
        "target_rate_limited_429",
        "visit_api_error",
    } or error_type.startswith("target_server_error_")


def fetch_url_with_fallback(
    url: str,
    token: str,
    *,
    api_url: str = DEFAULT_VISIT_API_URL,
    event_callback: Optional[Callable[[str], None]] = None,
) -> VisitFetchResult:
    """Fetch a public URL with PDF parsing, Jina retries, and direct fallbacks."""
    cached = _get_cached_success(url)
    if cached is not None:
        if event_callback:
            event_callback(cached.attempts[0])
        return cached
    attempts = []

    def record(label: str, result: VisitFetchResult) -> VisitFetchResult:
        attempts.append(f"{label}:{'ok' if result.ok else result.error_type}")
        if event_callback:
            event_callback(attempts[-1])
        result.attempts = list(attempts)
        _cache_success(url, result)
        return result

    pdf_candidate = is_pdf_candidate(url)
    failures = []
    if pdf_candidate:
        result = record("direct_pdf", direct_fetch(url, expect_pdf=True))
        if result.ok:
            return result
        failures.append(result)

    result = record("jina_new", _jina_fetch(url, token, "new", api_url))
    if result.ok:
        return result
    failures.append(result)

    if _retry_without_cache(result.error_type):
        if result.error_type == "target_rate_limited_429":
            time.sleep(2)
        elif result.error_type.startswith("target_server_error_"):
            time.sleep(1)
        result = record("jina_no_cache", _jina_fetch(url, token, "no", api_url))
        if result.ok:
            return result
        failures.append(result)

    if not pdf_candidate:
        result = record("direct_http", direct_fetch(url))
        if result.ok:
            return result
        failures.append(result)

    wrapper_url = _reader_wrapper_url(url)
    if wrapper_url and failures[-1].error_type not in {
        "target_not_found_404",
        "validator_error",
    }:
        result = record("jina_reader_wrapper", _jina_fetch(wrapper_url, token, "new", api_url))
        if result.ok:
            result.final_url = wrapper_url
            return result
        failures.append(result)

    preferred = next(
        (item for item in failures if item.error_type == "insufficient_balance"),
        failures[-1],
    )
    preferred.attempts = list(attempts)
    details = [f"{item.source}: {item.error_type}: {item.detail}" for item in failures]
    preferred.detail = " | ".join(details)[:3000]
    return preferred


def format_visit_failure(result: VisitFetchResult) -> str:
    attempts = ", ".join(result.attempts) or "none"
    return (
        f"[visit] Failed to read page. error_type={result.error_type}; "
        f"attempts={attempts}; detail={result.detail}"
    )
