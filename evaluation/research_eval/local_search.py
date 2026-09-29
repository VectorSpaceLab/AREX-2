import builtins
import asyncio, json, httpx
import logging
from typing import Any, Dict, Optional, List
import os
import time
import traceback
import re
import random
from prompts_no_subagent import EXTRACTOR_PROMPT as DEFAULT_EXTRACTOR_PROMPT
from pretty_console import get_pretty_console
from benchmark_leak_filter import (
    LEAK_BLOCK_NOTICE,
    filter_search_results,
    is_blocked_reference,
    leak_filter_enabled,
    strip_leaks,
)
from hle_vendor.visit_fallback import (
    fetch_url_with_fallback,
    format_visit_failure,
)
from hle_vendor.web_blocklist import (
    WEB_BLOCK_NOTICE,
    filter_search_results as filter_blocked_search_results,
    is_blocked_web_url,
)
from web_provider import (
    JINA_API_KEY,
    JINA_API_URL,
    SERPER_API_KEY,
    SERPER_API_URL,
    is_official_jina,
    is_official_serper,
    jina_reader_url,
    serper_url,
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class NeedRetryError(Exception):
    pass


MAX_RETRY_ATTEMPTS = 10
VISIT_PAGE_MAX_TOKENS = int(os.environ.get("VISIT_PAGE_MAX_TOKENS", "95000"))


def _is_retryable_request_error(exc: Exception) -> bool:
    return isinstance(exc, (NeedRetryError, httpx.RequestError, json.JSONDecodeError, KeyError))


async def _run_with_retries(operation: str, target: str, func, *, base_delay: int = 1, max_delay: int = 5, pretty_case_key: Optional[str] = None):
    last_error = None
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            return await func()
        except ValueError:
            raise
        except Exception as e:
            last_error = e
            should_retry = _is_retryable_request_error(e)
            if attempt >= MAX_RETRY_ATTEMPTS or not should_retry:
                raise
            sleep_seconds = min(max_delay, max(1, base_delay * (2 ** (attempt - 1))))
            get_pretty_console().warning(
                f"[{operation}] Attempt {attempt}/{MAX_RETRY_ATTEMPTS} failed for {target}: "
                f"{type(e).__name__}: {e}. Retrying in {sleep_seconds}s...",
                pretty_case_key,
            )
            await asyncio.sleep(sleep_seconds)
    raise last_error


def keep_first_n_tokens(text: str, max_tokens: int = 65536, _tokenizer=None, pretty_case_key: Optional[str] = None) -> str:
    try:
        text = text.replace('\x00', '')
        tokens = _tokenizer.encode(text)
        get_pretty_console().tool_event(
            "tokenize_page",
            f"tokens={len(tokens)} max_tokens={max_tokens}",
            case_key=pretty_case_key,
            style="bright_black",
        )
        if len(tokens) <= max_tokens:
            return text
        return _tokenizer.decode(tokens[:max_tokens])
    except Exception as e:
        error_str = traceback.format_exc()
        get_pretty_console().warning(
            f"trunc Error: {type(e).__name__}: {e}\n{error_str}",
            pretty_case_key,
        )
        return text


def _maybe_json_load(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    try:
        return json.loads(text)
    except Exception:
        return value


def _normalize_page_num(value, default: int = 1) -> int:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(10, page))


def _expand_page_nums(raw_page, num_queries: int, default: int = 1) -> List[int]:
    raw_page = _maybe_json_load(raw_page)
    if isinstance(raw_page, list):
        pages = [_normalize_page_num(item, default=default) for item in raw_page[:num_queries]]
        if len(pages) < num_queries:
            fill = pages[-1] if pages else default
            pages.extend([fill] * (num_queries - len(pages)))
        return pages
    return [_normalize_page_num(raw_page, default=default)] * num_queries

async def search_serper(query: str, page_num: int = 1, use_scholar: bool = False, client: httpx.AsyncClient=None, pretty_case_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    使用 rag.ac.cn 的 SERP 接口做网页搜索。

    输入保持与原 search_serper 一致：
        - query: 查询词
        - topk: 返回条数
    输出保持与原 search_serper 一致：
        [{"title": ..., "link": ..., "snippet": ...}, ...]
        失败时返回：[{"error": "..."}]
    """
    async def _search_once():
        get_pretty_console().tool_event(
            "serper_search",
            query,
            detail=f"page={page_num} scholar={use_scholar}",
            case_key=pretty_case_key,
            style="cyan",
        )
        url = serper_url(use_scholar)
        if not SERPER_API_KEY:
            raise RuntimeError("SERPER_API_KEY is not set")
        if is_official_serper(url):
            payload = {"q": query, "page": page_num}
            headers = {"Content-Type": "application/json", "X-API-KEY": SERPER_API_KEY}
        else:
            payload = {
                "query": query,
                "page": page_num,
                "use_cache": True,
                "token": SERPER_API_KEY,
            }
            headers = {"Content-Type": "application/json"}
        if use_scholar:
            payload["search_type"] = "scholar"
        else:
            payload["search_type"] = "search"
        def contains_chinese_basic(text: str) -> bool:
            return any('\u4E00' <= char <= '\u9FFF' for char in text)
        if use_scholar:
            pass
        elif contains_chinese_basic(query):
            payload["location"] = "China"
            payload["gl"] = "cn"
            payload["hl"] = "zh-cn"
        else:
            payload["location"] = "United States"
            payload["gl"] = "us"
            payload["hl"] = "en"

        response = await client.post(url, json=payload, headers=headers)

        if response.status_code == 429 or response.status_code >= 500:
            get_pretty_console().warning(f"Serper error ({response.status_code}), retry...", pretty_case_key)
            raise NeedRetryError(f"Server Error: {response.status_code}")

        elif response.status_code == 422:
            get_pretty_console().warning(f"Serper error: cannot process the query ({query})", pretty_case_key)
            raise ValueError(f"URL Unprocessable by Serper: {query}")

        data = response.json()
        if data == {"error":"Invalid or expired token"}:
            get_pretty_console().warning("Serper error: invalid or expired token", pretty_case_key)
            raise NeedRetryError("Server error: invalid or expired token")

        response.raise_for_status()

        if "organic" not in data:
            get_pretty_console().warning(f"Empty result for query {query}, no retry", pretty_case_key)
            return {"organic": []}
        # organic = data.get("organic", [])
        # results = [{
        #     "title": item.get("title"),
        #     "url": item.get("link"),
        #     "text": item.get("snippet")
        # } for item in organic]
        return data

    return await _run_with_retries("search", query, _search_once, base_delay=1, max_delay=5, pretty_case_key=pretty_case_key)

async def search(
    query: str,
    serper_client: httpx.AsyncClient = None,
    page_id_to_url={},
    page_url_to_id={},
    use_scholar: bool = False,
    page_num: int = 1,
    pretty_case_key: Optional[str] = None,
    leak_filter: Optional[str] = None,
    dataset_name: Optional[str] = None,
):
    try:
        results = await search_serper(query=query, client=serper_client, use_scholar=use_scholar, page_num=page_num, pretty_case_key=pretty_case_key)
        results = filter_blocked_search_results(results, dataset_name)
        results = filter_search_results(results, leak_filter)
        page_num_for_header = page_num
        if "organic" not in results or not results.get("organic"):
            get_pretty_console().warning(
                f"Error, No results found for query: '{query}' on page {page_num}. Try with a more general query.",
                pretty_case_key,
            )
            return f"No results found for query: '{query}' on page {page_num}. Try with a more general query."
        web_snippets = list()
        idx = 0

        for page in results["organic"]:
            idx += 1
            if use_scholar:
                date_published = ""
                if "year" in page:
                    date_published = "\nDate published: " + str(page["year"])

                publicationInfo = ""
                if "publicationInfo" in page:
                    publicationInfo = "\npublicationInfo: " + page["publicationInfo"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                link_info = "no available link"
                url = "no available link"
                if "pdfUrl" in page:
                    link_info = "pdfUrl: " + page["pdfUrl"]
                    url = page["pdfUrl"]

                citedBy = ""
                if "citedBy" in page:
                    citedBy = "\ncitedBy: " + str(page["citedBy"])

                redacted_version = f"{idx}. [{page['title']}]({link_info}){publicationInfo}{date_published}{citedBy}\n{snippet}"
            else:
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]
                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                url = page["link"]

            redacted_version = redacted_version.replace("Your browser can't play this video.", "")
            web_snippets.append(redacted_version)

            if url not in page_url_to_id:
                url_id = str(len(page_url_to_id))
                page_url_to_id[url] = len(page_url_to_id)
                page_id_to_url[url_id] = url

        info = "search"
        if use_scholar:
            info = "scholar"
        content = f"A Google {info} for '{query}' on page {page_num_for_header} found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
        return strip_leaks(content, leak_filter)
    except ValueError as e:
        observation = "Http Error. Retry next time."
        get_pretty_console().warning("Http Error. Retry next time.", pretty_case_key)
        return observation
    except Exception as e:
        error_str = traceback.format_exc()
        get_pretty_console().error(f"Serper Error: {type(e).__name__}: {e}\n{error_str}", pretty_case_key)
        observation = f"Search Error: {type(e).__name__}: {e}"
        return observation


async def read_url_jina(url: str, client: httpx.AsyncClient, pretty_case_key: Optional[str] = None) -> str:
    """
    使用 Jina Reader 将网页 URL 转换为干净的 Markdown 文本。
    适合：读取公众号、知乎、新闻网页的正文，去除广告。
    """
    async def _read_once():
        get_pretty_console().tool_event("jina_read", url, case_key=pretty_case_key, style="cyan")

        api_url = JINA_API_URL
        if is_official_jina(api_url):
            headers = {"Accept": "text/plain"}
            if JINA_API_KEY:
                headers["Authorization"] = f"Bearer {JINA_API_KEY}"
            response = await client.get(jina_reader_url(api_url, url), headers=headers)
            response.raise_for_status()
            text = re.sub(r"\(https?:.*?\)|\[https?:.*?\]", "", response.text)
            return text.replace("---", "-").replace("===", "=").replace("   ", " ")
        if not JINA_API_KEY:
            raise RuntimeError("JINA_API_KEY is not set")
        data = {
            "urls": [url],
            "token": JINA_API_KEY,
        }
        headers = {"Content-Type": "application/json"}
        response = await client.post(api_url, json=data, headers=headers)
        if response.status_code == 429 or response.status_code >= 500:
            get_pretty_console().warning(f"Jina error ({response.status_code}), retry...", pretty_case_key)
            raise NeedRetryError(f"Server Error: {response.status_code}")

        elif response.status_code == 422:
            get_pretty_console().warning(f"Jina error: cannot process the url ({url})", pretty_case_key)
            raise ValueError(f"URL Unprocessable by url: {url}")

        data = response.json()
        if data == {"error":"Invalid or expired token"}:
            get_pretty_console().warning("Jina error: invalid or expired token", pretty_case_key)
            raise NeedRetryError("Server error: invalid or expired token")

        response.raise_for_status()

        pattern = r"\(https?:.*?\)|\[https?:.*?\]"
        res = data["results"][data["urls"][0]]
        text = re.sub(pattern, '', res)
        text = text.replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
        return text

    return await _run_with_retries("visit", url, _read_once, base_delay=2, max_delay=60, pretty_case_key=pretty_case_key)


async def open(url: str, jina_client: httpx.AsyncClient, pretty_case_key: Optional[str] = None):
    try:
        observation = await read_url_jina(url=url, client=jina_client, pretty_case_key=pretty_case_key)
    except ValueError as e:
        observation = "[visit] Failed to read page."
        get_pretty_console().warning("Error, [visit] Failed to read page.", pretty_case_key)
        return observation
    except Exception as e:
        error_str = traceback.format_exc()
        get_pretty_console().error(f"Jina Error: {type(e).__name__}: {e}\n{error_str}", pretty_case_key)
        observation = f"[visit] Failed to read page."
        return observation
    return observation
    

def _tolerant_json_parse(text: str):
    """Best-effort JSON parse: strict json → brace-extract → json_repair → regex evidence/summary."""
    if not text:
        return None
    # Strip markdown fences
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    # 1) strict
    try:
        return json.loads(t)
    except Exception:
        pass
    # 2) brace extract
    left = t.find('{'); right = t.rfind('}')
    if left != -1 and right != -1 and left <= right:
        snippet = t[left:right+1]
        try:
            return json.loads(snippet)
        except Exception:
            # 3) json_repair
            try:
                import json_repair
                repaired = json_repair.loads(snippet)
                if isinstance(repaired, dict) and repaired:
                    return repaired
            except Exception:
                pass
    # 4) regex fallback: pull evidence / summary field values leniently
    try:
        ev_m = re.search(r'"evidence"\s*:\s*"([\s\S]*?)"\s*,\s*"summary"', t)
        sm_m = re.search(r'"summary"\s*:\s*"([\s\S]*?)"\s*\}?\s*$', t)
        ev = ev_m.group(1).strip() if ev_m else ""
        sm = sm_m.group(1).strip() if sm_m else ""
        if ev or sm:
            return {"evidence": ev, "summary": sm}
    except Exception:
        pass
    return None


def _choose_summary_client(client):
    if isinstance(client, (list, tuple)):
        available_clients = [c for c in client if c is not None]
        if available_clients:
            return random.choice(available_clients)
    return client


async def summarize(
    client,
    model,
    messages,
    max_retries=6,
    max_output_tokens=16384,
    pretty_case_key: Optional[str] = None,
    summary_enable_thinking: Optional[bool] = True,
) -> str:

    last_raw = ""
    for attempt in range(max_retries):
        content = ""
        try:
            selected_client = _choose_summary_client(client)
            request_kwargs = dict(
                model=model,
                messages=messages,
                max_tokens=max_output_tokens,
            )
            if summary_enable_thinking is not None:
                request_kwargs["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": bool(summary_enable_thinking),
                    }
                }
            completion = await selected_client.chat.completions.create(**request_kwargs)
            message_content = completion.choices[0].message.content or ""
            content = message_content.split("</think>")[-1].strip()
            last_raw = content
            if content:
                parsed = _tolerant_json_parse(content)
                if isinstance(parsed, dict):
                    return parsed
                # still no valid dict — retry
                raise ValueError("tolerant_parse_returned_none")
        except Exception as e:
            get_pretty_console().warning(
                f"[ERROR] summarize attempt {attempt + 1}/{max_retries} failed: "
                f"{type(e).__name__}: {e}",
                pretty_case_key,
            )
            if attempt == (max_retries - 1):
                # Last-ditch: return raw content as string so caller can still use it
                return last_raw or content or ""
            continue
    return last_raw or ""
def _visit_content_is_usable(content: Optional[str]) -> bool:
    if not content:
        return False
    return (
        not content.startswith("[visit] Failed to read page.")
        and content != "[visit] Empty content."
        and not content.startswith("[document_parser]")
    )


def _empty_visit_information(url: str, goal: str) -> str:
    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
    useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
    useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
    return useful_information


def _failed_visit_information(url: str, goal: str, failure: str) -> str:
    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
    useful_information += "Evidence in page: \n" + failure + "\n\n"
    useful_information += "Summary: \nThe page could not be retrieved after the configured visit fallbacks. Try another source.\n\n"
    return useful_information


async def _summarize_page_content(
    url: str,
    goal: str,
    content: str,
    summary_client,
    model_name,
    _tokenizer,
    extractor_prompt: str,
    pretty_case_key: Optional[str] = None,
    summary_enable_thinking: Optional[bool] = True,
) -> str:
    content = keep_first_n_tokens(content, max_tokens=VISIT_PAGE_MAX_TOKENS, _tokenizer=_tokenizer, pretty_case_key=pretty_case_key)
    messages = [{"role":"user","content": extractor_prompt.format(webpage_content=content, goal=goal)}]
    get_pretty_console().tool_event("summarize_page", url, case_key=pretty_case_key, style="cyan")
    raw = await summarize(
        summary_client,
        model_name,
        messages,
        max_output_tokens=16384,
        pretty_case_key=pretty_case_key,
        summary_enable_thinking=summary_enable_thinking,
    )
    if isinstance(raw, dict):
        try:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
            useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"
        except:
            get_pretty_console().warning(f"Error summary json: len={len(raw)} raw={raw}", pretty_case_key)
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Summary: \n" + str(raw) + "\n\n"
    else:
        if len(raw) < 10:
            get_pretty_console().warning(f"Error summary: len={len(raw)} raw={raw}", pretty_case_key)
            useful_information = _empty_visit_information(url, goal)
        else:
            get_pretty_console().warning(f"Error summary: len={len(raw)} raw={raw}", pretty_case_key)
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Summary: \n" + str(raw) + "\n\n"
    return useful_information


async def readpage_jina(
    url: str,
    goal: str,
    jina_client: httpx.AsyncClient,
    summary_client,
    model_name,
    _tokenizer,
    extractor_prompt: str = DEFAULT_EXTRACTOR_PROMPT,
    pretty_case_key: Optional[str] = None,
    leak_filter: Optional[str] = None,
    dataset_name: Optional[str] = None,
    summary_enable_thinking: Optional[bool] = True,
    enable_visit_fallback: bool = True,
) -> str:
    if is_blocked_web_url(url, dataset_name):
        get_pretty_console().warning(f"[visit blocked] built-in web blocklist: {url}", pretty_case_key)
        return WEB_BLOCK_NOTICE
    if is_blocked_reference(url, leak_filter):
        get_pretty_console().warning(f"[visit blocked] benchmark leak source: {url}", pretty_case_key)
        return LEAK_BLOCK_NOTICE

    if enable_visit_fallback:
        visit_result = await asyncio.to_thread(
            fetch_url_with_fallback,
            url,
            JINA_API_KEY,
            api_url=JINA_API_URL,
        )
        get_pretty_console().tool_event(
            "visit_fallback",
            f"source={visit_result.source} ok={visit_result.ok} attempts={','.join(visit_result.attempts)}",
            case_key=pretty_case_key,
            style="cyan" if visit_result.ok else "yellow",
        )
        if not visit_result.ok:
            failure = format_visit_failure(visit_result)
            get_pretty_console().warning(failure, pretty_case_key)
            return strip_leaks(_failed_visit_information(url, goal, failure), leak_filter)
        content = visit_result.content
    else:
        content = await open(url, jina_client, pretty_case_key=pretty_case_key)
    content = strip_leaks(content, leak_filter)
    if content == LEAK_BLOCK_NOTICE:
        return content

    if _visit_content_is_usable(content):
        summary = await _summarize_page_content(
            url,
            goal,
            content,
            summary_client,
            model_name,
            _tokenizer,
            extractor_prompt,
            pretty_case_key=pretty_case_key,
            summary_enable_thinking=summary_enable_thinking,
        )
        return strip_leaks(summary, leak_filter)

    get_pretty_console().warning(f"Error content {content}", pretty_case_key)
    return _empty_visit_information(url, goal)


def _safe_join_under_root(root: Optional[str], requested_path: str, extra_roots: Optional[list[str]] = None) -> str:
    if not requested_path:
        raise ValueError("empty path")
    roots = [item for item in [root, *(extra_roots or [])] if item]
    if roots:
        root_abs_list = [os.path.abspath(item) for item in roots]
        if os.path.isabs(requested_path):
            path_abs = os.path.abspath(requested_path)
            for root_abs in root_abs_list:
                if path_abs == root_abs or path_abs.startswith(root_abs + os.sep):
                    return path_abs
            raise ValueError(f"path escapes allowed attachment roots: {requested_path}")
        root_abs = root_abs_list[0]
        path_abs = os.path.abspath(os.path.join(root_abs, requested_path))
        if not (path_abs == root_abs or path_abs.startswith(root_abs + os.sep)):
            raise ValueError(f"path escapes attachment root: {requested_path}")
        return path_abs
    return os.path.abspath(requested_path)


def _read_local_text_file(path: str, max_chars: int = 200000) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".csv", ".json", ".jsonl", ".xml", ".md", ".html", ".htm", ".yaml", ".yml", ".tsv"):
        with builtins.open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    if ext == ".xlsx":
        try:
            import openpyxl  # type: ignore
        except Exception as e:
            return f"[read_local_file] Unsupported .xlsx because openpyxl is unavailable: {type(e).__name__}: {e}"
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        chunks = []
        for ws in wb.worksheets:
            chunks.append(f"# Sheet: {ws.title}")
            for row in ws.iter_rows(values_only=True):
                chunks.append("\t".join("" if cell is None else str(cell) for cell in row))
                if sum(len(chunk) for chunk in chunks) >= max_chars:
                    return "\n".join(chunks)[:max_chars]
        return "\n".join(chunks)[:max_chars]
    if ext == ".pdf":
        try:
            import pypdf  # type: ignore
        except Exception:
            try:
                import PyPDF2 as pypdf  # type: ignore
            except Exception as e:
                return f"[read_local_file] Unsupported .pdf because pypdf/PyPDF2 is unavailable: {type(e).__name__}: {e}"
        reader = pypdf.PdfReader(path)
        text = []
        for page in reader.pages[:20]:
            text.append(page.extract_text() or "")
            if sum(len(chunk) for chunk in text) >= max_chars:
                break
        return "\n".join(text)[:max_chars]
    if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"):
        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            return f"[read_local_file] Image file exists but image inspection is unavailable because Pillow is missing: {type(e).__name__}: {e}"
        with Image.open(path) as img:
            return (
                f"[read_local_file] Image file: {path}\n"
                f"format: {img.format}\n"
                f"size: {img.width}x{img.height}\n"
                "This text-only tool cannot solve visual content directly."
            )
    return f"[read_local_file] Unsupported file type {ext or '<none>'}: {path}"


class LocalSearch:
    def __init__(
        self,
        summary_client=None,
        search_client=None,
        jina_client=None,
        model_name=None,
        tokenizer=None,
        page_id_to_url=None,
        page_url_to_id=None,
        extractor_prompt: str = DEFAULT_EXTRACTOR_PROMPT,
        case_output_dir: Optional[str] = None,
        attachment_root: Optional[str] = None,
        pretty_case_key: Optional[str] = None,
        leak_filter: Optional[str] = None,
        dataset_name: Optional[str] = None,
        summary_enable_thinking: Optional[bool] = True,
        enable_visit_fallback: bool = True,
    ):
        self.search_client = search_client
        self.jina_client = jina_client
        self.summary_client = summary_client
        self.model_name = model_name
        self.tokenizer = tokenizer
        # Some callers do not pass these maps; keep local search usable anyway.
        self.page_id_to_url = page_id_to_url if page_id_to_url is not None else {}
        self.page_url_to_id = page_url_to_id if page_url_to_id is not None else {}
        self.extractor_prompt = extractor_prompt
        self.case_output_dir = case_output_dir
        self.attachment_root = attachment_root
        self.pretty_case_key = pretty_case_key
        self.leak_filter = leak_filter
        self.dataset_name = dataset_name
        self.summary_enable_thinking = summary_enable_thinking
        self.enable_visit_fallback = enable_visit_fallback
        if leak_filter_enabled(self.leak_filter):
            get_pretty_console().tool_event(
                "leak_filter",
                f"enabled={self.leak_filter}",
                case_key=self.pretty_case_key,
                style="bright_black",
            )

    def _append_case_error(self, error_type, message):
        if not self.case_output_dir:
            return
        try:
            os.makedirs(self.case_output_dir, exist_ok=True)
            error_path = os.path.join(self.case_output_dir, "runtime_errors.log")
            with builtins.open(error_path, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {error_type}\n")
                f.write(str(message).rstrip() + "\n\n")
        except Exception as e:
            get_pretty_console().warning(f"Failed to write case error log: {type(e).__name__}: {e}", self.pretty_case_key)

    def _coerce_to_text(self, value):
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.decode("utf-8", errors="replace")
        try:
            if isinstance(value, (dict, list, tuple)):
                return json.dumps(value, ensure_ascii=False, default=str)
            return str(value)
        except Exception as e:
            return f"[ERROR] Failed to serialize LocalSearch response: {type(e).__name__}: {e}"
        

    async def run_action(self, fn_call, row_id=None):
        observation = ''
        name = (fn_call.get('function') or '').strip()
        params = fn_call.get('arguments') or {}
        if not isinstance(params, dict):
            try:
                params = json.loads(params) if isinstance(params, str) else {}
            except Exception:
                params = {}

        # print('???', params)
        if name == 'search':
            try:
                query = params["query"]
            except:
                get_pretty_console().error("[ERROR] Invalid request format for Search: Input must be a JSON object containing 'query' field", self.pretty_case_key)
                return "[ERROR] Invalid request format for Search: Input must be a JSON object containing 'query' field"
            try:
                query = json.loads(query)
            except:
                pass
            page = _maybe_json_load(params.get("page", 1))
            # Handle nested dict format like {"query": [...]} from some models
            if isinstance(query, dict) and "query" in query:
                query = query["query"]
                # Handle double-nested JSON string
                if isinstance(query, str):
                    try:
                        query = json.loads(query)
                    except:
                        pass
            if isinstance(page, dict) and "page" in page:
                page = page["page"]
            if isinstance(query, str):
                response = await search(
                    query,
                    self.search_client,
                    self.page_id_to_url,
                    self.page_url_to_id,
                    page_num=_normalize_page_num(page),
                    pretty_case_key=self.pretty_case_key,
                    leak_filter=self.leak_filter,
                    dataset_name=self.dataset_name,
                )
            elif isinstance(query, list):
                page_nums = _expand_page_nums(page, len(query))
                responses = await asyncio.gather(*(
                    search(
                        q,
                        self.search_client,
                        self.page_id_to_url,
                        self.page_url_to_id,
                        page_num=page_num,
                        pretty_case_key=self.pretty_case_key,
                        leak_filter=self.leak_filter,
                        dataset_name=self.dataset_name,
                    )
                    for q, page_num in zip(query, page_nums)
                ))
                response = "\n=======\n".join(responses)
            else:
                get_pretty_console().error(f"[ERROR] Invalid query type for search: {type(query)}, value: {query}", self.pretty_case_key)
                response = await search(
                    str(query),
                    self.search_client,
                    self.page_id_to_url,
                    self.page_url_to_id,
                    pretty_case_key=self.pretty_case_key,
                    leak_filter=self.leak_filter,
                    dataset_name=self.dataset_name,
                )
        elif name == 'google_scholar':
            try:
                query = params["query"]
            except:
                get_pretty_console().error("[ERROR] Invalid request format for google_scholar: Input must be a JSON object containing 'query' field", self.pretty_case_key)
                return "[ERROR] Invalid request format for google_scholar: Input must be a JSON object containing 'query' field"
            try:
                query = json.loads(query)
            except:
                pass
            # Handle nested dict format like {"query": [...]} from some models
            if isinstance(query, dict) and "query" in query:
                query = query["query"]
                # Handle double-nested JSON string
                if isinstance(query, str):
                    try:
                        query = json.loads(query)
                    except:
                        pass
            if isinstance(query, str):
                response = await search(query, self.search_client, self.page_id_to_url, self.page_url_to_id, use_scholar=True, pretty_case_key=self.pretty_case_key, leak_filter=self.leak_filter, dataset_name=self.dataset_name)
            elif isinstance(query, list):
                responses = await asyncio.gather(*(
                    search(q, self.search_client, self.page_id_to_url, self.page_url_to_id, use_scholar=True, pretty_case_key=self.pretty_case_key, leak_filter=self.leak_filter, dataset_name=self.dataset_name)
                    for q in query
                ))
                response = "\n=======\n".join(responses)
            else:
                get_pretty_console().error(f"[ERROR] Invalid query type for google_scholar: {type(query)}, value: {query}", self.pretty_case_key)
                response = await search(str(query), self.search_client, self.page_id_to_url, self.page_url_to_id, use_scholar=True, pretty_case_key=self.pretty_case_key, leak_filter=self.leak_filter, dataset_name=self.dataset_name)
                       
        elif name == 'visit':
            try:
                url = params["url"]
                goal = params["goal"]
            except:
                get_pretty_console().error("[ERROR] Invalid request format for Visit: Input must be a JSON object containing 'url' and 'goal' fields", self.pretty_case_key)
                return "[ERROR] Invalid request format for Visit: Input must be a JSON object containing 'url' and 'goal' fields"
            try:
                url = json.loads(url)
            except:
                pass
            visit_urls = None
            if isinstance(url, str):
                visit_urls = [url]
            elif isinstance(url, list):
                visit_urls = []
                for item in url:
                    if isinstance(item, str):
                        visit_urls.append(item)
                    elif isinstance(item, list):
                        for nested_item in item:
                            if not isinstance(nested_item, str):
                                error_msg = (
                                    "[ERROR] Invalid request format for Visit: `url` must be a URL string "
                                    "or a list of URL strings. Nested lists may only contain strings."
                                )
                                get_pretty_console().error(error_msg, self.pretty_case_key)
                                return error_msg
                            visit_urls.append(nested_item)
                    else:
                        error_msg = (
                            "[ERROR] Invalid request format for Visit: `url` must be a URL string "
                            "or a list of URL strings."
                        )
                        get_pretty_console().error(error_msg, self.pretty_case_key)
                        return error_msg
            else:
                error_msg = (
                    "[ERROR] Invalid request format for Visit: `url` must be a URL string "
                    "or a list of URL strings."
                )
                get_pretty_console().error(error_msg, self.pretty_case_key)
                return error_msg

            if not visit_urls:
                error_msg = "[ERROR] Invalid request format for Visit: `url` is empty."
                get_pretty_console().error(error_msg, self.pretty_case_key)
                return error_msg

            for u in visit_urls:
                if is_blocked_web_url(u, self.dataset_name):
                    get_pretty_console().warning(f"[visit blocked] built-in web blocklist: {u}", self.pretty_case_key)
                if is_blocked_reference(u, self.leak_filter):
                    get_pretty_console().warning(f"[visit blocked] benchmark leak source: {u}", self.pretty_case_key)
                if u not in self.page_url_to_id:
                    get_pretty_console().warning(f"Error url {u} not in page_url_to_id", self.pretty_case_key)
            response = await asyncio.gather(*(
                readpage_jina(
                    u,
                    goal,
                    self.jina_client,
                    self.summary_client,
                    self.model_name,
                    self.tokenizer,
                    self.extractor_prompt,
                    pretty_case_key=self.pretty_case_key,
                    leak_filter=self.leak_filter,
                    dataset_name=self.dataset_name,
                    summary_enable_thinking=self.summary_enable_thinking,
                    enable_visit_fallback=self.enable_visit_fallback,
                )
                for u in visit_urls
            ))
            response = "\n=======\n".join(response)
            
            get_pretty_console().tool_event(
                "visit_summary",
                f"chars={len(response)} urls={len(visit_urls)}",
                case_key=self.pretty_case_key,
                style="bright_black",
            )

        elif name == 'read_local_file':
            path = params.get("path", "")
            goal = params.get("goal", "")
            if not path:
                return "[ERROR] Invalid request format for read_local_file: missing `path` field"
            try:
                safe_path = _safe_join_under_root(self.attachment_root, str(path), extra_roots=[self.case_output_dir] if self.case_output_dir else None)
                if is_blocked_reference(safe_path, self.leak_filter):
                    return LEAK_BLOCK_NOTICE
                if not os.path.exists(safe_path):
                    return f"[read_local_file] File not found: {safe_path}"
                content = _read_local_text_file(safe_path)
                if self.tokenizer is not None:
                    content = keep_first_n_tokens(content, max_tokens=VISIT_PAGE_MAX_TOKENS, _tokenizer=self.tokenizer, pretty_case_key=self.pretty_case_key)
                response = (
                    f"The useful information in local file {safe_path} for user goal {goal} as follows:\n\n"
                    f"{content}"
                )
            except Exception as e:
                error_str = traceback.format_exc()
                self._append_case_error("read_local_file_error", error_str)
                return f"[read_local_file] Error: {type(e).__name__}: {e}"

        else:
            tool_name = name or "UNKNOWN"
            error_msg = f"[ERROR] The tool `{tool_name}` does not exist or is not available in the current strategy."
            get_pretty_console().error(error_msg, self.pretty_case_key)
            return error_msg

        try:
            return strip_leaks(response.strip(), self.leak_filter)
        except Exception as e:
            error_str = traceback.format_exc()
            original_type = type(response).__name__
            repaired_response = self._coerce_to_text(response).strip()
            self._append_case_error(
                "local_search_response_repair",
                (
                    f"type={original_type}\n"
                    f"error={type(e).__name__}: {e}\n"
                    f"traceback:\n{error_str}"
                ),
            )
            get_pretty_console().warning(
                f"[WARN] LocalSearch response.strip() failed for type {original_type}, "
                "falling back to string serialization.",
                self.pretty_case_key,
            )
            return repaired_response
