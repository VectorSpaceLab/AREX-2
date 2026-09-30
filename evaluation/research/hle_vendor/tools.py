"""
Search API Tool - Search documents and retrieve full content.

Usage:
    from tools import search, open_page

    # Search for documents
    results = local_search("your query", topk=10)
    # Returns: {"results": [{"docid": "...", "text": "...", "score": ...}, ...]}

    # Open a specific document
    content = open_page("docid_here")
    # Returns: Full document content as dict
"""

import os
import json
import traceback
from typing import Dict, Any, List
import requests

from openai import AsyncOpenAI
from prompt import EXTRACTOR_PROMPT
from visit_fallback import VisitFetchResult, fetch_url_with_fallback
from web_provider import (
    JINA_API_KEY,
    JINA_API_URL,
    SERPER_API_KEY,
    SERPER_API_URL,
    is_official_jina,
    is_official_serper,
    jina_reader_url,
)

def read_url_jina(url: str) -> str:
    """Read a public page with Jina Reader; the key is supplied by the environment."""
    print(f"  [Visit] Reading: {url}")
    try:
        if is_official_jina(JINA_API_URL):
            headers = {"Accept": "text/plain"}
            if JINA_API_KEY:
                headers["Authorization"] = f"Bearer {JINA_API_KEY}"
            response = requests.get(
                jina_reader_url(JINA_API_URL, url), headers=headers, timeout=60
            )
            response.raise_for_status()
            return response.text
        if not JINA_API_KEY:
            raise RuntimeError("JINA_API_KEY is not set")
        response = requests.post(
            JINA_API_URL,
            json={"urls": [url], "token": JINA_API_KEY},
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return (data.get("results") or {}).get(url, "")
    except Exception as e:
        return f"[visit] Failed to read page: {e}"


def read_url_with_fallback(url: str) -> VisitFetchResult:
    """Use the resilient visit pipeline while preserving ``read_url_jina`` for legacy mode."""
    print(f"  [Visit fallback] Reading: {url}")
    return fetch_url_with_fallback(url, JINA_API_KEY, api_url=JINA_API_URL)


async def _call_summary_llm(client: AsyncOpenAI, model: str, messages: list, max_retries: int = 3) -> str:
    """Call LLM for page summarization with retries."""
    for attempt in range(max_retries):
        content = ""
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=32768,
            )
            content = completion.choices[0].message.content or ""
            reasoning = getattr(completion.choices[0].message, 'reasoning_content', None)
            if reasoning and not content:
                content = reasoning
            if "</think>" in content:
                content = content.split("</think>")[-1]
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    left = content.find('{')
                    right = content.rfind('}')
                    if left != -1 and right != -1 and left <= right:
                        return json.loads(content[left:right+1])
                    return content
        except Exception as e:
            print(f"  [Summarize] Error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return content
            continue
    return ""


async def summarize_page(url: str, goal: str, summary_client: AsyncOpenAI, summary_model: str) -> str:
    """Fetch page via jina, then use LLM to extract relevant info based on goal."""
    import asyncio
    raw_content = await asyncio.to_thread(read_url_jina, url)

    if not raw_content or raw_content.startswith("[visit] Failed"):
        return (
            f"Source: {url}\n"
            f"Goal: {goal}\n"
            f"Status: Inaccessible\n\n"
            f"Extracted evidence:\n"
            f"The page could not be accessed or did not return readable content.\n\n"
            f"Brief summary:\n"
            f"The webpage content could not be processed.\n"
        )

    max_chars = 95000 * 4
    if len(raw_content) > max_chars:
        raw_content = raw_content[:max_chars]

    messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(
        webpage_content=raw_content, goal=goal)}]

    raw = await _call_summary_llm(summary_client, summary_model, messages)

    if isinstance(raw, dict):
        try:
            evidence = raw.get("evidence", "") or "No relevant evidence found for the specified goal."
            summary = raw.get("summary", "") or "No useful summary could be extracted."
            info = f"Source: {url}\n"
            info += f"Goal: {goal}\n"
            info += "Status: Success\n\n"
            info += f"Extracted evidence:\n{evidence}\n\n"
            info += f"Brief summary:\n{summary}\n"
            return info
        except Exception:
            info = f"Source: {url}\n"
            info += f"Goal: {goal}\n"
            info += "Status: Success\n\n"
            info += f"Extracted evidence:\n{raw}\n\n"
            info += "Brief summary:\nNo separate summary could be extracted.\n"
            return info
    elif isinstance(raw, str) and len(raw) > 10:
        info = f"Source: {url}\n"
        info += f"Goal: {goal}\n"
        info += "Status: Success\n\n"
        info += f"Extracted evidence:\n{raw}\n\n"
        info += "Brief summary:\nNo separate summary could be extracted.\n"
        return info
    else:
        return (
            f"Source: {url}\n"
            f"Goal: {goal}\n"
            f"Status: Success\n\n"
            f"Extracted evidence:\n"
            f"No relevant evidence found for the specified goal.\n\n"
            f"Brief summary:\n"
            f"The webpage content could not be processed.\n"
        )


def search_serper(query: str, topk: int = 10) -> List[Dict[str, Any]]:
    """
    使用 rag.ac.cn 的 SERP 接口做网页搜索。

    输入保持与原 search_serper 一致：
        - query: 查询词
        - topk: 返回条数
        - api_key: 可选 token（兼容旧参数名）

    输出保持与原 search_serper 一致：
        [{"title": ..., "link": ..., "snippet": ...}, ...]
        失败时返回：[{"error": "..."}]
    """

    print(f"🔍 [Serper] Searching: {query}")
    url = SERPER_API_URL
    if not SERPER_API_KEY:
        return [{"error": "SERPER_API_KEY is not set"}]
    if is_official_serper(url):
        payload = {"q": query, "page": 1}
        headers = {"Content-Type": "application/json", "X-API-KEY": SERPER_API_KEY}
    else:
        payload = {
            "query": query,
            "page": 1,
            "search_type": "search",
            "use_cache": True,
            "token": SERPER_API_KEY,
        }
        headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            raise ValueError("Search API returned a non-object response")
        if data.get("error"):
            raise ValueError(f"Search API error: {data['error']}")

        organic = data.get("organic", [])
        return [
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            }
            for item in organic[:topk]
        ]
    except Exception as e:
        return [{"error": str(e)}]
