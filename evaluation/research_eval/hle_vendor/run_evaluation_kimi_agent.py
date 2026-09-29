"""
Qwen XML-tool agent evaluation on HLE.

The execution loop is inherited from the Kimi evaluator, but the model-facing
protocol is aligned with the BrowseComp/Qwen SFT data:
- system/user prompt from local prompts_no_subagent.py
- tool descriptions from local format_tools.py plus code_interpreter
- assistant tool calls parsed from XML text instead of OpenAI native tool_calls
- tool results returned as user <tool_response> messages

All tool outputs are filtered through _strip_hle_leaks(...) so the model cannot
answer by retrieving HLE dataset mirrors or answer-key fields.
"""

import os
import sys
import json
import re
import random
import argparse
import asyncio
import copy
import importlib.util
import traceback
import time
import io
import fcntl
import tempfile
import shutil
import subprocess
import textwrap
import requests
from datetime import datetime
from pathlib import Path
from typing import Any
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from tools import read_url_jina, read_url_with_fallback, search_serper
from web_blocklist import is_common_blocked_url

import os as _os
_os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoTokenizer
_qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


# ---------------------------------------------------------------------------
# HLE leak-blocking filter
#
# Blocklist used for Humanity's Last Exam: both URL and pattern are normalized
# by removing "/" and lowercasing; a URL is blocked when any normalized pattern
# is a substring of the normalized URL. Tool outputs are filtered before the
# model sees them, and explicit answer-key fields are also removed.
# ---------------------------------------------------------------------------
_HLE_URL_BLOCKLIST_PATTERNS = (
    # Domains hosting HLE content or solutions.
    "huggingface.co",
    "hf.co",
    "promptfoo.dev",
    "://scale.com",
    ".scale.com",
    "lastexam.ai",
    "agi.safe.ai",
    "last-exam",
    "hle-exam",
    "askfilo.com",
    "studocu.com",
    "coursehero.com",
    "qiita.com",
    # Grok / Grokipedia mirrors have repeatedly surfaced direct answer pages.
    "gr.inc",
    "grokipedia.com",
    # Specific URLs with HLE-related content.
    "arxiv.org/abs/2501.14249",
    "arxiv.org/pdf/2501.14249",
    "arxiv.org/html/2501.14249",
    "arxiv.org/abs/2507.05241",
    "arxiv.org/pdf/2507.05241",
    "arxiv.org/html/2507.05241",
    "arxiv.org/abs/2508.10173",
    "arxiv.org/pdf/2508.10173",
    "arxiv.org/html/2508.10173",
    "arxiv.org/abs/2510.08959",
    "arxiv.org/pdf/2510.08959",
    "arxiv.org/html/2510.08959",
    "nature.com/articles/s41586-025-09962-4",
    "openreview.net/pdf?id=46UGfq8kMI",
    "www.researchgate.net/publication/394488269_Benchmark-Driven_Selection_of_AI_Evidence_from_DeepSeek-R1",
    "openreview.net/pdf/a94b1a66a55ab89d0e45eb8ed891b115db8bf760.pdf",
    "scribd.com/document/866099862",
    "x.com/tbenst/status/1951089655191122204",
    "x.com/andrewwhite01/status/1948056183115493745",
    "news.ycombinator.com/item?id=44694191",
    "github.com/supaihq/hle",
    "github.com/centerforaisafety/hle",
    "mveteanu/HLE_PDF",
    "researchgate.net/scientific-contributions/Petr-Spelda-2170307851",
    "medium.com/@82deutschmark/o3-quiet-breakthrough-1bf9f0bafc84",
    "rahulpowar.medium.com/deepseek-triggers-1-trillion-slump-but-paves-a-bigger-future-for-ai",
    "www.bincial.com/news/tzTechnology/421026",
    "36kr.com/p/3481854274280581",
    "jb243.github.io/pages/1438",
    # Older observed mirrors / aliases retained for compatibility.
    "datasets-server.huggingface.co",
    "hf-mirror.com/datasets",
    "humanitys-last-exam",
    "humanity_last_exam",
    "humanity's last exam",
    "humanitylastexam",
    "hle-verified",
    "cais/hle",
    "meoconxinhxan",
    "medical-eval-humanitylastexam",
    "vk.com/wall-92248679",
    "t.me/s/blastim",
)
_NORMALIZED_HLE_URL_BLOCKLIST_PATTERNS = tuple(
    p.replace("/", "").lower()
    for p in _HLE_URL_BLOCKLIST_PATTERNS
    if p
)
_ANSWER_FIELD_RE = re.compile(
    r'"(?:answer|answer_idx|answer_index|rationale|canary)"\s*:\s*"',
    re.IGNORECASE,
)
_LEAK_BLOCK_NOTICE = (
    "[Tool result blocked: it contained a known HLE benchmark mirror "
    "or an explicit answer-key field. Solve from first principles "
    "instead of looking up the dataset.]"
)


def _normalize_blocklist_url(text: str) -> str:
    return (text or "").replace("/", "").lower()


def _is_blocked_url(url: str) -> bool:
    if not url:
        return False
    if is_common_blocked_url(url):
        return True
    normalized_url = _normalize_blocklist_url(url)
    return any(p in normalized_url for p in _NORMALIZED_HLE_URL_BLOCKLIST_PATTERNS)


def _looks_like_leak(text: str) -> bool:
    if not text:
        return False
    if _is_blocked_url(text):
        return True
    if _ANSWER_FIELD_RE.search(text):
        return True
    return False


def _strip_hle_leaks(text: str) -> str:
    """Remove HLE-mirror search results / pages that would leak the answer."""
    if not text or not _looks_like_leak(text):
        return text

    # Older search formatting returned a JSON list of dicts — try to filter
    # element-wise so legitimate results in the same response survive.
    try:
        data = json.loads(text)
        if isinstance(data, list):
            kept = []
            for d in data:
                blob = json.dumps(d, ensure_ascii=False)
                if not _looks_like_leak(blob):
                    kept.append(d)
            return json.dumps(kept, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Otherwise (fetch output, code_runner stdout, raw text) drop it.
    return _LEAK_BLOCK_NOTICE


# ---------------------------------------------------------------------------
# JSONL helpers (stream results to disk as they complete)
# ---------------------------------------------------------------------------
def load_predictions_jsonl(filepath):
    predictions = {}
    if not os.path.exists(filepath):
        return predictions
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                predictions[entry["id"]] = entry
    return predictions


async def save_result_jsonl(filepath, result_dict, lock):
    async with lock:
        line = json.dumps(result_dict, ensure_ascii=False) + "\n"
        with open(filepath, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
def build_openai_client(args=None) -> AsyncOpenAI:
    api_key = (
        getattr(args, "sdk_api_key", None)
        or os.environ.get("SDK_API_KEY")
        or os.environ.get("TARGET_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("MOONSHOT_API_KEY")
        or "EMPTY"
    )
    base_url = (
        getattr(args, "sdk_base_url", None)
        or os.environ.get("SDK_BASE_URL")
        or os.environ.get("TARGET_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("MOONSHOT_BASE_URL")
        or "https://api.moonshot.ai/v1"
    )
    if getattr(args, "general_max_attempts", None) is not None:
        from openai_retry_client import create_async_openai_with_retry

        return create_async_openai_with_retry(
            api_key=api_key,
            base_url=base_url,
            timeout=600,
            general_max_attempts=args.general_max_attempts,
        )
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=3600.0,
        max_retries=1,
    )


def build_chat_completion_kwargs(args, max_tokens: int, messages: list[dict]) -> dict:
    kwargs = {
        "model": args.model,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False,
    }
    if args.presence_penalty is not None:
        kwargs["presence_penalty"] = args.presence_penalty

    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.min_p is not None:
        extra_body["min_p"] = args.min_p
    if args.repetition_penalty is not None:
        extra_body["repetition_penalty"] = args.repetition_penalty
    if args.enable_thinking:
        extra_body["enable_thinking"] = True
    if args.preserve_thinking:
        extra_body["preserve_thinking"] = True
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


client = build_openai_client()

# ---------------------------------------------------------------------------
# Qwen/BrowseComp prompt and tool protocol
# ---------------------------------------------------------------------------
# This evaluator keeps the original agent loop and local tool execution, but the
# model-facing protocol is the BrowseComp/Qwen XML protocol used to generate the
# SFT trajectories in ``bc_format_from_correct.jsonl``.  The prompt/tool/parser
# files are local copies in this directory, so the script is self-contained.
SCRIPT_DIR = Path(__file__).resolve().parent

DEEPRESEARCH_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        "description": "Python code sandbox, which can be used to execute Python code.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The python code.",
                },
            },
            "required": ["code"],
        },
    },
}


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_qwen_bc_prompts() -> tuple[str, str, list[dict[str, Any]], Any]:
    prompts_path = SCRIPT_DIR / 'prompts_no_subagent.py'
    tools_path = SCRIPT_DIR / 'format_tools.py'
    utils_path = SCRIPT_DIR / 'utils.py'
    missing = [str(p) for p in (prompts_path, tools_path, utils_path) if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing local BrowseComp/Qwen helper files: {missing}")

    prompts = _load_module('hle_harness_prompts_no_subagent', prompts_path)
    format_tools = _load_module('hle_harness_format_tools', tools_path)
    utils = _load_module('hle_harness_utils', utils_path)

    tools = [copy.deepcopy(tool) for tool in format_tools.NATIVE_TOOLS]
    finish_idx = next(
        (idx for idx, tool in enumerate(tools) if tool.get('function', {}).get('name') == 'finish'),
        len(tools),
    )
    tools.insert(finish_idx, copy.deepcopy(DEEPRESEARCH_CODE_TOOL))

    tool_des = '<tools>\n'
    for tool in tools:
        tool_des += json.dumps(tool, indent=2, ensure_ascii=False) + '\n'
    tool_des += '</tools>\n'

    return prompts.SYSTEM_PROMPT_MAIN.format(tool_des=tool_des), prompts.USER_PROMPT_MAIN, tools, utils


QWEN_SYSTEM_PROMPT, QWEN_USER_PROMPT, QWEN_TOOLS, BROWSECOMP_UTILS = _load_qwen_bc_prompts()
HLE_REVIEW = _load_module('hle_solution_review', SCRIPT_DIR / 'hle_review.py')


def _content_to_text(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                parts.append(str(item.get('text', '')))
            else:
                parts.append(json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item))
        return '\n'.join(part for part in parts if part)
    if isinstance(content, (dict, tuple)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ''
    try:
        return json.loads(stripped)
    except Exception:
        return value


def _as_query_list(value: Any) -> list[str]:
    value = _parse_jsonish(value)
    if isinstance(value, dict) and 'query' in value:
        value = _parse_jsonish(value['query'])
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if value is None or value == '':
        return []
    return [str(value)]


def _as_url_list(value: Any) -> list[str]:
    value = _parse_jsonish(value)
    if isinstance(value, dict) and 'url' in value:
        value = _parse_jsonish(value['url'])
    if isinstance(value, list):
        urls: list[str] = []
        for item in value:
            parsed = _parse_jsonish(item)
            if isinstance(parsed, list):
                urls.extend(str(x) for x in parsed if str(x).strip())
            elif parsed is not None and str(parsed).strip():
                urls.append(str(parsed))
        return urls
    if value is None or value == '':
        return []
    return [str(value)]


def _parse_numeric_confidence(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)', text):
            return None
        number = float(text)
    if not 0 <= number <= 100:
        return None
    if number.is_integer():
        return int(number)
    return number


def normalize_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        arguments = {}
    if name in {'search', 'google_scholar'}:
        return {'query': _as_query_list(arguments.get('query', []))}
    if name == 'visit':
        return {
            'url': _as_url_list(arguments.get('url', [])),
            'goal': str(arguments.get('goal') or 'Retrieve information relevant to the question.'),
            **{key: arguments[key] for key in ('start_index', 'max_length') if key in arguments},
        }
    if name == 'code_interpreter':
        return {'code': str(arguments.get('code', ''))}
    if name == 'update_context':
        context = arguments.get('context', '')
        if not isinstance(context, str):
            context = json.dumps(context, ensure_ascii=False)
        return {'context': context.strip()}
    if name == 'finish':
        evidences = _parse_jsonish(arguments.get('evidences', []))
        if not isinstance(evidences, list):
            evidences = []
        return {
            'answer': str(arguments.get('answer', '')).strip(),
            'evidences': evidences,
            'confidence': _parse_numeric_confidence(arguments.get('confidence')),
        }
    return arguments


def parse_qwen_tool_calls(text: str) -> list[dict[str, Any]]:
    calls = BROWSECOMP_UTILS.extract_fn_call_multi(text) or []
    normalized = []
    for call in calls:
        name = call.get('function') or ''
        args = normalize_tool_arguments(name, call.get('arguments') or {})
        normalized.append({'function': name, 'arguments': args})
    return normalized


def tool_response_message(content: str) -> dict[str, str]:
    return {'role': 'user', 'content': f'<tool_response>\n{content}\n</tool_response>'}


def is_tool_response_message(message: dict) -> bool:
    if message.get('role') != 'user':
        return False
    content = _content_to_text(message.get('content')).strip()
    return content.startswith('<tool_response>') and content.endswith('</tool_response>')


# ---------------------------------------------------------------------------
# Stateful Python code interpreter (mimics IPython / code interpreter)
# ---------------------------------------------------------------------------
_CODE_RUNNER_NETNS_AVAILABLE: bool | None = None


def _code_runner_netns_available() -> bool:
    """Return whether this host can start a process in an isolated netns."""
    global _CODE_RUNNER_NETNS_AVAILABLE
    if _CODE_RUNNER_NETNS_AVAILABLE is not None:
        return _CODE_RUNNER_NETNS_AVAILABLE

    unshare = shutil.which("unshare")
    if not unshare:
        _CODE_RUNNER_NETNS_AVAILABLE = False
        return False

    try:
        result = subprocess.run(
            [unshare, "--user", "--map-root-user", "--net", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        _CODE_RUNNER_NETNS_AVAILABLE = result.returncode == 0
    except Exception:
        _CODE_RUNNER_NETNS_AVAILABLE = False
    return _CODE_RUNNER_NETNS_AVAILABLE


def create_python_interpreter(pkg_dir: str):
    """Start a persistent subprocess for stateful Python execution."""
    os.makedirs(pkg_dir, exist_ok=True)
    worker_code = textwrap.dedent(
        """
        import ast
        import io
        import json
        import os
        import socket
        import subprocess
        import sys
        import traceback

        _NETWORK_DISABLED_MESSAGE = "Network access is disabled inside code_runner."
        _NETWORK_ISOLATION = os.environ.get("HLE_AGENT_NETWORK_ISOLATION", "python")

        def _install_no_network_guards():
            def _raise_network_disabled(*args, **kwargs):
                raise OSError(_NETWORK_DISABLED_MESSAGE)

            socket.socket.connect = _raise_network_disabled
            socket.socket.connect_ex = _raise_network_disabled
            socket.create_connection = _raise_network_disabled
            socket.getaddrinfo = _raise_network_disabled
            socket.gethostbyname = _raise_network_disabled
            socket.gethostbyname_ex = _raise_network_disabled
            socket.gethostbyaddr = _raise_network_disabled

            if _NETWORK_ISOLATION != "netns":
                def _raise_subprocess_disabled(*args, **kwargs):
                    raise OSError(
                        "Subprocess execution is disabled inside code_runner "
                        "because OS network namespace isolation is unavailable."
                    )

                subprocess.Popen = _raise_subprocess_disabled
                subprocess.run = _raise_subprocess_disabled
                subprocess.call = _raise_subprocess_disabled
                subprocess.check_call = _raise_subprocess_disabled
                subprocess.check_output = _raise_subprocess_disabled

        _install_no_network_guards()

        pkg_dir = os.environ["HLE_AGENT_PKG_DIR"]
        if pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)

        class Context:
            def read_object(self, uri):
                path = uri if os.path.isabs(uri) else os.path.join(os.getcwd(), uri)
                with open(path, "rb") as f:
                    return f.read()

        ctx = Context()
        namespace = {"__builtins__": __builtins__, "ctx": ctx}
        plot_counter = 0

        def _compile_with_last_expr(code):
            tree = ast.parse(code, mode="exec")
            if not tree.body or not isinstance(tree.body[-1], ast.Expr):
                return compile(tree, "<code_runner>", "exec"), False
            last_expr = tree.body[-1]
            assign = ast.Assign(
                targets=[ast.Name(id="__hle_last_expr__", ctx=ast.Store())],
                value=last_expr.value,
            )
            ast.copy_location(assign, last_expr)
            tree.body[-1] = assign
            ast.fix_missing_locations(tree)
            return compile(tree, "<code_runner>", "exec"), True

        def _save_matplotlib_plots():
            global plot_counter
            try:
                import matplotlib
                matplotlib.use("Agg", force=True)
                import matplotlib.pyplot as plt
            except Exception:
                return []
            fig_nums = list(plt.get_fignums())
            paths = []
            for fig_num in fig_nums:
                plot_counter += 1
                path = os.path.join(pkg_dir, f"plot_{plot_counter}.png")
                plt.figure(fig_num).savefig(path, bbox_inches="tight")
                paths.append(path)
            if paths:
                plt.close("all")
            return paths

        while True:
            line = sys.stdin.readline()
            if not line:
                break

            try:
                request = json.loads(line)
            except Exception:
                sys.stdout.write(json.dumps({
                    "ok": False,
                    "result": "Invalid request",
                }) + "\\n")
                sys.stdout.flush()
                continue

            if request.get("type") == "shutdown":
                sys.stdout.write(json.dumps({"ok": True, "result": "shutdown"}) + "\\n")
                sys.stdout.flush()
                break

            code = request.get("code", "")
            old_stdout, old_stderr = sys.stdout, sys.stderr
            captured_out, captured_err = io.StringIO(), io.StringIO()
            old_pip_target = os.environ.get("PIP_TARGET")
            old_root_action = os.environ.get("PIP_ROOT_USER_ACTION")
            old_cache_dir = os.environ.get("PIP_CACHE_DIR")
            try:
                sys.stdout, sys.stderr = captured_out, captured_err
                os.environ["PIP_TARGET"] = pkg_dir
                os.environ["PIP_ROOT_USER_ACTION"] = "ignore"
                os.environ["PIP_CACHE_DIR"] = os.environ["HLE_AGENT_PIP_CACHE_DIR"]
                namespace.pop("__hle_last_expr__", None)
                compiled, has_last_expr = _compile_with_last_expr(code)
                exec(compiled, namespace)
                if has_last_expr and "__hle_last_expr__" in namespace:
                    captured_out.write(repr(namespace["__hle_last_expr__"]) + "\\n")
                plot_paths = _save_matplotlib_plots()
                if plot_paths:
                    captured_out.write("Saved plots:\\n")
                    for path in plot_paths:
                        captured_out.write(f"- {path}\\n")
            except Exception:
                captured_err.write(traceback.format_exc())
            finally:
                if old_pip_target is None:
                    os.environ.pop("PIP_TARGET", None)
                else:
                    os.environ["PIP_TARGET"] = old_pip_target
                if old_root_action is None:
                    os.environ.pop("PIP_ROOT_USER_ACTION", None)
                else:
                    os.environ["PIP_ROOT_USER_ACTION"] = old_root_action
                if old_cache_dir is None:
                    os.environ.pop("PIP_CACHE_DIR", None)
                else:
                    os.environ["PIP_CACHE_DIR"] = old_cache_dir
                sys.stdout, sys.stderr = old_stdout, old_stderr

            stdout = captured_out.getvalue()
            stderr = captured_err.getvalue()
            result = ""
            if stdout:
                result += stdout
            if stderr:
                result += ("\\n" if result else "") + stderr
            if not result:
                result = "(no output)"

            sys.stdout.write(json.dumps({"ok": True, "result": result}) + "\\n")
            sys.stdout.flush()
        """
    )
    env = os.environ.copy()
    env["HLE_AGENT_PKG_DIR"] = pkg_dir
    offline_cache_dir = os.path.join(pkg_dir, ".offline_cache")
    env["HLE_AGENT_PIP_CACHE_DIR"] = os.path.join(offline_cache_dir, "pip")
    env["HF_HOME"] = os.path.join(offline_cache_dir, "huggingface")
    env["HF_DATASETS_CACHE"] = os.path.join(env["HF_HOME"], "datasets")
    env["HUGGINGFACE_HUB_CACHE"] = os.path.join(env["HF_HOME"], "hub")
    env["TRANSFORMERS_CACHE"] = os.path.join(env["HF_HOME"], "transformers")
    env["HF_DATASETS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    os.makedirs(env["HLE_AGENT_PIP_CACHE_DIR"], exist_ok=True)
    os.makedirs(env["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(env["HUGGINGFACE_HUB_CACHE"], exist_ok=True)

    worker_cmd = [sys.executable, "-u", "-c", worker_code]
    if _code_runner_netns_available():
        worker_cmd = [
            shutil.which("unshare") or "unshare",
            "--user",
            "--map-root-user",
            "--net",
            *worker_cmd,
        ]
        env["HLE_AGENT_NETWORK_ISOLATION"] = "netns"
    else:
        env["HLE_AGENT_NETWORK_ISOLATION"] = "python"

    proc = subprocess.Popen(
        worker_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        env=env,
    )
    return {"proc": proc, "__pkg_dir__": pkg_dir}


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
def execute_tool(name: str, arguments: dict, namespace: dict) -> str:
    """Execute a BrowseComp/Qwen XML tool call and return the result as text."""
    try:
        if name == 'search':
            queries = _as_query_list(arguments.get('query', []))
            if not queries:
                return '[ERROR] Invalid request format for search: `query` must be a non-empty string or list.'
            outputs = []
            for query in queries:
                results = search_serper(query=query, topk=10)
                outputs.append(_format_search_results(query, _filter_search_result_leaks(results)))
            return '\n=======\n'.join(outputs)

        if name == 'google_scholar':
            queries = _as_query_list(arguments.get('query', []))
            if not queries:
                return '[ERROR] Invalid request format for google_scholar: `query` must be a non-empty string or list.'
            outputs = []
            for query in queries:
                scholar_query = f'{query} site:scholar.google.com OR site:semanticscholar.org OR site:arxiv.org OR site:pubmed.ncbi.nlm.nih.gov'
                results = search_serper(query=scholar_query, topk=10)
                outputs.append(_format_search_results(f'Google Scholar style query: {query}', _filter_search_result_leaks(results)))
            return '\n=======\n'.join(outputs)

        if name == 'visit':
            urls = _as_url_list(arguments.get('url', []))
            if not urls:
                return '[ERROR] Invalid request format for visit: `url` must be a non-empty string or list.'
            goal = str(arguments.get('goal') or 'Retrieve information relevant to the question.')
            outputs = []
            for url in urls:
                if _is_blocked_url(url):
                    outputs.append(
                        '[visit blocked: this URL is a known HLE benchmark mirror '
                        '(huggingface dataset / VK / Telegram / etc.). Solve from first principles instead.]'
                    )
                    continue
                outputs.append(_strip_hle_leaks(fetch_url_content({
                    'url': url,
                    'goal': goal,
                    **{key: arguments[key] for key in ('start_index', 'max_length') if key in arguments},
                    '_enable_visit_fallback': namespace.get('__enable_visit_fallback__', True),
                })))
            return '\n=======\n'.join(outputs)

        if name == 'code_interpreter':
            code = str(arguments.get('code', ''))
            if _looks_like_leak(code):
                return _LEAK_BLOCK_NOTICE
            return _strip_hle_leaks(execute_python(code, namespace))

        # Backward-compatible aliases for old Kimi trajectories / partial outputs.
        if name == 'web_search':
            return execute_tool('search', {'query': arguments.get('query', '')}, namespace)
        if name == 'fetch':
            return execute_tool('visit', arguments, namespace)
        if name in {'code_runner', 'PythonInterpreter'}:
            return execute_tool('code_interpreter', {'code': arguments.get('code', '')}, namespace)

        return f"Error: Unknown tool '{name}'"
    except Exception as e:
        return f"Error executing {name}: {e}\n{traceback.format_exc()}"


def _format_search_results(query: str, results: list[dict]) -> str:
    """Format SERP results as a compact, model-readable tool message."""
    if not results:
        return f'Search results for "{query}":\nNo results found.'

    if len(results) == 1 and isinstance(results[0], dict) and results[0].get("error"):
        return f"Search failed: {results[0].get('error')}"

    lines = [f'Search results for "{query}":']
    for i, item in enumerate(results, start=1):
        if item.get("error"):
            lines.append(f"{i}. Error: {item.get('error')}")
            continue
        title = item.get("title") or "(no title)"
        url = item.get("link") or item.get("url") or "(no url)"
        snippet = item.get("snippet") or "(no snippet)"
        lines.extend([
            f"{i}. Title: {title}",
            f"   URL: {url}",
            f"   Snippet: {snippet}",
            "",
        ])
    return "\n".join(lines).rstrip()


def _filter_search_result_leaks(results: list[dict]) -> list[dict]:
    """Remove blocked search results before converting them to text."""
    kept = []
    for item in results:
        blob = json.dumps(item, ensure_ascii=False)
        if not _looks_like_leak(blob):
            kept.append(item)
    return kept


def _restart_worker(namespace: dict):
    """Kill the old worker (if still around) and start a fresh one in the same pkg_dir."""
    old = namespace.get("proc")
    if old is not None:
        if old.poll() is None:
            try:
                old.kill()
            except Exception:
                pass
        try:
            old.wait(timeout=2)
        except Exception:
            pass
    new_ns = create_python_interpreter(namespace["__pkg_dir__"])
    namespace["proc"] = new_ns["proc"]


def execute_python(code: str, namespace: dict, timeout: int = 600) -> str:
    """Execute Python code in a persistent subprocess, capturing stdout/stderr."""
    proc = namespace["proc"]
    restart_notice = ""
    if proc.poll() is not None:
        _restart_worker(namespace)
        proc = namespace["proc"]
        restart_notice = (
            "[Python interpreter was restarted after a previous crash — "
            "all variables and imports have been reset. "
            "Re-import any modules you need.]\n"
        )

    request = json.dumps({"type": "exec", "code": code}, ensure_ascii=False) + "\n"
    try:
        assert proc.stdin is not None
        proc.stdin.write(request)
        proc.stdin.flush()
        line = _read_worker_line(proc, timeout)
        if line is None:
            _restart_worker(namespace)
            return (
                restart_notice
                + f"Error: execution timed out after {timeout}s. "
                f"The Python interpreter has been restarted. "
                f"All previous state (variables, imports, functions) has been lost. "
                f"Re-run any setup code before continuing."
            )
        if not line:
            return restart_notice + f"Error: empty response from python worker"
        try:
            response = json.loads(line)
        except Exception:
            return restart_notice + f"Error: invalid python worker response: {line.strip()}"
        return restart_notice + response.get("result", "(no output)")
    except Exception:
        _restart_worker(namespace)
        return restart_notice + f"Error communicating with python worker (interpreter restarted): {traceback.format_exc()}"


def _read_worker_line(proc: subprocess.Popen, timeout: int) -> str | None:
    """Read one line from the worker's stdout, with timeout. Never touches stderr."""
    import select
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        if proc.stdout is None:
            break
        ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 1.0))
        if ready:
            return proc.stdout.readline()
        if proc.poll() is not None:
            return None
    try:
        proc.kill()
    except Exception:
        pass
    return None


def _shutdown_python_interpreter(namespace: dict):
    proc = namespace.get("proc")
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        if proc.stdin is not None:
            proc.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
            proc.stdin.flush()
    except Exception:
        pass
    try:
        proc.wait(timeout=1)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


TOOL_TIMEOUT = 600

async def execute_tool_async(name: str, arguments: dict, namespace: dict) -> str:
    """Run execute_tool in a thread pool with a timeout, so it never blocks the event loop."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(execute_tool, name, arguments, namespace),
            timeout=TOOL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return f"Error: tool '{name}' timed out after {TOOL_TIMEOUT}s"
    except Exception as e:
        return f"Error executing {name}: {e}"


def fetch_url_content(arguments: dict) -> str:
    """Fetch URL content using the official fetch parameter shape."""
    url = arguments.get("url", "")
    raw = bool(arguments.get("raw", False))
    start_index = max(0, int(arguments.get("start_index", 0) or 0))
    default_max_length = int(os.environ.get("HLE_VISIT_MAX_LENGTH", "5000"))
    max_length = int(arguments.get("max_length", default_max_length) or default_max_length)
    max_length = min(max(1, max_length), 999999)
    enable_visit_fallback = bool(arguments.get("_enable_visit_fallback", True))
    fetch_source = ""
    fallback_attempts = []

    try:
        if raw:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            content = response.text
            content_type = "raw HTML"
        elif enable_visit_fallback:
            result = read_url_with_fallback(url)
            if not result.ok:
                attempts = ", ".join(result.attempts) or "none"
                return (
                    f"URL: {url}\n"
                    f"Status: Error\n"
                    f"Error-Type: {result.error_type}\n"
                    f"Fetch-Source: {result.source}\n"
                    f"Fallback-Attempts: {attempts}\n"
                    f"Error: {result.detail}\n"
                )
            content = result.content
            content_type = "markdown"
            fetch_source = result.source
            fallback_attempts = result.attempts
        else:
            content = read_url_jina(url)
            content_type = "markdown"
    except Exception as e:
        return f"URL: {url}\nStatus: Error\nError: {e}"

    if not content:
        return f"URL: {url}\nStatus: Empty\nContent-Type: {content_type}\n\n"

    sliced = content[start_index:start_index + max_length]
    end_index = start_index + len(sliced)
    truncated = end_index < len(content)
    header = (
        f"URL: {url}\n"
        f"Status: Success\n"
        f"Content-Type: {content_type}\n"
        + (f"Fetch-Source: {fetch_source}\n" if fetch_source else "")
        + (f"Fallback-Attempts: {', '.join(fallback_attempts)}\n" if fallback_attempts else "")
        + f"Start-Index: {start_index}\n"
        f"Returned-Chars: {len(sliced)}\n"
        f"Total-Chars: {len(content)}\n"
        f"Truncated: {str(truncated).lower()}\n\n"
    )
    if truncated:
        return header + sliced + f"\n\n[Content truncated. Call visit with the same url and start_index={end_index} to continue.]"
    return header + sliced


# ---------------------------------------------------------------------------
# Hide-Tool-Result context management
# ---------------------------------------------------------------------------
_HIDDEN_TOOL_RESULT_PLACEHOLDER = "[Previous tool result hidden to save context]"


def apply_context_management(
    messages: list,
    max_context_tokens: int,
    last_prompt_tokens: int | None = None,
    return_info: bool = False,
) -> list | tuple[list, dict]:
    """
    Hide-Tool-Result for the Qwen XML protocol.

    Qwen/BrowseComp trajectories store tool results as user messages containing
    <tool_response>...</tool_response>, so old tool-response user turns are
    replaced with a short placeholder once the context is too long.
    """
    if last_prompt_tokens is not None:
        estimated = last_prompt_tokens
    else:
        estimated = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) // 4

    info = {
        'applied': False,
        'estimated_tokens': estimated,
        'max_context_tokens': max_context_tokens,
        'last_tool_response_group_start': None,
        'hidden_tool_message_indices': [],
        'hidden_tool_call_ids': [],
        'placeholder': _HIDDEN_TOOL_RESULT_PLACEHOLDER,
    }

    if estimated <= max_context_tokens:
        return (messages, info) if return_info else messages

    tool_indices = [i for i, m in enumerate(messages) if is_tool_response_message(m)]
    if not tool_indices:
        return (messages, info) if return_info else messages

    last_tool_idx = tool_indices[-1]
    keep_start = last_tool_idx
    while keep_start - 1 >= 0 and is_tool_response_message(messages[keep_start - 1]):
        keep_start -= 1
    info['last_tool_response_group_start'] = keep_start

    managed = []
    for i, msg in enumerate(messages):
        if is_tool_response_message(msg) and i < keep_start:
            if _HIDDEN_TOOL_RESULT_PLACEHOLDER not in _content_to_text(msg.get('content')):
                info['hidden_tool_message_indices'].append(i)
            managed.append(tool_response_message(_HIDDEN_TOOL_RESULT_PLACEHOLDER))
        else:
            managed.append(msg)

    info['applied'] = bool(info['hidden_tool_message_indices'])
    return (managed, info) if return_info else managed


# ---------------------------------------------------------------------------
# Format initial messages
# ---------------------------------------------------------------------------
def format_initial_messages(question: dict) -> list:
    """Build initial BrowseComp/Qwen system + user messages for a question."""
    question_text = str(question['question']).strip()
    if question_text.startswith('Question:') and QWEN_USER_PROMPT.startswith('Question: {question}'):
        user_template = '{question}' + QWEN_USER_PROMPT[len('Question: {question}'):]
    else:
        user_template = QWEN_USER_PROMPT
    user_text = user_template.format(question=question_text)

    if question.get('image'):
        content = [
            {'type': 'text', 'text': user_text},
            {'type': 'image_url', 'image_url': {'url': question['image']}},
        ]
    else:
        content = user_text

    return [
        {'role': 'system', 'content': QWEN_SYSTEM_PROMPT},
        {'role': 'user', 'content': content},
    ]


def _cached_tokens_from_usage(usage: dict) -> int:
    cached = usage.get("cached_tokens")
    if cached is not None:
        return cached or 0
    details = usage.get("prompt_tokens_details") or {}
    return details.get("cached_tokens") or 0


KIMI_INPUT_CACHE_HIT_PRICE_PER_1M = 0.16
KIMI_INPUT_CACHE_MISS_PRICE_PER_1M = 0.95
KIMI_OUTPUT_PRICE_PER_1M = 4.00


def _token_price(tokens: int, price_per_1m: float) -> float:
    return tokens * price_per_1m / 1_000_000


def _usage_price(input_tokens: int, output_tokens: int, cache_read_tokens: int) -> dict:
    uncached_input_tokens = max(input_tokens - cache_read_tokens, 0)
    cached_input_cost = _token_price(cache_read_tokens, KIMI_INPUT_CACHE_HIT_PRICE_PER_1M)
    uncached_input_cost = _token_price(uncached_input_tokens, KIMI_INPUT_CACHE_MISS_PRICE_PER_1M)
    output_cost = _token_price(output_tokens, KIMI_OUTPUT_PRICE_PER_1M)
    total_cost = cached_input_cost + uncached_input_cost + output_cost
    return {
        "cached_input_cost_usd": round(cached_input_cost, 8),
        "uncached_input_cost_usd": round(uncached_input_cost, 8),
        "input_cost_usd": round(cached_input_cost + uncached_input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "total_cost_usd": round(total_cost, 8),
        "price_per_1m": {
            "cache_hit_input": KIMI_INPUT_CACHE_HIT_PRICE_PER_1M,
            "cache_miss_input": KIMI_INPUT_CACHE_MISS_PRICE_PER_1M,
            "output": KIMI_OUTPUT_PRICE_PER_1M,
        },
    }


def _build_model_call_usage(
    *,
    step: int,
    call_type: str,
    finish_reason: str,
    usage: dict,
    api_elapsed: float,
    tokens_per_sec: float,
    qwen_output_tokens: int,
    qwen_tokens_per_sec: float,
    tool_calls: list[str] | None = None,
) -> dict:
    input_tokens = usage.get("prompt_tokens", 0) or 0
    output_tokens = usage.get("completion_tokens", 0) or 0
    cache_read_tokens = _cached_tokens_from_usage(usage)
    uncached_input_tokens = max(input_tokens - cache_read_tokens, 0)
    cost = _usage_price(input_tokens, output_tokens, cache_read_tokens)
    return {
        "step": step,
        "type": call_type,
        "finish_reason": finish_reason,
        "tool_calls": tool_calls or [],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "total_tokens": usage.get("total_tokens", input_tokens + output_tokens) or 0,
        **cost,
        "api_time": round(api_elapsed, 2),
        "output_tokens_per_sec": round(tokens_per_sec, 1),
        "qwen_output_tokens": qwen_output_tokens,
        "qwen_output_tokens_per_sec": round(qwen_tokens_per_sec, 1),
    }


def _summarize_model_call_usage(model_call_usage: list[dict]) -> dict:
    input_tokens = sum(x.get("input_tokens", 0) for x in model_call_usage)
    output_tokens = sum(x.get("output_tokens", 0) for x in model_call_usage)
    cache_read_tokens = sum(x.get("cache_read_tokens", 0) for x in model_call_usage)
    total_tokens = sum(x.get("total_tokens", 0) for x in model_call_usage)
    api_time = sum(x.get("api_time", 0) for x in model_call_usage)
    cost = _usage_price(input_tokens, output_tokens, cache_read_tokens)
    return {
        "model_calls": len(model_call_usage),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "uncached_input_tokens": max(input_tokens - cache_read_tokens, 0),
        "total_tokens": total_tokens,
        **cost,
        "cache_hit_rate": round(cache_read_tokens / input_tokens, 6) if input_tokens else 0.0,
        "api_time": round(api_time, 2),
    }


# ---------------------------------------------------------------------------
# Agentic loop for a single question
# ---------------------------------------------------------------------------
async def run_agent(question: dict, args) -> dict | None:
    """
    Run the agentic loop for one HLE question.
    The model can call tools iteratively until it produces a final text answer
    or hits the step limit.
    """
    review_enabled = bool(getattr(args, 'enable_confidence_review', False))
    review_threshold = float(getattr(args, 'review_threshold', 95))
    review_middle = float(getattr(args, 'review_middle_threshold', 90))
    review_limit = int(getattr(args, 'review_max_rounds', 2))
    review_max_tokens = int(getattr(args, 'review_max_tokens', 4096))
    outer_resume = getattr(args, 'outer_resume', None)
    if (review_enabled or outer_resume) and (
        not 0 <= review_middle <= review_threshold <= 100
        or review_limit < 0 or review_max_tokens <= 0
    ):
        raise ValueError('Invalid HLE confidence review configuration')
    messages = format_initial_messages(question)
    initial_messages = copy.deepcopy(messages)
    output_stem = os.path.splitext(os.path.basename(args.output))[0]
    env_root = os.path.join(os.path.dirname(args.output) or ".", f"{output_stem}_envs")
    pkg_dir = os.path.join(env_root, str(question["id"]))
    namespace = create_python_interpreter(pkg_dir)
    namespace["__model__"] = args.model
    namespace["__enable_visit_fallback__"] = bool(
        getattr(args, "enable_visit_fallback", True)
    )

    try:
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        trajectory = []
        model_call_usage = []
        last_prompt_tokens = None
        managed_messages = list(messages)
        context_management_steps = []
        length_streak = 0
        final_answer_override = None
        final_reasoning_override = None
        completed_final_answer = False
        tool_call_regen_retry_count = 0
        saved_finish = None
        saved_reasoning = None
        pending_review = False
        review_count = 0
        review_meta = {
            'enabled': review_enabled,
            'prompt_version': HLE_REVIEW.PROMPT_VERSION,
            'threshold': review_threshold,
            'middle_threshold': review_middle,
            'max_rounds': review_limit,
            'drafts': [],
            'reviews': [],
        }
        outer_resume_meta = {
            'enabled': bool(outer_resume),
            'outer_round': int(getattr(args, 'outer_round', 1)),
        }

        def current_total_tokens(step_usage: dict) -> int:
            return step_usage.get("total_tokens") or (
                (step_usage.get("prompt_tokens", 0) or 0)
                + (step_usage.get("completion_tokens", 0) or 0)
            )

        def total_token_budget_reached(step_usage: dict) -> bool:
            return (
                args.max_total_tokens is not None
                and args.max_total_tokens > 0
                and current_total_tokens(step_usage) >= args.max_total_tokens
            )

        def failure_result(failure_content: str, event: dict) -> dict:
            trajectory.append(event)
            if saved_finish is not None:
                review_meta['stop_reason'] = event['type']
                review_meta['used_saved_finish'] = True
                trajectory.append({
                    'step': event['step'], 'type': 'finish',
                    'confidence': saved_finish['confidence'], 'restored_draft': True,
                })
            return {
                "id": question["id"],
                "content": saved_finish['answer'] if saved_finish is not None else failure_content,
                "reasoning": saved_reasoning if saved_finish is not None else None,
                "accepted_finish": saved_finish,
                "confidence_review": review_meta,
                "outer_resume": outer_resume_meta,
                "usage": total_usage,
                "model_call_usage": model_call_usage,
                "model_usage_summary": _summarize_model_call_usage(model_call_usage),
                "steps": len(trajectory),
                "trajectory": trajectory,
                "context_management_steps": context_management_steps,
                "messages": messages if saved_finish is not None else messages + [{"role": "assistant", "content": failure_content}],
            }

        first_solver_step = 0
        if outer_resume:
            draft = outer_resume.get('draft') or {}
            prior_messages = outer_resume.get('messages') or []
            review_input = HLE_REVIEW.build_review_messages(
                question, draft, prior_messages, review_middle,
                max_chars=max(1, min(180000, args.max_context_tokens)),
            )
            review_usage = {}
            review_finish_reason = 'error'
            review_content = ''
            review_reasoning = ''
            review_started = time.time()
            outer_record = {
                'step': 0,
                'source_outer': int(outer_resume.get('source_outer') or 1),
                'source_result_path': str(outer_resume.get('source_result_path') or ''),
                'source_confidence': draft.get('confidence'),
            }
            try:
                review_response = await client.chat.completions.create(
                    **build_chat_completion_kwargs(args, review_max_tokens, review_input)
                )
                if review_response.usage:
                    review_usage = json.loads(review_response.usage.model_dump_json())
                    for key in total_usage:
                        total_usage[key] += review_usage.get(key, 0) or 0
                review_choice = review_response.choices[0]
                review_finish_reason = review_choice.finish_reason
                review_content = review_choice.message.content or ''
                review_reasoning = getattr(review_choice.message, 'reasoning_content', None) or ''
                if review_finish_reason == 'length':
                    raise ValueError('Outer reviewer output was truncated')
                feedback = HLE_REVIEW.parse_review(review_content)
                outer_record['feedback'] = feedback
                feedback_payload = json.dumps(feedback, ensure_ascii=False)
            except Exception as error:
                outer_record['error'] = str(error)[:1000]
                feedback_payload = (
                    'No usable independent review was returned. Recheck the prior draft '
                    'yourself; reviewer failure is not evidence for or against it.'
                )
            review_elapsed = time.time() - review_started
            outer_record.update({
                'raw_content': review_content,
                'reasoning': review_reasoning,
                'usage': review_usage,
                'api_time': round(review_elapsed, 2),
            })
            outer_resume_meta.update({
                'source_outer': outer_record['source_outer'],
                'source_result_path': outer_record['source_result_path'],
                'source_confidence': outer_record['source_confidence'],
                'review': outer_record,
            })
            review_qwen_tokens = len(_qwen_tok.encode(review_content + review_reasoning))
            model_call_usage.append(_build_model_call_usage(
                step=0,
                call_type='outer_review',
                finish_reason=review_finish_reason,
                usage=review_usage,
                api_elapsed=review_elapsed,
                tokens_per_sec=(review_usage.get('completion_tokens', 0) or 0) / max(review_elapsed, 1e-9),
                qwen_output_tokens=review_qwen_tokens,
                qwen_tokens_per_sec=review_qwen_tokens / max(review_elapsed, 1e-9),
            ))
            trajectory.append({
                'step': 0,
                'type': 'outer_review',
                'source_outer': outer_record['source_outer'],
                'source_confidence': outer_record['source_confidence'],
                'assessment': outer_record.get('feedback', {}).get('assessment'),
                'error': outer_record.get('error'),
                'api_time': round(review_elapsed, 2),
                'usage': review_usage,
            })
            handoff = HLE_REVIEW.OUTER_HANDOFF_PROMPT.format(
                outer_round=outer_resume_meta['outer_round'],
                draft=json.dumps(draft, ensure_ascii=False),
                feedback=feedback_payload,
            )
            handoff_message = {'role': 'user', 'content': handoff}
            messages.append(handoff_message)
            managed_messages.append(handoff_message)
            first_solver_step = 1
            print(
                f"  [Q {question['id']}] Outer review from "
                f"{outer_record['source_outer']} to {outer_resume_meta['outer_round']}, "
                f"step=0/{args.max_steps}"
            )

        for step in range(first_solver_step, args.max_steps):
            if pending_review:
                # A review consumes this loop iteration just like a solver call.
                pending_review = False
                review_count += 1
                review_input = HLE_REVIEW.build_review_messages(
                    question, saved_finish, managed_messages, review_middle,
                    max_chars=max(1, min(180000, args.max_context_tokens)),
                )
                review_record = {'step': step, 'round': review_count, 'messages': review_input}
                review_usage = {}
                review_finish_reason = 'error'
                review_content = ''
                review_reasoning = ''
                review_started = time.time()
                try:
                    review_response = await client.chat.completions.create(
                        **build_chat_completion_kwargs(args, review_max_tokens, review_input)
                    )
                    if review_response.usage:
                        review_usage = json.loads(review_response.usage.model_dump_json())
                        for key in total_usage:
                            total_usage[key] += review_usage.get(key, 0) or 0
                    review_choice = review_response.choices[0]
                    review_finish_reason = review_choice.finish_reason
                    review_content = review_choice.message.content or ''
                    review_reasoning = getattr(review_choice.message, 'reasoning_content', None) or ''
                    if review_finish_reason == 'length':
                        raise ValueError('Reviewer output was truncated')
                    feedback = HLE_REVIEW.parse_review(review_content)
                    review_record['feedback'] = feedback
                    observation = HLE_REVIEW.HANDOFF_PROMPT.format(
                        feedback=json.dumps(feedback, ensure_ascii=False)
                    )
                except Exception as error:
                    review_record['error'] = str(error)[:1000]
                    observation = (
                        'The independent reviewer returned no usable feedback. Your previous '
                        'finish draft is saved. Check the weakest step of your solution and '
                        'submit finish again with an honest numeric confidence. '
                        'Reviewer failure is not evidence against your answer.'
                    )
                review_elapsed = time.time() - review_started
                review_record.update({
                    'raw_content': review_content, 'reasoning': review_reasoning,
                    'usage': review_usage, 'api_time': round(review_elapsed, 2),
                })
                review_meta['reviews'].append(review_record)
                review_qwen_tokens = len(_qwen_tok.encode(review_content + review_reasoning))
                model_call_usage.append(_build_model_call_usage(
                    step=step, call_type='review', finish_reason=review_finish_reason,
                    usage=review_usage, api_elapsed=review_elapsed,
                    tokens_per_sec=(review_usage.get('completion_tokens', 0) or 0) / max(review_elapsed, 1e-9),
                    qwen_output_tokens=review_qwen_tokens,
                    qwen_tokens_per_sec=review_qwen_tokens / max(review_elapsed, 1e-9),
                ))
                trajectory.append({
                    'step': step, 'type': 'review', 'round': review_count,
                    'assessment': review_record.get('feedback', {}).get('assessment'),
                    'error': review_record.get('error'),
                    'api_time': round(review_elapsed, 2), 'usage': review_usage,
                })
                feedback_message = tool_response_message(observation)
                messages.append(feedback_message)
                managed_messages.append(feedback_message)
                print(f"  [Q {question['id']}] Review {review_count}/{review_limit}, step={step}/{args.max_steps}")
                if total_token_budget_reached(review_usage):
                    return failure_result('Review exceeded token budget', {
                        'step': step, 'type': 'max_total_tokens_exceeded',
                    })
                continue

            if saved_finish is not None and step == args.max_steps - 1:
                budget_message = tool_response_message(
                    'This is your final model call within the shared solver and review budget. '
                    'Submit finish now with the best supported answer and an honest numeric '
                    'confidence from 0 to 100. State any unresolved uncertainty in the answer.'
                )
                messages.append(budget_message)
                managed_messages.append(budget_message)
            managed_messages, context_info = apply_context_management(
                managed_messages,
                args.max_context_tokens,
                last_prompt_tokens,
                return_info=True,
            )
            if context_info["applied"]:
                context_management_steps.append({
                    "step": step,
                    "trigger": "budget_check",
                    **context_info,
                })

            max_tokens = (
                min(args.max_completion_tokens, args.truncation_max_completion_tokens)
                if length_streak
                else args.max_completion_tokens
            )

            response = None
            max_retries = max(1, int(getattr(args, "llm_call_max_retries", 20)))
            t0 = time.time()
            for attempt in range(max_retries):
                try:
                    response = await client.chat.completions.create(
                        **build_chat_completion_kwargs(args, max_tokens, managed_messages)
                    )
                    break
                except Exception as e:
                    err_str = str(e)
                    is_context_overflow = any(k in err_str.lower() for k in [
                        "context length", "token limit", "maximum context",
                        "too many tokens", "context_length_exceeded",
                    ])
                    if is_context_overflow:
                        print(f"  [Q {question['id']}] Context overflow at step {step}, "
                              f"forcing context management and retrying")
                        managed_messages, context_info = apply_context_management(
                            managed_messages,
                            0,
                            last_prompt_tokens,
                            return_info=True,
                        )
                        if context_info["applied"]:
                            context_management_steps.append({
                                "step": step,
                                "trigger": "context_overflow_retry",
                                "attempt": attempt + 1,
                                **context_info,
                            })
                            continue
                        failure_content = "[EVALUATION_FAILED_CONTEXT_OVERFLOW] The prompt exceeded the serving context window even after context management."
                        return failure_result(failure_content, {
                            "step": step,
                            "type": "context_overflow",
                            "api_time": round(time.time() - t0, 2),
                            "error": err_str[:1000],
                            "context_management": context_info,
                        })
                    is_transient = any(k in err_str.lower() for k in [
                        "429", "overloaded", "timeout", "connection", "rate",
                        "server_error", "502", "503", "504",
                    ])
                    if is_transient and attempt < max_retries - 1:
                        wait = min(2 ** attempt * 5, 120)
                        print(f"  [Q {question['id']}] Transient error at step {step} "
                              f"(attempt {attempt+1}/{max_retries}), retrying in {wait}s: {e}")
                        await asyncio.sleep(wait)
                    else:
                        print(f"  [Q {question['id']}] API error at step {step} "
                              f"(attempt {attempt+1}/{max_retries}): {e}")
                        break
            api_elapsed = time.time() - t0

            if response is None:
                return failure_result('[EVALUATION_FAILED_API] No model response.', {
                    'step': step, 'type': 'api_error',
                })

            step_usage = {}
            if response.usage:
                step_usage = json.loads(response.usage.model_dump_json())
                for k in total_usage:
                    total_usage[k] += step_usage.get(k, 0)
                last_prompt_tokens = step_usage.get("prompt_tokens", 0) + step_usage.get("completion_tokens", 0)

            completion_tokens = step_usage.get("completion_tokens", 0)
            step_total_tokens = current_total_tokens(step_usage)
            tokens_per_sec = completion_tokens / api_elapsed if api_elapsed > 0 else 0

            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            _content = msg.content or ""
            _reasoning = getattr(msg, "reasoning_content", None) or ""
            qwen_out = len(_qwen_tok.encode(_content + _reasoning))
            qwen_tps = qwen_out / api_elapsed if api_elapsed > 0 else 0

            print(f"  [Q {question['id']}] Step {step}/{args.max_steps}, "
                  f"finish={finish_reason}, "
                  f"prompt={step_usage.get('prompt_tokens', '?')}, "
                  f"completion={completion_tokens} ({tokens_per_sec:.1f} tok/s), "
                  f"total={step_total_tokens}/{args.max_total_tokens}, "
                  f"qwen_out={qwen_out} ({qwen_tps:.1f} tok/s), "
                  f"api={api_elapsed:.1f}s")

            if finish_reason == "length":
                model_call_usage.append(_build_model_call_usage(
                    step=step,
                    call_type="truncated",
                    finish_reason=finish_reason,
                    usage=step_usage,
                    api_elapsed=api_elapsed,
                    tokens_per_sec=tokens_per_sec,
                    qwen_output_tokens=qwen_out,
                    qwen_tokens_per_sec=qwen_tps,
                ))
                length_streak += 1
                reasoning = getattr(msg, "reasoning_content", None)
                assistant_msg = {"role": "assistant", "content": msg.content}
                if reasoning:
                    assistant_msg["reasoning_content"] = reasoning
                messages.append(assistant_msg)
                managed_messages.append(assistant_msg)

                trajectory.append({
                    "step": step,
                    "type": "truncated",
                    "partial_content_chars": len(msg.content or ""),
                    "partial_reasoning_chars": len(reasoning or ""),
                    "api_time": round(api_elapsed, 2),
                    "usage": step_usage,
                    "tokens_per_sec": round(tokens_per_sec, 1),
                })
                if total_token_budget_reached(step_usage):
                    failure_content = "[EVALUATION_FAILED_MAX_TOTAL_TOKENS] Reached max_total_tokens after a truncated response."
                    return failure_result(failure_content, {
                        "step": step,
                        "type": "max_total_tokens_exceeded",
                        "current_total_tokens": step_total_tokens,
                        "cumulative_total_tokens": total_usage.get("total_tokens", 0),
                        "max_total_tokens": args.max_total_tokens,
                    })

                print(f"  [Q {question['id']}] Output truncated at step {step}, "
                      f"saved partial output and prompting agent to continue concisely")
                truncation_msg = {
                    "role": "user",
                    "content": (
                        "Your previous output was truncated because it exceeded the "
                        "length limit, but the partial output above has been preserved. "
                        "Continue from exactly where you stopped; do not restart or "
                        "repeat earlier reasoning. Be concise and either make the next "
                        "tool call or call the finish tool immediately."
                    ),
                }
                messages.append(truncation_msg)
                managed_messages.append(truncation_msg)
                continue
            length_streak = 0

            reasoning = getattr(msg, "reasoning_content", None)
            assistant_content = msg.content or ""
            assistant_msg = {"role": "assistant", "content": assistant_content}
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning
            messages.append(assistant_msg)
            managed_messages.append(assistant_msg)

            fn_call_list = parse_qwen_tool_calls(assistant_content)

            if not fn_call_list:
                model_call_usage.append(_build_model_call_usage(
                    step=step,
                    call_type="no_tool_call",
                    finish_reason=finish_reason,
                    usage=step_usage,
                    api_elapsed=api_elapsed,
                    tokens_per_sec=tokens_per_sec,
                    qwen_output_tokens=qwen_out,
                    qwen_tokens_per_sec=qwen_tps,
                ))
                tool_call_regen_retry_count += 1
                if tool_call_regen_retry_count <= args.tool_call_regen_max_retries:
                    if messages and messages[-1] is assistant_msg:
                        messages.pop()
                    elif messages and messages[-1].get("role") == "assistant" and messages[-1].get("content") == assistant_content:
                        messages.pop()
                    if managed_messages and managed_messages[-1] is assistant_msg:
                        managed_messages.pop()
                    elif managed_messages and managed_messages[-1].get("role") == "assistant" and managed_messages[-1].get("content") == assistant_content:
                        managed_messages.pop()
                    trajectory.append({
                        "step": step,
                        "type": "tool_call_regen_retry",
                        "reason": "no_valid_tool_call",
                        "retry_count": tool_call_regen_retry_count,
                        "max_retries": args.tool_call_regen_max_retries,
                        "removed_content_chars": len(assistant_content),
                        "api_time": round(api_elapsed, 2),
                        "usage": step_usage,
                        "tokens_per_sec": round(tokens_per_sec, 1),
                    })
                    print(
                        f"  [Q {question['id']}] No valid tool call at step {step}; "
                        f"regenerate assistant turn "
                        f"({tool_call_regen_retry_count}/{args.tool_call_regen_max_retries})"
                    )
                    continue

                observation = (
                    "[ERROR] No valid tool call was detected. Respond with exactly one "
                    "valid XML tool call. If you are ready to answer, call the finish "
                    "tool; otherwise call an available research/tool function. Do not "
                    "answer in plain text."
                )
                messages.append({"role": "user", "content": observation})
                managed_messages.append({"role": "user", "content": observation})
                trajectory.append({
                    "step": step,
                    "type": "tool_call_regen_fallback",
                    "reason": "no_valid_tool_call",
                    "retry_count": tool_call_regen_retry_count,
                    "max_retries": args.tool_call_regen_max_retries,
                    "api_time": round(api_elapsed, 2),
                    "usage": step_usage,
                    "tokens_per_sec": round(tokens_per_sec, 1),
                })
                tool_call_regen_retry_count = 0
                continue

            tool_call_regen_retry_count = 0
            tool_names = [call.get('function', '') for call in fn_call_list]
            model_call_usage.append(_build_model_call_usage(
                step=step,
                call_type="tool_call",
                finish_reason=finish_reason,
                usage=step_usage,
                api_elapsed=api_elapsed,
                tokens_per_sec=tokens_per_sec,
                qwen_output_tokens=qwen_out,
                qwen_tokens_per_sec=qwen_tps,
                tool_calls=tool_names,
            ))

            if any(call.get('function') == 'finish' for call in fn_call_list):
                if len(fn_call_list) != 1:
                    observation = "[ERROR] When calling `finish`, call it separately and do not call other tools in the same response."
                    tool_msg = tool_response_message(observation)
                    messages.append(tool_msg)
                    managed_messages.append(tool_msg)
                    trajectory.append({
                        "step": step,
                        "type": "tool_error",
                        "tool": "finish",
                        "args_summary": "finish mixed with other tools",
                        "api_time": round(api_elapsed, 2),
                        "usage": step_usage,
                        "tokens_per_sec": round(tokens_per_sec, 1),
                    })
                    continue
                finish_args = normalize_tool_arguments('finish', fn_call_list[0].get('arguments') or {})
                if finish_args.get('confidence') is None:
                    observation = (
                        "[ERROR] The `finish` confidence is required and must be a "
                        "numeric value from 0 to 100. Do not leave it blank or include "
                        "a percent sign. Continue the task and call `finish` again with "
                        "a valid numeric confidence."
                    )
                    tool_msg = tool_response_message(observation)
                    messages.append(tool_msg)
                    managed_messages.append(tool_msg)
                    trajectory.append({
                        "step": step,
                        "type": "tool_error",
                        "tool": "finish",
                        "args_summary": "missing, non-numeric, or out-of-range confidence",
                        "api_time": round(api_elapsed, 2),
                        "usage": step_usage,
                        "tokens_per_sec": round(tokens_per_sec, 1),
                    })
                    continue
                saved_finish = copy.deepcopy(finish_args)
                saved_reasoning = reasoning
                review_meta['drafts'].append({
                    'step': step, **copy.deepcopy(finish_args), 'reasoning': reasoning,
                })
                needs_review = review_enabled and finish_args['confidence'] < review_threshold
                if (needs_review and review_count < review_limit
                        and args.max_steps - step - 1 >= 3
                        and not total_token_budget_reached(step_usage)):
                    trajectory.append({
                        'step': step, 'type': 'finish_draft',
                        'confidence': finish_args['confidence'],
                        'api_time': round(api_elapsed, 2), 'usage': step_usage,
                    })
                    pending_review = True
                    continue
                review_meta['stop_reason'] = (
                    'review_disabled' if not review_enabled else
                    'confidence_threshold' if not needs_review else
                    'review_limit' if review_count >= review_limit else 'insufficient_budget'
                )
                final_answer_override = finish_args.get('answer') or assistant_content
                final_reasoning_override = reasoning
                trajectory.append({
                    "step": step,
                    "type": "finish",
                    "api_time": round(api_elapsed, 2),
                    "usage": step_usage,
                    "tokens_per_sec": round(tokens_per_sec, 1),
                    "confidence": finish_args.get('confidence', ''),
                    "evidences_count": len(finish_args.get('evidences') or []),
                })
                completed_final_answer = True
                break

            if any(call.get('function') == 'update_context' for call in fn_call_list):
                if len(fn_call_list) != 1:
                    observation = "[ERROR] When calling `update_context`, call it separately and do not call other tools in the same response."
                    tool_msg = tool_response_message(observation)
                    messages.append(tool_msg)
                    managed_messages.append(tool_msg)
                    trajectory.append({
                        "step": step,
                        "type": "tool_error",
                        "tool": "update_context",
                        "args_summary": "update_context mixed with other tools",
                        "api_time": round(api_elapsed, 2),
                        "usage": step_usage,
                        "tokens_per_sec": round(tokens_per_sec, 1),
                    })
                    continue

                update_args = normalize_tool_arguments(
                    'update_context', fn_call_list[0].get('arguments') or {}
                )
                new_context = update_args.get('context', '')
                if not new_context:
                    observation = "[ERROR] The `update_context` context parameter must be a non-empty string."
                    tool_msg = tool_response_message(observation)
                    messages.append(tool_msg)
                    managed_messages.append(tool_msg)
                    trajectory.append({
                        "step": step,
                        "type": "tool_error",
                        "tool": "update_context",
                        "args_summary": "empty context",
                        "api_time": round(api_elapsed, 2),
                        "usage": step_usage,
                        "tokens_per_sec": round(tokens_per_sec, 1),
                    })
                    continue

                messages_before_update = len(managed_messages)
                context_tokens = len(_qwen_tok.encode(new_context))
                updated_messages = copy.deepcopy(initial_messages)
                update_notice = (
                    "You have called the `update_context` function. The context "
                    "updated is:\n" + new_context +
                    "\n\nReflect on this retained information and continue researching "
                    "if the answer is not fully verified."
                )
                user_content = updated_messages[-1].get('content')
                if isinstance(user_content, str):
                    updated_messages[-1]['content'] = user_content + "\n\n" + update_notice
                elif isinstance(user_content, list):
                    user_content.append({'type': 'text', 'text': "\n\n" + update_notice})
                else:
                    updated_messages.append({'role': 'user', 'content': update_notice})
                managed_messages = updated_messages
                last_prompt_tokens = None
                audit_msg = tool_response_message(
                    "Context updated successfully. Continue from the retained context."
                )
                messages.append(audit_msg)
                trajectory.append({
                    "step": step,
                    "type": "update_context",
                    "tool": "update_context",
                    "context_chars": len(new_context),
                    "context_tokens": context_tokens,
                    "messages_before": messages_before_update,
                    "messages_after": len(managed_messages),
                    "api_time": round(api_elapsed, 2),
                    "usage": step_usage,
                    "tokens_per_sec": round(tokens_per_sec, 1),
                })
                context_management_steps.append({
                    "step": step,
                    "trigger": "model_update_context",
                    "context_chars": len(new_context),
                    "context_tokens": context_tokens,
                    "messages_before": messages_before_update,
                    "messages_after": len(managed_messages),
                })
                print(
                    f"    tool=update_context, context={len(new_context)} chars, "
                    f"managed_messages={messages_before_update}->{len(managed_messages)}"
                )
                continue

            if total_token_budget_reached(step_usage):
                failure_content = "[EVALUATION_FAILED_MAX_TOTAL_TOKENS] Reached max_total_tokens before producing a final answer."
                return failure_result(failure_content, {
                    "step": step,
                    "type": "max_total_tokens_exceeded",
                    "current_total_tokens": step_total_tokens,
                    "cumulative_total_tokens": total_usage.get("total_tokens", 0),
                    "max_total_tokens": args.max_total_tokens,
                })

            for fn_call in fn_call_list:
                fn_name = fn_call.get('function', '')
                fn_args = normalize_tool_arguments(fn_name, fn_call.get('arguments') or {})

                tool_t0 = time.time()
                result = await execute_tool_async(fn_name, fn_args, namespace)
                tool_elapsed = time.time() - tool_t0

                if len(result) > 8000:
                    result = result[:8000] + "\n... [truncated]"

                trajectory.append({
                    "step": step,
                    "type": "tool_call",
                    "tool": fn_name,
                    "args_summary": str(fn_args)[:200],
                    "tool_time": round(tool_elapsed, 2),
                    "api_time": round(api_elapsed, 2),
                    "usage": step_usage,
                    "tokens_per_sec": round(tokens_per_sec, 1),
                })
                print(f"    tool={fn_name}, {tool_elapsed:.1f}s")

                tool_msg = tool_response_message(result)
                messages.append(tool_msg)
                managed_messages.append(tool_msg)

        if not completed_final_answer and model_call_usage and len(model_call_usage) >= args.max_steps:
            failure_content = "[EVALUATION_FAILED_MAX_STEPS] Reached max_steps before producing a final answer."
            return failure_result(failure_content, {
                "step": args.max_steps,
                "type": "max_steps_exceeded",
                "max_steps": args.max_steps,
            })

        final_content = final_answer_override
        final_reasoning = final_reasoning_override
        if final_content is None and final_reasoning is None:
            for m in reversed(messages):
                if m.get("role") == "assistant":
                    content = m.get("content")
                    reasoning = m.get("reasoning_content")
                    if content or reasoning:
                        final_content = content or ""
                        final_reasoning = reasoning
                        break

        if final_content is None and final_reasoning is None:
            print(f"  [Q {question['id']}] No content or reasoning found in {len(messages)} messages")
            return None

        return {
            "id": question["id"],
            "content": final_content or "",
            "reasoning": final_reasoning,
            "accepted_finish": saved_finish,
            "confidence_review": review_meta,
            "outer_resume": outer_resume_meta,
            "usage": total_usage,
            "model_call_usage": model_call_usage,
            "model_usage_summary": _summarize_model_call_usage(model_call_usage),
            "steps": len(trajectory),
            "trajectory": trajectory,
            "context_management_steps": context_management_steps,
            "messages": messages,
        }
    finally:
        _shutdown_python_interpreter(namespace)


# ---------------------------------------------------------------------------
# Async orchestration
# ---------------------------------------------------------------------------
async def attempt_question(question: dict, args):
    try:
        return await run_agent(question, args)
    except Exception as e:
        print(f"Error on question {question['id']}: {e}")
        traceback.print_exc()
        return None


def cleanup_stale_pkg_dirs(output_filepath: str):
    """Remove empty leftover per-question package dirs."""
    output_stem = os.path.splitext(os.path.basename(output_filepath))[0]
    env_root = os.path.join(os.path.dirname(output_filepath) or ".", f"{output_stem}_envs")
    if not os.path.isdir(env_root):
        return
    for name in os.listdir(env_root):
        path = os.path.join(env_root, name)
        if os.path.isdir(path) and not os.listdir(path):
            shutil.rmtree(path, ignore_errors=True)
    if not os.listdir(env_root):
        shutil.rmtree(env_root, ignore_errors=True)


async def attempt_all(questions: list, args, output_filepath):
    semaphore = asyncio.Semaphore(args.num_workers)
    lock = asyncio.Lock()
    saved_count = 0

    async def bound_func(question):
        nonlocal saved_count
        async with semaphore:
            result = await attempt_question(question, args)
            if result is not None:
                result_dict = {
                    "id": result["id"],
                    "accepted_finish": result.get('accepted_finish'),
                    "confidence_review": result.get('confidence_review', {}),
                    "outer_resume": result.get('outer_resume', {}),
                    "model": args.model,
                    "response": result["content"],
                    "reasoning": result["reasoning"],
                    "usage": result["usage"],
                    "model_call_usage": result["model_call_usage"],
                    "model_usage_summary": result["model_usage_summary"],
                    "steps": result["steps"],
                    "trajectory": result["trajectory"],
                    "context_management_steps": result["context_management_steps"],
                    "messages": result["messages"],
                }
                await save_result_jsonl(output_filepath, result_dict, lock)
                saved_count += 1
            return result

    tasks = [bound_func(q) for q in questions]
    await tqdm_asyncio.gather(*tasks)
    return saved_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_jsonl(filepath):
    questions = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def main(args):
    global client
    assert args.num_workers >= 1, "num_workers must be >= 1"
    if args.max_total_tokens is None:
        args.max_total_tokens = args.max_context_tokens
    client = build_openai_client(args)

    all_questions = load_jsonl(args.dataset)
    print(f"Loaded {len(all_questions)} questions from {args.dataset}")

    if args.text_only:
        all_questions = [q for q in all_questions if not q.get("image")]
        print(f"After filtering image questions: {len(all_questions)}")

    random.seed(args.seed)
    if args.num_samples and args.num_samples < len(all_questions):
        questions = random.sample(all_questions, args.num_samples)
        print(f"Randomly sampled {args.num_samples} questions (seed={args.seed})")
    else:
        questions = all_questions

    output_filepath = (
        args.output if args.output
        else f"hle_agent_{os.path.basename(args.model)}.jsonl"
    )
    args.output = output_filepath
    os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)

    existing = load_predictions_jsonl(output_filepath)
    questions = [q for q in questions if q["id"] not in existing]
    if existing:
        print(f"Skipping {len(existing)} already completed, {len(questions)} remaining")

    if not questions:
        print("All questions already answered. Nothing to do.")
        return

    saved = asyncio.run(attempt_all(questions, args, output_filepath))
    cleanup_stale_pkg_dirs(output_filepath)
    total = len(existing) + saved
    print(f"Saved {saved} new predictions ({total} total) to {output_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen XML-tool Agent evaluation on HLE")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(SCRIPT_DIR / "data_json" / "text_items.jsonl"),
    )
    parser.add_argument("--model", type=str, default="Qwen3.5-4B")
    parser.add_argument("--sdk_base_url", type=str, default=None,
                        help="OpenAI-compatible base URL for the target Qwen model")
    parser.add_argument("--sdk_api_key", type=str, default=None,
                        help="API key for the OpenAI-compatible target model service")
    parser.add_argument("--max_completion_tokens", type=int, default=32768,
                        help="Max tokens per LLM call")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--presence_penalty", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--preserve_thinking", action="store_true")
    visit_fallback_group = parser.add_mutually_exclusive_group()
    visit_fallback_group.add_argument(
        "--enable-visit-fallback",
        "--enable_visit_fallback",
        dest="enable_visit_fallback",
        action="store_true",
        help="Use resilient visit handling with no-cache retry and local PDF parsing (default).",
    )
    visit_fallback_group.add_argument(
        "--disable-visit-fallback",
        "--disable_visit_fallback",
        dest="enable_visit_fallback",
        action="store_false",
        help="Use the legacy visit behavior.",
    )
    parser.set_defaults(enable_visit_fallback=True)
    parser.add_argument("--truncation_max_completion_tokens", type=int, default=8192,
                        help="Max tokens for calls after a length-truncated response")
    parser.add_argument("--max_context_tokens", type=int, default=200000,
                        help="Prompt context token budget for Hide-Tool-Result management")
    parser.add_argument("--max_total_tokens", type=int, default=None,
                        help="Per-request prompt+completion token budget; stop if a single call reaches this limit")
    parser.add_argument('--enable-confidence-review', action='store_true')
    parser.add_argument('--review-threshold', type=float, default=95)
    parser.add_argument('--review-middle-threshold', type=float, default=90)
    parser.add_argument('--review-max-rounds', type=int, default=2)
    parser.add_argument('--review-max-tokens', type=int, default=4096)
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Max model-call agentic rounds per question")
    parser.add_argument("--tool_call_regen_max_retries", type=int, default=20,
                        help="Assistant regeneration retries for responses without a valid tool call before injecting user feedback")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Async concurrency limit (lower than non-agent due to multi-step)")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--text_only", action="store_true")
    args = parser.parse_args()
    main(args)
