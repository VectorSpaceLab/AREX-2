import json
import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional


def _env_flag(name: str, default: str = "1") -> bool:
    value = os.environ.get(name, default)
    return str(value).strip().lower() not in ("0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False
    Console = None
    Panel = None
    Progress = None
    Text = None
    Table = None
    Syntax = None


def _shorten(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception:
            value = str(value)
    text = value.replace("\x00", "")
    if limit <= 0 or len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n...[truncated {omitted} chars]"


def _format_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    try:
        seconds = float(seconds)
    except Exception:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def summarize_tool_calls(fn_call_list: Any) -> str:
    calls = fn_call_list or []
    if not isinstance(calls, list):
        return _shorten(calls, 1200)
    lines = []
    for idx, call in enumerate(calls, 1):
        if not isinstance(call, dict):
            lines.append(f"{idx}. {call}")
            continue
        name = call.get("function") or "UNKNOWN"
        args = call.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        arg_bits = []
        if isinstance(args, dict):
            for key in ("query", "url", "goal", "path", "context", "answer", "confidence"):
                if key not in args:
                    continue
                val = args.get(key)
                if isinstance(val, (list, tuple)):
                    val = ", ".join(str(item)[:120] for item in val[:3])
                arg_bits.append(f"{key}={str(val)[:240]}")
        else:
            arg_bits.append(str(args)[:240])
        suffix = "; ".join(arg_bits) if arg_bits else "no args"
        lines.append(f"{idx}. {name}: {suffix}")
    return "\n".join(lines)


class PrettyConsole:
    def __init__(self):
        self.enabled = _env_flag("PRETTY_EVAL", "1")
        self.detail_limit = max(0, _env_int("PRETTY_EVAL_DETAIL_LIMIT", 3))
        self.min_interval = max(0.0, _env_float("PRETTY_EVAL_MIN_INTERVAL_SECONDS", 1.2))
        self.max_llm_chars = max(200, _env_int("PRETTY_EVAL_MAX_LLM_CHARS", 5000))
        self.max_tool_chars = max(200, _env_int("PRETTY_EVAL_MAX_TOOL_CHARS", 4000))
        self.easter_mode = os.environ.get("PRETTY_EVAL_EASTER_EGGS", "active").strip().lower()
        self.output_mode = os.environ.get("PRETTY_EVAL_OUTPUT", "tty").strip().lower()
        self._stream = self._open_stream()
        self.rich_enabled = bool(self.enabled and RICH_AVAILABLE)
        self.console = self._make_console()
        self.progress = None
        self.task_id = None
        self.total = 0
        self.completed = 0
        self.run_started_at = None
        self.case_started_at: Dict[str, float] = {}
        self.case_meta: Dict[str, Dict[str, Any]] = {}
        self.case_last_event: Dict[str, float] = {}
        self.case_last_print: Dict[str, float] = {}
        self._last_egg_at = 0.0
        self._eggs = [
            "Still searching the evidence graph. Good science takes a few careful hops.",
            "Tiny status update: the agent is negotiating with web pages and token budgets.",
            "Background hypothesis: better logs make long evals feel 37 percent less eternal.",
            "The progress bar is alive. The benchmark is thinking. We are supervising calmly.",
            "Evidence first, vibes second. But a little terminal color is allowed.",
            "Long context detected in spirit. Snacks and patience remain valid infrastructure.",
        ]

    def _open_stream(self):
        if self.output_mode == "tty":
            try:
                return open("/dev/tty", "w", encoding="utf-8", buffering=1)
            except Exception:
                return sys.stdout
        return sys.stdout

    def _make_console(self):
        if not self.rich_enabled:
            return None
        try:
            force_terminal = bool(getattr(self._stream, "isatty", lambda: False)())
            return Console(
                file=self._stream,
                force_terminal=force_terminal,
                color_system="auto",
                soft_wrap=True,
                highlight=False,
            )
        except Exception:
            self.rich_enabled = False
            return None

    def _plain(self, message: str):
        if not self.enabled:
            return
        print(message, file=self._stream, flush=True)

    def _print(self, renderable):
        if not self.enabled:
            return
        if self.rich_enabled and self.console is not None:
            self.console.print(renderable)
        else:
            self._plain(str(renderable))

    def _touch(self, case_key: Optional[str]):
        if not case_key:
            return
        self.case_last_event[case_key] = time.time()

    def _focused_keys(self):
        if self.detail_limit <= 0:
            return set()
        active = [
            (last, key)
            for key, last in self.case_last_event.items()
            if key in self.case_started_at
        ]
        active.sort(reverse=True)
        return {key for _, key in active[: self.detail_limit]}

    def is_focused(self, case_key: Optional[str]) -> bool:
        if not self.enabled or not case_key:
            return False
        return case_key in self._focused_keys()

    def should_print_detail(self, case_key: Optional[str], force: bool = False) -> bool:
        if not self.is_focused(case_key):
            return False
        if force:
            return True
        now = time.time()
        last = self.case_last_print.get(str(case_key), 0.0)
        if now - last < self.min_interval:
            return False
        self.case_last_print[str(case_key)] = now
        return True

    def start_run(self, total: int, metadata: Optional[Dict[str, Any]] = None):
        if not self.enabled:
            return
        self.total = max(0, int(total or 0))
        self.completed = 0
        self.run_started_at = time.time()
        title = "Pretty Eval started"
        body = metadata or {}
        if self.rich_enabled and self.console is not None and Progress is not None:
            table = self._kv_table(body)
            self._print(Panel(table, title=title, border_style="bright_cyan"))
            self.progress = Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                TextColumn("done={task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                refresh_per_second=1,
                transient=False,
            )
            self.progress.start()
            self.task_id = self.progress.add_task("Unified eval", total=self.total)
        else:
            self._plain(f"[PRETTY_EVAL] started total={self.total} metadata={body}")
        self.maybe_easter_egg(force=True)

    def finish_run(self, status: str = "finished"):
        if not self.enabled:
            return
        elapsed = _format_seconds(time.time() - self.run_started_at) if self.run_started_at else "n/a"
        if self.progress is not None:
            try:
                self.progress.stop()
            except Exception:
                pass
            self.progress = None
        style = "green" if status == "finished" else "red"
        self.panel("Run", f"status={status}\ncompleted={self.completed}/{self.total}\nelapsed={elapsed}", style=style)

    def dataset(self, name: str, loaded: int, selected: int, start_index: int, end_index: int, shuffle: bool, save_path: str):
        body = {
            "dataset": name,
            "loaded": loaded,
            "selected": selected,
            "start_index": start_index,
            "end_index": end_index,
            "shuffle": shuffle,
            "save_path": save_path,
        }
        if self.rich_enabled:
            self._print(Panel(self._kv_table(body), title="Dataset", border_style="magenta"))
        else:
            self._plain(f"[DATASET] {body}")

    def return_selection(self, name: str, selected: int):
        self.line("RETURN", f"dataset={name} selected={selected} score=0 no-finish indices", style="yellow")

    def case_start(self, case_key: str, dataset: str, idx: Any, sample_id: Any, attempt: int):
        self._touch(case_key)
        self.case_started_at[case_key] = time.time()
        self.case_meta[case_key] = {
            "dataset": dataset,
            "idx": idx,
            "sample_id": sample_id,
            "attempt": attempt,
        }
        if self.should_print_detail(case_key, force=True):
            self.line("CASE", f"start dataset={dataset} idx={idx} sample_id={sample_id} attempt={attempt}", style="cyan")

    def case_done(self, case_key: str, score: Any, status: str, call_stats: Optional[Dict[str, Any]] = None, case_duration: Optional[float] = None):
        self._touch(case_key)
        self.completed += 1
        if self.progress is not None and self.task_id is not None:
            try:
                self.progress.update(self.task_id, completed=min(self.completed, self.total))
            except Exception:
                pass
        stats = call_stats or {}
        meta = self.case_meta.get(case_key, {})
        message = (
            f"done dataset={meta.get('dataset')} idx={meta.get('idx')} "
            f"score={score} status={status} duration={_format_seconds(case_duration)} "
            f"assistant={stats.get('assistant_calls_total', 0)} "
            f"search={stats.get('search_calls_total', 0)} visit={stats.get('visit_calls_total', 0)} "
            f"llm_time={_format_seconds(stats.get('llm_elapsed_seconds_total'))} "
            f"search_time={_format_seconds(stats.get('search_elapsed_seconds_total'))} "
            f"visit_time={_format_seconds(stats.get('visit_elapsed_seconds_total'))} "
            f"update_context={stats.get('update_context_calls_total', 0)} "
            f"update_time={_format_seconds(stats.get('update_context_elapsed_seconds_total'))}"
        )
        self.line("EVAL", message, style="green" if str(status).lower() != "error" else "red")
        self.case_started_at.pop(case_key, None)
        self.maybe_easter_egg()

    def case_retry(self, case_key: str, message: str):
        self._touch(case_key)
        self.line("RETRY", message, style="yellow")
        self.case_started_at.pop(case_key, None)

    def case_error(self, case_key: str, error_type: str, message: str, case_duration: Optional[float] = None):
        self._touch(case_key)
        self.completed += 1
        if self.progress is not None and self.task_id is not None:
            try:
                self.progress.update(self.task_id, completed=min(self.completed, self.total))
            except Exception:
                pass
        meta = self.case_meta.get(case_key, {})
        self.line(
            "ERROR",
            (
                f"dataset={meta.get('dataset')} idx={meta.get('idx')} "
                f"type={error_type} duration={_format_seconds(case_duration)} message={message}"
            ),
            style="red",
        )
        self.case_started_at.pop(case_key, None)

    def llm(self, case_key: Optional[str], idx: Any, round_id: Any, elapsed: float, response: Any, fn_call_list: Any, tokens: Optional[Dict[str, Any]] = None):
        self._touch(case_key)
        if not self.should_print_detail(case_key):
            return
        body = [
            f"idx={idx} round={round_id} elapsed={_format_seconds(elapsed)}",
        ]
        if tokens:
            body.append(
                "tokens current={current} all={all_tokens} llm={llm} input={input} output={output}".format(
                    current=tokens.get("current_tokens"),
                    all_tokens=tokens.get("all_tokens"),
                    llm=tokens.get("llm_tokens"),
                    input=tokens.get("total_input_tokens"),
                    output=tokens.get("total_output_tokens"),
                )
            )
        body.append("\nTool calls:\n" + summarize_tool_calls(fn_call_list))
        body.append("\nLLM output preview:\n" + _shorten(response, self.max_llm_chars))
        self.panel("LLM Output", "\n".join(body), style="bright_blue")

    def tool_call(self, case_key: Optional[str], idx: Any, action: str, arguments: Any):
        self._touch(case_key)
        if not self.should_print_detail(case_key):
            return
        text = f"idx={idx} action={action}\narguments:\n{_shorten(arguments, 1600)}"
        self.panel("Tool Call", text, style="yellow")

    def tool_result(self, case_key: Optional[str], idx: Any, action: str, elapsed: float, observation: Any):
        self._touch(case_key)
        if not self.should_print_detail(case_key):
            return
        text = (
            f"idx={idx} action={action} elapsed={_format_seconds(elapsed)} "
            f"chars={len(str(observation or ''))}\n\n{_shorten(observation, self.max_tool_chars)}"
        )
        self.panel("Tool Result", text, style="green")

    def timing(self, case_key: Optional[str], idx: Any, round_id: Any, turn_elapsed: float):
        self._touch(case_key)
        if not self.should_print_detail(case_key):
            return
        self.line("TIMING", f"idx={idx} round={round_id} turn={_format_seconds(turn_elapsed)}", style="bright_black")

    def usage(self, case_key: Optional[str], message: str):
        self._touch(case_key)
        if self.should_print_detail(case_key):
            self.line("USAGE", message, style="bright_black")

    def tool_event(self, action: str, target: Any, detail: str = "", case_key: Optional[str] = None, style: str = "cyan"):
        self._touch(case_key)
        if case_key and not self.should_print_detail(case_key):
            return
        self.line("TOOL", f"{action}: {str(target)[:500]} {detail}".strip(), style=style)

    def warning(self, message: str, case_key: Optional[str] = None):
        self._touch(case_key)
        if case_key and not self.should_print_detail(case_key):
            return
        self.line("WARN", message, style="yellow")

    def error(self, message: str, case_key: Optional[str] = None):
        self._touch(case_key)
        if case_key and not self.should_print_detail(case_key):
            return
        self.line("ERROR", message, style="red")

    def line(self, label: str, message: str, style: str = "white"):
        if not self.enabled:
            return
        if self.rich_enabled and Text is not None:
            text = Text()
            text.append(f"[{label}] ", style=f"bold {style}")
            text.append(str(message))
            self._print(text)
        else:
            self._plain(f"[{label}] {message}")

    def panel(self, title: str, body: str, style: str = "white"):
        if not self.enabled:
            return
        if self.rich_enabled and Panel is not None:
            panel_body = _shorten(body, max(self.max_llm_chars, self.max_tool_chars))
            renderable = Text(panel_body) if Text is not None else panel_body
            self._print(Panel(renderable, title=title, border_style=style))
        else:
            self._plain(f"\n[{title}]\n{body}\n")

    def maybe_easter_egg(self, force: bool = False):
        if not self.enabled or self.easter_mode in ("0", "off", "none", "false"):
            return
        now = time.time()
        interval = 180.0 if self.easter_mode == "active" else 420.0
        if not force and now - self._last_egg_at < interval:
            return
        self._last_egg_at = now
        self.line("WAIT", random.choice(self._eggs), style="bright_magenta")

    def _kv_table(self, values: Dict[str, Any]):
        if not self.rich_enabled or Table is None:
            return str(values)
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("key", style="cyan")
        table.add_column("value", style="white")
        for key, value in values.items():
            table.add_row(Text(str(key)), Text(str(value)))
        return table

    @contextmanager
    def run_context(self, total: int, metadata: Optional[Dict[str, Any]] = None):
        self.start_run(total, metadata)
        status = "finished"
        try:
            yield self
        except Exception:
            status = "failed"
            raise
        finally:
            self.finish_run(status)


_GLOBAL_CONSOLE: Optional[PrettyConsole] = None


def get_pretty_console() -> PrettyConsole:
    global _GLOBAL_CONSOLE
    if _GLOBAL_CONSOLE is None:
        _GLOBAL_CONSOLE = PrettyConsole()
    return _GLOBAL_CONSOLE


def case_key(dataset: str, idx: Any, attempt: int = 0) -> str:
    return f"{dataset}:row{idx}:attempt{attempt}"
