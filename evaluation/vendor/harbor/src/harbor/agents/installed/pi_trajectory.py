"""Convert Pi's JSONL event stream to Harbor's ATIF trajectory format."""

from __future__ import annotations

import base64
import json
import uuid
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, override

from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.utils.trajectory_utils import format_trajectory_json


class _SessionDataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_session_data = False
        self.parts: list[str] = []

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script" and dict(attrs).get("id") == "session-data":
            self._in_session_data = True

    @override
    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_session_data:
            self._in_session_data = False

    @override
    def handle_data(self, data: str) -> None:
        if self._in_session_data:
            self.parts.append(data)


def _session_metadata_from_html(
    path: Path,
) -> tuple[str | None, list[dict[str, Any]] | None]:
    try:
        parser = _SessionDataParser()
        parser.feed(path.read_text(encoding="utf-8"))
        encoded = "".join(parser.parts).strip()
        if not encoded:
            return None, None
        session_data = json.loads(base64.b64decode(encoded).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None, None
    if not isinstance(session_data, dict):
        return None, None

    system_prompt = session_data.get("systemPrompt")
    if not isinstance(system_prompt, str) or not system_prompt:
        system_prompt = None

    raw_tools = session_data.get("tools")
    if not isinstance(raw_tools, list):
        return system_prompt, None

    tool_definitions: list[dict[str, Any]] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            continue
        name = raw_tool.get("name")
        description = raw_tool.get("description")
        parameters = raw_tool.get("parameters")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(description, str)
            or not isinstance(parameters, dict)
        ):
            continue
        tool_definitions.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )

    return system_prompt, tool_definitions


def _timestamp(value: Any) -> str | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, UTC).isoformat()
    if isinstance(value, str) and value:
        return value
    return None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)
        elif isinstance(part.get("thinking"), str):
            parts.append(part["thinking"])
    return "".join(parts)


def _assistant_content(content: Any) -> tuple[str, str | None, list[ToolCall]]:
    if not isinstance(content, list):
        return _content_text(content), None, []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text" and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        elif part_type in {"thinking", "reasoning"}:
            value = part.get("thinking") or part.get("text")
            if isinstance(value, str):
                reasoning_parts.append(value)
        elif part_type == "toolCall" and isinstance(part.get("name"), str):
            raw_arguments = part.get("arguments", {})
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments) if raw_arguments else {}
                except json.JSONDecodeError:
                    arguments = {"raw": raw_arguments}
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                arguments = {}
            tool_calls.append(
                ToolCall(
                    tool_call_id=str(part.get("id") or ""),
                    function_name=part["name"],
                    arguments=arguments,
                )
            )
    return "".join(text_parts), "\n".join(reasoning_parts) or None, tool_calls


def _metrics(usage: Any) -> Metrics | None:
    if not isinstance(usage, dict):
        return None
    prompt = int(usage.get("input") or 0)
    completion = int(usage.get("output") or 0)
    cached = int(usage.get("cacheRead") or 0)
    cache_write = int(usage.get("cacheWrite") or 0)
    cost = usage.get("cost") or {}
    total_cost = cost.get("total") if isinstance(cost, dict) else None
    if not any((prompt, completion, cached, cache_write, total_cost)):
        return None
    return Metrics(
        prompt_tokens=prompt + cached or None,
        completion_tokens=completion or None,
        cached_tokens=cached or None,
        cost_usd=float(total_cost) if total_cost is not None else None,
        extra={"cache_write_tokens": cache_write} if cache_write else None,
    )


def _tool_result_content(content: Any) -> str:
    text = _content_text(content)
    return text if text else json.dumps(content, ensure_ascii=False, default=str)


def _append_observation(step: Step, result: ObservationResult) -> None:
    if step.observation is None:
        step.observation = Observation(results=[result])
    else:
        step.observation.results.append(result)


def parse_pi_trajectory(
    path: Path,
    *,
    instruction: str,
    model_name: str | None,
    agent_version: str,
    system_prompt: str | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
) -> Trajectory | None:
    events: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    except OSError:
        return None

    if not events:
        return None

    session_id = next(
        (
            str(event.get("id"))
            for event in events
            if event.get("type") == "session" and event.get("id")
        ),
        str(uuid.uuid4()),
    )
    steps: list[Step] = []
    if system_prompt:
        steps.append(Step(step_id=1, source="system", message=system_prompt))
    call_to_step: dict[str, Step] = {}
    seen_results: set[str] = set()
    fallback_results: dict[str, dict[str, Any]] = {}
    total_prompt = total_completion = total_cached = 0
    total_cost = 0.0
    user_seen = False

    for event in events:
        event_type = event.get("type")
        if event_type == "tool_execution_end" and event.get("toolCallId"):
            fallback_results[str(event["toolCallId"])] = event

        if event_type == "message_start":
            message = event.get("message") or {}
            role = message.get("role")
            if role == "user":
                text = _content_text(message.get("content"))
                if text:
                    user_seen = True
                    steps.append(
                        Step(
                            step_id=len(steps) + 1,
                            timestamp=_timestamp(message.get("timestamp")),
                            source="user",
                            message=text,
                        )
                    )
            elif role == "toolResult":
                call_id = str(message.get("toolCallId") or "")
                if not call_id or call_id in seen_results:
                    continue
                step = call_to_step.get(call_id)
                if step is None:
                    continue
                result = ObservationResult(
                    source_call_id=call_id,
                    content=_tool_result_content(message.get("content")),
                    extra={"is_error": bool(message.get("isError"))}
                    if message.get("isError")
                    else None,
                )
                _append_observation(step, result)
                seen_results.add(call_id)

        if event_type != "message_end":
            continue
        message = event.get("message") or {}
        if message.get("role") != "assistant":
            continue
        text, reasoning, tool_calls = _assistant_content(message.get("content"))
        step = Step(
            step_id=len(steps) + 1,
            timestamp=_timestamp(message.get("timestamp")),
            source="agent",
            model_name=message.get("model") or model_name,
            message=text,
            reasoning_content=reasoning,
            tool_calls=tool_calls or None,
            metrics=_metrics(message.get("usage")),
            llm_call_count=1,
        )
        steps.append(step)
        for tool_call in tool_calls:
            if tool_call.tool_call_id:
                call_to_step[tool_call.tool_call_id] = step
        usage = message.get("usage") or {}
        if isinstance(usage, dict):
            total_prompt += int(usage.get("input") or 0)
            total_completion += int(usage.get("output") or 0)
            total_cached += int(usage.get("cacheRead") or 0)
            cost = usage.get("cost") or {}
            if isinstance(cost, dict):
                total_cost += float(cost.get("total") or 0)

    if not user_seen:
        insert_at = 1 if steps and steps[0].source == "system" else 0
        steps.insert(
            insert_at,
            Step(step_id=1, source="user", message=instruction),
        )
        for index, step in enumerate(steps, start=1):
            step.step_id = index

    for call_id, event in fallback_results.items():
        if call_id in seen_results:
            continue
        step = call_to_step.get(call_id)
        if step is None:
            continue
        _append_observation(
            step,
            ObservationResult(
                source_call_id=call_id,
                content=_tool_result_content(
                    (event.get("result") or {}).get("content")
                ),
                extra={"is_error": bool(event.get("isError"))}
                if event.get("isError")
                else None,
            ),
        )

    if not steps:
        return None
    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        agent=Agent(
            name="pi",
            version=agent_version,
            model_name=model_name,
            tool_definitions=tool_definitions,
        ),
        steps=steps,
        final_metrics=FinalMetrics(
            total_prompt_tokens=total_prompt + total_cached or None,
            total_completion_tokens=total_completion or None,
            total_cached_tokens=total_cached or None,
            total_cost_usd=total_cost or None,
            total_steps=len(steps),
        ),
    )


def write_pi_trajectory(
    path: Path,
    *,
    instruction: str,
    model_name: str | None,
    agent_version: str,
) -> Trajectory | None:
    system_prompt, tool_definitions = _session_metadata_from_html(
        path.parent / "trajectory.html"
    )
    trajectory = parse_pi_trajectory(
        path,
        instruction=instruction,
        model_name=model_name,
        agent_version=agent_version,
        system_prompt=system_prompt,
        tool_definitions=tool_definitions,
    )
    if trajectory is None:
        return None
    path.parent.joinpath("trajectory.json").write_text(
        format_trajectory_json(trajectory.to_json_dict()), encoding="utf-8"
    )
    return trajectory
