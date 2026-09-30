from typing import Iterable, List

from format_tools import get_tools_by_name


UPDATE_CONTEXT_TOOL = "update_context"


def mode_to_context_strategy(mode: str) -> str:
    if mode == "direct":
        return "discard_all"
    if mode in ("refine_summary", "return"):
        return "refine_summary"
    raise ValueError(f"Unsupported mode: {mode}")


def tool_names_for_mode(base_tool_names: Iterable[str], mode: str) -> List[str]:
    names = [name for name in base_tool_names if name]
    if not names:
        return []
    if mode == "direct":
        return [name for name in names if name != UPDATE_CONTEXT_TOOL]
    if mode in ("refine_summary", "return") and UPDATE_CONTEXT_TOOL not in names:
        names.append(UPDATE_CONTEXT_TOOL)
    return names


def tool_defs_for_mode(base_tool_names: Iterable[str], mode: str):
    return get_tools_by_name(tool_names_for_mode(base_tool_names, mode))
