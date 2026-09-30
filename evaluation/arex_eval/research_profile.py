"""Named research-evaluation profiles exposed by the checkout CLI.

The ``refine-equal`` profile shares the generation settings of the 0919
BrowseComp runner, with one concurrent case by default. Keeping values here makes the
subprocess command inspectable and prevents the four headline benchmarks from
silently drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


REFINE_EQUAL_DATASETS = frozenset(
    {"BrowseComp", "GAIA-2023-validation-text-103", "HLE", "DeepSearch-QA"}
)


@dataclass(frozen=True)
class ResearchProfile:
    name: str
    mode: str
    concurrency: int
    max_outer_rounds: int
    max_calls_per_outer: int
    max_total_calls: int
    confidence_threshold: float
    confidence_middle_threshold: float
    max_response_tokens: int
    max_context_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    tool_call_regen_retries: int
    llm_call_retries: int
    general_max_attempts: int
    max_attempts: int


REFINE_EQUAL = ResearchProfile(
    name="refine-equal",
    mode="direct",  # HLE uses its dedicated harness; other targets override this.
    concurrency=1,
    max_outer_rounds=10,
    max_calls_per_outer=300,
    max_total_calls=1500,
    confidence_threshold=95.0,
    confidence_middle_threshold=90.0,
    max_response_tokens=16384,
    max_context_tokens=240000,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
    tool_call_regen_retries=20,
    llm_call_retries=5,
    general_max_attempts=10,
    max_attempts=1,
)


def select_profile(requested: str, datasets: list[str]) -> ResearchProfile | None:
    """Resolve a named profile and validate its dataset scope.

    HLE keeps its dedicated agent loop. Mixed core-dataset runs use refine_summary
    for the unified backend and the independent outer chain for HLE.
    """

    requested = (requested or "auto").strip().lower()
    selected = set(datasets)
    if requested in {"", "auto"}:
        if not selected.intersection(REFINE_EQUAL_DATASETS):
            return None
        if not selected.issubset(REFINE_EQUAL_DATASETS):
            raise ValueError("Run core and other datasets separately, or use --profile default")
    elif requested in {"default", "none"}:
        return None
    elif requested != "refine-equal":
        raise ValueError("--profile must be one of: auto, default, refine-equal")
    elif not selected.issubset(REFINE_EQUAL_DATASETS):
        unsupported = sorted(selected - REFINE_EQUAL_DATASETS)
        raise ValueError(
            "profile refine-equal only supports BrowseComp, GAIA, HLE, and "
            f"DeepSearch-QA (unsupported: {', '.join(unsupported)})"
        )

    mode = "direct" if selected == {"HLE"} else "refine_summary"
    return replace(REFINE_EQUAL, mode=mode)
