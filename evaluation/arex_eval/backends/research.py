from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

from ..datasets import data_root as resolve_data_root, dataset_paths, path_is_ready, research_names
from ..research_profile import select_profile


def command(
    repo_root: Path, datasets: str | list[str], *, mode: str | None = None,
    profile: str = "auto",
    model: str = "", api_key_env: str = "MODEL_API_KEY", base_url: str = "",
    data_path: str = "", save_path: str = "", data_root: str = "",
    tokenizer_path: str = "", num_tasks: int | None = None, start_index: int = 0,
    concurrency: int | None = None, shuffle: bool = False,
    judge_model: str = "", judge_base_url: str = "", judge_api_key_env: str = "",
    summary_model: str = "", summary_base_url: str = "", summary_api_key_env: str = "",
    dry_run: bool = False, extra: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    selected = list(dict.fromkeys([datasets] if isinstance(datasets, str) else datasets))
    if not selected or start_index < 0 or (num_tasks is not None and num_tasks <= 0):
        raise ValueError("select a dataset, a nonnegative start index, and a positive task count")
    available = set(research_names())
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(
            "Unknown research dataset(s): " + ", ".join(unknown)
            + ". Run `python3 evaluate.py list` to see available datasets."
        )
    selected_profile = select_profile(profile, selected)
    if selected_profile is not None:
        if mode not in (None, selected_profile.mode):
            raise ValueError(
                f"profile {selected_profile.name} requires evaluator mode "
                f"{selected_profile.mode!r}; omit --mode or use --profile default"
            )
        mode = selected_profile.mode
    else:
        mode = mode or "direct"
    if concurrency is None:
        concurrency = selected_profile.concurrency if selected_profile else 4
    if concurrency <= 0:
        raise ValueError("concurrency must be greater than zero")
    if data_path and len(selected) != 1:
        raise ValueError("--data-path requires exactly one dataset; use --data-root for multiple datasets")
    env = os.environ.copy()
    if selected_profile is not None:
        for label, value, expected in (
            ("model", summary_model, model),
            ("base URL", summary_base_url, base_url),
            ("key environment", summary_api_key_env, api_key_env),
        ):
            if value and value != expected:
                raise ValueError(f"refine-equal uses the inference {label} for summary; remove the summary override")
        summary_model, summary_base_url, summary_api_key_env = model, base_url, api_key_env
    root = resolve_data_root(data_root)
    paths = dataset_paths(root, selected, use_legacy=not (data_root or env.get("AREX_DATA_ROOT")))
    if data_path:
        paths[selected[0]] = str(Path(data_path).expanduser().resolve())
    if not dry_run:
        for name in selected:
            path = paths.get(name)
            if not path or not path_is_ready(path):
                raise ValueError(f"Missing data for {name}. Run: python3 evaluate.py download {name}")
        if not model:
            raise ValueError("Set AREX_MODEL_NAME or --model-name")
        if not env.get(api_key_env):
            raise ValueError(f"Set the model key environment variable {api_key_env!r}")
        if any(name != "HLE" for name in selected) and not tokenizer_path:
            raise ValueError("Set AREX_TOKENIZER_PATH or --tokenizer-path to the agent model's tokenizer")
        if selected_profile is not None:
            if not judge_model or not judge_base_url or not judge_api_key_env:
                raise ValueError(
                    "refine-equal requires an externally specified judge: "
                    "--judge-model, --judge-base-url, and --judge-api-key-env"
                )
            if not env.get(judge_api_key_env):
                raise ValueError(f"Missing key environment variable {judge_api_key_env!r}")
    for value in extra or []:
        if "api_key" in value.lower() or "api-key" in value.lower():
            raise ValueError("Pass keys through --api-key-env/--judge-api-key-env/--summary-api-key-env, not --extra")
    evaluator_root = repo_root / "evaluation/research"
    args = [sys.executable, str(evaluator_root / "eval_unified.py"),
            "--datasets", *selected, "--mode", mode,
            "--start_index", str(start_index),
            "--end_index", str(start_index + num_tasks if num_tasks is not None else 9999999999999),
            "--concurrency_limit", str(concurrency)]
    # Explicit per-dataset ends bypass the legacy HLE default of only 200 tasks.
    end = start_index + num_tasks if num_tasks is not None else 9999999999999
    args += ["--dataset-end-indices", " ".join(f"{name}={end}" for name in selected)]
    if not shuffle:
        args.append("--no-shuffle")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = Path(save_path).expanduser().resolve() if save_path else repo_root / "runs" / timestamp
    args += ["--save_path", str(output)]
    for flag, value in (("model", model), ("tokenizer_path", tokenizer_path),
                        ("judge_model", judge_model), ("judge_base_url", judge_base_url),
                        ("summary_model", summary_model), ("summary_base_url", summary_base_url)):
        if value:
            args += [f"--{flag}", value]
    # Keys remain in the subprocess environment, never in the command preview.
    for variable, source in (("AREX_SDK_API_KEY", api_key_env),
                             ("AREX_JUDGE_API_KEY", judge_api_key_env),
                             ("AREX_SUMMARY_API_KEY", summary_api_key_env)):
        if source:
            if not dry_run and not env.get(source):
                raise ValueError(f"Missing key environment variable {source!r}")
            env[variable] = env.get(source, "")
    if base_url:
        env["AREX_SDK_BASE_URL"] = base_url
    env["AREX_DATA_PATHS"] = json.dumps(paths)
    env["UNIFY_EVAL_ROOT"] = str(evaluator_root)
    env["AREX_DATASET_CONFIG_ROOT"] = str(repo_root / "data/research")
    env["AREX_DATA_ROOT"] = str(root)
    env["PYTHONPATH"] = str(evaluator_root) + os.pathsep + env.get("PYTHONPATH", "")
    if selected_profile is not None:
        args += [
            "--judge-mode", "offical",
            "--enable-thinking", "--preserve_thinking",
            "--summary-enable-thinking", "--enable-visit-fallback",
            "--max_tokens", str(selected_profile.max_context_tokens),
            "--max_response_tokens", str(selected_profile.max_response_tokens),
            "--agent_temperature", str(selected_profile.temperature),
            "--agent_top_p", str(selected_profile.top_p),
            "--agent_top_k", str(selected_profile.top_k),
            "--agent_min_p", str(selected_profile.min_p),
            "--agent_presence_penalty", str(selected_profile.presence_penalty),
            "--agent_repetition_penalty", str(selected_profile.repetition_penalty),
            "--refine_summary_max_outer_rounds", str(selected_profile.max_outer_rounds),
            "--refine_summary_max_llm_calls", str(selected_profile.max_calls_per_outer),
            "--refine_summary_max_total_llm_calls", str(selected_profile.max_total_calls),
            "--refine_summary_trigger_tokens", "128000",
            "--refine_summary_max_updates", "24",
            "--enable_confidence_outer_retry",
            "--confidence_outer_retry_threshold", str(selected_profile.confidence_threshold),
            "--confidence_outer_retry_review_max_tokens", "4096",
            "--enable_confidence_tiered_review",
            "--confidence_tiered_review_middle_threshold", str(selected_profile.confidence_middle_threshold),
            "--general_max_attempts", str(selected_profile.general_max_attempts),
            "--max_attempts", str(selected_profile.max_attempts),
            "--case_timeout_seconds", "86400",
            "--tool_call_regen_max_retries", str(selected_profile.tool_call_regen_retries),
            "--llm_call_max_retries", str(selected_profile.llm_call_retries),
            "--skip-existing-mode", "all",
        ]
        if "HLE" in selected:
            # The HLE adapter receives the same generation and retry
            # values through its own namespaced arguments.  The outer chain is
            # started automatically by eval_unified when no prior run is given.
            hle_outer_root = str(output / "_hle_outer")
            args += [
                "--hle-review-threshold", str(selected_profile.confidence_threshold),
                "--hle-review-middle-threshold", str(selected_profile.confidence_middle_threshold),
                "--hle-review-max-tokens", "4096",
                "--hle-max-completion-tokens", str(selected_profile.max_response_tokens),
                "--hle-truncation-max-completion-tokens", str(selected_profile.max_response_tokens),
                "--hle-max-context-tokens", str(selected_profile.max_context_tokens),
                "--hle-max-total-tokens", "262144",
                "--hle-max-steps", str(selected_profile.max_calls_per_outer),
                "--hle-tool-call-regen-max-retries", str(selected_profile.tool_call_regen_retries),
                "--hle-temperature", str(selected_profile.temperature),
                "--hle-top-p", str(selected_profile.top_p),
                "--hle-top-k", str(selected_profile.top_k),
                "--hle-min-p", str(selected_profile.min_p),
                "--hle-presence-penalty", str(selected_profile.presence_penalty),
                "--hle-repetition-penalty", str(selected_profile.repetition_penalty),
                "--hle-enable-thinking", "--hle-preserve-thinking",
                "--hle-llm-call-max-retries", str(selected_profile.llm_call_retries),
                "--hle-general-max-attempts", str(selected_profile.general_max_attempts),
                "--hle-case-timeout-seconds", "86400",
                "--hle-per-case-outer-max", str(selected_profile.max_outer_rounds),
                "--hle-outer-round", "2",
                "--hle-per-case-total-max-steps", str(selected_profile.max_total_calls),
                "--hle-per-case-outer-root", hle_outer_root,
                "--hle-per-case-fill-missing-outer1",
                "--hle-rerun-confidence-threshold", str(selected_profile.confidence_threshold),
            ]
    if dry_run:
        args.append("--dry-run")
    args += list(extra or [])
    return args, env
