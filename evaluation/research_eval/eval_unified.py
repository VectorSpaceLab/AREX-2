import argparse
import asyncio
import base64
import contextlib
import datetime
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from unified_eval.loaders import load_samples
from unified_eval.registry import available_datasets, load_dataset_specs
from unified_eval.scorers import score_prediction
from unified_eval.tools import mode_to_context_strategy, tool_names_for_mode
from unified_eval.types import DatasetSpec, EvalSample
from direct_repair import maybe_run_direct_repair_retry
from hle_backend import (
    DEFAULT_HLE_HARNESS_DIR,
    DEFAULT_HLE_JUDGE_SCRIPT,
    HLEBackend,
    HLE_SCORER_SOURCE,
    select_hle_samples,
)
from hle_vendor.visit_fallback import visit_fallback_default_enabled
from pretty_console import case_key as pretty_case_key
from pretty_console import get_pretty_console


DEFAULT_END_INDEX = 9999999999999


def _sha256_path(path: str) -> Optional[str]:
    """Hash a prepared input file or directory deterministically."""
    target = Path(path).expanduser()
    if not target.exists():
        return None
    digest = hashlib.sha256()
    if target.is_file():
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for item in sorted(item for item in target.rglob("*") if item.is_file()):
        relative = item.relative_to(target).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root, text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return {"commit": commit, "dirty": dirty}


def write_run_metadata(save_path: str, metadata: Dict[str, object]) -> None:
    """Write the provenance record next to a dataset's reported scores."""
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def import_tokenizer(tokenizer_path: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError(f"transformers is required to run inference: {type(e).__name__}: {e}")
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def parse_dataset_names(text) -> List[str]:
    if isinstance(text, (list, tuple)):
        text = " ".join(str(item) for item in text)
    names = [part.strip() for part in re.split(r"[,\s]+", text or "") if part.strip()]
    if not names:
        raise ValueError("--datasets must contain at least one dataset name")
    return names


def parse_target_indices(text: str) -> Optional[set[int]]:
    if not text:
        return None
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def _parse_dataset_pairs(text: str) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for part in re.split(r"[,\s]+", text or ""):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
        elif ":" in part:
            key, value = part.split(":", 1)
        else:
            raise ValueError(f"Dataset override must be name=value, got: {part}")
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Dataset override must be name=value, got: {part}")
        pairs[key] = value
    return pairs


def parse_dataset_int_overrides(text: str) -> Dict[str, int]:
    return {key: int(value) for key, value in _parse_dataset_pairs(text).items()}


def parse_dataset_bool_overrides(text: str) -> Dict[str, bool]:
    truthy = {"1", "true", "yes", "y", "on", "shuffle"}
    falsy = {"0", "false", "no", "n", "off", "no_shuffle", "noshuffle"}
    parsed = {}
    for key, value in _parse_dataset_pairs(text).items():
        normalized = value.strip().lower()
        if normalized in truthy:
            parsed[key] = True
        elif normalized in falsy:
            parsed[key] = False
        else:
            raise ValueError(f"Dataset shuffle override for {key} must be 1/0 or true/false, got: {value}")
    return parsed


def dataset_selection_config(
    dataset_name: str,
    args,
    evaluation_backend: str = "unified",
) -> Tuple[int, int, bool]:
    start_overrides = parse_dataset_int_overrides(args.dataset_start_indices)
    end_overrides = parse_dataset_int_overrides(args.dataset_end_indices)
    shuffle_overrides = parse_dataset_bool_overrides(args.dataset_shuffle)
    start_index = start_overrides.get(dataset_name, args.start_index)
    end_index = end_overrides.get(dataset_name, args.end_index)
    if (
        evaluation_backend == "hle"
        and dataset_name not in end_overrides
        and args.end_index == DEFAULT_END_INDEX
    ):
        end_index = args.hle_num_samples
    shuffle_samples = shuffle_overrides.get(dataset_name, not args.no_shuffle)
    return start_index, end_index, shuffle_samples


def extract_row_index(path_name: str) -> Optional[int]:
    match = re.search(r"row(\d+)$", path_name)
    return int(match.group(1)) if match else None


def should_skip_existing_result(result: dict, skip_existing_mode: str) -> bool:
    if skip_existing_mode == "none":
        return False
    if skip_existing_mode == "all":
        return True
    if skip_existing_mode == "correct":
        return result_is_full_credit(result)
    raise ValueError(f"Unsupported skip_existing_mode: {skip_existing_mode}")


def result_is_full_credit(result: dict) -> bool:
    score_result = result.get("score_result") if isinstance(result.get("score_result"), dict) else {}
    metrics = score_result.get("metrics") if isinstance(score_result.get("metrics"), dict) else {}
    if metrics.get("full_credit") is True:
        return True
    score = score_result.get("score", result.get("score"))
    try:
        score_val = float(score)
    except Exception:
        return False
    return score_val == 1.0 or score_val == 100.0


def iter_result_json_paths(result_dirs: Iterable[str]):
    for result_dir in result_dirs:
        if not os.path.isdir(result_dir):
            continue
        for root, _, files in os.walk(result_dir):
            if "failure_attempt" in root.split(os.sep):
                continue
            if "temp.json" in files:
                yield os.path.join(root, "temp.json")


def collect_existing_indices(result_dirs: Iterable[str], skip_existing_mode: str) -> set[int]:
    if skip_existing_mode == "none":
        return set()
    idx_set = set()
    for path in iter_result_json_paths(result_dirs):
        try:
            with open(path, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception:
            continue
        if should_skip_existing_result(result, skip_existing_mode):
            row_idx = extract_row_index(os.path.basename(os.path.dirname(path)))
            if row_idx is not None:
                idx_set.add(row_idx)
    return idx_set


def collect_latest_result_paths_by_index(result_dir: str) -> Dict[int, str]:
    latest: Dict[int, str] = {}
    for path in sorted(iter_result_json_paths([result_dir])):
        row_idx = extract_row_index(os.path.basename(os.path.dirname(path)))
        if row_idx is not None:
            latest[row_idx] = path
    return latest


def parse_confidence_value(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def hle_result_meets_finish_confidence_threshold(
    result: dict,
    threshold: float,
) -> bool:
    if not _has_finish_call(result):
        return False
    confidence = parse_confidence_value(result.get("confidence"))
    if confidence is None:
        return False
    return confidence >= threshold


def hle_result_model_calls(result: dict) -> int:
    legacy_key = "hle" + "_0724"
    current = result.get("hle")
    hle = current if isinstance(current, dict) else result.get(legacy_key, {})
    if not isinstance(hle, dict):
        hle = {}
    raw = hle.get("raw_prediction") if isinstance(hle.get("raw_prediction"), dict) else {}
    summaries = (
        hle.get("model_usage_summary"),
        raw.get("model_usage_summary"),
        result.get("model_usage_summary"),
    )
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        value = summary.get("model_calls")
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, value)
    return 0


def collect_hle_rerun_indices_and_copy_retained(
    source_result_dir: str,
    dest_result_dir: str,
    confidence_threshold: float,
    copy_retained: bool,
    expected_indices: Optional[set[int]] = None,
) -> Tuple[set[int], dict]:
    if not os.path.isdir(source_result_dir):
        raise FileNotFoundError(f"HLE rerun source root not found: {source_result_dir}")

    rerun_indices: set[int] = set()
    retained_indices: set[int] = set()
    copied_indices: set[int] = set()
    source_paths = collect_latest_result_paths_by_index(source_result_dir)
    expected = set(expected_indices) if expected_indices is not None else None

    for row_idx, temp_path in source_paths.items():
        if expected is not None and row_idx not in expected:
            continue
        try:
            with open(temp_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception:
            rerun_indices.add(row_idx)
            continue

        if hle_result_meets_finish_confidence_threshold(result, confidence_threshold):
            retained_indices.add(row_idx)
            if copy_retained and dest_result_dir:
                source_case_dir = os.path.dirname(temp_path)
                dest_case_dir = os.path.join(dest_result_dir, os.path.basename(source_case_dir))
                if os.path.abspath(source_case_dir) != os.path.abspath(dest_case_dir):
                    if not os.path.exists(dest_case_dir):
                        shutil.copytree(source_case_dir, dest_case_dir)
                        copied_indices.add(row_idx)
        else:
            rerun_indices.add(row_idx)

    missing_indices = set()
    if expected is not None:
        missing_indices = expected - set(source_paths)
        rerun_indices.update(missing_indices)

    return rerun_indices, {
        "source_root": source_result_dir,
        "dest_root": dest_result_dir,
        "confidence_threshold": confidence_threshold,
        "expected_cases": len(expected) if expected is not None else None,
        "source_cases": len(source_paths),
        "missing_cases": len(missing_indices),
        "retained_cases": len(retained_indices),
        "copied_cases": len(copied_indices),
        "rerun_cases": len(rerun_indices),
    }


def confidence_outer_one_elapsed_seconds(prior_origin: Optional[dict]) -> float:
    if not isinstance(prior_origin, dict):
        return 0.0
    timing_stats = prior_origin.get("timing_stats")
    if not isinstance(timing_stats, dict):
        return 0.0
    rounds = timing_stats.get("rounds")
    if not isinstance(rounds, list):
        return 0.0
    return sum(
        float(record.get("turn_elapsed_seconds") or 0.0)
        for record in rounds
        if isinstance(record, dict) and record.get("outer_round") == 1
    )


def _message_contents(value):
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            yield content
        for nested in value.values():
            yield from _message_contents(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _message_contents(item)


def _has_structured_finish_call(value) -> bool:
    if isinstance(value, dict):
        for key in ("type", "name"):
            marker = value.get(key)
            if isinstance(marker, str) and marker.strip().lower() == "finish":
                return True
        return any(_has_structured_finish_call(nested) for nested in value.values())
    if isinstance(value, list):
        return any(_has_structured_finish_call(item) for item in value)
    return False


def _has_finish_call(data: dict) -> bool:
    for key in ("finish", "finished"):
        if key in data:
            return bool(data[key])
    if _has_structured_finish_call(data):
        return True
    for content in _message_contents(data):
        if re.search(r'<function\s*=\s*finish\b|<function_call>\s*finish\b|"name"\s*:\s*"finish"', content, re.I):
            return True
    return False


def collect_return_indices(prev_result_path: str) -> set[int]:
    selected = set()
    for path in iter_result_json_paths([prev_result_path]):
        row_idx = extract_row_index(os.path.basename(os.path.dirname(path)))
        if row_idx is None:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            selected.add(row_idx)
            continue
        score = data.get("score", 0)
        try:
            score_val = int(score)
        except Exception:
            score_val = 0
        if score_val == 0 and not _has_finish_call(data):
            selected.add(row_idx)
    return selected


def empty_case_call_stats() -> dict:
    return {
        "assistant_calls_total": 0,
        "search_calls_total": 0,
        "visit_calls_total": 0,
        "llm_calls_total": 0,
        "update_context_calls_total": 0,
        "llm_elapsed_seconds_total": 0.0,
        "tool_elapsed_seconds_total": 0.0,
        "search_elapsed_seconds_total": 0.0,
        "visit_elapsed_seconds_total": 0.0,
        "update_context_elapsed_seconds_total": 0.0,
        "round_elapsed_seconds_total": 0.0,
        "update_context_context_tokens_total": 0,
        "update_context_context_chars_total": 0,
        "tool_call_regen_retries_total": 0,
        "discard_all_reset_count": 0,
        "max_current_tokens_seen": 0,
    }


def merge_attempt_call_stats(case_call_stats: dict, attempt_call_stats: dict) -> dict:
    attempt_call_stats = attempt_call_stats or {}
    int_sum_keys = (
        "assistant_calls_total",
        "search_calls_total",
        "visit_calls_total",
        "llm_calls_total",
        "update_context_calls_total",
        "update_context_context_tokens_total",
        "update_context_context_chars_total",
        "tool_call_regen_retries_total",
        "discard_all_reset_count",
    )
    float_sum_keys = (
        "llm_elapsed_seconds_total",
        "tool_elapsed_seconds_total",
        "search_elapsed_seconds_total",
        "visit_elapsed_seconds_total",
        "update_context_elapsed_seconds_total",
        "round_elapsed_seconds_total",
    )
    for key in int_sum_keys:
        case_call_stats[key] = int(case_call_stats.get(key) or 0) + int(attempt_call_stats.get(key) or 0)
    for key in float_sum_keys:
        case_call_stats[key] = float(case_call_stats.get(key) or 0.0) + float(attempt_call_stats.get(key) or 0.0)
    case_call_stats["max_current_tokens_seen"] = max(
        int(case_call_stats.get("max_current_tokens_seen") or 0),
        int(attempt_call_stats.get("max_current_tokens_seen") or 0),
    )
    if case_call_stats.get("llm_calls_total"):
        case_call_stats["llm_elapsed_seconds_avg"] = (
            case_call_stats["llm_elapsed_seconds_total"] / max(1, case_call_stats["llm_calls_total"])
        )
    merged = dict(attempt_call_stats)
    merged.update(case_call_stats)
    return merged


def dataset_save_path(base_save_path: str, dataset_name: str) -> str:
    return os.path.join(base_save_path, dataset_name)


def select_samples(
    samples: Sequence[EvalSample],
    start_index: int,
    end_index: int,
    target_indices: Optional[set[int]],
    target_shard_rank: int,
    target_shard_count: int,
    skip_indices: set[int],
    shuffle_samples: bool = True,
) -> List[EvalSample]:
    indices = list(range(len(samples)))
    if shuffle_samples:
        random.seed(66)
        random.shuffle(indices)
    if target_indices is not None:
        selected = [i for i in indices if i in target_indices]
    else:
        selected = indices[start_index:end_index]
    if target_shard_count > 1:
        if target_shard_rank < 0 or target_shard_rank >= target_shard_count:
            raise ValueError(f"--target-shard-rank must be in [0, {target_shard_count}), got {target_shard_rank}")
        selected = selected[target_shard_rank::target_shard_count]
    return [samples[i] for i in selected if i not in skip_indices]


def select_samples_for_spec(
    spec: DatasetSpec,
    samples: Sequence[EvalSample],
    start_index: int,
    end_index: int,
    target_indices: Optional[set[int]],
    target_shard_rank: int,
    target_shard_count: int,
    skip_indices: set[int],
    shuffle_samples: bool,
    hle_seed: int = 125,
) -> List[EvalSample]:
    if spec.evaluation_backend == "hle":
        return select_hle_samples(
            samples,
            start_index,
            end_index,
            target_indices,
            target_shard_rank,
            target_shard_count,
            skip_indices,
            shuffle_samples,
            seed=hle_seed,
        )
    return select_samples(
        samples,
        start_index,
        end_index,
        target_indices,
        target_shard_rank,
        target_shard_count,
        skip_indices,
        shuffle_samples=shuffle_samples,
    )


def build_question(sample: EvalSample, extra_attachments: Optional[List[str]] = None) -> str:
    attachments = list(sample.attachments)
    attachments.extend(extra_attachments or [])
    if not attachments:
        return sample.question
    attachment_lines = "\n".join(f"- {path}" for path in attachments)
    return f"{sample.question}\n\nAvailable local attachment files:\n{attachment_lines}"


def materialize_inline_attachments(sample: EvalSample, case_dir: str) -> List[str]:
    materialized = []
    for key in ("image",):
        value = sample.raw.get(key)
        if not isinstance(value, str) or not value.lower().startswith("data:"):
            continue
        header, sep, payload = value.partition(",")
        if not sep or ";base64" not in header.lower():
            continue
        mime = header[5:].split(";", 1)[0].lower()
        ext = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
        }.get(mime, ".bin")
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample.sample_id)[:80] or str(sample.idx)
        out_dir = os.path.join(case_dir, "attachments")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{key}_{safe_id}{ext}")
        try:
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(payload))
            materialized.append(out_path)
        except Exception as e:
            get_pretty_console().warning(
                f"failed to materialize inline attachment dataset={sample.dataset_name} "
                f"idx={sample.idx} key={key}: {type(e).__name__}: {e}"
            )
    return materialized


async def run_one_sample(
    spec: DatasetSpec,
    sample: EvalSample,
    config: dict,
    llm_client,
    summary_client,
    serper_client,
    jina_client,
    tokenizer,
    sum_tokenizer,
    judge_client,
    judge_model: str,
    repair_judge_client,
    repair_judge_model: str,
    semaphore: asyncio.Semaphore,
):
    from Agent_no_subagent import MainAgent
    from local_search import LocalSearch
    from prompts_no_subagent import EXTRACTOR_PROMPT

    async with semaphore:
        save_path = config["save_path"]
        now = datetime.datetime.now()
        case_dir = os.path.join(save_path, f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_row{sample.idx}")
        os.makedirs(case_dir, exist_ok=True)

        max_attempts = max(1, int(config.get("max_attempts", 1)))
        case_timeout_seconds = float(config.get("case_timeout_seconds") or 0)
        case_start_time = time.time()
        case_call_stats = empty_case_call_stats()
        resume_deadline_adjusted = False

        async def await_with_case_deadline(awaitable):
            """Apply one cumulative deadline to search, repair, and scoring."""
            if case_timeout_seconds <= 0:
                return await awaitable
            remaining = case_timeout_seconds - (time.time() - case_start_time)
            if remaining <= 0:
                close = getattr(awaitable, "close", None)
                if callable(close):
                    close()
                raise asyncio.TimeoutError
            return await asyncio.wait_for(awaitable, timeout=remaining)

        for attempt_idx in range(max_attempts):
            attempt_start_time = time.time()
            case_key = pretty_case_key(spec.name, sample.idx, attempt_idx)
            pretty = get_pretty_console()
            pretty.case_start(case_key, spec.name, sample.idx, sample.sample_id, attempt_idx)
            page_id_to_url = {}
            page_url_to_id = {}
            env = LocalSearch(
                summary_client=summary_client,
                search_client=serper_client,
                jina_client=jina_client,
                tokenizer=sum_tokenizer,
                page_id_to_url=page_id_to_url,
                page_url_to_id=page_url_to_id,
                model_name=config.get("summary_model") or config.get("model"),
                extractor_prompt=config.get("extractor_prompt", EXTRACTOR_PROMPT),
                case_output_dir=case_dir,
                attachment_root=spec.attachments_root,
                pretty_case_key=case_key,
                leak_filter=spec.leak_filter,
                dataset_name=spec.name,
                summary_enable_thinking=config.get("summary_enable_thinking"),
                enable_visit_fallback=bool(config.get("enable_visit_fallback", True)),
            )
            agent_config = dict(config)
            agent_config["case_output_dir"] = case_dir
            agent_config["pretty_case_key"] = case_key
            agent_config["tool_names"] = tool_names_for_mode(spec.tools, config["mode"])
            if spec.tool_definitions:
                agent_config["tool_definitions"] = spec.tool_definitions
            if spec.generation:
                generation = dict(spec.generation)
                agent_config["generation"] = generation
                if generation.get("context_length") is not None:
                    agent_config["max_tokens"] = generation["context_length"]
                if generation.get("max_completion_tokens") is not None:
                    agent_config["max_response_tokens"] = generation["max_completion_tokens"]
                if generation.get("retry_token_threshold") is not None:
                    agent_config["retry_token_threshold"] = generation["retry_token_threshold"]
                if generation.get("retry_max_attempts") is not None:
                    agent_config["retry_max_attempts"] = generation["retry_max_attempts"]
            if spec.prompts:
                agent_config["prompt_bundle"] = spec.prompts
            main_agent = MainAgent(agent_config, env, llm_client, tokenizer)

            score_value = 0.0
            predicted_answer = ""
            evidence = ""
            confidence = ""
            full_credit = False
            direct_repair_meta = {}
            try:
                inline_attachments = materialize_inline_attachments(sample, case_dir)
                case_data = {"problem": build_question(sample, inline_attachments), "idx": sample.idx}
                resume_paths = config.get("confidence_outer_resume_paths") or {}
                resume_path = resume_paths.get(sample.idx)
                if resume_path:
                    with open(resume_path, "r", encoding="utf-8") as handle:
                        prior_result = json.load(handle)
                    prior_origin = None
                    prior_origin_path = os.path.join(os.path.dirname(resume_path), "temp_origin.json")
                    if os.path.isfile(prior_origin_path):
                        with open(prior_origin_path, "r", encoding="utf-8") as handle:
                            prior_origin = json.load(handle)
                    if not resume_deadline_adjusted:
                        case_start_time -= confidence_outer_one_elapsed_seconds(prior_origin)
                        resume_deadline_adjusted = True
                    case_coro = main_agent.run_confidence_outer_resume(
                        case_data,
                        prior_result,
                        prior_origin=prior_origin,
                        prior_result_path=resume_path,
                    )
                elif (
                    config.get("confidence_outer_resume_root")
                    and not config.get("confidence_outer_resume_missing_from_scratch")
                ):
                    raise FileNotFoundError(
                        f"no prior result found for confidence resume row {sample.idx}"
                    )
                else:
                    case_coro = main_agent.run(case_data)
                predicted_answer, evidence, confidence = await await_with_case_deadline(case_coro)

                predicted_answer, evidence, confidence, direct_repair_meta = await await_with_case_deadline(maybe_run_direct_repair_retry(
                    spec=spec,
                    sample=sample,
                    config=config,
                    main_agent=main_agent,
                    predicted_answer=predicted_answer,
                    evidence=evidence,
                    confidence=confidence,
                    judge_client=judge_client,
                    judge_model=judge_model,
                    repair_judge_client=repair_judge_client,
                    repair_judge_model=repair_judge_model,
                    pretty=pretty,
                    case_key=case_key,
                ))

                score_result = await await_with_case_deadline(score_prediction(
                    spec,
                    sample,
                    predicted_answer,
                    judge_client=judge_client,
                    judge_model=judge_model,
                ))
                score_value = score_result.score if score_result.score is not None else 0.0
                full_credit = bool(score_result.metrics.get("full_credit", score_value == 1 or score_value == 100))
                attempt_call_stats = main_agent.get_call_stats() if hasattr(main_agent, "get_call_stats") else {}
                call_stats = merge_attempt_call_stats(case_call_stats, attempt_call_stats)
                timing_stats = main_agent.get_timing_stats() if hasattr(main_agent, "get_timing_stats") else {}
                temp_json = {
                    "dataset_name": spec.name,
                    "sample_id": sample.sample_id,
                    "task_type": spec.task_type,
                    "question": sample.question,
                    "answer": sample.answer,
                    "predicted": predicted_answer,
                    "evidence": evidence,
                    "confidence": confidence,
                    "score": score_value,
                    "score_result": score_result.to_dict(),
                    "official_scorer": score_result.official_scorer,
                    "metadata": sample.metadata,
                    "trajectory": main_agent.messages,
                    "sub_traj": "",
                    "timing_summary": timing_stats.get("summary", {}),
                    "run_metadata": config.get("run_metadata"),
                }
                if direct_repair_meta:
                    temp_json["direct_repair"] = direct_repair_meta
                if getattr(main_agent, "confidence_outer_retry_meta", None):
                    temp_json["confidence_outer_retry"] = main_agent.confidence_outer_retry_meta
                with open(os.path.join(case_dir, "temp.json"), "w", encoding="utf-8") as f:
                    json.dump(temp_json, f, ensure_ascii=False, indent=2)

                token_usage = main_agent.get_total_usage()
                temp_origin = {
                    "origin_traj": main_agent.origin_messages,
                    "origin_sub_traj": "",
                    "token_usage": token_usage,
                    "call_stats": call_stats,
                    "attempt_call_stats": attempt_call_stats,
                    "timing_stats": timing_stats,
                    "case_duration_seconds": time.time() - case_start_time,
                    "attempt_duration_seconds": time.time() - attempt_start_time,
                    "attempt_index": attempt_idx,
                    "run_metadata": config.get("run_metadata"),
                }
                if getattr(main_agent, "confidence_outer_retry_meta", None):
                    temp_origin["confidence_outer_retry"] = main_agent.confidence_outer_retry_meta
                temp_origin.update(call_stats)
                with open(os.path.join(case_dir, "temp_origin.json"), "w", encoding="utf-8") as f:
                    json.dump(temp_origin, f, ensure_ascii=False, indent=2)
                is_final_attempt = full_credit or attempt_idx + 1 >= max_attempts
                if is_final_attempt:
                    pretty.case_done(
                        case_key,
                        score_value,
                        score_result.status,
                        call_stats,
                        time.time() - case_start_time,
                    )
                else:
                    pretty.case_retry(
                        case_key,
                        (
                            f"dataset={spec.name} idx={sample.idx} attempt={attempt_idx} "
                            f"score={score_value} status={score_result.status}"
                        ),
                    )
            except asyncio.TimeoutError:
                await write_error_case(spec, sample, case_dir, attempt_idx, main_agent, "CaseTimeout", f"case exceeded timeout_seconds={case_timeout_seconds}", case_start_time, attempt_start_time, case_call_stats, config.get("run_metadata"))
                if attempt_idx + 1 >= max_attempts:
                    pretty.case_error(case_key, "CaseTimeout", f"case exceeded timeout_seconds={case_timeout_seconds}", time.time() - case_start_time)
                else:
                    pretty.case_retry(case_key, f"dataset={spec.name} idx={sample.idx} attempt={attempt_idx} timeout")
            except Exception as e:
                await write_error_case(spec, sample, case_dir, attempt_idx, main_agent, type(e).__name__, str(e), case_start_time, attempt_start_time, case_call_stats, config.get("run_metadata"))
                if attempt_idx + 1 >= max_attempts:
                    pretty.case_error(case_key, type(e).__name__, str(e), time.time() - case_start_time)
                else:
                    pretty.case_retry(
                        case_key,
                        f"dataset={spec.name} idx={sample.idx} attempt={attempt_idx} error={type(e).__name__}: {e}",
                    )

            should_retry = not full_credit
            if not should_retry or attempt_idx + 1 >= max_attempts:
                break
            failure_dir = os.path.join(case_dir, "failure_attempt", str(attempt_idx))
            os.makedirs(failure_dir, exist_ok=True)
            for fname in ("temp.json", "temp_origin.json", "fatal_error.txt"):
                src = os.path.join(case_dir, fname)
                if os.path.exists(src):
                    os.replace(src, os.path.join(failure_dir, fname))


async def write_error_case(
    spec: DatasetSpec,
    sample: EvalSample,
    case_dir: str,
    attempt_idx: int,
    main_agent,
    error_type: str,
    message: str,
    case_start_time: float,
    attempt_start_time: float,
    case_call_stats: dict,
    run_metadata: Optional[dict] = None,
):
    error_trace = traceback.format_exc()
    timing_stats = main_agent.get_timing_stats() if hasattr(main_agent, "get_timing_stats") else {}
    with open(os.path.join(case_dir, "fatal_error.txt"), "w", encoding="utf-8") as f:
        f.write(f"dataset: {spec.name}\n")
        f.write(f"idx: {sample.idx}\n")
        f.write(f"sample_id: {sample.sample_id}\n")
        f.write(f"attempt: {attempt_idx}\n")
        f.write(f"question: {sample.question}\n")
        f.write(f"error: {error_type}: {message}\n\n")
        f.write(error_trace)
    traj = getattr(main_agent, "messages", [])
    origin_traj = getattr(main_agent, "origin_messages", [])
    with open(os.path.join(case_dir, "temp.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dataset_name": spec.name,
            "sample_id": sample.sample_id,
            "task_type": spec.task_type,
            "question": sample.question,
            "answer": sample.answer,
            "predicted": "",
            "score": 0,
            "score_result": {
                "status": "error",
                "score": None,
                "official_scorer": spec.scorer.get("source", ""),
                "error_type": error_type,
                "error_message": message,
            },
            "metadata": sample.metadata,
            "trajectory": traj,
            "sub_traj": "",
            "timing_summary": timing_stats.get("summary", {}),
            "run_metadata": run_metadata,
            "error": {"type": error_type, "message": message, "traceback": error_trace},
        }, f, ensure_ascii=False, indent=2)
    attempt_call_stats = main_agent.get_call_stats() if hasattr(main_agent, "get_call_stats") else {}
    call_stats = merge_attempt_call_stats(case_call_stats, attempt_call_stats)
    temp_origin = {
        "origin_traj": origin_traj,
        "origin_sub_traj": "",
        "token_usage": main_agent.get_total_usage() if hasattr(main_agent, "get_total_usage") else {},
        "call_stats": call_stats,
        "attempt_call_stats": attempt_call_stats,
        "timing_stats": timing_stats,
        "case_duration_seconds": time.time() - case_start_time,
        "attempt_duration_seconds": time.time() - attempt_start_time,
        "attempt_index": attempt_idx,
        "run_metadata": run_metadata,
    }
    temp_origin.update(call_stats)
    with open(os.path.join(case_dir, "temp_origin.json"), "w", encoding="utf-8") as f:
        json.dump(temp_origin, f, ensure_ascii=False, indent=2)


def hle_call_stats(result: dict) -> dict:
    usage_summary = result.get("model_usage_summary") or {}
    trajectory = result.get("hle_trajectory") or []
    tool_names = [
        str(event.get("tool") or "")
        for event in trajectory
        if event.get("type") == "tool_call"
    ]
    model_calls = int(usage_summary.get("model_calls") or 0)
    update_events = [
        event for event in trajectory if event.get("type") == "update_context"
    ]
    return {
        "assistant_calls_total": model_calls,
        "llm_calls_total": model_calls,
        "review_calls_total": sum(
            event.get('type') in {'review', 'outer_review'} for event in trajectory
        ),
        "search_calls_total": sum(name in ("search", "google_scholar") for name in tool_names),
        "visit_calls_total": tool_names.count("visit"),
        "update_context_calls_total": len(update_events),
        "update_context_context_tokens_total": sum(
            int(event.get("context_tokens") or 0) for event in update_events
        ),
        "update_context_context_chars_total": sum(
            int(event.get("context_chars") or 0) for event in update_events
        ),
        "tool_call_regen_retries_total": sum(
            event.get("type") == "tool_call_regen_retry" for event in trajectory
        ),
        "input_tokens_total": int(usage_summary.get("input_tokens") or 0),
        "output_tokens_total": int(usage_summary.get("output_tokens") or 0),
        "cache_read_tokens_total": int(usage_summary.get("cache_read_tokens") or 0),
        "total_tokens": int(usage_summary.get("total_tokens") or 0),
        "llm_elapsed_seconds_total": float(usage_summary.get("api_time") or 0.0),
    }


def hle_score_result(result: dict) -> dict:
    judge_response = result.get("judge_response") or {}
    full_credit = bool(result.get("full_credit"))
    metrics = {
        "method": "external_hle_judge",
        "full_credit": full_credit,
        "model_answer": judge_response.get("model_answer"),
        "judge_reasoning": judge_response.get("reasoning"),
        "judge_confidence": judge_response.get("confidence"),
        "response_truncated_for_judge": judge_response.get(
            "response_truncated_for_judge", False
        ),
    }
    return {
        "status": "scored" if full_credit else "incorrect",
        "score": float(result.get("score") or 0.0),
        "official_scorer": HLE_SCORER_SOURCE,
        "metrics": metrics,
        "judge_raw": judge_response.get("raw_judge_outputs"),
        "error_type": "HLEJudgeError" if judge_response.get("judge_error") else None,
        "error_message": judge_response.get("judge_error"),
    }


async def write_hle_error_case(
    spec: DatasetSpec,
    sample: EvalSample,
    case_dir: str,
    attempt_idx: int,
    error_type: str,
    message: str,
    case_start_time: float,
    attempt_start_time: float,
    run_metadata: Optional[dict] = None,
) -> None:
    error_trace = traceback.format_exc()
    with open(os.path.join(case_dir, "fatal_error.txt"), "w", encoding="utf-8") as f:
        f.write(f"dataset: {spec.name}\n")
        f.write(f"idx: {sample.idx}\n")
        f.write(f"sample_id: {sample.sample_id}\n")
        f.write(f"attempt: {attempt_idx}\n")
        f.write(f"question: {sample.question}\n")
        f.write(f"error: {error_type}: {message}\n\n")
        f.write(error_trace)
    temp_json = {
        "dataset_name": spec.name,
        "sample_id": sample.sample_id,
        "task_type": spec.task_type,
        "question": sample.question,
        "answer": sample.answer,
        "predicted": "",
        "score": 0,
        "score_result": {
            "status": "error",
            "score": None,
            "official_scorer": HLE_SCORER_SOURCE,
            "metrics": {},
            "judge_raw": None,
            "error_type": error_type,
            "error_message": message,
        },
        "official_scorer": HLE_SCORER_SOURCE,
        "metadata": sample.metadata,
        "trajectory": [],
        "sub_traj": "",
        "timing_summary": {},
        "run_metadata": run_metadata,
        "evaluation_backend": "hle",
        "error": {"type": error_type, "message": message, "traceback": error_trace},
    }
    with open(os.path.join(case_dir, "temp.json"), "w", encoding="utf-8") as f:
        json.dump(temp_json, f, ensure_ascii=False, indent=2)
    with open(os.path.join(case_dir, "temp_origin.json"), "w", encoding="utf-8") as f:
        json.dump({
            "origin_traj": [],
            "origin_sub_traj": "",
            "token_usage": {},
            "call_stats": {},
            "attempt_call_stats": {},
            "timing_stats": {},
            "case_duration_seconds": time.time() - case_start_time,
            "attempt_duration_seconds": time.time() - attempt_start_time,
            "attempt_index": attempt_idx,
            "evaluation_backend": "hle",
            "run_metadata": run_metadata,
        }, f, ensure_ascii=False, indent=2)


async def run_one_hle_sample(
    spec: DatasetSpec,
    sample: EvalSample,
    config: dict,
    backend: HLEBackend,
    semaphore: asyncio.Semaphore,
    *,
    outer_resume_path: Optional[str] = None,
    outer_round: Optional[int] = None,
    use_semaphore: bool = True,
) -> str:
    gate = semaphore if use_semaphore else contextlib.AsyncExitStack()
    async with gate:
        save_path = config["save_path"]
        now = datetime.datetime.now()
        case_dir = os.path.join(save_path, f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_row{sample.idx}")
        os.makedirs(case_dir, exist_ok=True)
        # The HLE harness attempts each selected question once and has no
        # per-case wall-clock cutoff. Unified-eval only changes persistence.
        max_attempts = 1
        case_timeout_seconds = float(config.get("hle_case_timeout_seconds") or 0)
        case_start_time = time.time()

        for attempt_idx in range(max_attempts):
            attempt_start_time = time.time()
            case_key = pretty_case_key(spec.name, sample.idx, attempt_idx)
            pretty = get_pretty_console()
            pretty.case_start(case_key, spec.name, sample.idx, sample.sample_id, attempt_idx)
            full_credit = False
            try:
                case_coro = backend.run(
                    sample,
                    case_dir,
                    outer_resume_path=outer_resume_path,
                    outer_round=outer_round,
                    max_steps=config.get("hle_max_steps"),
                )
                if case_timeout_seconds > 0:
                    remaining = case_timeout_seconds - (time.time() - case_start_time)
                    if remaining <= 0:
                        case_coro.close()
                        raise asyncio.TimeoutError
                    result = await asyncio.wait_for(case_coro, timeout=remaining)
                else:
                    result = await case_coro

                score_result = hle_score_result(result)
                full_credit = bool(result.get("full_credit"))
                call_stats = hle_call_stats(result)
                temp_json = {
                    "dataset_name": spec.name,
                    "sample_id": sample.sample_id,
                    "task_type": spec.task_type,
                    "question": sample.question,
                    "answer": sample.answer,
                    "predicted": result.get("predicted", ""),
                    "reasoning": result.get("reasoning"),
                    "evidence": result.get("evidence", []),
                    "confidence": result.get("confidence", ""),
                    "score": result.get("score", 0.0),
                    "score_result": score_result,
                    "official_scorer": HLE_SCORER_SOURCE,
                    "metadata": sample.metadata,
                    "trajectory": result.get("messages", []),
                    "sub_traj": "",
                    "timing_summary": {},
                    "evaluation_backend": "hle",
                    "run_metadata": config.get("run_metadata"),
                    "hle": {
                        "outer_round": outer_round,
                        "confidence_review": result.get('confidence_review', {}),
                        "outer_resume": result.get('outer_resume', {}),
                        "steps": result.get("steps", 0),
                        "trajectory": result.get("hle_trajectory", []),
                        "context_management_steps": result.get(
                            "context_management_steps", []
                        ),
                        "judge_response": result.get("judge_response", {}),
                        "model_usage_summary": result.get("model_usage_summary", {}),
                        "raw_prediction": result.get("raw_prediction", {}),
                    },
                }
                with open(os.path.join(case_dir, "temp.json"), "w", encoding="utf-8") as f:
                    json.dump(temp_json, f, ensure_ascii=False, indent=2)
                temp_origin = {
                    "origin_traj": result.get("messages", []),
                    "origin_sub_traj": "",
                    "token_usage": result.get("usage", {}),
                    "call_stats": call_stats,
                    "attempt_call_stats": call_stats,
                    "timing_stats": {},
                    "case_duration_seconds": time.time() - case_start_time,
                    "attempt_duration_seconds": time.time() - attempt_start_time,
                    "attempt_index": attempt_idx,
                    "evaluation_backend": "hle",
                    "run_metadata": config.get("run_metadata"),
                    "model_call_usage": result.get("model_call_usage", []),
                    "model_usage_summary": result.get("model_usage_summary", {}),
                }
                temp_origin.update(call_stats)
                with open(os.path.join(case_dir, "temp_origin.json"), "w", encoding="utf-8") as f:
                    json.dump(temp_origin, f, ensure_ascii=False, indent=2)

                is_final_attempt = full_credit or attempt_idx + 1 >= max_attempts
                if is_final_attempt:
                    pretty.case_done(
                        case_key,
                        result.get("score", 0.0),
                        score_result["status"],
                        call_stats,
                        time.time() - case_start_time,
                    )
                else:
                    pretty.case_retry(
                        case_key,
                        f"dataset={spec.name} idx={sample.idx} attempt={attempt_idx} score=0",
                    )
            except asyncio.TimeoutError:
                await write_hle_error_case(
                    spec,
                    sample,
                    case_dir,
                    attempt_idx,
                    "CaseTimeout",
                    f"case exceeded timeout_seconds={case_timeout_seconds}",
                    case_start_time,
                    attempt_start_time,
                    config.get("run_metadata"),
                )
                if attempt_idx + 1 >= max_attempts:
                    pretty.case_error(
                        case_key,
                        "CaseTimeout",
                        f"case exceeded timeout_seconds={case_timeout_seconds}",
                        time.time() - case_start_time,
                    )
            except Exception as exc:
                await write_hle_error_case(
                    spec,
                    sample,
                    case_dir,
                    attempt_idx,
                    type(exc).__name__,
                    str(exc),
                    case_start_time,
                    attempt_start_time,
                    config.get("run_metadata"),
                )
                if attempt_idx + 1 >= max_attempts:
                    pretty.case_error(
                        case_key,
                        type(exc).__name__,
                        str(exc),
                        time.time() - case_start_time,
                    )

            if full_credit or attempt_idx + 1 >= max_attempts:
                break
            failure_dir = os.path.join(case_dir, "failure_attempt", str(attempt_idx))
            os.makedirs(failure_dir, exist_ok=True)
            for fname in ("temp.json", "temp_origin.json", "fatal_error.txt"):
                src = os.path.join(case_dir, fname)
                if os.path.exists(src):
                    os.replace(src, os.path.join(failure_dir, fname))

        return os.path.join(case_dir, "temp.json")


async def run_hle_per_case_outer_chain(
    spec: DatasetSpec,
    sample: EvalSample,
    config: dict,
    backend: HLEBackend,
    semaphore: asyncio.Semaphore,
) -> None:
    prior_paths = config.get("hle_outer_resume_paths") or {}
    prior_path = prior_paths.get(sample.idx)
    source_root = str(config.get("hle_outer_resume_source_root") or "")
    fill_missing_outer1 = bool(
        config.get("hle_per_case_fill_missing_outer1", False)
    )
    if not prior_path:
        if not source_root:
            raise FileNotFoundError(f"no prior HLE outer result for row {sample.idx}")
        if not fill_missing_outer1:
            print(f"[HLE outer] row={sample.idx} waiting for its outer1 result")
            poll_seconds = max(
                0.01,
                float(config.get("hle_outer_source_poll_seconds", 5)),
            )
            while not prior_path:
                prior_path = collect_latest_result_paths_by_index(source_root).get(
                    sample.idx
                )
                if not prior_path:
                    await asyncio.sleep(poll_seconds)

    start_outer = int(config.get("hle_outer_round", 2))
    max_outer = int(config.get("hle_per_case_outer_max", start_outer))
    threshold = float(config.get("hle_rerun_confidence_threshold", 95))
    per_outer_max_steps = max(1, int(config.get("hle_max_steps", 100)))
    total_max_steps = max(
        0,
        int(config.get("hle_per_case_total_max_steps", 0) or 0),
    )
    outer_root = str(config.get("hle_per_case_outer_root") or "")
    if not outer_root:
        raise ValueError("hle_per_case_outer_root is required for per-case outer mode")
    if not 2 <= start_outer <= max_outer:
        raise ValueError("invalid per-case HLE outer range")

    def publish_final(result_path):
        final_root = config.get("hle_final_save_path")
        if final_root:
            source = os.path.dirname(result_path)
            destination = os.path.join(final_root, os.path.basename(source))
            shutil.copytree(source, destination, dirs_exist_ok=True)

    used_steps = 0
    async with semaphore:
        chain_started = time.monotonic()
        timeout = float(config.get("hle_case_timeout_seconds") or 0)
        if not prior_path:
            # Recheck after acquiring the shared slot so a resumed run never
            # duplicates outer1 work completed while this task was queued.
            prior_path = collect_latest_result_paths_by_index(source_root).get(
                sample.idx
            )
        if not prior_path:
            outer1_config = dict(config)
            outer1_config["save_path"] = source_root
            if total_max_steps:
                outer1_config["hle_max_steps"] = min(per_outer_max_steps, total_max_steps)
            os.makedirs(source_root, exist_ok=True)
            print(
                f"[HLE outer] row={sample.idx} starting missing outer1; "
                f"max_steps={per_outer_max_steps}"
            )
            prior_path = await run_one_hle_sample(
                spec,
                sample,
                outer1_config,
                backend,
                semaphore,
                outer_resume_path=None,
                outer_round=1,
                use_semaphore=False,
            )

        for outer in range(start_outer, max_outer + 1):
            try:
                with open(prior_path, "r", encoding="utf-8") as handle:
                    prior_result = json.load(handle)
            except Exception:
                prior_result = {}
            if outer == start_outer:
                used_steps = hle_result_model_calls(prior_result)
                if used_steps == 0 and not hle_result_meets_finish_confidence_threshold(
                    prior_result, threshold
                ):
                    used_steps = per_outer_max_steps
            if hle_result_meets_finish_confidence_threshold(prior_result, threshold):
                print(
                    f"[HLE outer] row={sample.idx} stopped before outer{outer}: "
                    f"confidence threshold {threshold:g} reached"
                )
                publish_final(prior_path)
                return

            remaining_steps = (
                total_max_steps - used_steps if total_max_steps > 0 else per_outer_max_steps
            )
            if total_max_steps > 0 and remaining_steps < 2:
                print(
                    f"[HLE outer] row={sample.idx} stopped before outer{outer}: "
                    f"total model-call budget {total_max_steps} reached "
                    f"(used={used_steps})"
                )
                publish_final(prior_path)
                return

            outer_config = dict(config)
            if timeout > 0:
                remaining_seconds = timeout - (time.monotonic() - chain_started)
                if remaining_seconds <= 0:
                    publish_final(prior_path)
                    return
                outer_config["hle_case_timeout_seconds"] = remaining_seconds
            outer_config["hle_max_steps"] = min(per_outer_max_steps, remaining_steps)
            outer_config["save_path"] = dataset_save_path(
                os.path.join(outer_root, f"outer{outer}"),
                spec.name,
            )
            os.makedirs(outer_config["save_path"], exist_ok=True)
            print(
                f"[HLE outer] row={sample.idx} starting outer{outer}/{max_outer} "
                f"from {os.path.basename(os.path.dirname(prior_path))}; "
                f"max_steps={outer_config['hle_max_steps']} used_total={used_steps}"
            )
            existing_path = collect_latest_result_paths_by_index(
                outer_config["save_path"]
            ).get(sample.idx)
            if existing_path:
                print(f"[HLE outer] row={sample.idx} reusing saved outer{outer}")
                prior_path = existing_path
            else:
                prior_path = await run_one_hle_sample(
                    spec,
                    sample,
                    outer_config,
                    backend,
                    semaphore,
                    outer_resume_path=prior_path,
                    outer_round=outer,
                    use_semaphore=False,
                )
            try:
                with open(prior_path, "r", encoding="utf-8") as handle:
                    completed_outer_result = json.load(handle)
            except Exception:
                completed_outer_result = {}
            completed_outer_steps = hle_result_model_calls(completed_outer_result)
            used_steps += completed_outer_steps or outer_config["hle_max_steps"]
        publish_final(prior_path)


def dry_run(specs: Dict[str, DatasetSpec], args) -> None:
    for spec in specs.values():
        try:
            samples = load_samples(spec, no_decrypt=args.no_decrypt)
            target_indices = parse_target_indices(args.target_indices)
            start_index, end_index, shuffle_samples = dataset_selection_config(
                spec.name,
                args,
                spec.evaluation_backend,
            )
            hle_rerun_stats = None
            if args.hle_rerun_source_root and spec.evaluation_backend == "hle":
                source_save_path = dataset_save_path(args.hle_rerun_source_root, spec.name)
                candidate_samples = select_samples_for_spec(
                    spec,
                    samples,
                    start_index,
                    end_index,
                    target_indices,
                    args.target_shard_rank,
                    args.target_shard_count,
                    set(),
                    shuffle_samples=shuffle_samples,
                    hle_seed=args.hle_seed,
                )
                expected_indices = {sample.idx for sample in candidate_samples}
                rerun_indices, hle_rerun_stats = collect_hle_rerun_indices_and_copy_retained(
                    source_save_path,
                    "",
                    args.hle_rerun_confidence_threshold,
                    False,
                    expected_indices,
                )
                target_indices = (
                    rerun_indices
                    if target_indices is None
                    else target_indices & rerun_indices
                )
            selected = select_samples_for_spec(
                spec,
                samples,
                start_index,
                end_index,
                target_indices,
                args.target_shard_rank,
                args.target_shard_count,
                set(),
                shuffle_samples=shuffle_samples,
                hle_seed=args.hle_seed,
            )
            preview = selected[0] if selected else None
            print(json.dumps({
                "dataset_name": spec.name,
                "num_samples": len(samples),
                "num_selected": len(selected),
                "start_index": start_index,
                "end_index": end_index,
                "shuffle": shuffle_samples,
                "task_type": spec.task_type,
                "evaluation_backend": spec.evaluation_backend,
                "data_path": spec.data_path,
                "prompt_source": (
                    f"{args.hle_harness_dir}/prompts_no_subagent.py"
                    if spec.evaluation_backend == "hle"
                    else spec.prompt_source or "src/prompts_no_subagent.py"
                ),
                "tools_direct": (
                    spec.tools
                    if spec.evaluation_backend == "hle"
                    else tool_names_for_mode(spec.tools, "direct")
                ),
                "tools_refine_summary": (
                    spec.tools
                    if spec.evaluation_backend == "hle"
                    else tool_names_for_mode(spec.tools, "refine_summary")
                ),
                "official_scorer": spec.scorer,
                "leak_filter": spec.leak_filter,
                "enable_visit_fallback": args.enable_visit_fallback,
                "hle_rerun": hle_rerun_stats,
                "hle": {
                    "harness_dir": args.hle_harness_dir,
                    "judge_script": args.hle_judge_script,
                    "seed": args.hle_seed,
                    "num_samples": args.hle_num_samples,
                    "max_completion_tokens": args.hle_max_completion_tokens,
                    "truncation_max_completion_tokens": args.hle_truncation_max_completion_tokens,
                    "max_context_tokens": args.hle_max_context_tokens,
                    "max_total_tokens": args.hle_max_total_tokens,
                    "max_steps": args.hle_max_steps,
                    "tool_call_regen_max_retries": args.hle_tool_call_regen_max_retries,
                    "enable_thinking": args.hle_enable_thinking,
                    "preserve_thinking": args.hle_preserve_thinking,
                    "per_case_outer_max": args.hle_per_case_outer_max,
                    "per_case_outer_root": args.hle_per_case_outer_root,
                } if spec.evaluation_backend == "hle" else None,
                "first_sample": {
                    "sample_id": preview.sample_id,
                    "idx": preview.idx,
                    "question_preview": preview.question[:240],
                    "answer_preview": str(preview.answer)[:120],
                    "metadata": preview.metadata,
                    "attachments": preview.attachments,
                } if preview else None,
            }, ensure_ascii=False, indent=2))
        except Exception as e:
            print(json.dumps({
                "dataset_name": spec.name,
                "status": "load_error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "official_scorer": spec.scorer,
            }, ensure_ascii=False, indent=2))


async def run_all(args) -> None:
    dataset_names = parse_dataset_names(args.datasets)
    specs = load_dataset_specs(dataset_names, data_path_override=args.data_path or None, judge_mode=args.judge_mode)
    if args.dry_run:
        dry_run(specs, args)
        return

    if not args.save_path:
        raise ValueError("--save_path is required unless --dry_run is set")
    os.makedirs(args.save_path, exist_ok=True)

    judge_model = args.judge_model or args.summary_model or args.model
    summary_model = args.summary_model or args.model
    repair_judge_model = args.direct_repair_judge_model or judge_model
    has_unified_backend = any(
        spec.evaluation_backend != "hle" for spec in specs.values()
    )
    tokenizer = None
    sum_tokenizer = None
    llm_client = None
    summary_client = None
    judge_client = None
    repair_judge_client = None

    if has_unified_backend:
        try:
            import httpx  # type: ignore
        except Exception as e:
            raise RuntimeError(f"httpx is required to run inference: {type(e).__name__}: {e}")
        from openai_retry_client import create_async_openai_with_retry

        tokenizer = import_tokenizer(args.tokenizer_path)
        sum_tokenizer = tokenizer if not args.sum_tokenizer_path else import_tokenizer(args.sum_tokenizer_path)
        llm_client = create_async_openai_with_retry(
            api_key=args.sdk_api_key,
            base_url=args.sdk_base_url or None,
            timeout=600,
            general_max_attempts=args.general_max_attempts,
        )
        summary_client = create_async_openai_with_retry(
            api_key=args.summary_api_key or args.sdk_api_key,
            base_url=args.summary_base_url or args.sdk_base_url or None,
            timeout=600,
            general_max_attempts=args.general_max_attempts,
        )
        judge_client = create_async_openai_with_retry(
            api_key=args.judge_api_key or args.summary_api_key or args.sdk_api_key,
            base_url=args.judge_base_url or args.summary_base_url or args.sdk_base_url or None,
            timeout=600,
            general_max_attempts=args.general_max_attempts,
        )
        if args.direct_repair_judge_base_url or args.direct_repair_judge_api_key or args.direct_repair_judge_model:
            repair_judge_client = create_async_openai_with_retry(
                api_key=args.direct_repair_judge_api_key or args.judge_api_key or args.summary_api_key or args.sdk_api_key,
                base_url=args.direct_repair_judge_base_url or args.judge_base_url or args.summary_base_url or args.sdk_base_url or None,
                timeout=600,
                general_max_attempts=args.general_max_attempts,
            )

    semaphore = asyncio.Semaphore(args.concurrency_limit)
    pretty = get_pretty_console()
    hle_backends: List[HLEBackend] = []
    async with contextlib.AsyncExitStack() as stack:
        if getattr(args, "concurrency_control_file", ""):
            from live_concurrency import LiveConcurrency

            semaphore = await stack.enter_async_context(
                LiveConcurrency(
                    args.concurrency_limit, args.concurrency_control_file
                ).monitor()
            )
        serper_client = None
        jina_client = None
        if has_unified_backend:
            search_limits = httpx.Limits(max_connections=100, max_keepalive_connections=80)
            jina_limits = httpx.Limits(max_connections=100, max_keepalive_connections=80)
            serper_client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=600, follow_redirects=True, limits=search_limits)
            )
            jina_client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=600, follow_redirects=True, limits=jina_limits)
            )
        all_tasks = []
        for spec in specs.values():
            save_path = dataset_save_path(args.save_path, spec.name)
            os.makedirs(save_path, exist_ok=True)
            samples = load_samples(spec, no_decrypt=args.no_decrypt)
            target_indices = parse_target_indices(args.target_indices)
            if args.mode == "return":
                prev_path = dataset_save_path(args.prev_result_path, spec.name) if args.prev_result_path else save_path
                return_indices = collect_return_indices(prev_path)
                target_indices = return_indices if target_indices is None else (target_indices & return_indices)
                pretty.return_selection(spec.name, len(target_indices))

            start_index, end_index, shuffle_samples = dataset_selection_config(
                spec.name,
                args,
                spec.evaluation_backend,
            )
            hle_outer_resume_paths = {}
            hle_outer_resume_source_root = ""
            if args.hle_rerun_source_root and spec.evaluation_backend == "hle":
                source_save_path = dataset_save_path(args.hle_rerun_source_root, spec.name)
                candidate_samples = select_samples_for_spec(
                    spec,
                    samples,
                    start_index,
                    end_index,
                    target_indices,
                    args.target_shard_rank,
                    args.target_shard_count,
                    set(),
                    shuffle_samples=shuffle_samples,
                    hle_seed=args.hle_seed,
                )
                expected_indices = {sample.idx for sample in candidate_samples}
                rerun_indices, hle_rerun_stats = collect_hle_rerun_indices_and_copy_retained(
                    source_save_path,
                    save_path,
                    args.hle_rerun_confidence_threshold,
                    args.hle_copy_retained_results,
                    expected_indices,
                )
                target_indices = (
                    rerun_indices
                    if target_indices is None
                    else target_indices & rerun_indices
                )
                if args.hle_outer_review_resume:
                    source_paths = collect_latest_result_paths_by_index(source_save_path)
                    hle_outer_resume_paths = {
                        idx: path for idx, path in source_paths.items()
                        if idx in rerun_indices
                    }
                    hle_outer_resume_source_root = source_save_path
                print(json.dumps({
                    "dataset_name": spec.name,
                    "hle_rerun": hle_rerun_stats,
                }, ensure_ascii=False, indent=2))

            skip_indices = collect_existing_indices([save_path], args.skip_existing_mode)
            selected = select_samples_for_spec(
                spec,
                samples,
                start_index,
                end_index,
                target_indices,
                args.target_shard_rank,
                args.target_shard_count,
                skip_indices,
                shuffle_samples=shuffle_samples,
                hle_seed=args.hle_seed,
            )
            pretty.dataset(
                spec.name,
                len(samples),
                len(selected),
                start_index,
                end_index,
                shuffle_samples,
                save_path,
            )
            config = vars(args).copy()
            config.update({
                "save_path": save_path,
                "summary_model": summary_model,
                "judge_model": judge_model,
                "context_limit_strategy": mode_to_context_strategy(args.mode),
                "mode": args.mode,
            })
            run_metadata = {
                "dataset": spec.name,
                "git": _git_provenance(),
                "model": args.model or "unknown",
                "endpoint": args.sdk_base_url or "provider-default",
                "summary_model": summary_model or "unknown",
                "summary_endpoint": args.summary_base_url or args.sdk_base_url or "provider-default",
                "judge_model": judge_model or "unknown",
                "judge_endpoint": args.judge_base_url or args.summary_base_url or args.sdk_base_url or "provider-default",
                "task_range": {
                    "start": start_index,
                    "end": min(end_index, len(samples)),
                    "requested_end": end_index,
                    "count": len(selected),
                    "shuffle": shuffle_samples,
                },
                "evaluator": {
                    "mode": args.mode,
                    "judge_mode": args.judge_mode,
                },
                "data": {
                    "path": spec.data_path,
                    "sha256": _sha256_path(spec.data_path),
                },
            }
            run_metadata.update({
                "git_commit": run_metadata["git"]["commit"],
                "evaluator_mode": args.mode,
                "data_checksum": run_metadata["data"]["sha256"],
            })
            config["run_metadata"] = run_metadata
            write_run_metadata(save_path, run_metadata)
            if hle_outer_resume_paths:
                config["hle_outer_resume_paths"] = hle_outer_resume_paths
            if hle_outer_resume_source_root:
                config["hle_outer_resume_source_root"] = hle_outer_resume_source_root
            if args.confidence_outer_resume_root and spec.evaluation_backend != "hle":
                resume_dataset_root = dataset_save_path(
                    args.confidence_outer_resume_root,
                    spec.name,
                )
                config["confidence_outer_resume_paths"] = collect_latest_result_paths_by_index(
                    resume_dataset_root
                )
            if spec.evaluation_backend == "hle":
                if args.hle_per_case_outer_max > 1 and not args.hle_rerun_source_root:
                    # A fresh public run starts outer1 itself. Store unfinished
                    # rounds outside HLE/ so restarting cannot skip an incomplete chain.
                    outer_root = args.hle_per_case_outer_root or os.path.join(args.save_path, "_hle_outer")
                    config["hle_outer_resume_source_root"] = dataset_save_path(
                        os.path.join(outer_root, "outer1"), spec.name,
                    )
                    config["hle_per_case_outer_root"] = outer_root
                    config["hle_per_case_fill_missing_outer1"] = True
                    config["hle_outer_round"] = 2
                    config["hle_final_save_path"] = save_path
                backend = HLEBackend(config)
                hle_backends.append(backend)
                for sample in selected:
                    if args.hle_per_case_outer_max > 1:
                        all_tasks.append(run_hle_per_case_outer_chain(
                            spec,
                            sample,
                            config,
                            backend,
                            semaphore,
                        ))
                    else:
                        all_tasks.append(run_one_hle_sample(
                            spec,
                            sample,
                            config,
                            backend,
                            semaphore,
                        ))
            else:
                for sample in selected:
                    all_tasks.append(run_one_sample(
                        spec,
                        sample,
                        config,
                        llm_client,
                        summary_client,
                        serper_client,
                        jina_client,
                        tokenizer,
                        sum_tokenizer,
                        judge_client,
                        judge_model,
                        repair_judge_client,
                        repair_judge_model,
                        semaphore,
                    ))
        pretty.start_run(
            len(all_tasks),
            {
                "datasets": " ".join(dataset_names),
                "mode": args.mode,
                "model": args.model,
                "concurrency": args.concurrency_limit,
                "save_path": args.save_path,
            },
        )
        run_status = "finished"
        try:
            await asyncio.gather(*all_tasks)
        except Exception:
            run_status = "failed"
            raise
        finally:
            pretty.finish_run(run_status)
            for backend in hle_backends:
                await backend.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified multi-dataset agent inference and official evaluation")
    parser.add_argument("--datasets", nargs="+", default=["BrowseComp"], help=f"Comma/space separated dataset names. Available: {', '.join(available_datasets())}")
    parser.add_argument("--mode", type=str, default="direct", choices=["direct", "refine_summary", "return"])
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--prev-result-path", "--prev_result_path", dest="prev_result_path", type=str, default="")

    parser.add_argument(
        "--sdk_base_url",
        type=str,
        default=os.environ.get("AREX_SDK_BASE_URL", os.environ.get("BASE_URL", "")),
    )
    parser.add_argument(
        "--sdk_api_key",
        type=str,
        default=os.environ.get("AREX_SDK_API_KEY", os.environ.get("API_KEY", "")),
    )
    parser.add_argument("--model", type=str, default=os.environ.get("AREX_MODEL_NAME", ""))
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=240000)
    parser.add_argument("--max_response_tokens", type=int, default=None)
    parser.add_argument("--agent_temperature", "--main_agent_temperature", dest="agent_temperature", type=float, default=None)
    parser.add_argument("--agent_top_p", type=float, default=None)
    parser.add_argument("--agent_top_k", type=int, default=None)
    parser.add_argument("--agent_min_p", type=float, default=None)
    parser.add_argument("--agent_presence_penalty", type=float, default=None)
    parser.add_argument("--agent_repetition_penalty", type=float, default=None)
    agent_thinking_group = parser.add_mutually_exclusive_group()
    agent_thinking_group.add_argument(
        "--enable-thinking",
        "--enable_thinking",
        dest="enable_thinking",
        action="store_true",
        help="Enable thinking for agent and reviewer requests (default).",
    )
    agent_thinking_group.add_argument(
        "--disable-thinking",
        "--disable_thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable thinking for agent and reviewer requests.",
    )
    parser.set_defaults(enable_thinking=True)
    parser.add_argument("--preserve_thinking", action="store_true")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--summary_base_url", type=str, default="")
    parser.add_argument("--summary_api_key", type=str, default=os.environ.get("AREX_SUMMARY_API_KEY", ""))
    parser.add_argument("--summary_model", type=str, default="")
    parser.add_argument("--sum_tokenizer_path", type=str, default="")
    summary_thinking_group = parser.add_mutually_exclusive_group()
    summary_thinking_group.add_argument(
        "--summary-enable-thinking",
        "--summary_enable_thinking",
        dest="summary_enable_thinking",
        action="store_true",
        help="Enable thinking for summary-model requests (default).",
    )
    summary_thinking_group.add_argument(
        "--summary-disable-thinking",
        "--summary_disable_thinking",
        dest="summary_enable_thinking",
        action="store_false",
        help="Explicitly disable thinking for summary-model requests.",
    )
    parser.set_defaults(summary_enable_thinking=True)
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
    parser.set_defaults(enable_visit_fallback=visit_fallback_default_enabled())
    parser.add_argument("--judge_base_url", type=str, default="")
    parser.add_argument("--judge_api_key", type=str, default=os.environ.get("AREX_JUDGE_API_KEY", ""))
    parser.add_argument("--judge_model", type=str, default="")
    parser.add_argument("--judge-mode", "--judge_mode", dest="judge_mode", type=str, default="local", choices=["local", "offical", "official"])

    hle_group = parser.add_argument_group("HLE-Harness compatibility backend")
    hle_group.add_argument('--hle-enable-confidence-review', action='store_true',
                           help='Review low-confidence HLE finishes within the shared model-call budget.')
    hle_group.add_argument('--hle-review-threshold', type=float, default=95)
    hle_group.add_argument('--hle-review-middle-threshold', type=float, default=90)
    hle_group.add_argument('--hle-review-max-rounds', type=int, default=2)
    hle_group.add_argument('--hle-review-max-tokens', type=int, default=4096)
    hle_group.add_argument(
        '--hle-outer-review-resume',
        action='store_true',
        help='For selected HLE reruns, review the prior outer result and pass the feedback to the new attempt.',
    )
    hle_group.add_argument('--hle-outer-round', type=int, default=1)
    hle_group.add_argument(
        '--hle-per-case-outer-max',
        type=int,
        default=1,
        help='Advance each selected HLE case independently through this outer round without a batch barrier.',
    )
    hle_group.add_argument(
        '--hle-per-case-outer-root',
        type=str,
        default='',
        help='Root containing outer2, outer3, and later per-case HLE result directories.',
    )
    hle_group.add_argument(
        '--hle-per-case-total-max-steps',
        type=int,
        default=0,
        help='Cross-outer HLE model-call cap per case, including outer reviewers; 0 disables the cap.',
    )
    hle_group.add_argument(
        '--hle-per-case-fill-missing-outer1',
        action='store_true',
        help=(
            'In per-case outer mode, generate a missing outer1 result in the '
            'rerun source root and immediately continue that case through later '
            'outers under the same global concurrency semaphore.'
        ),
    )
    hle_group.add_argument(
        "--hle-harness-dir",
        "--hle_harness_dir",
        dest="hle_harness_dir",
        default=DEFAULT_HLE_HARNESS_DIR,
    )
    hle_group.add_argument(
        "--hle-judge-script",
        "--hle_judge_script",
        dest="hle_judge_script",
        default=DEFAULT_HLE_JUDGE_SCRIPT,
    )
    hle_group.add_argument("--hle-seed", "--hle_seed", dest="hle_seed", type=int, default=125)
    hle_group.add_argument(
        "--hle-num-samples",
        "--hle_num_samples",
        dest="hle_num_samples",
        type=int,
        default=200,
    )
    hle_group.add_argument(
        "--hle-max-completion-tokens",
        "--hle_max_completion_tokens",
        dest="hle_max_completion_tokens",
        type=int,
        default=49152,
    )
    hle_group.add_argument(
        "--hle-truncation-max-completion-tokens",
        "--hle_truncation_max_completion_tokens",
        dest="hle_truncation_max_completion_tokens",
        type=int,
        default=8192,
    )
    hle_group.add_argument(
        "--hle-max-context-tokens",
        "--hle_max_context_tokens",
        dest="hle_max_context_tokens",
        type=int,
        default=200000,
    )
    hle_group.add_argument(
        "--hle-max-total-tokens",
        "--hle_max_total_tokens",
        dest="hle_max_total_tokens",
        type=int,
        default=262144,
    )
    hle_group.add_argument(
        "--hle-max-steps",
        "--hle_max_steps",
        dest="hle_max_steps",
        type=int,
        default=100,
    )
    hle_group.add_argument(
        "--hle-case-timeout-seconds",
        "--hle_case_timeout_seconds",
        dest="hle_case_timeout_seconds",
        type=float,
        default=0,
        help="Optional safety override; zero preserves the original no-timeout behavior.",
    )
    hle_group.add_argument(
        "--hle-tool-call-regen-max-retries",
        "--hle_tool_call_regen_max_retries",
        dest="hle_tool_call_regen_max_retries",
        type=int,
        default=20,
    )
    hle_group.add_argument("--hle-general-max-attempts", type=int, default=None,
                           help="Use the unified client's request retry policy for HLE when specified.")
    hle_group.add_argument("--hle-llm-call-max-retries", type=int, default=20,
                           help="Maximum HLE solver attempts per logical call.")
    hle_group.add_argument("--hle-temperature", "--hle_temperature", dest="hle_temperature", type=float, default=1.0)
    hle_group.add_argument("--hle-top-p", "--hle_top_p", dest="hle_top_p", type=float, default=0.95)
    hle_group.add_argument("--hle-top-k", "--hle_top_k", dest="hle_top_k", type=int, default=None)
    hle_group.add_argument("--hle-min-p", "--hle_min_p", dest="hle_min_p", type=float, default=None)
    hle_group.add_argument(
        "--hle-presence-penalty",
        "--hle_presence_penalty",
        dest="hle_presence_penalty",
        type=float,
        default=None,
    )
    hle_group.add_argument(
        "--hle-repetition-penalty",
        "--hle_repetition_penalty",
        dest="hle_repetition_penalty",
        type=float,
        default=None,
    )
    hle_thinking_group = hle_group.add_mutually_exclusive_group()
    hle_thinking_group.add_argument(
        "--hle-enable-thinking",
        "--hle_enable_thinking",
        dest="hle_enable_thinking",
        action="store_true",
    )
    hle_thinking_group.add_argument(
        "--hle-disable-thinking",
        "--hle_disable_thinking",
        dest="hle_enable_thinking",
        action="store_false",
    )
    parser.set_defaults(hle_enable_thinking=False)
    hle_group.add_argument(
        "--hle-preserve-thinking",
        "--hle_preserve_thinking",
        dest="hle_preserve_thinking",
        action="store_true",
    )
    hle_group.add_argument(
        "--hle-judge-num-workers",
        "--hle_judge_num_workers",
        dest="hle_judge_num_workers",
        type=int,
        default=4,
    )
    hle_group.add_argument(
        "--hle-judge-timeout",
        "--hle_judge_timeout",
        dest="hle_judge_timeout",
        type=float,
        default=600,
    )
    hle_group.add_argument(
        "--hle-judge-max-retries",
        "--hle_judge_max_retries",
        dest="hle_judge_max_retries",
        type=int,
        default=0,
    )
    hle_group.add_argument(
        "--hle-judge-attempts",
        "--hle_judge_attempts",
        dest="hle_judge_attempts",
        type=int,
        default=3,
    )
    hle_group.add_argument(
        "--hle-judge-max-tokens",
        "--hle_judge_max_tokens",
        dest="hle_judge_max_tokens",
        type=int,
        default=8192,
    )
    hle_group.add_argument(
        "--hle-judge-truncate-response-token-threshold",
        "--hle_judge_truncate_response_token_threshold",
        dest="hle_judge_truncate_response_token_threshold",
        type=int,
        default=100000,
    )
    hle_group.add_argument(
        "--hle-judge-truncate-response-chars",
        "--hle_judge_truncate_response_chars",
        dest="hle_judge_truncate_response_chars",
        type=int,
        default=100000,
    )
    hle_group.add_argument(
        "--hle-rerun-source-root",
        "--hle_rerun_source_root",
        dest="hle_rerun_source_root",
        type=str,
        default="",
        help="For the HLE backend, read prior results from this root and rerun only unfinished or low-confidence cases.",
    )
    hle_group.add_argument(
        "--hle-rerun-confidence-threshold",
        "--hle_rerun_confidence_threshold",
        dest="hle_rerun_confidence_threshold",
        type=float,
        default=90.0,
        help="For --hle-rerun-source-root, retain only HLE cases with finish confidence greater than or equal to this threshold.",
    )
    hle_group.add_argument(
        "--hle-copy-retained-results",
        "--hle_copy_retained_results",
        dest="hle_copy_retained_results",
        action="store_true",
        help="Copy retained HLE prior-result directories into the current save path before rerunning failed/low-confidence cases.",
    )

    parser.add_argument("--discard_all_tokens_limit", type=int, default=None, help="Deprecated direct-mode context limit override. Defaults to --max_tokens.")
    parser.add_argument("--refine_summary_max_outer_rounds", type=int, default=5)
    parser.add_argument("--refine_summary_max_llm_calls", type=int, default=300)
    parser.add_argument(
        "--refine_summary_max_total_llm_calls",
        type=int,
        default=0,
        help="Cross-outer logical LLM-call cap for refine_summary; 0 disables the cap.",
    )
    parser.add_argument("--refine_summary_trigger_tokens", type=int, default=None, help="Token threshold for prompting update_context in refine_summary mode. Defaults to --max_tokens.")
    parser.add_argument("--refine_summary_max_updates", type=int, default=5)
    parser.add_argument("--enable-confidence-outer-retry", "--enable_confidence_outer_retry",
                        dest="enable_confidence_outer_retry", action="store_true",
                        help="Retry refine-summary outer rounds until finish confidence reaches the threshold; use a trajectory reviewer to choose restart or refine.")
    parser.add_argument("--confidence-outer-retry-threshold", "--confidence_outer_retry_threshold",
                        dest="confidence_outer_retry_threshold", type=float, default=95.0)
    parser.add_argument("--confidence-outer-retry-review-max-tokens", "--confidence_outer_retry_review_max_tokens",
                        dest="confidence_outer_retry_review_max_tokens", type=int, default=4096)
    parser.add_argument("--enable-confidence-tiered-review", "--enable_confidence_tiered_review",
                        dest="enable_confidence_tiered_review", action="store_true",
                        help="When confidence outer retry is enabled, route sub-threshold reviews to separate middle- and low-confidence prompts while preserving the same four-field handoff schema.")
    parser.add_argument("--confidence-tiered-review-middle-threshold", "--confidence_tiered_review_middle_threshold",
                        dest="confidence_tiered_review_middle_threshold", type=float, default=90.0,
                        help="Lower bound for the middle-confidence reviewer. Lower or invalid confidence uses the low-confidence alternative-search reviewer.")
    parser.add_argument("--confidence-outer-resume-root", "--confidence_outer_resume_root",
                        dest="confidence_outer_resume_root", type=str, default="",
                        help="Load normal results from this summary root, recover outer-1, and continue confidence-aware retries through the configured maximum outer round.")
    parser.add_argument("--confidence-outer-resume-missing-from-scratch", "--confidence_outer_resume_missing_from_scratch",
                        dest="confidence_outer_resume_missing_from_scratch", action="store_true",
                        help="When a resume root lacks a selected case, run that case from outer round 1 instead of failing. Cases with a saved source still resume from it.")
    parser.add_argument("--enable-env-feedback", "--enable_env_feedback", dest="enable_env_feedback", action="store_true",
                        help="Inject a soft user feedback message during agent search.")
    parser.add_argument("--env-feedback-round", "--env_feedback_round", dest="env_feedback_round", type=int, default=20,
                        help="Inject env feedback before the kth LLM call.")
    parser.add_argument("--env-feedback-message", "--env_feedback_message", dest="env_feedback_message", type=str, default="",
                        help="Optional custom env feedback message.")
    parser.add_argument("--general_max_attempts", type=int, default=5)
    parser.add_argument("--max_attempts", type=int, default=1)
    parser.add_argument("--case-timeout-seconds", "--case_timeout_seconds", dest="case_timeout_seconds", type=float, default=0)
    parser.add_argument("--tool_call_regen_max_retries", type=int, default=20)
    parser.add_argument("--llm_call_max_retries", type=int, default=300)
    parser.add_argument("--direct_context_limit_max_retries", type=int, default=8)
    parser.add_argument("--refine_summary_context_limit_max_retries", type=int, default=8)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=DEFAULT_END_INDEX)
    parser.add_argument("--target_indices", type=str, default="")
    parser.add_argument("--no-shuffle", "--no_shuffle", dest="no_shuffle", action="store_true")
    parser.add_argument("--dataset-start-indices", "--dataset_start_indices", dest="dataset_start_indices", type=str, default="", help="Per-dataset start overrides, e.g. 'BrowseComp=0 GAIA-2023-validation-text-103=10'")
    parser.add_argument("--dataset-end-indices", "--dataset_end_indices", dest="dataset_end_indices", type=str, default="", help="Per-dataset end overrides, e.g. 'BrowseComp=20 GAIA-2023-validation-text-103=30'")
    parser.add_argument("--dataset-shuffle", "--dataset_shuffle", dest="dataset_shuffle", type=str, default="", help="Per-dataset shuffle overrides, e.g. 'BrowseComp=0 GAIA-2023-validation-text-103=1'")
    parser.add_argument("--target-shard-rank", type=int, default=0)
    parser.add_argument("--target-shard-count", type=int, default=1)
    parser.add_argument("--concurrency_limit", type=int, default=16)
    parser.add_argument(
        "--concurrency-control-file", default="",
        help="Read the case concurrency limit from this file every 2 seconds; 0 pauses new cases.",
    )
    parser.add_argument("--no-decrypt", action="store_true")
    parser.add_argument("--skip-existing-mode", type=str, default="correct", choices=["none", "correct", "all"])
    parser.add_argument("--enable_ds_harness", action="store_true",
                        help="Use DeepSeek V4 native Chat Completions tool calling instead of XML prompt parsing.")
    parser.add_argument("--direct-repair-retry", "--direct_repair_retry", dest="direct_repair_retry", action="store_true",
                        help="Enable one gold-free direct repair retry after the first finish for matching datasets.")
    parser.add_argument("--direct-repair-datasets", "--direct_repair_datasets", dest="direct_repair_datasets", type=str,
                        default="BrowseComp",
                        help="Comma/space separated case-insensitive substrings for datasets eligible for direct repair retry.")
    parser.add_argument("--direct-repair-audit-max-tokens", "--direct_repair_audit_max_tokens",
                        dest="direct_repair_audit_max_tokens", type=int, default=4096)
    parser.add_argument("--direct-repair-judge-base-url", "--direct_repair_judge_base_url",
                        dest="direct_repair_judge_base_url", type=str, default="",
                        help="Optional OpenAI-compatible base URL used only for direct repair audit.")
    parser.add_argument("--direct-repair-judge-api-key", "--direct_repair_judge_api_key",
                        dest="direct_repair_judge_api_key", type=str, default="",
                        help="Optional API key used only for direct repair audit.")
    parser.add_argument("--direct-repair-judge-model", "--direct_repair_judge_model",
                        dest="direct_repair_judge_model", type=str, default="",
                        help="Optional model used only for direct repair audit.")
    return parser


def redacted_args(args) -> dict:
    return {key: ("[REDACTED]" if "api_key" in key and value else value)
            for key, value in vars(args).items()}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pretty = get_pretty_console()
    if pretty.enabled:
        pretty.panel("Args", json.dumps(redacted_args(args), ensure_ascii=False, indent=2), style="bright_black")
    else:
        print(redacted_args(args))
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
