"""Adapter for the exact HLE agent and judge used by the evaluation suite.

The adapter deliberately delegates model-facing behavior to the original
HLE-Harness modules.  This module only maps unified-eval configuration and
results to those modules.
"""

import argparse
import asyncio
import contextlib
import copy
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from unittest import mock

from unified_eval.types import EvalSample


UNIFY_EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_HLE_HARNESS_DIR = str(UNIFY_EVAL_ROOT / "hle_vendor")
DEFAULT_HLE_JUDGE_SCRIPT = str(
    UNIFY_EVAL_ROOT / "hle_vendor" / "HLE_judge.py"
)
HLE_SCORER_SOURCE = f"{DEFAULT_HLE_JUDGE_SCRIPT} (HLE judge adapter)"
QWEN_USAGE_TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"
_LEGACY_RESULT_KEY = "hle" + "_0724"
_LEGACY_TOKENIZER_ENV = "HLE" + "_0724_QWEN_TOKENIZER_PATH"


def select_hle_samples(
    samples: Sequence[EvalSample],
    start_index: int,
    end_index: int,
    target_indices: Optional[set[int]],
    target_shard_rank: int,
    target_shard_count: int,
    skip_indices: set[int],
    shuffle_samples: bool,
    seed: int = 125,
) -> List[EvalSample]:
    """Match the benchmark ``random.sample(..., NUM_SAMPLES)`` selection order."""
    indices = list(range(len(samples)))
    if target_indices is not None:
        selected = [idx for idx in indices if idx in target_indices]
    elif shuffle_samples:
        capped_end = min(max(0, end_index), len(indices))
        if capped_end < len(indices):
            sampled_prefix = random.Random(seed).sample(indices, capped_end)
            selected = sampled_prefix[start_index:capped_end]
        else:
            # The original harness does not shuffle when NUM_SAMPLES is at
            # least the dataset size.
            selected = indices[start_index:end_index]
    else:
        selected = indices[start_index:end_index]

    if target_shard_count > 1:
        if target_shard_rank < 0 or target_shard_rank >= target_shard_count:
            raise ValueError(
                f"--target-shard-rank must be in [0, {target_shard_count}), "
                f"got {target_shard_rank}"
            )
        selected = selected[target_shard_rank::target_shard_count]
    return [samples[idx] for idx in selected if idx not in skip_indices]


def _cached_qwen_usage_tokenizer() -> Optional[str]:
    override = os.environ.get("HLE_QWEN_TOKENIZER_PATH") or os.environ.get(_LEGACY_TOKENIZER_ENV)
    if override and Path(override).is_dir():
        return override
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(QWEN_USAGE_TOKENIZER_ID, local_files_only=True)
    except Exception:
        return None


def _tokenizer_import_patch(enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    local_path = _cached_qwen_usage_tokenizer()
    if not local_path:
        return contextlib.nullcontext()
    from transformers import AutoTokenizer

    original = AutoTokenizer.from_pretrained

    def load_from_local_snapshot(name_or_path, *args, **kwargs):
        if str(name_or_path) == QWEN_USAGE_TOKENIZER_ID:
            name_or_path = local_path
        return original(name_or_path, *args, **kwargs)

    return mock.patch.object(AutoTokenizer, "from_pretrained", load_from_local_snapshot)


def _load_module(
    module_name: str,
    path: Path,
    import_root: Optional[Path] = None,
    patch_usage_tokenizer: bool = False,
):
    if not path.is_file():
        raise FileNotFoundError(f"HLE module not found: {path}")
    old_sys_path = list(sys.path)
    saved_tools = sys.modules.pop("tools", None) if import_root else None
    try:
        if import_root:
            sys.path.insert(0, str(import_root))
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import HLE module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        with _tokenizer_import_patch(patch_usage_tokenizer):
            spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_sys_path
        if import_root:
            sys.modules.pop("tools", None)
            if saved_tools is not None:
                sys.modules["tools"] = saved_tools


def _extract_finish_fields(agent_module, result: Dict[str, Any]) -> tuple[Any, int | float | None]:
    if 'accepted_finish' in result:
        accepted = result['accepted_finish'] or {}
        return accepted.get('evidences', []), accepted.get('confidence')
    for message in reversed(result.get("messages") or []):
        if message.get("role") != "assistant":
            continue
        calls = agent_module.parse_qwen_tool_calls(message.get("content") or "")
        for call in calls:
            if call.get("function") != "finish":
                continue
            arguments = call.get("arguments") or {}
            confidence = arguments.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                confidence = None
            return arguments.get("evidences") or [], confidence
    return [], None


class HLEBackend:
    def __init__(self, config: Dict[str, Any], agent_module=None, judge_module=None):
        self.config = dict(config)
        self.harness_dir = Path(
            self.config.get("hle_harness_dir") or DEFAULT_HLE_HARNESS_DIR
        )
        self.judge_script = Path(
            self.config.get("hle_judge_script") or DEFAULT_HLE_JUDGE_SCRIPT
        )
        self.agent_module = agent_module or _load_module(
            "unify_eval_hle_agent",
            self.harness_dir / "run_evaluation_kimi_agent.py",
            import_root=self.harness_dir,
            patch_usage_tokenizer=True,
        )
        self.judge_module = judge_module or _load_module(
            "unify_eval_hle_judge",
            self.judge_script,
        )

        self.base_agent_args = self._build_agent_args(output="hle.jsonl")
        self.agent_client = self.agent_module.build_openai_client(self.base_agent_args)
        self.agent_module.client = self.agent_client

        judge_args = argparse.Namespace(
            timeout=float(self.config.get("hle_judge_timeout", 600)),
            max_retries=int(self.config.get("hle_judge_max_retries", 0)),
            base_url=self.config.get("judge_base_url")
            or self.config.get("summary_base_url")
            or self.config.get("sdk_base_url") or None,
            api_key=self.config.get("judge_api_key")
            or self.config.get("summary_api_key")
            or self.config.get("sdk_api_key"),
        )
        self.judge_client = self.judge_module.build_client(judge_args)
        self.judge_model = (
            self.config.get("judge_model")
            or self.config.get("summary_model")
            or self.config.get("model")
        )
        self.judge_semaphore = asyncio.Semaphore(
            max(1, int(self.config.get("hle_judge_num_workers", 4)))
        )

    def _build_agent_args(self, output: str) -> argparse.Namespace:
        return argparse.Namespace(
            dataset=str(self.harness_dir / "data_json" / "text_items.jsonl"),
            model=self.config.get("model") or "",
            sdk_base_url=self.config.get("sdk_base_url") or None,
            sdk_api_key=self.config.get("sdk_api_key") or None,
            max_completion_tokens=int(
                self.config.get("hle_max_completion_tokens", 49152)
            ),
            temperature=float(self.config.get("hle_temperature", 1.0)),
            top_p=float(self.config.get("hle_top_p", 0.95)),
            top_k=self.config.get("hle_top_k"),
            min_p=self.config.get("hle_min_p"),
            presence_penalty=self.config.get("hle_presence_penalty"),
            repetition_penalty=self.config.get("hle_repetition_penalty"),
            enable_thinking=bool(self.config.get("hle_enable_thinking", False)),
            preserve_thinking=bool(self.config.get("hle_preserve_thinking", False)),
            enable_visit_fallback=bool(
                self.config.get("enable_visit_fallback", True)
            ),
            truncation_max_completion_tokens=int(
                self.config.get("hle_truncation_max_completion_tokens", 8192)
            ),
            max_context_tokens=int(self.config.get("hle_max_context_tokens", 200000)),
            max_total_tokens=int(self.config.get("hle_max_total_tokens", 262144)),
            max_steps=int(self.config.get("hle_max_steps", 100)),
            general_max_attempts=self.config.get("hle_general_max_attempts"),
            llm_call_max_retries=int(self.config.get("hle_llm_call_max_retries", 20)),
            enable_confidence_review=bool(self.config.get('hle_enable_confidence_review', False)),
            review_threshold=float(self.config.get('hle_review_threshold', 95)),
            review_middle_threshold=float(self.config.get('hle_review_middle_threshold', 90)),
            review_max_rounds=int(self.config.get('hle_review_max_rounds', 2)),
            review_max_tokens=int(self.config.get('hle_review_max_tokens', 4096)),
            outer_resume=None,
            outer_round=int(self.config.get('hle_outer_round', 1)),
            tool_call_regen_max_retries=int(
                self.config.get("hle_tool_call_regen_max_retries", 20)
            ),
            num_workers=1,
            num_samples=1,
            seed=int(self.config.get("hle_seed", 125)),
            output=output,
            text_only=True,
        )

    async def run(
        self,
        sample: EvalSample,
        case_dir: str,
        outer_resume_path: Optional[str] = None,
        outer_round: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        question = copy.deepcopy(sample.raw)
        question["id"] = str(question.get("id") or sample.sample_id)
        question["question"] = str(question.get("question") or sample.question)
        question["answer"] = str(question.get("answer") or sample.answer)

        agent_args = self._build_agent_args(
            output=os.path.join(case_dir, "hle_agent.jsonl")
        )
        if outer_round is not None:
            agent_args.outer_round = int(outer_round)
        if max_steps is not None:
            agent_args.max_steps = max(1, int(max_steps))
        resume_paths = self.config.get("hle_outer_resume_paths") or {}
        resume_path = outer_resume_path or resume_paths.get(sample.idx)
        if resume_path:
            with open(resume_path, "r", encoding="utf-8") as handle:
                prior = json.load(handle)
            prior_payload = prior.get("hle") or prior.get(_LEGACY_RESULT_KEY) or {}
            raw_prediction = (prior_payload if isinstance(prior_payload, dict) else {}).get("raw_prediction") or {}
            prior_messages = raw_prediction.get("messages") or prior.get("trajectory") or []
            confidence = prior.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                confidence = 0
            agent_args.outer_resume = {
                "source_result_path": resume_path,
                "source_outer": max(1, int(agent_args.outer_round) - 1),
                "draft": {
                    "answer": str(prior.get("predicted") or ""),
                    "confidence": float(confidence),
                    "evidences": prior.get("evidence") or [],
                },
                "messages": prior_messages,
            }
        result = await self.agent_module.attempt_question(question, agent_args)
        if result is None:
            raise RuntimeError("HLE-Harness returned no result")

        prediction = {
            "id": result["id"],
            "accepted_finish": result.get('accepted_finish'),
            "confidence_review": result.get('confidence_review', {}),
            "outer_resume": result.get('outer_resume', {}),
            "model": agent_args.model,
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
        judge_question = {
            "id": question["id"],
            "question": question["question"],
            "_answer": question["answer"],
        }
        async with self.judge_semaphore:
            _, judged_prediction = await self.judge_module.add_judge_response(
                client=self.judge_client,
                judge_model=self.judge_model,
                question=judge_question,
                predictions={question["id"]: prediction},
                max_tokens=int(self.config.get("hle_judge_max_tokens", 8192)),
                attempts=int(self.config.get("hle_judge_attempts", 3)),
                use_local_endpoint=self.judge_module._is_local_endpoint(
                    self.config.get("judge_base_url")
                    or self.config.get("summary_base_url")
                    or self.config.get("sdk_base_url")
                ),
                truncate_response_token_threshold=int(
                    self.config.get("hle_judge_truncate_response_token_threshold", 100000)
                ),
                truncate_response_chars=int(
                    self.config.get("hle_judge_truncate_response_chars", 100000)
                ),
            )
        if judged_prediction is None:
            raise RuntimeError("HLE judge returned no result")

        evidence, confidence = _extract_finish_fields(self.agent_module, result)
        judge_response = judged_prediction.get("judge_response") or {}
        full_credit = str(judge_response.get("correct") or "").lower() == "yes"
        return {
            "predicted": result["content"],
            "reasoning": result["reasoning"],
            "evidence": evidence,
            "confidence": confidence,
            "score": 1.0 if full_credit else 0.0,
            "full_credit": full_credit,
            "judge_response": judge_response,
            "messages": result["messages"],
            "hle_trajectory": result["trajectory"],
            "usage": result["usage"],
            "model_call_usage": result["model_call_usage"],
            "model_usage_summary": result["model_usage_summary"],
            "context_management_steps": result["context_management_steps"],
            "steps": result["steps"],
            "raw_prediction": prediction,
            "confidence_review": result.get('confidence_review', {}),
            "outer_resume": result.get('outer_resume', {}),
        }

    async def close(self) -> None:
        for client in (self.agent_client, self.judge_client):
            close = getattr(client, "close", None)
            if close is None:
                continue
            result = close()
            if hasattr(result, "__await__"):
                await result
