import argparse
import asyncio
import json
import os
import random
import time
import traceback

from Agent_no_subagent import MainAgent
from prompts_no_subagent import EXTRACTOR_PROMPT, GRADER_TEMPLATE
from local_search import LocalSearch
from utils import em_score, parse_judge_response
import base64
import hashlib
import re
import csv
from transformers import AutoTokenizer
import datetime
import httpx
from openai_retry_client import create_async_openai_with_retry
from hle_vendor.visit_fallback import visit_fallback_default_enabled

def _patch_browsecomp_typos(correct_answer: str, predicted_answer: str):
    # Keep consistent with FoldAgent/envs/local_search.py:judge()
    correct_answer = "ttellomS saiboT"[::-1] if "tellomS saiboT"[::-1] in correct_answer else correct_answer
    correct_answer = "yayhdapottahC najnarawsiB"[::-1] if "yayhdapattahC najnarawsiB"[::-1] in correct_answer else correct_answer
    predicted_answer = "yrtnuoC a fo htaP ehT :sedirelC sokfalG"[::-1] if "yrtnuoC a fo htaP ehT :sedirelC socfalG"[::-1] in predicted_answer else predicted_answer
    return correct_answer, predicted_answer

async def llm_judge(question: str, correct_answer: str, predicted_answer: str, llm_client, model) -> int:
    """Call LLM judge to check if the answer is correct."""
    # Hardcoded grader endpoint (match evaluate.py)
    # url = "http://172.24.160.223:38610/v1/chat/completions"
    # api_key = "inspectai"
    # model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    prompt = GRADER_TEMPLATE.format(question=question, response=predicted_answer, correct_answer=correct_answer)
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    for _ in range(3):
        try:
            response = await llm_client.chat.completions.create(**payload)
            content = response.choices[0].message.content.split("</think>")[-1]
            parsed = parse_judge_response(content)
            if parsed.get("parse_error"):
                continue
            return int(bool(parsed.get("correct")))
        except Exception:
            continue
    return 0

def should_skip_existing_result(result: dict, skip_existing_mode: str) -> bool:
    if skip_existing_mode == "none":
        return False
    if skip_existing_mode == "all":
        return True
    if skip_existing_mode == "correct":
        return result.get("score") == 1
    raise ValueError(f"Unsupported skip_existing_mode: {skip_existing_mode}")

def extract_row_index(path_name: str):
    match = re.search(r"row(\d+)$", path_name)
    return int(match.group(1)) if match else None

def remove_empty_dir_tree(path: str) -> bool:
    """Remove path if it contains no files, including empty nested dirs."""
    if not os.path.isdir(path):
        return False

    for _, _, files in os.walk(path):
        if files:
            return False

    for root, dirs, _ in os.walk(path, topdown=False):
        for dirname in dirs:
            child = os.path.join(root, dirname)
            if os.path.isdir(child):
                try:
                    os.rmdir(child)
                except OSError:
                    return False

    try:
        os.rmdir(path)
        return True
    except OSError:
        return False

def cleanup_empty_case_dirs(save_path: str, idx: int) -> int:
    if not os.path.isdir(save_path):
        return 0

    removed_count = 0
    for name in os.listdir(save_path):
        case_dir = os.path.join(save_path, name)
        if not os.path.isdir(case_dir):
            continue
        if extract_row_index(name) != idx:
            continue
        if remove_empty_dir_tree(case_dir):
            removed_count += 1
            print(f"[CLEANUP] removed empty previous case dir: {case_dir}")
    return removed_count

def iter_result_temp_json_paths(result_dirs):
    for result_dir in result_dirs:
        if not os.path.isdir(result_dir):
            continue
        for root, _, files in os.walk(result_dir):
            if "failure_attempt" in root.split(os.sep):
                continue
            if "temp.json" in files:
                yield os.path.join(root, "temp.json")

def load_dataset_rows(data_path: str):
    _, ext = os.path.splitext(data_path.lower())
    if ext == ".jsonl":
        with open(data_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(data_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def infer_leak_filter(config: dict) -> str:
    configured = str(config.get("leak_filter") or "").strip().lower()
    if configured:
        return configured
    data_path = str(config.get("data_path") or "").lower()
    if "gaia" in data_path:
        return "gaia"
    return ""

def parse_target_indices(target_indices_str: str):
    if not target_indices_str:
        return None
    return {int(x.strip()) for x in target_indices_str.split(",") if x.strip()}

def collect_existing_indices(result_dirs, skip_existing_mode: str):
    idx_set = set()
    if skip_existing_mode == "none":
        return idx_set

    for temp_json_path in iter_result_temp_json_paths(result_dirs):
        with open(temp_json_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        if should_skip_existing_result(result, skip_existing_mode):
            row_idx = extract_row_index(os.path.basename(os.path.dirname(temp_json_path)))
            if row_idx is not None:
                idx_set.add(row_idx)
    return idx_set

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
































def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]

def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()

async def worker(config, row_list, llm_client, summary_client, serper_client, jina_client, _tokenizer, sum_tokenizer, judge_client, judge_model, semaphore: asyncio.Semaphore):
    # Create summarizer and search client (same pattern as workflow.py)
    
    model = config.get("model")
    summary_model=config.get("summary_model")
    if summary_model == "":
        summary_model = model

    for row, idx in row_list:
       
        if config.get("no_decrypt", False):
            problem = row.get("question", "")
            answer = row.get("answer", "")
        else:
            problem = decrypt(row.get("problem", ""), row.get("canary", ""))
            answer = decrypt(row.get("answer", ""), row.get("canary", ""))

        async with semaphore:
            print("problem: ", problem, " answer: ", answer, " idx: ", idx)
            if config.get("skip_existing_mode") == "all":
                cleanup_empty_case_dirs(config["save_path"], idx)
            now = datetime.datetime.now()
            time_step = now.strftime("%Y-%m-%d_%H-%M-%S")
            case_dir = os.path.join(config["save_path"], time_step + '_' + 'row' + str(idx))
            os.makedirs(case_dir, exist_ok=True)

            max_attempts = max(1, int(config.get("max_attempts", 1)))
            case_timeout_seconds = float(config.get("case_timeout_seconds") or 0)
            case_start_time = time.time()
            case_call_stats = empty_case_call_stats()
            for attempt_idx in range(max_attempts):
                attempt_start_time = time.time()
                page_id_to_url = {}
                page_url_to_id = {}

                env = LocalSearch(
                    summary_client=summary_client,  # type: ignore[call-arg]
                    search_client=serper_client,
                    jina_client=jina_client,
                    tokenizer=sum_tokenizer,
                    page_id_to_url=page_id_to_url,
                    page_url_to_id=page_url_to_id,
                    model_name=summary_model,
                    extractor_prompt=EXTRACTOR_PROMPT,
                    case_output_dir=case_dir,
                    leak_filter=config.get("leak_filter") or None,
                    enable_visit_fallback=bool(config.get("enable_visit_fallback", True)),
                )

                agent_config = config.copy()
                agent_config["case_output_dir"] = case_dir
                main_agent = MainAgent(agent_config, env, llm_client, _tokenizer)

                score = 0
                confidence_val = None
                try:
                    case_coro = main_agent.run({"problem": problem, "idx": idx})
                    if case_timeout_seconds > 0:
                        predicted_answer, evidence, confidence = await asyncio.wait_for(
                            case_coro,
                            timeout=case_timeout_seconds,
                        )
                    else:
                        predicted_answer, evidence, confidence = await case_coro
                    conf_m = re.search(r"(\d+)", str(confidence))
                    if conf_m:
                        confidence_val = int(conf_m.group(1))

                    origin_traj = main_agent.origin_messages
                    traj = main_agent.messages
                    origin_sub_traj = ""
                    sub_traj = ""
                    traj_path = os.path.join(case_dir, "temp.json")
                    # async with save_lock:
                        # all_scores.append(score)
                    answer, predicted_answer = _patch_browsecomp_typos(answer, predicted_answer)
                    if em_score(answer, predicted_answer):
                        score = 1
                    elif len(predicted_answer.strip()) == 0:
                        score = 0
                    else:
                        score = await llm_judge(problem, answer, predicted_answer, judge_client, judge_model)
                    print("score: ", score, " attempt: ", attempt_idx)
                    timing_stats = main_agent.get_timing_stats() if hasattr(main_agent, "get_timing_stats") else {}

                    with open(traj_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "question": problem,
                            "answer":answer,
                            "predicted": predicted_answer,
                            "score": score,
                            "trajectory": traj,
                            "sub_traj": sub_traj,
                            "leak_filter": config.get("leak_filter") or None,
                            "timing_summary": timing_stats.get("summary", {}),
                        }, f, ensure_ascii=False, indent=2)
                    # Collect token usage stats
                    token_usage = main_agent.get_total_usage()
                    attempt_call_stats = main_agent.get_call_stats() if hasattr(main_agent, "get_call_stats") else {}
                    call_stats = merge_attempt_call_stats(case_call_stats, attempt_call_stats)
                    attempt_duration_seconds = time.time() - attempt_start_time
                    case_elapsed_seconds = time.time() - case_start_time
                    temp_origin_data = {
                        "origin_traj": origin_traj,
                        "origin_sub_traj": origin_sub_traj,
                        "token_usage": token_usage,
                        "call_stats": call_stats,
                        "attempt_call_stats": attempt_call_stats,
                        "timing_stats": timing_stats,
                        "case_duration_seconds": case_elapsed_seconds,
                        "attempt_duration_seconds": attempt_duration_seconds,
                        "attempt_index": attempt_idx,
                    }
                    temp_origin_data.update(call_stats)
                    with open(os.path.join(case_dir, "temp_origin.json"), "w", encoding="utf-8") as f:
                        json.dump(temp_origin_data, f, ensure_ascii=False, indent=2)
                    # Save training segments for refine_summary strategy when answer is correct
                    if config["context_limit_strategy"] == "refine_summary" and score == 1:
                        train_segments = getattr(main_agent, "refine_summary_train_segments", [])
                        for seg_idx, segment in enumerate(train_segments):
                            with open(os.path.join(case_dir, f"train_{seg_idx}.json"), "w", encoding="utf-8") as f:
                                json.dump(segment, f, ensure_ascii=False, indent=2)
                        update_segments = getattr(main_agent, "refine_summary_update_segments", [])
                        for seg_idx, segment in enumerate(update_segments):
                            with open(os.path.join(case_dir, f"update_{seg_idx}.json"), "w", encoding="utf-8") as f:
                                json.dump(segment, f, ensure_ascii=False, indent=2)
                    print(f"[EVAL] score={score} | confidence={confidence_val} | label={answer!r} | pred={predicted_answer!r}", "idx: ", idx, " attempt: ", attempt_idx)
                except asyncio.TimeoutError as e:
                    error_trace = traceback.format_exc()
                    timeout_msg = f"case exceeded timeout_seconds={case_timeout_seconds}"
                    print(f"[CASE TIMEOUT] idx={idx} attempt={attempt_idx}: {timeout_msg}")

                    traj = getattr(main_agent, "messages", [])
                    origin_traj = getattr(main_agent, "origin_messages", [])
                    sub_traj = ""
                    origin_sub_traj = ""

                    with open(os.path.join(case_dir, "fatal_error.txt"), "w", encoding="utf-8") as f:
                        f.write(f"idx: {idx}\n")
                        f.write(f"attempt: {attempt_idx}\n")
                        f.write(f"question: {problem}\n")
                        f.write(f"error: CaseTimeout: {timeout_msg}\n\n")
                        f.write(error_trace)

                    with open(os.path.join(case_dir, "temp.json"), "w", encoding="utf-8") as f:
                        timing_stats = main_agent.get_timing_stats() if hasattr(main_agent, "get_timing_stats") else {}
                        json.dump({
                            "question": problem,
                            "answer": answer,
                            "predicted": "",
                            "score": 0,
                            "trajectory": traj,
                            "sub_traj": sub_traj,
                            "leak_filter": config.get("leak_filter") or None,
                            "timing_summary": timing_stats.get("summary", {}),
                            "error": {
                                "type": "CaseTimeout",
                                "message": timeout_msg,
                                "timeout_seconds": case_timeout_seconds,
                                "traceback": error_trace,
                            },
                        }, f, ensure_ascii=False, indent=2)

                    token_usage = main_agent.get_total_usage() if hasattr(main_agent, "get_total_usage") else {}
                    attempt_call_stats = main_agent.get_call_stats() if hasattr(main_agent, "get_call_stats") else {}
                    call_stats = merge_attempt_call_stats(case_call_stats, attempt_call_stats)
                    attempt_duration_seconds = time.time() - attempt_start_time
                    case_elapsed_seconds = time.time() - case_start_time
                    err_origin = {
                        "origin_traj": origin_traj,
                        "origin_sub_traj": origin_sub_traj,
                        "token_usage": token_usage,
                        "call_stats": call_stats,
                        "attempt_call_stats": attempt_call_stats,
                        "timing_stats": timing_stats,
                        "case_duration_seconds": case_elapsed_seconds,
                        "attempt_duration_seconds": attempt_duration_seconds,
                        "attempt_index": attempt_idx,
                    }
                    err_origin.update(call_stats)
                    with open(os.path.join(case_dir, "temp_origin.json"), "w", encoding="utf-8") as f:
                        json.dump(err_origin, f, ensure_ascii=False, indent=2)

                except Exception as e:
                    error_trace = traceback.format_exc()
                    print(f"[CASE ERROR] idx={idx} attempt={attempt_idx}: {type(e).__name__}: {e}")

                    traj = getattr(main_agent, "messages", [])
                    origin_traj = getattr(main_agent, "origin_messages", [])
                    sub_traj = ""
                    origin_sub_traj = ""

                    with open(os.path.join(case_dir, "fatal_error.txt"), "w", encoding="utf-8") as f:
                        f.write(f"idx: {idx}\n")
                        f.write(f"attempt: {attempt_idx}\n")
                        f.write(f"question: {problem}\n")
                        f.write(f"error: {type(e).__name__}: {e}\n\n")
                        f.write(error_trace)

                    with open(os.path.join(case_dir, "temp.json"), "w", encoding="utf-8") as f:
                        timing_stats = main_agent.get_timing_stats() if hasattr(main_agent, "get_timing_stats") else {}
                        json.dump({
                            "question": problem,
                            "answer": answer,
                            "predicted": "",
                            "score": 0,
                            "trajectory": traj,
                            "sub_traj": sub_traj,
                            "leak_filter": config.get("leak_filter") or None,
                            "timing_summary": timing_stats.get("summary", {}),
                            "error": {
                                "type": type(e).__name__,
                                "message": str(e),
                                "traceback": error_trace,
                            },
                        }, f, ensure_ascii=False, indent=2)

                    token_usage = main_agent.get_total_usage() if hasattr(main_agent, "get_total_usage") else {}
                    attempt_call_stats = main_agent.get_call_stats() if hasattr(main_agent, "get_call_stats") else {}
                    call_stats = merge_attempt_call_stats(case_call_stats, attempt_call_stats)
                    attempt_duration_seconds = time.time() - attempt_start_time
                    case_elapsed_seconds = time.time() - case_start_time
                    err_origin = {
                        "origin_traj": origin_traj,
                        "origin_sub_traj": origin_sub_traj,
                        "token_usage": token_usage,
                        "call_stats": call_stats,
                        "attempt_call_stats": attempt_call_stats,
                        "timing_stats": timing_stats,
                        "case_duration_seconds": case_elapsed_seconds,
                        "attempt_duration_seconds": attempt_duration_seconds,
                        "attempt_index": attempt_idx,
                    }
                    err_origin.update(call_stats)
                    with open(os.path.join(case_dir, "temp_origin.json"), "w", encoding="utf-8") as f:
                        json.dump(err_origin, f, ensure_ascii=False, indent=2)
                should_retry = score != 1
                if (not should_retry) or attempt_idx + 1 >= max_attempts:
                    break
                print(f"[RETRY] score={score}, retrying idx={idx} attempt={attempt_idx+1}")

                # Move this failed attempt's outputs into failure_attempt/{attempt_idx}/
                failure_dir = os.path.join(case_dir, "failure_attempt", str(attempt_idx))
                os.makedirs(failure_dir, exist_ok=True)
                for fname in ("temp.json", "temp_origin.json", "fatal_error.txt"):
                    src = os.path.join(case_dir, fname)
                    if os.path.exists(src):
                        os.replace(src, os.path.join(failure_dir, fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--sdk_base_url", type=str, default="")
    parser.add_argument("--sdk_api_key", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=240000)
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--summary_base_url", type=str, default="")
    parser.add_argument("--summary_api_key", type=str, default="")
    parser.add_argument("--summary_model", type=str, default="")
    parser.add_argument("--sum_tokenizer_path", type=str, default="")
    parser.add_argument("--judge_base_url", type=str, default="")
    parser.add_argument("--judge_api_key", type=str, default="")
    parser.add_argument("--judge_model", type=str, default="")
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

    parser.add_argument(
        "--context_limit_strategy",
        type=str,
        default="discard_all",
        choices=["discard_all", "refine_summary"],
    )
    parser.add_argument("--discard_all_tokens_limit", type=int, default=240000)
    parser.add_argument("--refine_summary_max_outer_rounds", type=int, default=5,
                        help="Maximum number of outer rounds in refine_summary strategy")
    parser.add_argument("--refine_summary_max_llm_calls", type=int, default=300,
                        help="Maximum number of LLM calls per inner loop in refine_summary strategy")
    parser.add_argument("--refine_summary_trigger_tokens", type=int, default=128000,
                        help="Token threshold to trigger update_context prompt in refine_summary strategy")
    parser.add_argument("--refine_summary_max_updates", type=int, default=5,
                        help="Maximum number of update_context calls per inner loop in refine_summary strategy")
    parser.add_argument("--general_max_attempts", type=int, default=5,
                        help="Maximum retry attempts for LLM API calls")
    parser.add_argument("--max_attempts", type=int, default=1,
                        help="每题在 score!=1 时的最大尝试次数。1=不重试；2=首次失败后再试一次。失败的 temp.json/temp_origin.json/fatal_error.txt 会被搬到 case_dir/failure_attempt/{N}/ (N 从 0 开始)。")
    parser.add_argument("--case-timeout-seconds", "--case_timeout_seconds",
                        dest="case_timeout_seconds", type=float, default=0,
                        help="Per-row timeout in seconds. 0 disables the watchdog.")
    parser.add_argument("--tool_call_regen_max_retries", type=int, default=20,
                        help="Maximum assistant regeneration retries for invalid tool-call outputs before injecting user-visible error feedback")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=9999999999999)
    parser.add_argument("--target_indices", type=str, default="",
                        help="Comma-separated row indices to run. When set, only these indices are processed (still respects skip-existing).")
    parser.add_argument(
        "--target-shard-rank",
        type=int,
        default=0,
        help="Shard rank used to split target indices after target filtering.",
    )
    parser.add_argument(
        "--target-shard-count",
        type=int,
        default=1,
        help="Number of shards used to split target indices after target filtering.",
    )
    parser.add_argument("--concurrency_limit", type=int, default=16)
    parser.add_argument("--leak_filter", type=str, default="",
                        help="Benchmark leak filter to apply to tool outputs. Defaults to gaia when data_path contains GAIA.")
    parser.add_argument("--no-decrypt", action='store_true', help="不解密，直接读取明文question和answer字段")
    parser.add_argument(
        "--skip-existing-mode",
        type=str,
        default="correct",
        choices=["none", "correct", "all"],
        help="已有结果的跳过模式：none=不跳过，correct=仅跳过score为1，all=只要已有temp.json就跳过",
    )
    args = parser.parse_args()
    
    config = vars(args)
    config["leak_filter"] = infer_leak_filter(config)
    print(config)

    os.makedirs(config["save_path"], exist_ok=True)
    skip_existing_mode = config["skip_existing_mode"]
    idx_list = collect_existing_indices([config["save_path"]], skip_existing_mode)
    print('????', idx_list, len(idx_list))
    concurrency_limit = config.get("concurrency_limit", 16)
    semaphore = asyncio.Semaphore(concurrency_limit)


    base_url = config.get("sdk_base_url")
    api_key = config.get("sdk_api_key")
    summary_base_url = config.get("summary_base_url")
    summary_api_key = config.get("summary_api_key") or api_key
    model = config.get("model")
    tokenizer_path = config.get("tokenizer_path")
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    llm_client = create_async_openai_with_retry(
        api_key=api_key,
        base_url=base_url,
        timeout=600,
        general_max_attempts=config.get("general_max_attempts", 5),
    )
    summary_client = create_async_openai_with_retry(
        api_key=summary_api_key,
        base_url=summary_base_url,
        timeout=600,
        general_max_attempts=config.get("general_max_attempts", 5),
    )
    print(f"[SUMMARY] configured summary base url: {summary_base_url}")
    judge_base_url = config.get("judge_base_url") or summary_base_url
    judge_api_key = config.get("judge_api_key") or summary_api_key
    judge_model = config.get("judge_model") or model
    config["judge_model"] = judge_model
    judge_client = create_async_openai_with_retry(
        api_key=judge_api_key,
        base_url=judge_base_url,
        timeout=600,
        general_max_attempts=config.get("general_max_attempts", 5),
    )
    sum_tokenizer_path = config.get("sum_tokenizer_path")
    if sum_tokenizer_path == "":
        sum_tokenizer=_tokenizer
    else:
        # print('???', sum_tokenizer_path)
        sum_tokenizer=AutoTokenizer.from_pretrained(sum_tokenizer_path, trust_remote_code=True)
    async def run():
        reader = load_dataset_rows(config["data_path"])
        random.seed(66)
        indices = list(range(len(reader)))
        random.shuffle(indices)
        selected_reader = indices[config["start_index"]: config["end_index"]]

        target_set = parse_target_indices(config.get("target_indices", ""))
        target_shard_count = config.get("target_shard_count", 1)
        target_shard_rank = config.get("target_shard_rank", 0)
        if target_shard_count > 1:
            if target_shard_rank < 0 or target_shard_rank >= target_shard_count:
                raise ValueError(
                    f"--target-shard-rank must be in [0, {target_shard_count}), got {target_shard_rank}"
                )
        if target_set is not None:
            selected_reader = [i for i in indices if i in target_set]
            if target_shard_count > 1:
                selected_reader = selected_reader[target_shard_rank::target_shard_count]
                print(
                    f"[TARGET] shard {target_shard_rank}/{target_shard_count} "
                    f"selected {len(selected_reader)} target indices"
                )
            row_list = [[(reader[i], i)] for i in selected_reader if i not in idx_list and i in target_set]
        else:
            if target_shard_count > 1:
                selected_reader = selected_reader[target_shard_rank::target_shard_count]
                print(
                    f"[SHARD] shard {target_shard_rank}/{target_shard_count} "
                    f"selected {len(selected_reader)} indices"
                )
            row_list = [[(reader[i], i)] for i in selected_reader if i not in idx_list]

        print('!!!!', len(row_list))
        search_limits = httpx.Limits(max_connections=100, max_keepalive_connections=80)
        jina_limits = httpx.Limits(max_connections=100, max_keepalive_connections=80)
        async with httpx.AsyncClient(timeout=600, follow_redirects=True, limits=search_limits) as serper_client, \
                   httpx.AsyncClient(timeout=600, follow_redirects=True, limits=jina_limits) as jina_client :

            tasks = [
                worker(
                    config,
                    row,
                    llm_client,
                    summary_client,
                    serper_client,
                    jina_client,
                    _tokenizer,
                    sum_tokenizer,
                    judge_client,
                    judge_model,
                    semaphore,
                )
                for row in row_list
            ]
            await asyncio.gather(*tasks)
        
    
    asyncio.run(run())

    
