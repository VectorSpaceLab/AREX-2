import os
import re
import json
import copy
import math
import argparse
import asyncio
import numpy as np
from glob import glob
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel
from openai import AsyncOpenAI
from tqdm import tqdm


JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
"""


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int


JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "hle_judge_response",
        "schema": {
            "type": "object",
            "properties": {
                "extracted_final_answer": {"type": "string"},
                "reasoning": {"type": "string"},
                "correct": {"type": "string", "enum": ["yes", "no"]},
                "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["extracted_final_answer", "reasoning", "correct", "confidence"],
        },
    },
}


ANSWER_KEYS = ["answer", "correct_answer", "gold_answer"]
DEFAULT_RESPONSE_TOKEN_THRESHOLD = 100000
DEFAULT_RESPONSE_KEEP_CHARS = 100000


def build_client(args) -> AsyncOpenAI:
    return AsyncOpenAI(
        timeout=args.timeout,
        max_retries=args.max_retries,
        base_url=args.base_url,
        api_key=args.api_key,
    )


def load_questions_from_json(json_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"题目文件不存在: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{json_path} 不是 JSON list 格式")

    questions: List[Dict[str, Any]] = []

    for idx, item in enumerate(data, 1):
        if not isinstance(item, dict):
            print(f"[Warning] 第 {idx} 条不是 dict，跳过")
            continue

        qid = item.get("id")
        qtext = item.get("question")

        if qid is None or qtext is None:
            print(f"[Warning] 第 {idx} 条缺少 id 或 question，跳过")
            continue

        answer = None
        for k in ANSWER_KEYS:
            if k in item and item[k] is not None:
                answer = item[k]
                break

        if answer is None:
            print(f"[Warning] 第 {idx} 条缺少标准答案字段 {ANSWER_KEYS}，跳过")
            continue

        normalized = dict(item)
        normalized["id"] = str(qid)
        normalized["question"] = str(qtext)
        normalized["_answer"] = str(answer)

        questions.append(normalized)

    return questions


def load_json_dict(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"文件不是 JSON dict 格式: {path}")

    return {str(k): v for k, v in data.items()}


def natural_part_key(path: str) -> int:
    """
    保证 part0, part1, ..., part11 按数字顺序排序。
    避免普通字符串排序出现 part10 排在 part2 前面。
    """
    name = os.path.basename(path)
    m = re.search(r"_part(\d+)\.json$", name)
    if m:
        return int(m.group(1))

    return 10**9


def load_prediction_files(predictions_pattern: str) -> Dict[str, Any]:
    """
    支持三种输入：

    1. 单个 JSON 文件：
       --predictions xxx.json

    2. glob 模式：
       --predictions "xxx_part*.json"

    3. 逗号分隔多个文件：
       --predictions "part0.json,part1.json,part2.json"
    """
    prediction_files: List[str] = []

    for x in predictions_pattern.split(","):
        x = x.strip()
        if not x:
            continue

        # glob 模式
        if any(ch in x for ch in ["*", "?", "["]):
            matched = glob(x)
            prediction_files.extend(matched)
        else:
            prediction_files.append(x)

    prediction_files = sorted(set(prediction_files), key=natural_part_key)

    if not prediction_files:
        raise FileNotFoundError(f"没有找到任何 prediction 文件: {predictions_pattern}")

    merged_predictions: Dict[str, Any] = {}
    duplicate_ids: List[str] = []

    print("Prediction files to merge:")
    for path in prediction_files:
        print(" ", path)

    print()

    for path in prediction_files:
        if not os.path.exists(path):
            print(f"[Warning] prediction 文件不存在，跳过: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"prediction 文件不是 JSON dict 格式: {path}")

        data = {str(k): v for k, v in data.items()}

        for qid, pred in data.items():
            if qid in merged_predictions:
                duplicate_ids.append(qid)

            # 如果重复，后面的 part 会覆盖前面的 part
            merged_predictions[qid] = pred

        print(f"Loaded prediction part: {path} | n={len(data)}")

    print()
    print(f"Total merged predictions: {len(merged_predictions)}")

    if duplicate_ids:
        print(f"[Warning] 合并时发现重复 id 数量: {len(duplicate_ids)}")
        print("重复 id 示例:", duplicate_ids[:20])

    print()

    return merged_predictions


def build_output_path(args) -> str:
    if args.output:
        return args.output

    # 如果 predictions 是 glob，不能直接用 os.path.splitext 生成输出名
    if any(ch in args.predictions for ch in ["*", "?", "[", ","]):
        return "merged_predictions_judged.json"

    root, ext = os.path.splitext(args.predictions)
    return f"{root}_judged.json" if ext else f"{args.predictions}_judged.json"


def save_json_dict(path: str, data: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _is_local_endpoint(base_url: Optional[str]) -> bool:
    if not base_url:
        return False
    from urllib.parse import urlparse
    host = urlparse(base_url).hostname or ""
    return "." in host


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.split("</think>")[-1]
    first = text.find("{")
    if first >= 0:
        text = text[first:]
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None


def _is_context_length_error(error: str) -> bool:
    lowered = error.lower()
    return "longer than the model's context length" in lowered or "context length" in lowered


def prediction_token_count(prediction: Dict[str, Any]) -> tuple[Optional[int], Optional[str]]:
    usage = prediction.get("usage")
    if not isinstance(usage, dict):
        return None, None
    for key in ("completion_tokens", "output_tokens", "response_tokens", "total_tokens"):
        value = usage.get(key)
        if value is None:
            continue
        try:
            return int(value), f"usage.{key}"
        except (TypeError, ValueError):
            continue
    return None, None


def response_for_judge(
    response_text: str,
    prediction: Dict[str, Any],
    truncate_response_token_threshold: int,
    truncate_response_chars: int,
) -> tuple[str, Dict[str, Any]]:
    token_count, token_field = prediction_token_count(prediction)
    original_chars = len(response_text)
    keep_chars = max(1, int(truncate_response_chars))
    threshold = int(truncate_response_token_threshold)
    should_truncate = (
        threshold > 0
        and token_count is not None
        and token_count > threshold
        and original_chars > keep_chars
    )
    if not should_truncate:
        return response_text, {
            "response_truncated_for_judge": False,
            "response_original_tokens": token_count,
            "response_token_field": token_field,
            "response_original_chars": original_chars,
            "response_judge_chars": original_chars,
        }
    return response_text[-keep_chars:], {
        "response_truncated_for_judge": True,
        "response_truncate_reason": (
            f"chars>{keep_chars}; {token_field}={token_count}; token_threshold={threshold}"
            if token_field
            else f"chars>{keep_chars}; token_threshold={threshold}"
        ),
        "response_original_tokens": token_count,
        "response_token_field": token_field,
        "response_original_chars": original_chars,
        "response_judge_chars": keep_chars,
    }


async def extract_answer(
    client: AsyncOpenAI,
    judge_model: str,
    question: str,
    correct_answer: str,
    response_text: str,
    max_tokens: int,
    attempts: int,
    use_local_endpoint: bool = False,
):
    prompt = JUDGE_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        response=response_text,
    )
    last_error = None
    raw_outputs = []
    for attempt in range(attempts):
        raw_output = {"attempt": attempt + 1, "mode": "beta_parse"}
        try:
            resp = await client.beta.chat.completions.parse(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                response_format=ExtractedAnswer,
            )
            raw = resp.choices[0].message.content or ""
            raw_output["content"] = raw
            content = resp.choices[0].message.parsed
            if content is None:
                raise ValueError(f"non-parseable judge response: {raw[:500]}")

            return {
                "correct_answer": correct_answer,
                "model_answer": content.extracted_final_answer,
                "reasoning": content.reasoning,
                "correct": content.correct,
                "confidence": content.confidence,
                "judge_type": "external_hle_judge",
                "raw_judge_outputs": raw_outputs + [raw_output],
            }, None

        except Exception as e:
            last_error = str(e)
            raw_output["error"] = last_error
            raw_outputs.append(raw_output)
            print(f"[Error] Judge failed: {e}")

        fallback_output = {"attempt": attempt + 1, "mode": "json_schema_fallback"}
        try:
            resp = await client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                response_format=JUDGE_RESPONSE_FORMAT,
            )
            raw = resp.choices[0].message.content or ""
            fallback_output["content"] = raw
            parsed = parse_json_object(raw)
            if not parsed:
                raise ValueError(f"non-json judge response: {raw[:500]}")
            correct = str(parsed.get("correct", "")).strip().lower()
            if correct not in ("yes", "no"):
                raise ValueError(f"invalid correct field: {correct!r}")
            confidence = parsed.get("confidence", 100)
            return {
                "correct_answer": correct_answer,
                "model_answer": str(parsed.get("extracted_final_answer", "None")),
                "reasoning": str(parsed.get("reasoning", ""))[:2000],
                "correct": correct,
                "confidence": max(0, min(100, int(confidence or 100))),
                "judge_type": "external_hle_judge",
                "raw_judge_outputs": raw_outputs + [fallback_output],
            }, None

        except Exception as e:
            last_error = str(e)
            fallback_output["error"] = last_error
            raw_outputs.append(fallback_output)
            print(f"[Error] Judge fallback failed: {e}")
            await asyncio.sleep(min(2 * (attempt + 1), 10))

    return {
        "correct_answer": correct_answer,
        "model_answer": "None",
        "reasoning": f"JUDGE_ERROR: {last_error}",
        "correct": "no",
        "confidence": 0,
        "judge_type": "external_hle_judge",
        "judge_error": last_error,
        "raw_judge_outputs": raw_outputs,
    }, last_error


async def add_judge_response(
    client: AsyncOpenAI,
    judge_model: str,
    question: Dict[str, Any],
    predictions: Dict[str, Any],
    max_tokens: int,
    attempts: int,
    use_local_endpoint: bool = False,
    truncate_response_token_threshold: int = DEFAULT_RESPONSE_TOKEN_THRESHOLD,
    truncate_response_chars: int = DEFAULT_RESPONSE_KEEP_CHARS,
):
    unique_id = str(question["id"])

    if unique_id not in predictions:
        return None, None

    prediction = copy.deepcopy(predictions[unique_id])

    if "judge_response" in prediction:
        return unique_id, prediction

    response_text = prediction.get("response")
    if not response_text:
        print(f"[Warning] id={unique_id} prediction 中没有 response，跳过")
        return None, None

    judge_response_text, truncate_meta = response_for_judge(
        response_text=str(response_text),
        prediction=prediction,
        truncate_response_token_threshold=truncate_response_token_threshold,
        truncate_response_chars=truncate_response_chars,
    )

    content, error = await extract_answer(
        client=client,
        judge_model=judge_model,
        question=question["question"],
        correct_answer=question["_answer"],
        response_text=judge_response_text,
        max_tokens=max_tokens,
        attempts=attempts,
        use_local_endpoint=use_local_endpoint,
    )

    if content is None and error and _is_context_length_error(error) and len(judge_response_text) > 100000:
        fallback_chars = min(100000, len(str(response_text)))
        judge_response_text = str(response_text)[-fallback_chars:]
        fallback_meta = dict(truncate_meta)
        fallback_meta.update({
            "response_truncated_for_judge": True,
            "response_truncate_reason": (
                f"context_length_retry; chars>{fallback_chars}; "
                f"original_reason={truncate_meta.get('response_truncate_reason')}"
            ),
            "response_judge_chars": fallback_chars,
        })
        content, error = await extract_answer(
            client=client,
            judge_model=judge_model,
            question=question["question"],
            correct_answer=question["_answer"],
            response_text=judge_response_text,
            max_tokens=max_tokens,
            attempts=attempts,
            use_local_endpoint=use_local_endpoint,
        )
        truncate_meta = fallback_meta

    if content is None:
        prediction["judge_response"] = {
            "correct_answer": question["_answer"],
            "model_answer": None,
            "reasoning": f"Judge failed: {error}",
            "correct": "no",
            "confidence": 0,
            "judge_error": error,
            **truncate_meta,
        }
        return unique_id, prediction

    content.update(truncate_meta)
    prediction["judge_response"] = content

    return unique_id, prediction


async def judge_all_responses(
    client: AsyncOpenAI,
    judge_model: str,
    questions: List[Dict[str, Any]],
    predictions: Dict[str, Any],
    judged_predictions: Dict[str, Any],
    output_filepath: str,
    num_workers: int,
    max_tokens: int,
    attempts: int,
    use_local_endpoint: bool = False,
    truncate_response_token_threshold: int = DEFAULT_RESPONSE_TOKEN_THRESHOLD,
    truncate_response_chars: int = DEFAULT_RESPONSE_KEEP_CHARS,
):
    sem = asyncio.Semaphore(num_workers)
    write_lock = asyncio.Lock()

    async def bound_func(question):
        async with sem:
            return await add_judge_response(
                client=client,
                judge_model=judge_model,
                question=question,
                predictions=predictions,
                max_tokens=max_tokens,
                attempts=attempts,
                use_local_endpoint=use_local_endpoint,
                truncate_response_token_threshold=truncate_response_token_threshold,
                truncate_response_chars=truncate_response_chars,
            )

    tasks = [asyncio.create_task(bound_func(q)) for q in questions]
    saved = 0

    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        unique_id, pred = await task
        if unique_id is None or pred is None:
            continue

        judged_predictions[unique_id] = pred
        saved += 1

        async with write_lock:
            await asyncio.to_thread(save_json_dict, output_filepath, judged_predictions)

        jr = pred.get("judge_response", {})
        print(
            f"[Saved] id={unique_id} correct={jr.get('correct')} "
            f"saved_this_run={saved} total={len(judged_predictions)} -> {output_filepath}",
            flush=True,
        )

    return judged_predictions


def calib_err(confidence, correct, p="2", beta=100):
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]

    if len(confidence) == 0:
        return 0.0

    num_bins = max(1, len(confidence) // beta)

    bins = [
        [i * beta, min((i + 1) * beta, len(confidence))]
        for i in range(num_bins)
    ]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)

    for i in range(len(bins)):
        start, end = bins[i]

        bin_confidence = confidence[start:end]
        bin_correct = correct[start:end]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(
                np.nanmean(bin_confidence) - np.nanmean(bin_correct)
            )

            if p == "2":
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p in ("infty", "infinity", "max"):
                cerr = np.maximum(cerr, difference)
            else:
                raise ValueError("p must be '1', '2', or 'infty'")

    if p == "2":
        cerr = np.sqrt(cerr)

    return cerr


def dump_metrics(judged_predictions: Dict[str, Any], eval_ids: Set[str]):
    correct = []
    confidence = []

    for qid in eval_ids:
        if qid not in judged_predictions:
            continue

        v = judged_predictions[qid]

        if "judge_response" not in v:
            continue

        jr = v["judge_response"]

        correct.append("yes" in jr["correct"])
        confidence.append(jr["confidence"])

    n = len(eval_ids)
    judged_n = len(correct)

    if n == 0:
        print("No evaluation samples.")
        return

    if judged_n == 0:
        print("No judged samples available.")
        return

    correct = np.array(correct)
    confidence = np.array(confidence) / 100

    accuracy = round(100 * float(np.sum(correct)) / judged_n, 2)

    confidence_half_width = round(
        1.96 * math.sqrt(accuracy * (100 - accuracy) / judged_n),
        2,
    )

    calibration_error = 100 * round(
        calib_err(confidence, correct, p="2", beta=100),
        4,
    )

    print("*** Metrics ***")
    print(f"Evaluated target questions: {n}")
    print(f"Available judged predictions: {judged_n}")
    print("correct:", np.sum(correct))
    print(f"Accuracy: {accuracy}% +/- {confidence_half_width}% | judged_n = {judged_n}")
    print(f"Calibration Error: {calibration_error}")


def main(args):
    if args.num_workers < 1:
        raise ValueError("num_workers 必须 >= 1")

    use_local_endpoint = _is_local_endpoint(args.base_url)
    print(f"use_local_endpoint={use_local_endpoint} (base_url={args.base_url})")

    client = build_client(args)

    questions = load_questions_from_json(args.questions_json)
    print(f"Loaded questions from {args.questions_json}: {len(questions)}")

    if args.max_samples is not None:
        questions = questions[:args.max_samples]
        print(f"After max_samples limit: {len(questions)}")

    eval_ids = {str(q["id"]) for q in questions}

    # 核心修改：支持多个 prediction part 文件合并
    predictions = load_prediction_files(args.predictions)
    print(f"Loaded merged predictions from {args.predictions}: {len(predictions)}")

    output_filepath = build_output_path(args)
    os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)

    judged_predictions = load_json_dict(output_filepath)
    print(f"Loaded existing judged file: {output_filepath} | total={len(judged_predictions)}")

    pending_questions = [
        q for q in questions
        if str(q["id"]) in predictions and str(q["id"]) not in judged_predictions
    ]

    missing_pred_ids = [
        str(q["id"]) for q in questions
        if str(q["id"]) not in predictions
    ]

    if missing_pred_ids:
        print(f"[Warning] 有 {len(missing_pred_ids)} 道题在 predictions 中找不到，judge 时会跳过")
        print("缺失 id 示例:", missing_pred_ids[:20])

    print(f"Need to judge now: {len(pending_questions)}")

    if pending_questions:
        asyncio.run(
            judge_all_responses(
                client=client,
                judge_model=args.judge,
                questions=pending_questions,
                predictions=predictions,
                judged_predictions=judged_predictions,
                output_filepath=output_filepath,
                num_workers=args.num_workers,
                max_tokens=args.max_tokens,
                attempts=args.attempts,
                use_local_endpoint=use_local_endpoint,
                truncate_response_token_threshold=args.truncate_response_token_threshold,
                truncate_response_chars=args.truncate_response_chars,
            )
        )
    else:
        save_json_dict(output_filepath, judged_predictions)

    print(f"Saved -> {output_filepath} | total={len(judged_predictions)}")

    dump_metrics(judged_predictions, eval_ids=eval_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--questions_json",
        type=str,
        default="/share/project/wanghui/hle_text_only_questions.json",
        help="本地题目 JSON 文件，格式为 JSON list，且需包含答案字段",
    )

    parser.add_argument(
        "--predictions",
        type=str,
        default="hle_Qwen3.5-122B-A10B.json",
        help=(
            "inference 输出的 predictions JSON。"
            "支持单文件、逗号分隔多文件、glob 模式，例如 *_part*.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="judge_hle_Qwen3.5-122B-A10B.json",
        help="judge 输出路径",
    )

    parser.add_argument(
        "--judge",
        type=str,
        default="ep-20260317021208-5zzv3",
        help="Judge model",
    )

    parser.add_argument(
        "--base_url",
        type=str,
        default=os.getenv("JUDGE_BASE_URL", "https://kspmas.ksyun.com/v1/responses"),
        help="Judge API base_url，也可以通过环境变量 JUDGE_BASE_URL 设置",
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("JUDGE_API_KEY", "xxxx"),
        help="Judge API key，建议通过环境变量 JUDGE_API_KEY 设置",
    )

    parser.add_argument("--timeout", type=float, default=20000.0)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--truncate_response_token_threshold",
        type=int,
        default=DEFAULT_RESPONSE_TOKEN_THRESHOLD,
        help="Kept for compatibility; responses are truncated by trailing character count.",
    )
    parser.add_argument(
        "--truncate_response_chars",
        type=int,
        default=DEFAULT_RESPONSE_KEEP_CHARS,
        help="Number of trailing response characters to keep for judge when the response is longer.",
    )

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError(
            "缺少 api_key。请通过 --api_key 传入，或设置环境变量 JUDGE_API_KEY。"
        )

    main(args)
