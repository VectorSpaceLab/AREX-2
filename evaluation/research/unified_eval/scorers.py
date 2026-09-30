import asyncio
import importlib.util
import json
import os
import re
import string
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from utils import em_score, parse_judge_response

from .types import DatasetSpec, EvalSample, ScoreResult, ScorerUnavailable


BROWSECOMP_GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, contains all the essential information from [correct_answer], is equivalent despite minor wording/order differences (such as name order, inclusion or omission of middle names/initials, common honorifics, standard shortenings of first names, inclusion/omission of non-contradictory date parts like year, minor articles like "a"/"the", extra descriptive context, non-essential descriptive prefixes/suffixes such as "Restaurant", "Inc.", "Ltd.", or sports suffixes like "FC", "CF", "SC", inclusion/omission of subtitles in titles, minor spacing/punctuation differences — including presence/absence of quotation marks, interchangeable punctuation such as ":" / "-" / "–", case-only differences, or presence/absence of diacritics), or is within a small margin of error for numerical problems. Answer 'no' only if the extracted answer is factually incorrect, missing essential identifying information, or contradicts the [correct_answer].

confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
""".strip()


BROWSECOMP_OFFICIAL_GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
""".strip()


BROWSECOMP_ZH_JUDGE_PROMPT = """根据以下精确且明确的[response]，判断以下对[question]的[correct_answer]是否正确。

[question]:  {question}

[response]:  {response}

您的判断必须符合以下指定的格式和标准：

extracted_final_answer: 从[response]中提取的最终准确答案。如果无法从答案中提取出准确的最终答案，则将提取的答案填写为"None"。

[correct_answer]: {correct_answer}

reasoning: 根据[correct_answer]解释提取的最终答案正确或错误的原因， 仅关注[correct_answer]和提取的最终答案之间是否存在有意义的差异。请勿评论问题的任何背景，请勿尝试解决问题，请勿争论任何与[correct_answer]不同的答案，仅关注答案是否匹配。

correct: 如果提取的最终答案与上面给出的[correct_answer]相符，或者在数值问题的误差范围内，则回答"yes"。否则，例如，如果存在任何不一致、歧义、不等同，或者提取的答案不正确，则回答"no"。

confidence: 从[response]中提取的置信度分数，介于0% 到100% 之间。如果没有可用的置信度分数，则填写100%。
""".strip()


XBENCH_LLM_JUDGE_PROMPT = """
你是一个通用人工智能助手。根据下面给出的[正确答案], 判断以下对[原问题]的[回答]的回答是否正确。

[原问题]: {question}

[正确答案]: {correct_answer}

[回答]:{response}

你的判断必须按照以下格式和标准进行:

最终答案: 从[回答]中提取出的最终准确答案。如果[回答]中没有明确的最终答案, 则填写'无'。

解释: 根据[正确]解释为什么[最终答案]是正确的或错误的。只关注[最终答案]与[正确答案]之间是否存在实质性差异, 不要评论题目的背景, 不要尝试重新解题, 不要为任何不同于[正确答案]的答案辩护, 只专注于判断答案是否一致。

结论: 如果[最终答案]与上方给出的[正确答案]一致, 或者在数值题目中处于可接受的微小误差范围内, 则填写'正确'; 否则（即存在任何不一致、歧义、不等价或提取出的答案错误的情况）填写'错误'。
""".strip()


HLE_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
""".strip()


GAIA_TEXT_103_JUDGE_PROMPT = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {correct_answer}

Predicted Answer: {response}

Did the model give an answer equivalent to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
""".strip()


DEEPSEARCH_QA_PROMPT = """\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" \
arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on \
the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses \
the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided \
ground truth answers. The order might not matter unless specified otherwise. The response \
might include more answers than the list. Determine the correctness *only* based on the \
list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer \
correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer \
part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part \
in the "Correct Answer". Each key will be a string indicating the expected answer part, \
and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive \
answer part. If the response provides answers that are **not** in the "Correct Answer" \
list, add these answers as excessive answers. Return an empty list when there's no \
excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following \
top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", \
"AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` \
(a string), `"Correctness Details"` (a dictionary where each key is the expected correct \
answer, and the value is a boolean indicating whether the response contains the correct \
answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and \
special characters in the JSON string. Make sure to escape all special characters and \
quotes in the JSON string.

"""


DEEPSEARCH_QA_RATING_EXAMPLE = r"""**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


MONACO_SINGLE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.
[question]: {question}
[response]: '{response}'

Your judgment must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems.

precision: Answer '1' if extracted_final_answer matches the [correct_answer] given above. Answer '0' otherwise.

final precision: Extract the precision score from above, just the final score (number).
""".strip()


MONACO_OFFICIAL_SINGLE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.
[question]: {question}
[response]: '{response}'

Your judgment must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems, a margin of 1 to 3.5 percentage points is acceptable. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

precision: Answer '1' if extracted_final_answer matches the [correct_answer] given above. Answer '0' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect. In the case where [correct_answer] is a number or percentage, then answer with the following formula to compute the normalized similarity score: [1 - (abs([correct_answer] - extracted_final_answer) / max(abs([correct_answer]), abs(extracted_final_answer)))]

final precision: Extract the precision score from above, just the final score (number).
""".strip()


MONACO_MULTI_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.
[question]: {question}
[response]: '{response}'

Your judgment must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

final answer length: Provide the overall number of unique answers that appear in [response], not just the correct ones. Be sure to provide a number, not an estimate!

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems.

overlapping answers: List all of the answers in [response] that also appear in [correct_answer]. List answers with each answer delimited by '###'. If the number of overlapping answers is zero, output 'NULL'.
""".strip()


MONACO_OFFICIAL_MULTI_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.
[question]: {question}
[response]: '{response}'

Your judgment must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

final answer length: Provide the overall number of unique answers that appear in [response], not just the correct ones. Be sure to provide a number, not an estimate!

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems, a margin of 1 to 5.5 percentage points is acceptable. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

overlapping answers: List all of the answers in [response] that also appear in [correct_answer]. You can consider an answer from [response] to match with an answer in [correct_answer] if it is equivalent or is within a small margin of error for numerical problems, a margin of 1 to 5.5 percentage points is acceptable. List all of the [response] answer appearing in [correct_answer] with each answer delimited by '###'. If the number of overlapping answers is zero, output 'NULL'.
""".strip()


DEEPWIDESEARCH_ENTITY_PROMPT = """你是一个专业的人工标注人员，现在你需要仔细检查一下是否针对查询 entity ({entity}) 相关信息的 question 的 response 中是否正确猜测对了实体的内容。

# Question
{question}

# 针对该查询问题的 Entity
{entity}

# Response
{response}

# 输出格式
请你直接输出你的判断结果，如果你认为 response 中信息正确猜测对了 entity——{entity}，请你直接输出 Yes，否则输出 No。不要添加任务无关的解释信息，只输出 Yes 或者 No。
""".strip()


def _patch_browsecomp_typos(correct_answer: str, predicted_answer: str) -> Tuple[str, str]:
    correct_answer = "ttellomS saiboT"[::-1] if "tellomS saiboT"[::-1] in correct_answer else correct_answer
    correct_answer = "yayhdapottahC najnarawsiB"[::-1] if "yayhdapattahC najnarawsiB"[::-1] in correct_answer else correct_answer
    predicted_answer = "yrtnuoC a fo htaP ehT :sedirelC sokfalG"[::-1] if "yrtnuoC a fo htaP ehT :sedirelC socfalG"[::-1] in predicted_answer else predicted_answer
    return correct_answer, predicted_answer


async def _call_judge(prompt: str, judge_client, judge_model: str) -> str:
    if judge_client is None or not judge_model:
        raise ScorerUnavailable("official scorer requires a judge_client and judge_model")
    response = await judge_client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").split("</think>")[-1].strip()


def _parse_xbench_response(text: str) -> Dict[str, Any]:
    conclusion = re.search(r"结论:*.*?(正确|错误)", text, re.S)
    extracted = re.search(r"最终答案:*(.*)", text)
    explanation = re.search(r"解释:*(.*)", text)
    return {
        "correct": conclusion.group(1) == "正确" if conclusion else None,
        "extracted_final_answer": extracted.group(1).strip() if extracted else "",
        "reasoning": explanation.group(1).strip() if explanation else "",
        "parse_error": conclusion is None,
    }


def _scorer_mode(spec: DatasetSpec) -> str:
    mode = str(spec.scorer.get("mode") or "local").strip().lower()
    if mode == "official":
        return "offical"
    return mode


def _scorer_options(spec: DatasetSpec) -> Dict[str, Any]:
    options = spec.scorer.get("options", {})
    return options if isinstance(options, dict) else {}


def _scorer_bool(spec: DatasetSpec, key: str, default: bool = False) -> bool:
    return bool(_scorer_options(spec).get(key, default))


def _xbench_extract_final_answer(text: str) -> Optional[str]:
    simple_match = re.search(
        r"(?:Final\s+Answer|最终答案)\s*[:：]*\s*(.*)",
        text or "",
        re.IGNORECASE,
    )
    if simple_match is None:
        return None
    return simple_match.group(1).strip()


def _parse_markdown_json(text: str) -> Optional[Any]:
    stripped = (text or "").strip()
    if not stripped:
        return None
    start_marker = "```json"
    start_idx = stripped.find(start_marker)
    if start_idx != -1:
        stripped = stripped[start_idx + len(start_marker):].strip()
        end_idx = stripped.rfind("```")
        if end_idx != -1:
            stripped = stripped[:end_idx].strip()
    elif stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
    try:
        return json.loads(stripped)
    except Exception:
        pass
    left = stripped.find("{")
    right = stripped.rfind("}")
    if left >= 0 and right >= left:
        try:
            return json.loads(stripped[left:right + 1])
        except Exception:
            return None
    return None


def _deepsearchqa_prompt(sample: EvalSample, response: str) -> str:
    answer_type = (
        sample.metadata.get("answer_type")
        or sample.raw.get("answer_type")
        or "Single Answer"
    )
    return DEEPSEARCH_QA_PROMPT + DEEPSEARCH_QA_RATING_EXAMPLE.format(
        prompt=sample.question,
        prompt_type=answer_type,
        answer=sample.answer,
        response=response,
    )


def _parse_deepsearchqa_judge(judge_raw: str) -> Dict[str, Any]:
    parsed = _parse_markdown_json(judge_raw)
    if not isinstance(parsed, dict):
        return {"parse_error": True, "error": "invalid_json"}
    node = parsed.get("Answer Correctness")
    if not isinstance(node, dict):
        return {"parse_error": True, "error": "missing_answer_correctness"}
    details = node.get("Correctness Details")
    if not isinstance(details, dict):
        return {"parse_error": True, "error": "invalid_correctness_details"}
    clean_details = {}
    for key, value in details.items():
        if not isinstance(key, str) or not isinstance(value, bool):
            return {"parse_error": True, "error": "invalid_correctness_details"}
        clean_details[key] = value
    excessive = node.get("Excessive Answers", [])
    if excessive is None:
        excessive = []
    if not isinstance(excessive, list) or not all(isinstance(item, str) for item in excessive):
        return {"parse_error": True, "error": "invalid_excessive_answers"}
    explanation = node.get("Explanation")
    if not isinstance(explanation, str):
        return {"parse_error": True, "error": "invalid_explanation"}

    ratings = list(clean_details.values())
    num_correct = sum(1 for item in ratings if item)
    total = len(ratings)
    false_positives = len(excessive)
    false_negatives = total - num_correct
    precision = num_correct / (num_correct + false_positives) if (num_correct + false_positives) else 0.0
    recall = num_correct / (num_correct + false_negatives) if (num_correct + false_negatives) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    if total and num_correct == total and not excessive:
        score = 100.0
    elif total and num_correct == total and excessive:
        score = 75.0
    elif num_correct > 0 and total:
        score = (num_correct / total) * 50.0
    else:
        score = 0.0
    return {
        "parse_error": False,
        "score": round(score, 2),
        "full_credit": score == 100.0,
        "explanation": explanation,
        "correctness_details": clean_details,
        "excessive_answers": excessive,
        "num_correct": num_correct,
        "num_expected": total,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def _normalize_gaia_str(input_str: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", str(input_str))
    if remove_punct:
        return no_spaces.lower().translate(str.maketrans("", "", string.punctuation))
    return no_spaces.lower()


def _normalize_gaia_number(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def _gaia_question_scorer(model_answer: str, ground_truth: str) -> Tuple[bool, str]:
    def is_float(element: Any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    if is_float(ground_truth):
        return _normalize_gaia_number(model_answer) == float(ground_truth), f"Evaluated {model_answer} as a number."
    if any(char in ground_truth for char in [",", ";"]):
        gt_elems = re.split("[,;]", ground_truth)
        ma_elems = re.split("[,;]", model_answer)
        if len(gt_elems) != len(ma_elems):
            return False, "Evaluated as a list; lists have different lengths."
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                comparisons.append(_normalize_gaia_number(ma_elem) == float(gt_elem))
            else:
                comparisons.append(_normalize_gaia_str(ma_elem, remove_punct=False) == _normalize_gaia_str(gt_elem, remove_punct=False))
        return all(comparisons), "Evaluated as a comma separated list."
    return _normalize_gaia_str(model_answer) == _normalize_gaia_str(ground_truth), f"Evaluated {model_answer} as a string."


def _extract_markdown_table(response: str, table_select: str = "first"):
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise ScorerUnavailable(f"WideSearch official scorer requires pandas: {type(e).__name__}: {e}")
    markdown_blocks = re.findall(r"```markdown(.*?)```", response or "", re.DOTALL)
    if not markdown_blocks:
        pipe_positions = [m.start() for m in re.finditer(r"\|", response or "")]
        if len(pipe_positions) >= 4:
            first_pipe = pipe_positions[0]
            last_pipe = pipe_positions[-1]
            start = response.rfind("\n", 0, first_pipe)
            start = 0 if start == -1 else start
            end = response.find("\n", last_pipe)
            end = len(response) if end == -1 else end
            table_candidate = response[start:end]
            markdown_blocks = re.findall(r"((?:\|.*\n?)+)", table_candidate)
    if not markdown_blocks:
        return None
    selected_block = markdown_blocks[-1] if table_select == "last" else markdown_blocks[0]
    lines = selected_block.strip().split("\n")
    if not lines:
        return None
    lines[0] = lines[0].replace(" ", "").lower()
    clean_lines = []
    for line in (line.strip() for line in lines):
        if set(line.strip()).issubset(set("|- :")) or "|" not in line:
            continue
        clean_lines.append("|".join(part.strip() for part in line.split("|")))
    if not clean_lines:
        return None
    df = pd.read_csv(StringIO("\n".join(clean_lines)), sep="|")
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def _norm_column(col: str) -> str:
    return str(col).strip().lower().replace(" ", "")


def _norm_cell(value: Any) -> str:
    return str(value).lower().strip().replace(" ", "").replace("*", "")


def _widesearch_extract_number(content: Any) -> str:
    numbers = re.findall(r"[-+]?\d*\.\d+%?|[-+]?\d+\.?\d*%?", str(content).replace(",", ""))
    return numbers[0] if numbers else "NULL"


def _parse_date(value: Any):
    text = str(value)
    try:
        import dateparser  # type: ignore
        return dateparser.parse(text, settings={"PREFER_DAY_OF_MONTH": "first"})
    except Exception:
        pass
    from datetime import datetime
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y-%m", "%Y/%m", "%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except Exception:
            continue
    return None


def _widesearch_preprocess(content: Any, preprocess_func_name: str) -> str:
    if preprocess_func_name == "extract_number":
        return _widesearch_extract_number(content)
    if preprocess_func_name == "norm_str":
        return _norm_cell(content)
    if preprocess_func_name == "norm_date":
        parsed = _parse_date(content)
        return parsed.strftime("%Y-%m-%d") if parsed is not None else str(content)
    raise ScorerUnavailable(f"WideSearch preprocess function not implemented: {preprocess_func_name}")


def _widesearch_metric(response: str, target: str, criterion: Any, metric_func_name: str) -> Tuple[float, str]:
    if metric_func_name == "exact_match":
        if response.lower() == target.lower():
            return 1.0, f"exact match, response: {response}, target: {target}"
        return 0.0, f"exact not match, response: {response}, target: {target}"

    if metric_func_name == "url_match":
        url_pattern = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
        response_urls = [urlparse(url).netloc for url in url_pattern.findall(response)]
        target_urls = [urlparse(url).netloc for url in url_pattern.findall(target)]
        if set(response_urls) == set(target_urls):
            return 1.0, f"url match, response: {response}, target: {target}"
        return 0.0, f"url not match, response: {response}, target: {target}"

    if metric_func_name == "in_match":
        if response in target:
            return 1.0, f"response in target, response: {response}, target: {target}"
        return 0.0, f"response not in target, response: {response}, target: {target}"

    if metric_func_name == "number_near":
        def to_number(text: str):
            if "%" in text:
                try:
                    return float(text.replace("%", "")) / 100.0
                except (ValueError, TypeError):
                    return None
            try:
                return float(text)
            except (ValueError, TypeError):
                return None

        response_num = to_number(response)
        target_num = to_number(target)
        if response_num is None or target_num is None:
            if response_num is None and target_num is None and response == target:
                return 1.0, f"number equal, response: {response}, target: {target}"
            return 0.0, f"number not convertable, response: {response_num}, target: {target_num}"
        try:
            criterion_float = float(criterion)
        except Exception:
            criterion_float = 0.0
        if abs(response_num - target_num) <= abs(target_num) * criterion_float:
            return 1.0, f"number near in range {criterion_float * 100}%, response: {response_num}, target: {target_num}"
        return 0.0, f"number not near, response: {response_num}, target: {target_num}"

    if metric_func_name == "date_near":
        response_date = _parse_date(response)
        target_date = _parse_date(target)
        if response_date is None or target_date is None:
            if response_date is None and target_date is None:
                return 1.0, f"date near, response: {response}, target: {target}"
            return 0.0, f"date not convertable, response: {response}, target: {target}"
        if abs((response_date - target_date).days) <= 31:
            return 1.0, f"date near, response: {response_date}, target: {target_date}"
        return 0.0, f"date not near, response: {response_date}, target: {target_date}"

    raise ScorerUnavailable(f"WideSearch metric function not implemented: {metric_func_name}")


WIDESEARCH_PRIMARY_KEY_PROMPT = """Your task is to align two vocabularies. The inputs are the vocabulary to be aligned and the reference vocabulary respectively. Note that you need to perform semantic alignment (not positional alignment). If two strings are exactly the same, they must correspond to each other. These two strings are supposed to represent the same entity, with differences only in the expression forms and formats.


The vocabulary to be aligned is as follows:
{response}

The reference vocabulary is as follows:
{reference}

The alignment rules are as follows:
List the values in the vocabulary to be aligned one by one. If there is a value in the reference vocabulary that has the same meaning as this value, `transform` should be represented as the value from the reference vocabulary; otherwise, `transform` should be represented as the original value from the vocabulary to be aligned.

Note that `origin` must be taken from the vocabulary to be aligned keeping the original format, and `transform` must be taken from the reference vocabulary. For example: Some words in the vocabulary to be aligned might be the words in the reference vocabulary with Markdown formatting added, keep the to be aligned format in `origin` and the reference format in `transform`.

For the `origin`, first find the `transform` that is the closest in meaning and then judge whether they correspond to each other. Those entities not correspond to each other could not output.

Please output the alignment results in the following format:
```json
{{
    "origin_str1": "transform_str1",
    "origin_str2": "transform_str2"
}}
```
"""


WIDESEARCH_EVAL_COLUMN_PROMPT = """You are an expert in grading answers. Your task is to score the responses to a certain question. Below, you will be provided with a set of standard answers, a set of responses to be graded, and specific grading criteria.

Each answer and each response has an idx. Please score each pair of answers and responses in this set according to the following methods:
1. The scoring range is from 0 to 1. A score of 1 indicates a completely correct answer. For deduction items, please refer to the specific grading criteria section.
2. After reading the standard answers, responses to be graded, and grading criteria, please first analyze and judge them item by item according to the grading criteria.
3. The score can only be an integer of 0 or 1.
4. After the analysis and judgment, please provide the final scoring results. Each pair should have a score. Output in Markdown JSON format, as shown below:
```json
{{
    "idx_xxx": score,
    "idx_yyy": score,
    ...
}}
```

====== criterion-start ======
{criterion}
====== criterion-end ======

====== response-start ======
{response}
====== response-end ======

Now start scoring. Please make sure to analyze each item step by step before providing the final scoring results.

"""


async def _widesearch_primary_key_preprocess(response: List[str], reference: List[str], judge_client, judge_model: str) -> Dict[str, str]:
    if set(response) == set(reference):
        return {}
    judge_raw = await _call_judge(
        WIDESEARCH_PRIMARY_KEY_PROMPT.format(response=response, reference=reference),
        judge_client,
        judge_model,
    )
    parsed = _parse_markdown_json(judge_raw)
    if isinstance(parsed, dict):
        return {str(key): str(value) for key, value in parsed.items()}
    return {}


async def _widesearch_llm_judge_column(response: List[str], target: List[str], criterion: str, judge_client, judge_model: str) -> Tuple[List[float], List[str]]:
    response_dict = {
        f"idx_{idx}": {"response": resp, "target": tar}
        for idx, (resp, tar) in enumerate(zip(response, target))
    }
    judge_raw = await _call_judge(
        WIDESEARCH_EVAL_COLUMN_PROMPT.format(criterion=criterion, response=response_dict),
        judge_client,
        judge_model,
    )
    score_dict = _parse_markdown_json(judge_raw)
    if not isinstance(score_dict, dict):
        return [0.0] * len(response), ["llm judge failed due to parse error"] * len(response)
    scores = []
    for idx in range(len(response)):
        raw_score = score_dict.get(f"idx_{idx}", 0)
        if isinstance(raw_score, bool):
            scores.append(1.0 if raw_score else 0.0)
        else:
            try:
                scores.append(1.0 if int(raw_score) == 1 else 0.0)
            except Exception:
                scores.append(0.0)
    if len(scores) != len(response):
        return [0.0] * len(response), ["llm judge failed due to length"] * len(response)
    return scores, [judge_raw] * len(response)


def _load_gold_csv(path: str, required_columns: List[str]):
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise ScorerUnavailable(f"WideSearch official scorer requires pandas: {type(e).__name__}: {e}")
    if not os.path.exists(path):
        raise ScorerUnavailable(f"gold answer csv not found: {path}")
    df = pd.read_csv(path)
    df.columns = [_norm_column(col) for col in df.columns]
    required = [_norm_column(col) for col in required_columns]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ScorerUnavailable(f"gold csv missing required columns {missing}: {path}")
    return df[required]


async def _official_table_score(
    spec: DatasetSpec,
    sample: EvalSample,
    predicted_answer: str,
    judge_client=None,
    judge_model: str = "",
) -> ScoreResult:
    if not spec.gold_root:
        raise ScorerUnavailable("WideSearch official scorer requires gold_root")
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise ScorerUnavailable(f"WideSearch official scorer requires pandas: {type(e).__name__}: {e}")

    source = spec.scorer.get("source", "widesearch_official")
    try:
        if _scorer_bool(spec, "entity_gate", False):
            entity = sample.metadata.get("entity") or sample.raw.get("entity")
            if entity:
                entity_judge_raw = await _call_judge(
                    DEEPWIDESEARCH_ENTITY_PROMPT.format(
                        question=sample.question,
                        response=predicted_answer,
                        entity=entity,
                    ),
                    judge_client,
                    judge_model,
                )
                if "yes" not in entity_judge_raw.lower():
                    return ScoreResult(
                        status="incorrect",
                        score=0.0,
                        official_scorer=source,
                        metrics={
                            "entity_acc": 0.0,
                            "entity_judge_raw": entity_judge_raw,
                            "msg": "the entity is wrong, all the results are wrong, set as 0",
                            "full_credit": False,
                        },
                        judge_raw=entity_judge_raw,
                    )

        evaluation = sample.raw.get("evaluation", {})
        if isinstance(evaluation, str):
            evaluation = json.loads(evaluation)
        if not isinstance(evaluation, dict):
            raise ValueError("evaluation is not a dict")

        required_columns = [_norm_column(col) for col in evaluation.get("required", [])]
        unique_columns = [_norm_column(col) for col in evaluation.get("unique_columns", [])]
        eval_pipeline = {
            _norm_column(col): item
            for col, item in (evaluation.get("eval_pipeline") or {}).items()
            if isinstance(item, dict)
        }
        if not required_columns:
            raise ValueError("evaluation.required is empty")
        if not unique_columns:
            raise ValueError("evaluation.unique_columns is empty")

        table_select = str(_scorer_options(spec).get("markdown_table_select") or "").strip().lower()
        if not table_select:
            table_select = "last" if spec.scorer.get("type") == "deepwidesearch_official" else "first"
        if table_select not in ("first", "last"):
            table_select = "first"
        response_df = _extract_markdown_table(predicted_answer, table_select=table_select)
        if response_df is None:
            return ScoreResult(status="incorrect", score=0.0, official_scorer=source, metrics={"msg": "response_df is None"})
        response_df.columns = [_norm_column(col) for col in response_df.columns]

        if set(required_columns) != set(response_df.columns):
            column_map = await _widesearch_primary_key_preprocess(
                response_df.columns.tolist(),
                required_columns,
                judge_client,
                judge_model,
            )
            response_df.rename(columns=column_map, inplace=True)

        if set(required_columns) != set(response_df.columns):
            msg = f"required_columns {required_columns} != response_df {list(response_df.columns)}"
            return ScoreResult(status="incorrect", score=0.0, official_scorer=source, metrics={"msg": msg})

        gold_path = os.path.join(spec.gold_root, f"{sample.sample_id}.csv")
        answer_df = _load_gold_csv(gold_path, required_columns)
        response_df = response_df[required_columns]
        answer_df = answer_df[required_columns]

        for col in required_columns:
            try:
                answer_type = answer_df[col].dtype
                response_type = response_df[col].dtype
            except Exception:
                answer_type = None
                response_type = None
            if (response_type == float and answer_type == int) or (response_type == int and answer_type == float):
                if response_type == int:
                    response_df[col] = response_df[col].astype(float)
                elif answer_type == int:
                    answer_df[col] = answer_df[col].astype(float)
            answer_df[col] = answer_df[col].astype(str)
            response_df[col] = response_df[col].astype(str)

        response_df.drop_duplicates(subset=unique_columns, inplace=True)
        answer_df.drop_duplicates(subset=unique_columns, inplace=True)

        for col in unique_columns:
            item = eval_pipeline.get(col)
            if item is None:
                continue
            metric_func_name_list = item.get("metric", [])
            if "llm_judge" in metric_func_name_list or "exact_match" in metric_func_name_list:
                primary_key_map = await _widesearch_primary_key_preprocess(
                    response_df[col].tolist(),
                    answer_df[col].tolist(),
                    judge_client,
                    judge_model,
                )
                response_df[col + "_before_map"] = response_df[col]
                response_df[col] = response_df[col].apply(lambda x: primary_key_map.get(x, x))

        for col, item in eval_pipeline.items():
            if col not in required_columns:
                continue
            for preprocess_func_name in item.get("preprocess", []):
                response_df[col] = response_df[col].apply(lambda x, fn=preprocess_func_name: _widesearch_preprocess(x, fn))
                answer_df[col] = answer_df[col].apply(lambda x, fn=preprocess_func_name: _widesearch_preprocess(x, fn))

        exact_score = 0.0
        if answer_df[required_columns].shape == response_df[required_columns].shape:
            gt_sorted = answer_df[required_columns].sort_values(by=required_columns).reset_index(drop=True)
            pred_sorted = response_df[required_columns].sort_values(by=required_columns).reset_index(drop=True)
            if gt_sorted.equals(pred_sorted):
                exact_score = 1.0

        df_inner = pd.merge(
            answer_df,
            response_df,
            on=unique_columns,
            how="inner",
            suffixes=("_query", "_response"),
        )

        df_inner_score = pd.DataFrame(index=df_inner.index)
        df_inner_msg = pd.DataFrame(index=df_inner.index)
        for col in required_columns:
            if col in unique_columns:
                df_inner_score[f"{col}_exact_match"] = 1.0
                df_inner_msg[f"{col}_exact_match_eval_msg"] = "key_match"
                continue

            item = eval_pipeline.get(col)
            if item is None:
                raise ValueError(f"missing eval_pipeline for required column {col}")
            criterion = item.get("criterion")
            for metric_func_name in item.get("metric", []):
                if metric_func_name == "llm_judge":
                    score_list, msg_list = await _widesearch_llm_judge_column(
                        df_inner[col + "_response"].tolist(),
                        df_inner[col + "_query"].tolist(),
                        str(criterion or ""),
                        judge_client,
                        judge_model,
                    )
                    df_inner_score[f"{col}_{metric_func_name}"] = score_list
                    df_inner_msg[f"{col}_{metric_func_name}_eval_msg"] = msg_list
                else:
                    metric_values = df_inner.apply(
                        lambda x, metric=metric_func_name: _widesearch_metric(
                            x[col + "_response"],
                            x[col + "_query"],
                            criterion,
                            metric,
                        ),
                        axis=1,
                    )
                    df_inner_score[f"{col}_{metric_func_name}"] = metric_values.apply(lambda x: x[0])
                    df_inner_msg[f"{col}_{metric_func_name}_eval_msg"] = metric_values.apply(lambda x: x[1])

        row_scores = df_inner_score.min(axis=1) if not df_inner_score.empty else pd.Series(dtype=float)
        tp_by_row = float(row_scores.sum())
        tp_by_item = float(df_inner_score.sum().sum()) if not df_inner_score.empty else 0.0
        num_pred_rows = len(response_df)
        num_gt_rows = len(answer_df)
        num_pred_items = num_pred_rows * len(required_columns)
        num_gt_items = num_gt_rows * len(required_columns)
        precision_by_row = tp_by_row / num_pred_rows if num_pred_rows > 0 else 0.0
        recall_by_row = tp_by_row / num_gt_rows if num_gt_rows > 0 else 0.0
        precision_by_item = tp_by_item / num_pred_items if num_pred_items > 0 else 0.0
        recall_by_item = tp_by_item / num_gt_items if num_gt_items > 0 else 0.0

        def calc_f1(precision, recall):
            return (2 * precision * recall / (precision + recall)) if (precision + recall > 1e-9) else 0.0

        f1_by_row = calc_f1(precision_by_row, recall_by_row)
        f1_by_item = calc_f1(precision_by_item, recall_by_item)
        unique_col_score_cols = [f"{col}_exact_match" for col in unique_columns if f"{col}_exact_match" in df_inner_score]
        if unique_col_score_cols:
            unique_col_row_scores = df_inner_score[unique_col_score_cols].min(axis=1)
            tp_by_unique_col = float(unique_col_row_scores.sum())
        else:
            tp_by_unique_col = 0.0
        column_precision = tp_by_unique_col / num_pred_rows if num_pred_rows > 0 else 0.0
        column_recall = tp_by_unique_col / num_gt_rows if num_gt_rows > 0 else 0.0
        column_f1 = calc_f1(column_precision, column_recall)
        score = exact_score
        if precision_by_item == recall_by_item == f1_by_item == 1.0 and precision_by_row == recall_by_row == f1_by_row == 1.0:
            score = 1.0

        msg = df_inner_score.to_string()
        return ScoreResult(
            status="scored" if score == 1.0 else "incorrect",
            score=float(score),
            official_scorer=source,
            metrics={
                "precision_by_row": precision_by_row,
                "recall_by_row": recall_by_row,
                "f1_by_row": f1_by_row,
                "precision_by_item": precision_by_item,
                "recall_by_item": recall_by_item,
                "f1_by_item": f1_by_item,
                "column_precision": column_precision,
                "column_recall": column_recall,
                "column_f1": column_f1,
                "num_pred_rows": num_pred_rows,
                "num_gold_rows": num_gt_rows,
                "num_pred_items": num_pred_items,
                "num_gold_items": num_gt_items,
                "exact_table_score": exact_score,
                "full_credit": score == 1.0,
                "msg": msg,
            },
        )
    except ScorerUnavailable:
        raise
    except Exception as e:
        return ScoreResult(
            status="error",
            score=None,
            official_scorer=source,
            error_type=type(e).__name__,
            error_message=str(e),
        )


def _parse_monaco_score(text: str, gold_answers_length: int) -> Dict[str, Any]:
    text = text.replace("final_answer_length", "final answer length")
    text = text.replace("overlapping_answers", "overlapping answers")
    text = text.replace("final_precision", "final precision")
    if gold_answers_length == 1:
        try:
            precision = float(text.split("final precision:")[1].split("\n")[0].replace("...", "").strip())
        except Exception:
            precision = 0.0
        return {"judge_score": precision, "precision": precision}
    length_match = re.search(r"final answer length:\s*(\d+)", text, re.I)
    predicted_length = int(length_match.group(1)) if length_match else 0
    overlap = ""
    if "\noverlapping answers:" in text:
        overlap = text.split("\noverlapping answers:", 1)[1].strip()
    answers = [] if not overlap or overlap == "NULL" else [item for item in overlap.strip("#").split("###") if item and item != "NULL"]
    num_correct = len(answers)
    recall = min(num_correct, gold_answers_length) / gold_answers_length if gold_answers_length else 0.0
    predicted_length = max(predicted_length, num_correct)
    precision = num_correct / predicted_length if predicted_length else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "judge_score": f1,
        "precision": precision,
        "recall": recall,
        "gold_answers_length": gold_answers_length,
        "predicted_answers_num": predicted_length,
        "correct_predictions": answers,
        "num_correct": num_correct,
    }


def _parse_deepresearch_json(text: str) -> Optional[Dict[str, Any]]:
    parsed = _parse_markdown_json(text)
    return parsed if isinstance(parsed, dict) else None


def _format_deepresearch_criteria(criteria_data: Dict[str, Any]) -> str:
    criteria_for_prompt = {}
    for dim, criterions_list in (criteria_data.get("criterions") or {}).items():
        if not isinstance(criterions_list, list):
            continue
        criteria_for_prompt[dim] = [
            {"criterion": item["criterion"], "explanation": item["explanation"]}
            for item in criterions_list
            if isinstance(item, dict) and "criterion" in item and "explanation" in item
        ]
    return json.dumps(criteria_for_prompt, ensure_ascii=False, indent=2)


DEEPRESEARCH_SCORE_PROMPT_EN = """
<system_role>You are a strict, meticulous, and objective research article evaluation expert. You excel at using specific assessment criteria to deeply compare two articles on the same task, providing precise scores and clear justifications.</system_role>

<user_prompt>
**Task Background**
There is a deep research task, and you need to evaluate two research articles written for this task. We will assess the articles across four dimensions: Comprehensiveness, Insight, Instruction Following, and Readability. The content is as follows:
<task>
"{task_prompt}"
</task>

**Articles to Evaluate**
<article_1>
"{article_1}"
</article_1>

<article_2>
"{article_2}"
</article_2>

**Evaluation Criteria**
Now, you need to evaluate and compare these two articles based on the following **evaluation criteria list**, providing comparative analysis and scoring each on a scale of 0-10. Each criterion includes an explanation, please understand carefully.

<criteria_list>
{criteria_list}
</criteria_list>

<Instruction>
**Your Task**
Please strictly evaluate and compare `<article_1>` and `<article_2>` based on **each criterion** in the `<criteria_list>`. You need to:
1. **Analyze Each Criterion**: Consider how each article fulfills the requirements of each criterion.
2. **Comparative Evaluation**: Analyze how the two articles perform on each criterion, referencing the content and criterion explanation.
3. **Score Separately**: Based on your comparative analysis, score each article on each criterion (0-10 points).

**Scoring Rules**
For each criterion, score both articles on a scale of 0-10 (continuous values). The score should reflect the quality of performance on that criterion:
* 0-2 points: Very poor performance. Almost completely fails to meet the criterion requirements.
* 2-4 points: Poor performance. Minimally meets the criterion requirements with significant deficiencies.
* 4-6 points: Average performance. Basically meets the criterion requirements, neither good nor bad.
* 6-8 points: Good performance. Largely meets the criterion requirements with notable strengths.
* 8-10 points: Excellent/outstanding performance. Fully meets or exceeds the criterion requirements.

**Output Format Requirements**
Please **strictly** follow the `<output_format>` below for each criterion evaluation. **Do not include any other unrelated content, introduction, or summary**.
</Instruction>

<output_format>
{{
  "comprehensiveness": [{{"criterion": "...", "analysis": "...", "article_1_score": 0, "article_2_score": 0}}],
  "insight": [{{"criterion": "...", "analysis": "...", "article_1_score": 0, "article_2_score": 0}}],
  "instruction_following": [{{"criterion": "...", "analysis": "...", "article_1_score": 0, "article_2_score": 0}}],
  "readability": [{{"criterion": "...", "analysis": "...", "article_1_score": 0, "article_2_score": 0}}]
}}
</output_format>

Now, please evaluate the two articles based on the research task and criteria, providing detailed comparative analysis and scores according to the requirements above. Ensure your output follows the specified `<output_format>` and that the JSON format is parsable, with all characters that might cause JSON parsing errors properly escaped.
</user_prompt>
""".strip()


DEEPRESEARCH_SCORE_PROMPT_ZH = """
<system_role>你是一名严格、细致、客观的调研文章评估专家。你擅长根据具体的评估标准，深入比较两篇针对同一任务的文章，并给出精确的评分和清晰的理由。</system_role>

<user_prompt>
**任务背景**
有一个深度调研任务，你需要评估针对该任务撰写的两篇调研文章。我们会从以下四个维度评估文章：全面性、洞察力、指令遵循能力和可读性。内容如下：
<task>
"{task_prompt}"
</task>

**待评估文章**
<article_1>
"{article_1}"
</article_1>

<article_2>
"{article_2}"
</article_2>

**评估标准**
现在，你需要根据以下**评判标准列表**，逐条评估并比较这两篇文章的表现，输出对比分析，然后给出0-10的分数。每个标准都附有其解释，请仔细理解。

<criteria_list>
{criteria_list}
</criteria_list>

<Instruction>
**你的任务**
请严格按照 `<criteria_list>` 中的**每一条标准**，对比评估 `<article_1>` 和 `<article_2>` 在该标准上的具体表现。你需要：
1. **逐条分析**：针对列表中的每一条标准，分别思考两篇文章是如何满足该标准要求的。
2. **对比评估**：结合文章内容与标准解释，对比分析两篇文章在每一条标准上的表现。
3. **分别打分**：基于你的对比分析，为两篇文章在该条标准上的表现分别打分（0-10分）。

**打分规则**
对每一条标准，分别为两篇文章打分，打分范围为 0-10 分（连续的数值）。分数高低应体现文章在该标准上表现的好坏：
* 0-2分：表现很差。几乎完全不符合标准要求。
* 2-4分：表现较差。少量符合标准要求，但有明显不足。
* 4-6分：表现中等。基本符合标准要求，不好不坏。
* 6-8分：表现较好。大部分符合标准要求，有可取之处。
* 8-10分：表现出色/极好。完全或超预期符合标准要求。

**输出格式要求**
请**严格**按照下列`<output_format>`格式输出每一条标准的评估结果，**不要包含任何其他无关内容、引言或总结**。
</Instruction>

<output_format>
{{
  "comprehensiveness": [{{"criterion": "...", "analysis": "...", "article_1_score": 0, "article_2_score": 0}}],
  "insight": [{{"criterion": "...", "analysis": "...", "article_1_score": 0, "article_2_score": 0}}],
  "instruction_following": [{{"criterion": "...", "analysis": "...", "article_1_score": 0, "article_2_score": 0}}],
  "readability": [{{"criterion": "...", "analysis": "...", "article_1_score": 0, "article_2_score": 0}}]
}}
</output_format>

现在，请根据调研任务和标准，对两篇文章进行评估，并按照上述要求给出详细的对比分析和评分，请确保输出格式遵守上述`<output_format>`，而且保证其中的json格式可以解析，注意所有可能导致json解析错误的要转义的符号。
</user_prompt>
""".strip()


DEEPRESEARCH_EXPECTED_DIMS = ["comprehensiveness", "insight", "instruction_following", "readability"]
_DEEPRESEARCH_PROMPT_CACHE: Dict[str, str] = {}


def _load_official_deepresearch_prompt(language: str) -> Optional[str]:
    lang = "zh" if language == "zh" else "en"
    if lang in _DEEPRESEARCH_PROMPT_CACHE:
        return _DEEPRESEARCH_PROMPT_CACHE[lang]
    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "reference_github"
        / "deep_research_bench"
        / "prompt"
        / f"score_prompt_{lang}.py"
    )
    if not prompt_path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location(f"deepresearch_score_prompt_{lang}", prompt_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prompt = getattr(module, "generate_merged_score_prompt", None)
        if isinstance(prompt, str) and prompt.strip():
            _DEEPRESEARCH_PROMPT_CACHE[lang] = prompt.strip()
            return _DEEPRESEARCH_PROMPT_CACHE[lang]
    except Exception:
        return None
    return None


def _deepresearch_prompt_template(language: str) -> str:
    official_prompt = _load_official_deepresearch_prompt(language)
    if official_prompt:
        return official_prompt
    return DEEPRESEARCH_SCORE_PROMPT_ZH if language == "zh" else DEEPRESEARCH_SCORE_PROMPT_EN


def _validate_deepresearch_output(parsed: Any) -> Optional[str]:
    if not isinstance(parsed, dict):
        return "DeepResearch RACE JSON output could not be parsed"
    missing_dims = [dim for dim in DEEPRESEARCH_EXPECTED_DIMS if dim not in parsed]
    if missing_dims:
        return f"DeepResearch RACE JSON output missing dimensions: {missing_dims}"
    bad_dims = [dim for dim in DEEPRESEARCH_EXPECTED_DIMS if not isinstance(parsed.get(dim), list)]
    if bad_dims:
        return f"DeepResearch RACE JSON output dimensions must be lists: {bad_dims}"
    return None


def _calculate_deepresearch_scores(llm_output_json: Dict[str, Any], criteria_data: Dict[str, Any]) -> Dict[str, Any]:
    dimension_weights = criteria_data.get("dimension_weight", {})
    criterion_weights = {
        dim: {crit["criterion"]: crit["weight"] for crit in criterions if isinstance(crit, dict) and "criterion" in crit and "weight" in crit}
        for dim, criterions in (criteria_data.get("criterions") or {}).items()
        if isinstance(criterions, list)
    }
    target_total = 0.0
    reference_total = 0.0
    dims = {}
    for dim, scores_list in llm_output_json.items():
        if not isinstance(scores_list, list) or dim not in dimension_weights:
            continue
        weights = criterion_weights.get(dim, {})
        target_sum = reference_sum = total_weight = 0.0
        for item in scores_list:
            if not isinstance(item, dict):
                continue
            criterion = str(item.get("criterion", "")).strip()
            weight = weights.get(criterion)
            if weight is None and weights:
                criterion_lower = criterion.lower()
                for key, val in weights.items():
                    if key.lower() == criterion_lower:
                        weight = val
                        break
            if weight is None and weights:
                criterion_lower = criterion.lower()
                for key, val in weights.items():
                    key_lower = key.lower()
                    if criterion_lower in key_lower or key_lower in criterion_lower:
                        weight = val
                        break
            if weight is None and weights:
                weight = sum(weights.values()) / len(weights)
            if weight is None:
                continue
            try:
                target_score = float(item.get("article_1_score"))
                reference_score = float(item.get("article_2_score"))
            except Exception:
                continue
            target_sum += target_score * weight
            reference_sum += reference_score * weight
            total_weight += weight
        target_avg = target_sum / total_weight if total_weight else 0.0
        reference_avg = reference_sum / total_weight if total_weight else 0.0
        dims[dim] = {"target": target_avg, "reference": reference_avg}
        target_total += target_avg * float(dimension_weights.get(dim, 0))
        reference_total += reference_avg * float(dimension_weights.get(dim, 0))
    overall = target_total / (target_total + reference_total) if (target_total + reference_total) > 0 else 0.0
    return {"target_total": target_total, "reference_total": reference_total, "overall_score": overall, "dims": dims}


def _load_jsonl_by_prompt(path: str) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if "prompt" in row:
                    out[row["prompt"]] = row
    return out


async def score_prediction(
    spec: DatasetSpec,
    sample: EvalSample,
    predicted_answer: str,
    judge_client=None,
    judge_model: str = "",
) -> ScoreResult:
    scorer_type = spec.scorer.get("type", "")
    source = spec.scorer.get("source", scorer_type or "unknown")
    scorer_mode = _scorer_mode(spec)
    answer = str(sample.answer or "")
    pred = str(predicted_answer or "")

    try:
        if scorer_type in ("browsecomp_official", "qa_official", "browsecomp_zh_official", "hle_official"):
            correct_answer = answer
            if scorer_type == "browsecomp_official":
                if _scorer_bool(spec, "patch_browsecomp_typos", scorer_mode == "local"):
                    correct_answer, pred = _patch_browsecomp_typos(correct_answer, pred)
                prompt_template = BROWSECOMP_GRADER_TEMPLATE if scorer_mode == "local" else BROWSECOMP_OFFICIAL_GRADER_TEMPLATE
            elif scorer_type == "browsecomp_zh_official":
                prompt_template = BROWSECOMP_ZH_JUDGE_PROMPT
            elif scorer_type == "hle_official":
                prompt_template = HLE_JUDGE_PROMPT
            else:
                prompt_template = BROWSECOMP_GRADER_TEMPLATE

            if _scorer_bool(spec, "use_em_shortcut", scorer_mode == "local") and em_score(correct_answer, pred):
                return ScoreResult(status="scored", score=1.0, official_scorer=source, metrics={"method": "exact_match"})
            if not pred.strip():
                return ScoreResult(status="incorrect", score=0.0, official_scorer=source, metrics={"method": "empty_prediction"})
            judge_raw = await _call_judge(
                prompt_template.format(question=sample.question, response=pred, correct_answer=correct_answer),
                judge_client,
                judge_model,
            )
            parsed = parse_judge_response(judge_raw)
            if parsed.get("parse_error"):
                return ScoreResult(status="error", score=None, official_scorer=source, judge_raw=judge_raw, error_type="JudgeParseError", error_message="official judge response could not be parsed")
            return ScoreResult(
                status="scored" if parsed.get("correct") else "incorrect",
                score=1.0 if parsed.get("correct") else 0.0,
                official_scorer=source,
                metrics={k: v for k, v in parsed.items() if k != "parse_error"},
                judge_raw=judge_raw,
            )

        if scorer_type == "deepsearchqa_official":
            if not pred.strip():
                return ScoreResult(status="incorrect", score=0.0, official_scorer=source, metrics={"method": "empty_prediction", "full_credit": False})
            judge_raw = await _call_judge(_deepsearchqa_prompt(sample, pred), judge_client, judge_model)
            parsed = _parse_deepsearchqa_judge(judge_raw)
            if parsed.get("parse_error"):
                return ScoreResult(status="error", score=None, official_scorer=source, judge_raw=judge_raw, error_type="JudgeParseError", error_message=f"DeepSearchQA judge response could not be parsed: {parsed.get('error')}")
            score = float(parsed.get("score", 0.0))
            return ScoreResult(
                status="scored" if parsed.get("full_credit") else "incorrect",
                score=score,
                official_scorer=source,
                metrics={k: v for k, v in parsed.items() if k != "parse_error"},
                judge_raw=judge_raw,
            )

        if scorer_type == "xbench_official":
            if _scorer_bool(spec, "extract_final_answer_match", scorer_mode == "offical"):
                extracted_for_match = _xbench_extract_final_answer(pred)
                if extracted_for_match == answer:
                    return ScoreResult(status="scored", score=1.0, official_scorer=source, metrics={"method": "official_extracted_final_answer_match", "full_credit": True})
            elif pred.strip() == answer.strip():
                return ScoreResult(status="scored", score=1.0, official_scorer=source, metrics={"method": "local_simple_match", "full_credit": True})
            judge_raw = await _call_judge(
                XBENCH_LLM_JUDGE_PROMPT.format(question=sample.question, correct_answer=answer, response=pred),
                judge_client,
                judge_model,
            )
            parsed = _parse_xbench_response(judge_raw)
            if parsed.get("parse_error"):
                return ScoreResult(status="error", score=None, official_scorer=source, judge_raw=judge_raw, error_type="JudgeParseError", error_message="xBench judge response could not be parsed")
            return ScoreResult(
                status="scored" if parsed.get("correct") else "incorrect",
                score=1.0 if parsed.get("correct") else 0.0,
                official_scorer=source,
                metrics={**parsed, "full_credit": bool(parsed.get("correct"))},
                judge_raw=judge_raw,
            )

        if scorer_type == "gaia_official":
            is_correct, explanation = _gaia_question_scorer(pred, answer)
            return ScoreResult(status="scored" if is_correct else "incorrect", score=1.0 if is_correct else 0.0, official_scorer=source, metrics={"explanation": explanation})

        if scorer_type == "gaia_text_103_judge":
            if _scorer_bool(spec, "use_gaia_exact_shortcut", scorer_mode == "local") and _gaia_question_scorer(pred, answer)[0]:
                return ScoreResult(status="scored", score=1.0, official_scorer=source, metrics={"method": "gaia_exact"})
            if not pred.strip():
                return ScoreResult(status="incorrect", score=0.0, official_scorer=source, metrics={"method": "empty_prediction"})
            judge_raw = await _call_judge(
                GAIA_TEXT_103_JUDGE_PROMPT.format(question=sample.question, response=pred, correct_answer=answer),
                judge_client,
                judge_model,
            )
            normalized = judge_raw.strip().rstrip(".").lower()
            if normalized == "correct":
                return ScoreResult(status="scored", score=1.0, official_scorer=source, metrics={"method": "llm_judge", "full_credit": True}, judge_raw=judge_raw)
            if normalized == "incorrect":
                return ScoreResult(status="incorrect", score=0.0, official_scorer=source, metrics={"method": "llm_judge", "full_credit": False}, judge_raw=judge_raw)
            return ScoreResult(status="error", score=None, official_scorer=source, judge_raw=judge_raw, error_type="JudgeParseError", error_message="GAIA Text-103 judge response must be Correct or Incorrect")

        if scorer_type in ("widesearch_official", "deepwidesearch_official"):
            return await _official_table_score(spec, sample, pred, judge_client=judge_client, judge_model=judge_model)

        if scorer_type == "monaco_official":
            gold_answers = [item.strip() for item in answer.split(",") if item.strip()] if isinstance(answer, str) else [answer]
            if scorer_mode == "offical":
                prompt = MONACO_OFFICIAL_SINGLE_PROMPT if len(gold_answers) <= 1 else MONACO_OFFICIAL_MULTI_PROMPT
            else:
                prompt = MONACO_SINGLE_PROMPT if len(gold_answers) <= 1 else MONACO_MULTI_PROMPT
            judge_raw = await _call_judge(
                prompt.format(question=sample.question, response=pred, correct_answer=answer),
                judge_client,
                judge_model,
            )
            metrics = _parse_monaco_score(judge_raw, max(1, len(gold_answers)))
            return ScoreResult(status="scored", score=float(metrics.get("judge_score", 0.0)), official_scorer=source, metrics=metrics, judge_raw=judge_raw)

        if scorer_type == "deepresearch_race_official":
            criteria_map = _load_jsonl_by_prompt(spec.criteria_path or "")
            reference_map = _load_jsonl_by_prompt(spec.reference_path or "")
            criteria_data = criteria_map.get(sample.question)
            reference_data = reference_map.get(sample.question)
            if not criteria_data:
                raise ScorerUnavailable(f"DeepResearch criteria not found for prompt: {sample.question[:80]}")
            if not reference_data:
                raise ScorerUnavailable(f"DeepResearch reference article not found for prompt: {sample.question[:80]}")
            criteria_list = _format_deepresearch_criteria(criteria_data)
            language = (sample.metadata.get("language") or sample.raw.get("language") or criteria_data.get("language") or "en")
            prompt_template = _deepresearch_prompt_template(str(language))
            judge_prompt = prompt_template.format(
                task_prompt=sample.question,
                article_1=pred,
                article_2=reference_data.get("article", ""),
                criteria_list=criteria_list,
            )
            max_retries = int(_scorer_options(spec).get("max_retries", 10) or 10)
            max_retries = max(1, max_retries)
            judge_raw = ""
            parsed = None
            validation_error = "DeepResearch RACE JSON output could not be parsed"
            for attempt_idx in range(max_retries):
                try:
                    judge_raw = await _call_judge(judge_prompt, judge_client, judge_model)
                    parsed = _parse_deepresearch_json(judge_raw)
                    validation_error = _validate_deepresearch_output(parsed)
                    if validation_error is None:
                        break
                except ScorerUnavailable:
                    raise
                except Exception as e:
                    validation_error = f"{type(e).__name__}: {e}"
                if attempt_idx < max_retries - 1:
                    await asyncio.sleep(min(1.5 ** attempt_idx, 10.0))
            if parsed is None or validation_error is not None:
                return ScoreResult(
                    status="error",
                    score=None,
                    official_scorer=source,
                    judge_raw=judge_raw,
                    error_type="JudgeParseError",
                    error_message=validation_error,
                )
            metrics = _calculate_deepresearch_scores(parsed, criteria_data)
            return ScoreResult(status="scored", score=float(metrics.get("overall_score", 0.0)), official_scorer=source, metrics=metrics, judge_raw=judge_raw)

        raise ScorerUnavailable(f"Unsupported official scorer type: {scorer_type}")
    except ScorerUnavailable as e:
        return ScoreResult(status="scorer_unavailable", score=None, official_scorer=source, error_type=type(e).__name__, error_message=str(e))
    except Exception as e:
        return ScoreResult(status="error", score=None, official_scorer=source, error_type=type(e).__name__, error_message=str(e))
