import json
import re
from typing import Optional, Tuple

from unified_eval.types import DatasetSpec, EvalSample


def direct_repair_enabled_for_dataset(spec: DatasetSpec, config: dict) -> bool:
    if not config.get("direct_repair_retry"):
        return False
    if str(config.get("mode") or "").strip().lower() != "direct":
        return False
    patterns = [
        part.strip().lower()
        for part in re.split(r"[,\s]+", str(config.get("direct_repair_datasets") or "BrowseComp"))
        if part.strip()
    ]
    dataset_name = spec.name.lower()
    prompt_source = (spec.prompt_source or "").lower()
    return any(pattern in dataset_name or pattern in prompt_source for pattern in patterns)


def _truncate_text(text, limit: int) -> str:
    text = "" if text is None else str(text)
    if limit <= 0 or len(text) <= limit:
        return text
    head = max(0, limit // 2)
    tail = max(0, limit - head)
    return (
        text[:head]
        + f"\n\n...[truncated {len(text) - limit} chars]...\n\n"
        + text[-tail:]
    )


def _extract_json_object(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            return data if isinstance(data, dict) else None
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


def _coerce_audit_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "ready", "pass", "ok"}
    return False


def build_direct_repair_audit_prompt(question: str, draft_answer: str, evidence: str) -> str:
    return f"""\
You are a strict table-answer auditor. You do NOT know the gold answer and must NOT assume hidden reference answers.
Evaluate the draft only against the user's question and the available evidence.

Check:
1. Exact required columns and Markdown table validity.
2. Whether the row universe implied by the question is complete.
3. Whether every row satisfies the scope constraints.
4. Whether high-risk cells such as dates, fees, rankings, URLs, prices, numeric values, and statuses are specific and evidence-backed.
5. Whether there are vague placeholders such as "varies", "unknown", "N/A", "/", or approximate values where exact values are required.

Return JSON only. Use valid JSON booleans (`true`/`false`) and `null`; do not include comments or Markdown fences.
{{
  "ready_to_finish": false,
  "confidence": 0,
  "schema_ok": false,
  "coverage_ok": false,
  "evidence_ok": false,
  "format_ok": false,
  "expected_row_rule": null,
  "observed_row_count": null,
  "likely_missing_scope": ["..."],
  "weak_columns": ["..."],
  "suspicious_cells_or_rows": ["..."],
  "repair_instruction": "concrete instruction for a repair pass, or empty string",
  "suggested_search_queries": ["..."]
}}

Original question:
{_truncate_text(question, 12000)}

Draft answer:
{_truncate_text(draft_answer, 20000)}

Existing evidence:
{_truncate_text(evidence, 12000)}
"""


def build_direct_repair_prompt(question: str, draft_answer: str, evidence: str, audit: dict) -> str:
    audit_json = json.dumps(audit, ensure_ascii=False, indent=2)
    return f"""\
The previous answer is a draft and may be incomplete. You do not have access to any gold answer.

Repair the draft using only the original question, the draft, the audit report, existing evidence, and any additional web research you perform.

Rules:
- Do not restart from scratch unless the audit says the row universe is wrong.
- Preserve well-supported rows and cells.
- Focus on missing rows, weak columns, vague cells, unsupported numeric/date/URL/fee/ranking values, and exact formatting.
- If more research is needed, call search or visit.
- When ready, call finish with exactly one Markdown table and no extra commentary.

Original question:
{_truncate_text(question, 12000)}

Draft answer:
{_truncate_text(draft_answer, 20000)}

Gold-free audit report:
{audit_json}

Existing evidence:
{_truncate_text(evidence, 12000)}
"""


async def audit_direct_repair_answer(
    *,
    question: str,
    draft_answer: str,
    evidence: str,
    judge_client,
    judge_model: str,
    max_tokens: int,
) -> dict:
    prompt = build_direct_repair_audit_prompt(question, draft_answer, evidence)
    response = await judge_client.chat.completions.create(
        model=judge_model,
        messages=[
            {
                "role": "system",
                "content": "You audit table answers for completeness and formatting. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content or ""
    parsed = _extract_json_object(raw)
    if parsed is None:
        return {
            "ready_to_finish": True,
            "audit_parse_error": True,
            "audit_raw": raw[:4000],
            "repair_instruction": "",
        }
    parsed["ready_to_finish"] = _coerce_audit_bool(parsed.get("ready_to_finish"))
    parsed["audit_raw"] = raw[:4000]
    return parsed


async def maybe_run_direct_repair_retry(
    *,
    spec: DatasetSpec,
    sample: EvalSample,
    config: dict,
    main_agent,
    predicted_answer: str,
    evidence: str,
    confidence: str,
    judge_client,
    judge_model: str,
    repair_judge_client=None,
    repair_judge_model: str = "",
    pretty,
    case_key: str,
) -> Tuple[str, str, str, dict]:
    audit_client = repair_judge_client or judge_client
    audit_model = repair_judge_model or judge_model
    repair_meta = {
        "enabled": direct_repair_enabled_for_dataset(spec, config),
        "attempted": False,
        "repaired": False,
        "judge_model": audit_model,
    }
    if not repair_meta["enabled"]:
        return predicted_answer, evidence, confidence, repair_meta

    try:
        audit = await audit_direct_repair_answer(
            question=sample.question,
            draft_answer=predicted_answer,
            evidence=evidence,
            judge_client=audit_client,
            judge_model=audit_model,
            max_tokens=int(config.get("direct_repair_audit_max_tokens") or 4096),
        )
    except Exception as e:
        repair_meta.update({
            "audit_error_type": type(e).__name__,
            "audit_error_message": str(e),
        })
        return predicted_answer, evidence, confidence, repair_meta

    repair_meta["audit"] = audit
    if audit.get("ready_to_finish", True):
        repair_meta["skipped_reason"] = "audit_ready_to_finish"
        return predicted_answer, evidence, confidence, repair_meta

    repair_meta["attempted"] = True
    repair_prompt = build_direct_repair_prompt(
        sample.question,
        predicted_answer,
        evidence,
        audit,
    )
    repair_meta["first_predicted_preview"] = _truncate_text(predicted_answer, 12000)
    repair_meta["first_evidence_preview"] = _truncate_text(evidence, 8000)
    repair_meta["repair_prompt_preview"] = repair_prompt[:4000]
    try:
        pretty.warning(f"direct repair retry dataset={spec.name} idx={sample.idx}", case_key)
        repaired_answer, repaired_evidence, repaired_confidence = await main_agent.continue_with_user_message(
            repair_prompt,
            sample.idx,
        )
    except Exception as e:
        repair_meta.update({
            "repair_error_type": type(e).__name__,
            "repair_error_message": str(e),
        })
        return predicted_answer, evidence, confidence, repair_meta

    if str(repaired_answer or "").strip():
        repair_meta["repaired"] = True
        return repaired_answer, repaired_evidence, repaired_confidence, repair_meta
    repair_meta["skipped_reason"] = "empty_repair_answer"
    return predicted_answer, evidence, confidence, repair_meta
