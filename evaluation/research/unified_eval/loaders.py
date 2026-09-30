import base64
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from .types import DatasetSpec, EvalSample, ScorerUnavailable


def _derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


def decrypt_xbench(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = password.encode("utf-8")
    if not key:
        raise ValueError("xBench canary is empty")
    return bytes(encrypted[i] ^ key[i % len(key)] for i in range(len(encrypted))).decode("utf-8")


def decrypt_by_scheme(ciphertext_b64: str, password: str, scheme: str) -> str:
    if scheme == "browsecomp_sha256_xor":
        return decrypt(ciphertext_b64, password)
    if scheme == "xbench_xor":
        return decrypt_xbench(ciphertext_b64, password)
    raise ValueError(f"Unsupported encryption scheme: {scheme}")


def _load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON root type in {path}: {type(data).__name__}")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_parquet(path: str) -> List[Dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except Exception as pandas_error:
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as pyarrow_error:
            raise ScorerUnavailable(
                "Parquet loading requires pandas or pyarrow. "
                f"pandas_error={type(pandas_error).__name__}: {pandas_error}; "
                f"pyarrow_error={type(pyarrow_error).__name__}: {pyarrow_error}"
            )
        table = pq.read_table(path)
        return table.to_pylist()
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def _stringify_answer(answer: Any) -> str:
    if isinstance(answer, str):
        return answer
    if isinstance(answer, (list, tuple)):
        return ", ".join(str(item) for item in answer)
    if answer is None:
        return ""
    return str(answer)


def _extract_monaco_answer(payload: Dict[str, Any]) -> Any:
    non_canary_keys = [key for key in payload.keys() if key != "canary"]
    if not non_canary_keys:
        return "", {}
    question = non_canary_keys[0]
    body = payload.get(question) or {}
    provenance = body.get("provenance") or {}
    if not provenance:
        return "", {}
    last_key = sorted(provenance.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))[-1]
    entries = provenance.get(last_key) or []
    answers = []
    for entry in entries:
        for answer in entry.get("answers") or []:
            if answer not in answers:
                answers.append(answer)
    return answers, {"question": question, "body": body}


def _load_monaco_main_benchmark(path: str) -> List[Dict[str, Any]]:
    trace_root = Path(path)
    dataset_root = trace_root if trace_root.is_dir() and trace_root.name != "execution_traces" else trace_root.parent
    candidates = [
        dataset_root / "monaco_version_1_release.jsonl",
        dataset_root / "monaco_version_1_release.json",
    ]
    main_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if main_path is None:
        return []
    rows = _load_jsonl(str(main_path)) if main_path.suffix == ".jsonl" else _load_json(str(main_path))
    normalized = []
    for idx, row in enumerate(rows):
        question = row.get("question")
        answer = row.get("validated_answer", row.get("answer", row.get("answers", "")))
        if not question:
            continue
        ex_num = row.get("ex_num", row.get("id", idx))
        trace_path = trace_root / f"dataset_ex_{ex_num}.json"
        out = dict(row)
        out.update({
            "trace_path": str(trace_path) if trace_path.exists() else str(ex_num),
            "question": question,
            "answer": answer,
            "canary": row.get("canary", ""),
            "source_file": str(main_path),
        })
        normalized.append(out)
    return normalized


def _is_data_uri(value: str) -> bool:
    return value.lower().startswith("data:")


def _has_attachment_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in ("none", "nan", "null")


def _load_monaco_traces(path: str) -> List[Dict[str, Any]]:
    main_rows = _load_monaco_main_benchmark(path)
    if main_rows:
        return main_rows
    rows = []
    for trace_path in sorted(Path(path).glob("dataset_ex_*.json")):
        with open(trace_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        answers, meta = _extract_monaco_answer(payload)
        if not meta:
            continue
        rows.append({
            "trace_path": str(trace_path),
            "question": meta["question"],
            "answer": answers,
            "canary": payload.get("canary", ""),
        })
    return rows


def load_raw_rows(spec: DatasetSpec) -> List[Dict[str, Any]]:
    fmt = spec.data_format.lower()
    if fmt == "csv":
        rows = _load_csv(spec.data_path)
    elif fmt == "json":
        rows = _load_json(spec.data_path)
    elif fmt == "jsonl":
        rows = _load_jsonl(spec.data_path)
    elif fmt == "parquet":
        rows = _load_parquet(spec.data_path)
    elif fmt == "monaco_traces":
        rows = _load_monaco_traces(spec.data_path)
    else:
        raise ValueError(f"Unsupported data_format for {spec.name}: {spec.data_format}")
    rows = _filter_rows_by_row_filters(spec, rows)
    rows = _filter_rows_by_image_presence(spec, rows)
    return _filter_rows_by_include_ids(spec, rows)


def _row_id(row: Dict[str, Any], id_field: str) -> str:
    value = row.get(id_field)
    return "" if value is None else str(value)


def _load_include_ids(path: str) -> Set[str]:
    ids: Set[str] = set()
    _, ext = os.path.splitext(path.lower())
    if ext == ".jsonl":
        for row in _load_jsonl(path):
            value = row.get("task_id", row.get("id"))
            if value is not None:
                ids.add(str(value))
        return ids
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            values = payload.get("task_ids", payload.get("ids", [payload]))
        else:
            values = payload
        if not isinstance(values, list):
            values = [values]
        for item in values:
            if isinstance(item, dict):
                value = item.get("task_id", item.get("id"))
            else:
                value = item
            if value is not None:
                ids.add(str(value))
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                ids.add(value)
    return ids


def _filter_rows_by_include_ids(spec: DatasetSpec, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not spec.include_ids_path:
        return rows
    if not spec.id_field:
        raise ValueError(f"{spec.name} sets include_ids_path but has no id_field")
    include_ids = _load_include_ids(spec.include_ids_path)
    return [row for row in rows if _row_id(row, spec.id_field) in include_ids]


def _matches_row_filter(row_value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_matches_row_filter(row_value, item) for item in expected)
    if row_value is None:
        return expected is None
    return str(row_value) == str(expected)


def _filter_rows_by_row_filters(spec: DatasetSpec, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not spec.row_filters:
        return rows
    return [
        row
        for row in rows
        if all(_matches_row_filter(row.get(key), expected) for key, expected in spec.row_filters.items())
    ]


def _has_image_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        if isinstance(value.get("bytes"), (bytes, bytearray)):
            return True
        return any(_has_image_value(value.get(key)) for key in ("url", "image_url", "path", "image_path"))
    if isinstance(value, (list, tuple)):
        return any(_has_image_value(item) for item in value)
    return False


def _row_has_image(row: Dict[str, Any]) -> bool:
    for key in ("image", "image_url", "image_path", "images"):
        if key in row and _has_image_value(row[key]):
            return True
    return False


def _filter_rows_by_image_presence(spec: DatasetSpec, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not spec.exclude_image_rows:
        return rows
    return [row for row in rows if not _row_has_image(row)]


def rows_to_samples(spec: DatasetSpec, rows: Iterable[Dict[str, Any]], no_decrypt: bool = False) -> List[EvalSample]:
    samples = []
    for idx, row in enumerate(rows):
        row = dict(row)
        if spec.encrypted and not no_decrypt:
            canary = row.get(spec.canary_field, "")
            question = decrypt_by_scheme(str(row.get(spec.question_field, "")), str(canary), spec.encryption_scheme)
            answer = decrypt_by_scheme(str(row.get(spec.answer_field, "")), str(canary), spec.encryption_scheme)
        else:
            question = row.get(spec.question_field, "")
            answer = row.get(spec.answer_field, "")

        sample_id = str(row.get(spec.id_field) if spec.id_field else idx)
        if sample_id == "None":
            sample_id = str(idx)

        metadata = {}
        for key in spec.metadata_fields:
            if key in row:
                value = row[key]
                if isinstance(value, str) and _is_data_uri(value):
                    metadata[key] = value.split(",", 1)[0] + ",[omitted]"
                else:
                    metadata[key] = value
        attachments = []
        if spec.attachments_root:
            for key in ("file_path", "file_name", "image"):
                value = row.get(key)
                if not _has_attachment_value(value):
                    continue
                value_str = str(value)
                if _is_data_uri(value_str):
                    continue
                path = os.path.join(spec.attachments_root, value_str)
                if os.path.exists(path) and path not in attachments:
                    attachments.append(path)
        samples.append(
            EvalSample(
                dataset_name=spec.name,
                sample_id=sample_id,
                idx=idx,
                question=str(question or ""),
                answer=_stringify_answer(answer),
                metadata=metadata,
                raw=row,
                attachments=attachments,
            )
        )
    return samples


def load_samples(spec: DatasetSpec, no_decrypt: bool = False) -> List[EvalSample]:
    return rows_to_samples(spec, load_raw_rows(spec), no_decrypt=no_decrypt)
