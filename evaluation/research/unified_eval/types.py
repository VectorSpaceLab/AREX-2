from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetSpec:
    name: str
    data_path: str
    data_format: str
    task_type: str
    question_field: str
    answer_field: str
    id_field: Optional[str] = None
    encrypted: bool = False
    encryption_scheme: str = "browsecomp_sha256_xor"
    canary_field: str = "canary"
    metadata_fields: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=lambda: ["search", "google_scholar", "visit", "finish"])
    tool_definitions: List[Dict[str, Any]] = field(default_factory=list)
    scorer: Dict[str, Any] = field(default_factory=dict)
    prompts: Dict[str, Any] = field(default_factory=dict)
    prompt_source: Optional[str] = None
    generation: Dict[str, Any] = field(default_factory=dict)
    gold_root: Optional[str] = None
    criteria_path: Optional[str] = None
    reference_path: Optional[str] = None
    attachments_root: Optional[str] = None
    include_ids_path: Optional[str] = None
    row_filters: Dict[str, Any] = field(default_factory=dict)
    exclude_image_rows: bool = False
    leak_filter: Optional[str] = None
    evaluation_backend: str = "unified"


@dataclass
class EvalSample:
    dataset_name: str
    sample_id: str
    idx: int
    question: str
    answer: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    attachments: List[str] = field(default_factory=list)


@dataclass
class ScoreResult:
    status: str
    score: Optional[float]
    official_scorer: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    judge_raw: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "score": self.score,
            "official_scorer": self.official_scorer,
            "metrics": self.metrics,
            "judge_raw": self.judge_raw,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


class ScorerUnavailable(RuntimeError):
    pass
