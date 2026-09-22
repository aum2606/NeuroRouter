"""Validated contracts for routing datasets, predictions, and measured reports."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neurorouter.schemas.routing import Intent, RoutingDecision


class EvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoutingEvalCase(EvalModel):
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_intent: Intent
    expected_needs_web: bool
    expected_needs_rag: bool
    expected_needs_code: bool
    expected_needs_data: bool
    expected_complexity: int | None = Field(default=None, ge=0, le=3)
    attachment_types: list[str] = Field(default_factory=list)
    indexed_document_count: int = Field(default=0, ge=0)


class RoutingPrediction(EvalModel):
    example_id: str
    decision: RoutingDecision | None = None
    observed_latency_ms: float = Field(ge=0)
    llm_calls: int = Field(default=0, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @model_validator(mode="after")
    def require_decision_or_error(self) -> "RoutingPrediction":
        if (self.decision is None) == (self.error is None):
            raise ValueError("prediction must contain exactly one of decision or error")
        return self


class BinaryMetrics(EvalModel):
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)
    brier_score: float = Field(ge=0, le=1)


class CalibrationBin(EvalModel):
    lower_bound: float = Field(ge=0, le=1)
    upper_bound: float = Field(ge=0, le=1)
    mean_probability: float = Field(ge=0, le=1)
    observed_rate: float = Field(ge=0, le=1)
    count: int = Field(ge=1)


class RouterMetrics(EvalModel):
    evaluated_examples: int = Field(ge=0)
    failed_examples: int = Field(ge=0)
    intent_accuracy: float = Field(ge=0, le=1)
    intent_confusion_matrix: dict[str, dict[str, int]]
    capabilities: dict[str, BinaryMetrics]
    macro_precision: float = Field(ge=0, le=1)
    macro_recall: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    mean_brier_score: float = Field(ge=0, le=1)
    calibration: dict[str, list[CalibrationBin]]
    complexity_mae: float | None = Field(default=None, ge=0)
    complexity_accuracy: float | None = Field(default=None, ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)
    total_llm_calls: int = Field(ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    unnecessary_tool_activations: int = Field(ge=0)


class EvaluationReport(EvalModel):
    run_id: UUID = Field(default_factory=uuid4)
    router_name: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime
    predictions: list[RoutingPrediction]
    metrics: RouterMetrics


class ComparisonMetric(EvalModel):
    metric: str
    baseline: float | int | None
    candidate: float | int | None
    delta: float | int | None
    lower_is_better: bool = False


class RouterComparison(EvalModel):
    baseline_run_id: UUID
    candidate_run_id: UUID
    baseline_name: str
    candidate_name: str
    metrics: list[ComparisonMetric]
