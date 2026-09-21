"""Normalized probabilistic outputs from the future Jev routing adapter."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Intent(StrEnum):
    GENERAL_QA = "general_qa"
    RESEARCH = "research"
    DOCUMENT_QA = "document_qa"
    CODING = "coding"
    DATA_ANALYSIS = "data_analysis"
    FINANCE = "finance"
    MIXED = "mixed"


class RoutingDecision(BaseModel):
    """Provider-neutral routing values consumed by deterministic policy."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    intent_confidence: float = Field(ge=0.0, le=1.0)
    intent_probabilities: dict[Intent, float]

    needs_web: float = Field(ge=0.0, le=1.0)
    needs_rag: float = Field(ge=0.0, le=1.0)
    needs_code: float = Field(ge=0.0, le=1.0)
    needs_data_analysis: float = Field(ge=0.0, le=1.0)
    needs_current_information: float = Field(ge=0.0, le=1.0)
    needs_citations: float = Field(ge=0.0, le=1.0)
    needs_multi_source_research: float = Field(ge=0.0, le=1.0)

    complexity_score: float = Field(ge=0, le=3)
    complexity_confidence: float = Field(ge=0.0, le=1.0)
    complexity_probabilities: dict[int, float]
    risk_score: float = Field(ge=0, le=3)
    risk_confidence: float = Field(ge=0.0, le=1.0)
    risk_probabilities: dict[int, float]

    jev_latency_ms: float = Field(ge=0.0)
    jev_model: str | None = None

    @model_validator(mode="after")
    def validate_distributions(self) -> "RoutingDecision":
        self._validate_distribution(self.intent_probabilities, "intent")
        self._validate_distribution(self.complexity_probabilities, "complexity")
        self._validate_distribution(self.risk_probabilities, "risk")
        if self.intent not in self.intent_probabilities:
            raise ValueError("intent must be present in intent_probabilities")
        return self

    @staticmethod
    def _validate_distribution(values: dict[Any, float], name: str) -> None:
        if not values:
            raise ValueError(f"{name}_probabilities cannot be empty")
        if any(value < 0.0 or value > 1.0 for value in values.values()):
            raise ValueError(f"{name}_probabilities values must be between 0 and 1")
        if abs(sum(values.values()) - 1.0) > 0.02:
            raise ValueError(f"{name}_probabilities must sum to approximately 1")


class JevRoutingResult(BaseModel):
    """Normalized decision plus separately retained provider response for debugging."""

    model_config = ConfigDict(extra="forbid")

    decision: RoutingDecision
    raw_response: dict[str, Any] = Field(default_factory=dict)
