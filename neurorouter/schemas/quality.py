"""Contracts for probabilistic response evaluation and bounded retry control."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from neurorouter.schemas.context import EvidencePacket
from neurorouter.schemas.execution import AgentName


class QualityModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QualityState(QualityModel):
    original_request: str = Field(min_length=1)
    candidate_response: str = Field(min_length=1)
    supporting_evidence: EvidencePacket
    agents_used: list[AgentName] = Field(default_factory=list)
    execution_metadata: dict[str, Any] = Field(default_factory=dict)


class QualityDecision(QualityModel):
    answers_request: float = Field(ge=0, le=1)
    supported_by_evidence: float = Field(ge=0, le=1)
    contains_unsupported_claims: float = Field(ge=0, le=1)
    missing_important_information: float = Field(ge=0, le=1)
    contradicts_evidence: float = Field(ge=0, le=1)
    needs_additional_retrieval: float = Field(ge=0, le=1)
    jev_latency_ms: float = Field(ge=0)
    jev_model: str | None = None


class JevQualityResult(QualityModel):
    decision: QualityDecision
    raw_response: dict[str, Any] = Field(default_factory=dict)
    evaluation_succeeded: bool = True
    error_type: str | None = None


class QualityAction(StrEnum):
    ACCEPT = "accept"
    ADDITIONAL_RETRIEVAL = "additional_retrieval"
    REGENERATE = "regenerate"
    RECONCILE = "reconcile"
    REVIEW = "review"


class QualityPolicyDecision(QualityModel):
    action: QualityAction
    accepted: bool = False
    requires_review: bool = False
    reasoning: list[str] = Field(default_factory=list)


class QualityAttempt(QualityModel):
    attempt_number: int = Field(ge=0)
    candidate_response: str
    quality: JevQualityResult
    policy: QualityPolicyDecision


class QualityStatus(StrEnum):
    ACCEPTED = "accepted"
    REVIEW = "review"


class QualityOutcome(QualityModel):
    final_response: str
    status: QualityStatus
    retry_count: int = Field(ge=0)
    attempts: list[QualityAttempt] = Field(default_factory=list)
    uncertainty_note: str | None = None
    controller_errors: list[str] = Field(default_factory=list)
