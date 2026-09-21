"""Deterministic execution-plan contract used from Phase 3 onward."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ModelTier(StrEnum):
    FAST = "fast"
    STANDARD = "standard"
    REASONING = "reasoning"


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agents: list[str] = Field(default_factory=list)
    parallel_agents: list[list[str]] = Field(default_factory=list)
    model_tier: ModelTier = ModelTier.FAST
    use_web: bool = False
    use_rag: bool = False
    use_code: bool = False
    use_data_analysis: bool = False
    citations_required: bool = False
    quality_gate_required: bool = True
    requires_review: bool = False
    reasoning: list[str] = Field(default_factory=list)
