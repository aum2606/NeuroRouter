"""Pydantic telemetry records shared by storage and UI layers."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class TraceStatus(StrEnum):
    RUNNING = "running"
    ACCEPTED = "accepted"
    REVIEW = "review"
    FAILED = "failed"


class StageStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class TraceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: UUID = Field(default_factory=uuid4)
    request_text: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: TraceStatus = TraceStatus.RUNNING
    route: str | None = None
    total_latency_ms: float | None = Field(default=None, ge=0.0)
    retry_count: int = Field(default=0, ge=0)
    state: dict[str, Any] | None = None
    routing_decision: dict[str, Any] | None = None
    raw_jev_response: dict[str, Any] | None = None
    execution_plan: dict[str, Any] | None = None
    synthesis_result: str | None = None
    quality_gate_result: dict[str, Any] | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    model_tier: str | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    agent_executions: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    final_response: str | None = None
    error: str | None = None


class StageEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: UUID = Field(default_factory=uuid4)
    trace_id: UUID
    stage: str = Field(min_length=1)
    component: str = Field(min_length=1)
    status: StageStatus
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0.0)
    dependencies: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)
    error: str | None = None
