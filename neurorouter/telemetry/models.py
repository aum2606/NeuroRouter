"""Database-facing telemetry update models."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from neurorouter.schemas.trace import TraceStatus


class TraceUpdate(BaseModel):
    """Partial mutation accepted by the trace repository."""

    model_config = ConfigDict(extra="forbid")

    status: TraceStatus | None = None
    route: str | None = None
    total_latency_ms: float | None = Field(default=None, ge=0)
    retry_count: int | None = Field(default=None, ge=0)
    state: dict[str, Any] | None = None
    routing_decision: dict[str, Any] | None = None
    raw_jev_response: dict[str, Any] | None = None
    execution_plan: dict[str, Any] | None = None
    synthesis_result: str | None = None
    quality_gate_result: dict[str, Any] | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    model_tier: str | None = None
    token_usage: dict[str, int] | None = None
    agent_executions: list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    final_response: str | None = None
    error: str | None = None
