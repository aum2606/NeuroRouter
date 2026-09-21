"""Common structured contract and failure boundary for specialist agents."""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neurorouter.schemas.execution import AgentName, ExecutionPlan
from neurorouter.schemas.state import RouterState


class AgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TokenUsage(AgentModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class Source(AgentModel):
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evidence(AgentModel):
    evidence_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_id: str | None = None
    relevance_score: float = Field(ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentError(AgentModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    error_type: str
    retriable: bool = False


class AgentInput(AgentModel):
    trace_id: UUID
    request_text: str = Field(min_length=1)
    state: RouterState
    plan: ExecutionPlan
    context: dict[str, Any] = Field(default_factory=dict)


class AgentPayload(AgentModel):
    output: str
    evidence: list[Evidence] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_usage: TokenUsage | None = None


class AgentResult(AgentModel):
    agent_name: AgentName
    output: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    ended_at: datetime
    latency_ms: float = Field(ge=0)
    token_usage: TokenUsage | None = None
    success: bool
    error: AgentError | None = None

    @model_validator(mode="after")
    def error_matches_status(self) -> "AgentResult":
        if self.success and self.error is not None:
            raise ValueError("successful agent results cannot contain an error")
        if not self.success and self.error is None:
            raise ValueError("failed agent results must contain an error")
        return self


class BaseAgent(ABC):
    """Run one bounded operation and convert exceptions into structured failures."""

    name: AgentName

    async def run(self, agent_input: AgentInput) -> AgentResult:
        started_at = datetime.now(UTC)
        started = perf_counter()
        try:
            payload = await self.execute(agent_input)
            ended_at = datetime.now(UTC)
            return AgentResult(
                agent_name=self.name,
                output=payload.output,
                evidence=payload.evidence,
                sources=payload.sources,
                metadata=payload.metadata,
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=(perf_counter() - started) * 1000,
                token_usage=payload.token_usage,
                success=True,
            )
        except Exception as error:
            ended_at = datetime.now(UTC)
            return AgentResult(
                agent_name=self.name,
                metadata={"failure_isolated": True},
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=(perf_counter() - started) * 1000,
                success=False,
                error=AgentError(
                    code="agent_execution_failed",
                    message=str(error) or type(error).__name__,
                    error_type=type(error).__name__,
                    retriable=isinstance(error, (ConnectionError, TimeoutError)),
                ),
            )

    @abstractmethod
    async def execute(self, agent_input: AgentInput) -> AgentPayload:
        """Perform the bounded specialist operation."""
        raise NotImplementedError
