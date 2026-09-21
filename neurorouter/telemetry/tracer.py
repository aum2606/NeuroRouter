"""High-level request lifecycle tracing built on the SQLite repository."""

from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import UUID

from neurorouter.schemas.trace import StageEvent, StageStatus, TraceRecord, TraceStatus
from neurorouter.telemetry.database import TraceRepository
from neurorouter.telemetry.models import TraceUpdate


class RequestTracer:
    """Create a trace immediately and safely finalize its lifecycle."""

    def __init__(self, repository: TraceRepository, request_text: str) -> None:
        self.repository = repository
        self.trace = repository.create_trace(TraceRecord(request_text=request_text))
        self._started = perf_counter()

    @property
    def trace_id(self) -> UUID:
        return self.trace.trace_id

    def record_stage(
        self,
        *,
        stage: str,
        component: str,
        status: StageStatus,
        started_at: datetime,
        metadata: dict[str, Any] | None = None,
        dependencies: list[str] | None = None,
        token_usage: dict[str, int] | None = None,
        error: str | None = None,
    ) -> StageEvent:
        ended_at = datetime.now(UTC)
        duration_ms = max((ended_at - started_at).total_seconds() * 1000, 0.0)
        event = StageEvent(
            trace_id=self.trace_id,
            stage=stage,
            component=component,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            metadata=metadata or {},
            dependencies=dependencies or [],
            token_usage=token_usage or {},
            error=error,
        )
        return self.repository.add_stage_event(event)

    def finalize(
        self,
        *,
        status: TraceStatus,
        final_response: str | None = None,
        error: str | None = None,
    ) -> TraceRecord:
        self.trace = self.repository.update_trace(
            self.trace_id,
            TraceUpdate(
                status=status,
                total_latency_ms=(perf_counter() - self._started) * 1000,
                final_response=final_response,
                error=error,
            ),
        )
        return self.trace

    def __enter__(self) -> "RequestTracer":
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        if exception is not None:
            self.finalize(status=TraceStatus.FAILED, error=str(exception))
