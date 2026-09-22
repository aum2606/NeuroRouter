from datetime import UTC, datetime

from neurorouter.schemas.trace import StageStatus, TraceStatus
from neurorouter.telemetry.database import TraceRepository
from neurorouter.telemetry.models import TraceUpdate
from neurorouter.telemetry.tracer import RequestTracer


def test_request_tracer_records_and_finalizes(repository: TraceRepository) -> None:
    tracer = RequestTracer(repository, "How should this route?")
    tracer.record_stage(
        stage="state_building",
        component="StateBuilder",
        status=StageStatus.SUCCEEDED,
        started_at=datetime.now(UTC),
    )
    final = tracer.finalize(status=TraceStatus.ACCEPTED, final_response="Accepted")

    assert final.status is TraceStatus.ACCEPTED
    assert final.total_latency_ms is not None
    assert len(repository.list_stage_events(tracer.trace_id)) == 1


def test_request_tracer_marks_context_exception_failed(repository: TraceRepository) -> None:
    tracer: RequestTracer | None = None
    try:
        with RequestTracer(repository, "Fail safely") as active_tracer:
            tracer = active_tracer
            raise RuntimeError("expected failure")
    except RuntimeError:
        pass

    assert tracer is not None
    stored = repository.get_trace(tracer.trace_id)
    assert stored is not None
    assert stored.status is TraceStatus.FAILED
    assert stored.error == "expected failure"


def test_request_tracer_persists_lifecycle_snapshot(repository: TraceRepository) -> None:
    tracer = RequestTracer(repository, "Persist the route")

    snapshot = tracer.snapshot(
        TraceUpdate(route="coding", model_tier="standard", token_usage={"total_tokens": 42})
    )

    assert snapshot.route == "coding"
    assert tracer.trace.model_tier == "standard"
    assert repository.get_trace(tracer.trace_id).token_usage == {"total_tokens": 42}
