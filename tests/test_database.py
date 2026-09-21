from datetime import UTC, datetime

import pytest

from neurorouter.schemas.trace import StageEvent, StageStatus, TraceRecord, TraceStatus
from neurorouter.telemetry.database import TraceRepository
from neurorouter.telemetry.models import TraceUpdate


def test_trace_round_trip(repository: TraceRepository) -> None:
    created = repository.create_trace(
        TraceRecord(request_text="Explain the architecture", state={"safe": True})
    )
    updated = repository.update_trace(
        created.trace_id,
        TraceUpdate(status=TraceStatus.ACCEPTED, route="general_qa", final_response="Done"),
    )

    assert updated.status is TraceStatus.ACCEPTED
    assert updated.state == {"safe": True}
    assert repository.list_traces()[0].trace_id == created.trace_id


def test_stage_event_round_trip(repository: TraceRepository) -> None:
    trace = repository.create_trace(TraceRecord(request_text="Trace this"))
    event = StageEvent(
        trace_id=trace.trace_id,
        stage="state_building",
        component="StateBuilder",
        status=StageStatus.SUCCEEDED,
        started_at=datetime.now(UTC),
        dependencies=["request"],
        metadata={"messages": 2},
    )
    repository.add_stage_event(event)

    stored = repository.list_stage_events(trace.trace_id)

    assert stored == [event]


def test_unknown_trace_update_fails(repository: TraceRepository) -> None:
    with pytest.raises(KeyError):
        repository.update_trace("00000000-0000-0000-0000-000000000000", TraceUpdate(route="x"))
