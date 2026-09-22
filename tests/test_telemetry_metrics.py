from datetime import UTC, datetime, timedelta

from neurorouter.schemas.trace import StageEvent, StageStatus, TraceRecord, TraceStatus
from neurorouter.telemetry.metrics import (
    capability_probabilities,
    dashboard_metrics,
    execution_graph_dot,
    timeline_items,
)


def _trace() -> TraceRecord:
    return TraceRecord(
        request_text="Build a router",
        status=TraceStatus.ACCEPTED,
        total_latency_ms=125.0,
        retry_count=1,
        route="coding",
        routing_decision={"needs_web": 0.2, "needs_code": 0.91},
        execution_plan={
            "agents": ["CodeAgent"],
            "agent_dependencies": {},
            "quality_gate_required": True,
        },
        agent_executions=[{"agent_name": "CodeAgent"}],
        token_usage={"total_tokens": 55},
    )


def _events(trace: TraceRecord) -> list[StageEvent]:
    started = datetime.now(UTC)
    return [
        StageEvent(
            trace_id=trace.trace_id,
            stage="agent_execution",
            component="CodeAgent",
            status=StageStatus.SUCCEEDED,
            started_at=started,
            ended_at=started + timedelta(milliseconds=20),
            duration_ms=20,
        ),
        StageEvent(
            trace_id=trace.trace_id,
            stage="quality_gate",
            component="JevQualityGate",
            status=StageStatus.SUCCEEDED,
            started_at=started + timedelta(milliseconds=25),
            ended_at=started + timedelta(milliseconds=35),
            duration_ms=10,
        ),
    ]


def test_dashboard_metrics_use_only_measured_trace_values() -> None:
    trace = _trace()
    metrics = dashboard_metrics(trace, _events(trace))

    assert metrics.total_latency_ms == 125
    assert metrics.agent_count == 1
    assert metrics.retry_count == 1
    assert metrics.total_tokens == 55


def test_probability_and_timeline_rows_are_stable() -> None:
    trace = _trace()
    probabilities = capability_probabilities(trace)
    timeline = timeline_items(_events(trace))

    assert {item.label: item.probability for item in probabilities}["Code"] == 0.91
    assert timeline[0].offset_ms == 0
    assert timeline[1].offset_ms == 25


def test_execution_graph_reflects_plan_and_outcome() -> None:
    trace = _trace()
    graph = execution_graph_dot(trace, _events(trace))

    assert '"policy" -> "CodeAgent"' in graph
    assert '"CodeAgent" -> "aggregate"' in graph
    assert '"quality" -> "outcome"' in graph
    assert "ACCEPTED" in graph
