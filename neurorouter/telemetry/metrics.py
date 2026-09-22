"""Pure presentation metrics derived from persisted trace records."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from neurorouter.schemas.trace import StageEvent, TraceRecord


class TelemetryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProbabilityMetric(TelemetryModel):
    label: str
    probability: float = Field(ge=0, le=1)


class TimelineItem(TelemetryModel):
    stage: str
    component: str
    status: str
    started_at: datetime
    ended_at: datetime
    offset_ms: float = Field(ge=0)
    duration_ms: float = Field(ge=0)
    dependencies: list[str]


class DashboardMetrics(TelemetryModel):
    total_latency_ms: float
    stage_count: int
    failed_stage_count: int
    agent_count: int
    retry_count: int
    total_tokens: int | None = None


def capability_probabilities(trace: TraceRecord) -> list[ProbabilityMetric]:
    """Return normalized Noul signals in a stable dashboard order."""
    decision = trace.routing_decision or {}
    fields = (
        ("Web", "needs_web"),
        ("RAG", "needs_rag"),
        ("Code", "needs_code"),
        ("Data analysis", "needs_data_analysis"),
        ("Current information", "needs_current_information"),
        ("Citations", "needs_citations"),
        ("Multi-source research", "needs_multi_source_research"),
    )
    return [
        ProbabilityMetric(label=label, probability=float(decision.get(field, 0.0)))
        for label, field in fields
    ]


def timeline_items(events: list[StageEvent]) -> list[TimelineItem]:
    """Convert absolute event timestamps to millisecond offsets for Plotly."""
    completed = [event for event in events if event.ended_at is not None]
    if not completed:
        return []
    origin = min(event.started_at for event in completed)
    return [
        TimelineItem(
            stage=event.stage,
            component=event.component,
            status=event.status.value,
            started_at=event.started_at,
            ended_at=event.ended_at,  # type: ignore[arg-type]
            offset_ms=max((event.started_at - origin).total_seconds() * 1000, 0.0),
            duration_ms=event.duration_ms or 0.0,
            dependencies=event.dependencies,
        )
        for event in completed
    ]


def dashboard_metrics(trace: TraceRecord, events: list[StageEvent]) -> DashboardMetrics:
    total_tokens = trace.token_usage.get("total_tokens")
    return DashboardMetrics(
        total_latency_ms=trace.total_latency_ms or 0.0,
        stage_count=len(events),
        failed_stage_count=sum(event.status.value == "failed" for event in events),
        agent_count=len(trace.agent_executions),
        retry_count=trace.retry_count,
        total_tokens=total_tokens,
    )


def execution_graph_dot(trace: TraceRecord, events: list[StageEvent]) -> str:
    """Build a compact dependency graph using only persisted execution data."""
    stage_status = {event.stage: event.status.value for event in events}
    component_status = {
        event.component: event.status.value for event in events if event.stage == "agent_execution"
    }
    plan: dict[str, Any] = trace.execution_plan or {}
    agents = [str(agent) for agent in plan.get("agents", [])]
    dependencies = plan.get("agent_dependencies", {})

    lines = [
        "digraph NeuroRouter {",
        'graph [bgcolor="transparent", rankdir=LR, pad="0.2", nodesep="0.35"]',
        'node [shape=box, style="rounded,filled", color="#334155", '
        'fontcolor="#e2e8f0", fontname="Arial"]',
        'edge [color="#64748b", arrowsize="0.7"]',
    ]

    def node(node_id: str, label: str, status: str = "succeeded") -> None:
        colors = {
            "succeeded": "#123d39",
            "failed": "#4a2028",
            "skipped": "#3b3548",
            "running": "#173b5e",
            "review": "#4a3b16",
            "accepted": "#123d39",
        }
        fill = colors.get(status, "#172033")
        lines.append(f'"{_dot(node_id)}" [label="{_dot(label)}", fillcolor="{fill}"]')

    node("user", "USER")
    node("state", "STATE", stage_status.get("state_building", "running"))
    node("jev", "JEV ROUTER", stage_status.get("jev_routing", "running"))
    node("policy", "POLICY", stage_status.get("policy_planning", "running"))
    lines.extend(['"user" -> "state"', '"state" -> "jev"', '"jev" -> "policy"'])

    for agent in agents:
        node(agent, agent, component_status.get(agent, "running"))
        agent_dependencies = [str(item) for item in dependencies.get(agent, [])]
        if agent_dependencies:
            for dependency in agent_dependencies:
                lines.append(f'"{_dot(dependency)}" -> "{_dot(agent)}"')
        else:
            lines.append(f'"policy" -> "{_dot(agent)}"')

    node("aggregate", "CONTEXT", stage_status.get("context_aggregation", "running"))
    for agent in agents:
        lines.append(f'"{_dot(agent)}" -> "aggregate"')
    node("synthesis", "SYNTHESIS", stage_status.get("synthesis", "running"))
    lines.append('"aggregate" -> "synthesis"')
    if plan.get("quality_gate_required"):
        node("quality", "QUALITY GATE", stage_status.get("quality_gate", "running"))
        lines.append('"synthesis" -> "quality"')
        previous = "quality"
    else:
        previous = "synthesis"
    node("outcome", trace.status.value.upper(), trace.status.value)
    lines.append(f'"{previous}" -> "outcome"')
    lines.append("}")
    return "\n".join(lines)


def _dot(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
