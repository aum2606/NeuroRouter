"""Inspect persisted routing, execution, and quality-control traces."""

import plotly.graph_objects as go
import streamlit as st

from neurorouter.schemas.trace import TraceRecord
from neurorouter.telemetry import TraceRepository, execution_graph_dot, timeline_items
from neurorouter.utils.config import PROJECT_ROOT, load_settings


def _distribution_chart(trace: TraceRecord) -> go.Figure:
    decision = trace.routing_decision or {}
    intent = decision.get("intent_probabilities", {})
    complexity = decision.get("complexity_probabilities", {})
    risk = decision.get("risk_probabilities", {})
    figure = go.Figure()
    for name, values, color in (
        ("Intent", intent, "#55e6c1"),
        ("Complexity", complexity, "#60a5fa"),
        ("Risk", risk, "#f7c65d"),
    ):
        figure.add_trace(
            go.Bar(
                name=name,
                x=[f"{name}: {key}" for key in values],
                y=[float(value) for value in values.values()],
                marker_color=color,
                hovertemplate="%{x}<br>%{y:.1%}<extra></extra>",
            )
        )
    figure.update_layout(
        height=330,
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1],
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        showlegend=False,
    )
    return figure


def _timeline_chart(trace: TraceRecord, repository: TraceRepository) -> go.Figure:
    items = timeline_items(repository.list_stage_events(trace.trace_id))
    colors = {"succeeded": "#55e6c1", "failed": "#ff7185", "skipped": "#a78bfa"}
    figure = go.Figure()
    for item in items:
        figure.add_trace(
            go.Bar(
                x=[max(item.duration_ms, 0.2)],
                y=[f"{item.component} · {item.stage}"],
                base=[item.offset_ms],
                orientation="h",
                marker_color=colors.get(item.status, "#60a5fa"),
                hovertemplate=f"{item.duration_ms:.2f} ms<extra></extra>",
                showlegend=False,
            )
        )
    figure.update_layout(
        barmode="overlay",
        height=max(300, len(items) * 42),
        xaxis_title="Milliseconds from trace start",
        margin=dict(l=10, r=10, t=15, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
    )
    return figure


st.set_page_config(page_title="Trace Explorer - NeuroRouter", page_icon="🔎", layout="wide")
st.title("🔎 Trace Explorer")
st.caption("Search, compare, and inspect every persisted control-plane decision.")

settings = load_settings()
repository = TraceRepository(PROJECT_ROOT / settings.telemetry.database_path)
traces = repository.list_traces(limit=500)

if not traces:
    st.info("No traces yet. Run a request from Command Center first.")
    st.stop()

filter_col, status_col, route_col = st.columns([2, 1, 1])
query_filter = filter_col.text_input("Search request", placeholder="Filter query text...")
statuses = sorted({trace.status.value for trace in traces})
selected_statuses = status_col.multiselect("Status", statuses, default=statuses)
routes = sorted({trace.route for trace in traces if trace.route})
selected_routes = route_col.multiselect("Route", routes, default=routes)

filtered = [
    trace
    for trace in traces
    if trace.status.value in selected_statuses
    and (not trace.route or trace.route in selected_routes)
    and query_filter.casefold() in trace.request_text.casefold()
]

st.dataframe(
    [
        {
            "trace_id": str(trace.trace_id),
            "query": trace.request_text,
            "route": trace.route or "—",
            "status": trace.status.value,
            "latency_ms": round(trace.total_latency_ms or 0, 2),
            "retries": trace.retry_count,
            "timestamp": trace.created_at,
        }
        for trace in filtered
    ],
    use_container_width=True,
    hide_index=True,
)

if not filtered:
    st.warning("No traces match the current filters.")
    st.stop()

selected = st.selectbox(
    "Inspect trace",
    filtered,
    format_func=lambda trace: (
        f"{str(trace.trace_id)[:8]} · {trace.status.value.upper()} · {trace.request_text[:75]}"
    ),
)
events = repository.list_stage_events(selected.trace_id)

st.divider()
summary = st.columns(6)
summary[0].metric("Status", selected.status.value.upper())
summary[1].metric("Intent", (selected.route or "unknown").replace("_", " ").title())
summary[2].metric("Latency", f"{selected.total_latency_ms or 0:,.0f} ms")
summary[3].metric("Model tier", (selected.model_tier or "n/a").upper())
summary[4].metric("Agents", len(selected.agent_executions))
summary[5].metric("Retries", selected.retry_count)

graph_col, probabilities_col = st.columns([1.25, 1], gap="large")
with graph_col:
    st.subheader("Execution graph")
    st.graphviz_chart(execution_graph_dot(selected, events), use_container_width=True)
with probabilities_col:
    st.subheader("Jev distributions")
    st.plotly_chart(_distribution_chart(selected), use_container_width=True)

st.subheader("Execution timeline")
st.plotly_chart(_timeline_chart(selected, repository), use_container_width=True)

agents_tab, quality_tab, response_tab, debug_tab = st.tabs(
    ["Agent executions", "Quality gate", "Responses", "Raw debug"]
)
with agents_tab:
    if selected.agent_executions:
        st.dataframe(
            [
                {
                    "agent": item.get("agent_name"),
                    "success": item.get("success"),
                    "latency_ms": round(float(item.get("latency_ms", 0)), 2),
                    "evidence": len(item.get("evidence", [])),
                    "sources": len(item.get("sources", [])),
                    "error": (item.get("error") or {}).get("message"),
                }
                for item in selected.agent_executions
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No agent execution snapshots were recorded for this trace.")
with quality_tab:
    if selected.quality_gate_result:
        st.json(selected.quality_gate_result)
    else:
        st.info("The deterministic plan did not require a quality gate.")
with response_tab:
    st.markdown("**Candidate synthesis**")
    st.write(selected.synthesis_result or "Not recorded")
    st.markdown("**Final response**")
    st.write(selected.final_response or "Not recorded")
with debug_tab:
    with st.expander("Validated state"):
        st.json(selected.state or {})
    with st.expander("Normalized routing decision"):
        st.json(selected.routing_decision or {})
    with st.expander("Execution plan"):
        st.json(selected.execution_plan or {})
    with st.expander("Raw Jev response"):
        st.json(selected.raw_jev_response or {})
    with st.expander("Stage events"):
        st.json([event.model_dump(mode="json") for event in events])
