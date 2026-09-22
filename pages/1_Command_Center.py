"""Live operations cockpit backed by the complete NeuroRouter control plane."""

import asyncio
from uuid import UUID

import plotly.graph_objects as go
import streamlit as st

from neurorouter.core.control_plane import NeuroRouterControlPlane
from neurorouter.schemas.trace import TraceRecord
from neurorouter.telemetry import (
    TraceRepository,
    capability_probabilities,
    dashboard_metrics,
    execution_graph_dot,
    timeline_items,
)
from neurorouter.utils.config import PROJECT_ROOT, load_settings


def _selected_trace(repository: TraceRepository, value: str | None) -> TraceRecord | None:
    if value:
        try:
            selected = repository.get_trace(UUID(value))
        except ValueError:
            selected = None
        if selected is not None:
            return selected
    traces = repository.list_traces(limit=1)
    return traces[0] if traces else None


def _render_decision_cards(trace: TraceRecord) -> None:
    decision = trace.routing_decision or {}
    plan = trace.execution_plan or {}
    first = st.columns(3)
    first[0].metric("Primary intent", (trace.route or "unknown").replace("_", " ").title())
    first[1].metric("Intent confidence", f"{decision.get('intent_confidence', 0):.0%}")
    first[2].metric("Model tier", (trace.model_tier or "n/a").upper())
    second = st.columns(3)
    second[0].metric("Complexity", f"{decision.get('complexity_score', 0):.2f} / 3")
    second[1].metric("Risk", f"{decision.get('risk_score', 0):.2f} / 3")
    second[2].metric("Review", "YES" if plan.get("requires_review") else "NO")


st.set_page_config(page_title="Command Center - NeuroRouter", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {background:#070b14;}
    [data-testid="stSidebar"] {background:#0b1220;}
    .nr-eyebrow {color:#55e6c1; letter-spacing:.18em; font-size:.72rem; font-weight:750;}
    .nr-panel {background:linear-gradient(145deg,#101a2b,#0d1524); border:1px solid #25334a;
      border-radius:14px; padding:1rem 1.1rem; margin:.25rem 0 .8rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

settings = load_settings()
repository = TraceRepository(PROJECT_ROOT / settings.telemetry.database_path)
control_plane = NeuroRouterControlPlane(repository)

if "nr_messages" not in st.session_state:
    st.session_state.nr_messages = []
if "nr_trace_id" not in st.session_state:
    st.session_state.nr_trace_id = None

st.markdown('<div class="nr-eyebrow">LIVE AI OPERATIONS</div>', unsafe_allow_html=True)
st.title("NeuroRouter Command Center")
st.caption("Probabilistic routing, deterministic policy, bounded execution, and quality control.")

with st.sidebar:
    st.subheader("Runtime")
    st.metric("LLM provider", settings.llm.provider.upper())
    st.metric("Jev model", settings.jev.model)
    st.caption("Billing fallback is disabled. Hosted providers must use free-tier models.")
    if not settings.telemetry.log_raw_jev_responses:
        st.info("Raw Jev response logging is disabled.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.nr_messages = []
        st.session_state.nr_trace_id = None
        st.rerun()

left, right = st.columns([1.2, 1], gap="large")
with left:
    st.subheader("Conversation")
    conversation = st.container(height=440, border=True)
    with conversation:
        if not st.session_state.nr_messages:
            st.info("Submit a request to watch the control plane build and execute its route.")
        for message in st.session_state.nr_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    with st.form("command_form", clear_on_submit=True):
        request_text = st.text_area(
            "Request",
            placeholder="Research, analyze, explain, or build something...",
            height=110,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "Route request",
            type="primary",
            use_container_width=True,
            disabled=not repository.healthcheck(),
        )

    if submitted and request_text.strip():
        prior_messages = list(st.session_state.nr_messages[-8:])
        st.session_state.nr_messages.append({"role": "user", "content": request_text.strip()})
        with st.status("Executing probabilistic control plane...", expanded=True) as status:
            st.write("Building validated request state")
            try:
                result = asyncio.run(
                    control_plane.execute(
                        request_text.strip(),
                        recent_messages=prior_messages,
                    )
                )
            except Exception as error:
                status.update(label="Request failed", state="error", expanded=True)
                st.error(f"{type(error).__name__}: {error}")
            else:
                st.write("Persisting route, execution, and quality telemetry")
                st.session_state.nr_trace_id = str(result.trace.trace_id)
                st.session_state.nr_messages.append(
                    {"role": "assistant", "content": result.pipeline.final_response}
                )
                status.update(
                    label=f"Trace {result.trace.status.value}",
                    state="complete",
                    expanded=False,
                )
                st.rerun()

trace = _selected_trace(repository, st.session_state.nr_trace_id)
events = repository.list_stage_events(trace.trace_id) if trace else []

with right:
    st.subheader("Live decisions")
    if trace is None:
        st.info("Routing probabilities will appear after the first request.")
    else:
        _render_decision_cards(trace)
        st.caption(f"Trace {str(trace.trace_id)[:8]} · {trace.status.value.upper()}")
        for metric in capability_probabilities(trace):
            label_col, value_col = st.columns([3, 1])
            label_col.caption(metric.label)
            value_col.caption(f"{metric.probability:.0%}")
            st.progress(metric.probability)

if trace is not None:
    metrics = dashboard_metrics(trace, events)
    st.divider()
    summary_columns = st.columns(5)
    summary_columns[0].metric("Total latency", f"{metrics.total_latency_ms:,.0f} ms")
    summary_columns[1].metric("Recorded stages", metrics.stage_count)
    summary_columns[2].metric("Agents", metrics.agent_count)
    summary_columns[3].metric("Retries", metrics.retry_count)
    summary_columns[4].metric(
        "Tokens", metrics.total_tokens if metrics.total_tokens is not None else "n/a"
    )

    graph_col, detail_col = st.columns([1.45, 1], gap="large")
    with graph_col:
        st.subheader("Execution graph")
        st.graphviz_chart(execution_graph_dot(trace, events), use_container_width=True)
    with detail_col:
        st.subheader("Deterministic plan")
        plan = trace.execution_plan or {}
        agents = plan.get("agents", [])
        st.write("Agents: " + (", ".join(agents) if agents else "None"))
        st.write(f"Model tier: `{trace.model_tier or 'n/a'}`")
        st.write(f"Citations: `{'required' if plan.get('citations_required') else 'optional'}`")
        st.write(f"Quality gate: `{'on' if plan.get('quality_gate_required') else 'off'}`")
        with st.expander("Policy explanation"):
            for reason in plan.get("reasoning", []):
                st.write(f"- {reason}")

    st.subheader("Execution timeline")
    timeline = timeline_items(events)
    if timeline:
        colors = {
            "succeeded": "#55e6c1",
            "failed": "#ff7185",
            "skipped": "#a78bfa",
            "running": "#60a5fa",
        }
        figure = go.Figure()
        for item in timeline:
            figure.add_trace(
                go.Bar(
                    x=[max(item.duration_ms, 0.2)],
                    y=[f"{item.component} · {item.stage}"],
                    base=[item.offset_ms],
                    orientation="h",
                    marker_color=colors.get(item.status, "#94a3b8"),
                    name=item.status,
                    hovertemplate=(
                        f"{item.component}<br>{item.stage}<br>"
                        f"{item.duration_ms:.2f} ms<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
        figure.update_layout(
            barmode="overlay",
            height=max(300, 42 * len(timeline)),
            margin=dict(l=10, r=10, t=15, b=10),
            xaxis_title="Milliseconds from trace start",
            yaxis_title=None,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
        )
        st.plotly_chart(figure, use_container_width=True)

    if trace.quality_gate_result:
        st.subheader("Quality gate")
        attempts = trace.quality_gate_result.get("attempts", [])
        if attempts:
            latest = attempts[-1]
            decision = latest["quality"]["decision"]
            quality_columns = st.columns(3)
            quality_columns[0].metric("Answers request", f"{decision['answers_request']:.0%}")
            quality_columns[1].metric(
                "Evidence support", f"{decision['supported_by_evidence']:.0%}"
            )
            quality_columns[2].metric("Policy action", latest["policy"]["action"])
        with st.expander("Quality attempts and debug data"):
            st.json(trace.quality_gate_result)
