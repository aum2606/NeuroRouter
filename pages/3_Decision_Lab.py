"""Counterfactual deterministic routing laboratory."""

import streamlit as st

from neurorouter.core.decision_lab import replay_policy
from neurorouter.schemas.execution import ExecutionPlan
from neurorouter.telemetry import TraceRepository
from neurorouter.utils.config import PROJECT_ROOT, load_settings, load_thresholds


def _render_plan(title: str, plan: ExecutionPlan) -> None:
    st.markdown(f"### {title}")
    agents = ", ".join(agent.value for agent in plan.agents)
    st.markdown(f"**Agents:** {agents or 'None'}")
    columns = st.columns(3)
    columns[0].metric("Model", plan.model_tier.value.upper())
    columns[1].metric("Quality gate", "ON" if plan.quality_gate_required else "OFF")
    columns[2].metric("Review", "YES" if plan.requires_review else "NO")
    st.write(
        {
            "web": plan.use_web,
            "rag": plan.use_rag,
            "code": plan.use_code,
            "data_analysis": plan.use_data_analysis,
            "citations": plan.citations_required,
        }
    )
    with st.expander(f"{title} policy explanation"):
        for reason in plan.reasoning:
            st.write(f"- {reason}")


st.set_page_config(page_title="Decision Lab - NeuroRouter", page_icon="🎛️", layout="wide")
st.title("🎛️ Decision Lab")
st.caption("Change business policy and replay a stored Jev result without another model call.")

settings = load_settings()
thresholds = load_thresholds()
repository = TraceRepository(PROJECT_ROOT / settings.telemetry.database_path)
eligible = [
    trace
    for trace in repository.list_traces(limit=500)
    if trace.routing_decision and trace.state and trace.execution_plan
]

if not eligible:
    st.info("No complete routing traces are available. Run a Command Center request first.")
    st.stop()

selected = st.selectbox(
    "Stored Jev result",
    eligible,
    format_func=lambda trace: (
        f"{str(trace.trace_id)[:8]} · {(trace.route or 'unknown').upper()} · "
        f"{trace.request_text[:70]}"
    ),
)

decision = selected.routing_decision or {}
signal_columns = st.columns(4)
signal_columns[0].metric("needs_web", f"{decision.get('needs_web', 0):.0%}")
signal_columns[1].metric("needs_rag", f"{decision.get('needs_rag', 0):.0%}")
signal_columns[2].metric("needs_code", f"{decision.get('needs_code', 0):.0%}")
signal_columns[3].metric("needs_data_analysis", f"{decision.get('needs_data_analysis', 0):.0%}")

st.subheader("Counterfactual thresholds")
left, right = st.columns(2)
with left:
    web_threshold = st.slider("Web threshold", 0.0, 1.0, thresholds.routing.web_threshold, 0.01)
    rag_threshold = st.slider("RAG threshold", 0.0, 1.0, thresholds.routing.rag_threshold, 0.01)
with right:
    code_threshold = st.slider("Code threshold", 0.0, 1.0, thresholds.routing.code_threshold, 0.01)
    data_threshold = st.slider("Data threshold", 0.0, 1.0, thresholds.routing.data_threshold, 0.01)

result = replay_policy(
    selected,
    thresholds,
    web_threshold=web_threshold,
    rag_threshold=rag_threshold,
    code_threshold=code_threshold,
    data_threshold=data_threshold,
)

st.success("Counterfactual computed with deterministic Python policy. Jev calls: 0 · LLM calls: 0")
original_col, counterfactual_col = st.columns(2, gap="large")
with original_col:
    _render_plan("Original route", result.original_plan)
with counterfactual_col:
    _render_plan("Counterfactual route", result.counterfactual_plan)

st.subheader("Route delta")
if result.changed:
    st.dataframe(
        [change.model_dump(mode="json") for change in result.changes],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("These threshold values do not change the stored execution plan.")

with st.expander("Persisted probabilities and applied thresholds"):
    st.json(
        {
            "trace_id": result.trace_id,
            "routing_decision": selected.routing_decision,
            "original_thresholds": thresholds.routing.model_dump(mode="json"),
            "counterfactual_thresholds": result.applied_thresholds,
        }
    )
