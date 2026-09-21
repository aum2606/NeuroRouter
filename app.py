"""NeuroRouter Streamlit entry point."""

import streamlit as st

from neurorouter.telemetry import TraceRepository
from neurorouter.utils.config import PROJECT_ROOT, load_settings

settings = load_settings()
database_path = PROJECT_ROOT / settings.telemetry.database_path
repository = TraceRepository(database_path)

st.set_page_config(
    page_title=settings.ui.page_title,
    page_icon=settings.ui.page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {background: #080d18;}
    [data-testid="stSidebar"] {background: #0c1424;}
    .nr-kicker {color:#54e5c2; letter-spacing:.18em; font-size:.75rem; font-weight:700;}
    .nr-card {border:1px solid #26344e; background:#101a2c; border-radius:14px;
              padding:1rem 1.1rem; min-height:116px;}
    .nr-muted {color:#91a0b8;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="nr-kicker">PROBABILISTIC AI CONTROL PLANE</div>', unsafe_allow_html=True)
st.title("NeuroRouter")
st.caption("Jev judgments → deterministic policy → bounded execution → quality control")

health = "ONLINE" if repository.healthcheck() else "DEGRADED"
col1, col2, col3, col4 = st.columns(4)
col1.metric("Control plane", health)
col2.metric("Jev model", settings.jev.model)
col3.metric("Stored traces", len(repository.list_traces(limit=1000)))
col4.metric("Environment", settings.app.environment.upper())

st.divider()
left, right = st.columns([1.55, 1], gap="large")
with left:
    st.subheader("Command Center")
    st.info("Phase 5 routing, policy, bounded execution, and local RAG are online.")
    st.text_area(
        "Request",
        placeholder="Ask NeuroRouter to research, analyze, or build something…",
        height=150,
        disabled=True,
    )
    st.button("Route request", type="primary", disabled=True, use_container_width=True)

with right:
    st.subheader("System readiness")
    readiness = {
        "Validated state contracts": True,
        "SQLite telemetry": repository.healthcheck(),
        "Jev routing layer": True,
        "Policy engine": True,
        "Initial agent runtime": True,
        "Local RAG pipeline": True,
        "Code / Finance agents": False,
    }
    for label, ready in readiness.items():
        st.write(f"{'🟢' if ready else '⚪'}  {label}")

st.divider()
st.subheader("Execution topology")
st.graphviz_chart(
    """
    digraph {
      graph [bgcolor="transparent", rankdir=LR]
      node [shape=box, style="rounded,filled", fillcolor="#142238", color="#344766",
            fontcolor="#dce7f7"]
      edge [color="#54e5c2"]
      User -> StateBuilder -> JevRouter -> PolicyEngine -> ExecutionPlan
      ExecutionPlan -> Agents -> Aggregator -> Synthesizer -> QualityGate -> Outcome
    }
    """,
    use_container_width=True,
)
