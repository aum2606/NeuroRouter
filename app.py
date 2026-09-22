"""NeuroRouter Streamlit entry point."""

import streamlit as st

from neurorouter.telemetry import TraceRepository
from neurorouter.utils.config import PROJECT_ROOT, load_settings

settings = load_settings()
repository = TraceRepository(PROJECT_ROOT / settings.telemetry.database_path)

st.set_page_config(
    page_title=settings.ui.page_title,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {background:#070b14;}
    [data-testid="stSidebar"] {background:#0b1220;}
    .nr-kicker {color:#55e6c1; letter-spacing:.18em; font-size:.75rem; font-weight:750;}
    .nr-card {border:1px solid #26344e; background:linear-gradient(145deg,#101a2c,#0c1423);
      border-radius:14px; padding:1rem 1.1rem; min-height:116px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="nr-kicker">PROBABILISTIC AI CONTROL PLANE</div>', unsafe_allow_html=True)
st.title("NeuroRouter")
st.caption(
    "Jev judgments → deterministic policy → bounded agents → evidence synthesis → quality control"
)

traces = repository.list_traces(limit=1000)
health = "ONLINE" if repository.healthcheck() else "DEGRADED"
accepted = sum(trace.status.value == "accepted" for trace in traces)
review = sum(trace.status.value == "review" for trace in traces)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Control plane", health)
col2.metric("Stored traces", len(traces))
col3.metric("Accepted", accepted)
col4.metric("Review queue", review)

st.divider()
left, right = st.columns([1.45, 1], gap="large")
with left:
    st.subheader("Operational architecture")
    st.write(
        "NeuroRouter separates probabilistic judgment from deterministic execution policy. "
        "Every decision and stage is persisted, inspectable, and replayable at the policy layer."
    )
    st.graphviz_chart(
        """
        digraph {
          graph [bgcolor="transparent", rankdir=LR]
          node [shape=box, style="rounded,filled", fillcolor="#142238", color="#344766",
                fontcolor="#dce7f7"]
          edge [color="#55e6c1"]
          User -> State -> Jev -> Policy -> Agents -> Context -> LLM -> Quality -> Outcome
        }
        """,
        use_container_width=True,
    )
    link_columns = st.columns(2)
    link_columns[0].page_link("pages/1_Command_Center.py", label="Open Command Center", icon="⚡")
    link_columns[1].page_link("pages/2_Trace_Explorer.py", label="Explore traces", icon="🔎")

with right:
    st.subheader("Phase 11 readiness")
    readiness = (
        "Validated state contracts",
        "Atomic Jev routing",
        "Deterministic policy engine",
        "Bounded specialist execution",
        "Local RAG and safe code boundary",
        "Free-tier LLM provider abstraction",
        "Atomic Jev quality gate",
        "Bounded retry controller",
        "Complete SQLite lifecycle traces",
        "Execution graph and timeline",
        "Live operations dashboard",
        "Deep trace inspection",
        "Counterfactual policy laboratory",
        "Knowledge-base ingestion interface",
        "Measured routing evaluations",
        "Brier and calibration analysis",
        "Baseline comparison contracts",
    )
    for label in readiness:
        st.write(f"🟢  {label}")

st.divider()
st.caption(
    f"Environment: {settings.app.environment.upper()} · LLM: {settings.llm.provider.upper()} · "
    f"Jev: {settings.jev.model} · "
    f"Paid fallback: {'ON' if settings.llm.allow_paid_models else 'OFF'}"
)
