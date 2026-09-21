"""Trace Explorer backed by the Phase 1 SQLite repository."""

import streamlit as st

from neurorouter.telemetry import TraceRepository
from neurorouter.utils.config import PROJECT_ROOT, load_settings

st.set_page_config(page_title="Trace Explorer · NeuroRouter", page_icon="🔎", layout="wide")
st.title("🔎 Trace Explorer")
settings = load_settings()
repository = TraceRepository(PROJECT_ROOT / settings.telemetry.database_path)
traces = repository.list_traces(limit=200)
if not traces:
    st.info("No traces yet. Requests recorded by the control plane will appear here.")
else:
    st.dataframe(
        [
            {
                "trace_id": str(trace.trace_id),
                "query": trace.request_text,
                "route": trace.route or "—",
                "status": trace.status.value,
                "latency_ms": trace.total_latency_ms,
                "timestamp": trace.created_at,
            }
            for trace in traces
        ],
        use_container_width=True,
        hide_index=True,
    )
