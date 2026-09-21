"""Decision Lab shell for deterministic routing counterfactuals."""

import streamlit as st

from neurorouter.utils.config import load_thresholds

st.set_page_config(page_title="Decision Lab · NeuroRouter", page_icon="🎛️", layout="wide")
st.title("🎛️ Decision Lab")
st.caption("Adjust policy without recalling Jev. Counterfactual comparison arrives in Phase 10.")
thresholds = load_thresholds().routing
col1, col2 = st.columns(2)
with col1:
    st.slider("Web threshold", 0.0, 1.0, thresholds.web_threshold, 0.01)
    st.slider("RAG threshold", 0.0, 1.0, thresholds.rag_threshold, 0.01)
with col2:
    st.slider("Code threshold", 0.0, 1.0, thresholds.code_threshold, 0.01)
    st.slider("Data threshold", 0.0, 1.0, thresholds.data_threshold, 0.01)
st.info("Select an existing Jev result after routing telemetry is implemented.")
