"""Evaluation dashboard shell."""

import streamlit as st

st.set_page_config(page_title="Evaluations · NeuroRouter", page_icon="📈", layout="wide")
st.title("📈 Evaluations")
st.caption("Measured routing accuracy, calibration, latency, and cost will appear here.")
col1, col2, col3 = st.columns(3)
col1.metric("Evaluation examples", "0")
col2.metric("Measured runs", "0")
col3.metric("Fabricated results", "Never")
st.info("Evaluation infrastructure is scheduled for Phase 11.")
