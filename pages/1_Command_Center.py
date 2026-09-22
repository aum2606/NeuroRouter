"""Command Center shell for live routing and execution."""

import streamlit as st

st.set_page_config(page_title="Command Center · NeuroRouter", page_icon="⚡", layout="wide")
st.title("⚡ Command Center")
st.caption("The live request cockpit will connect to the router and orchestrator in later phases.")
left, right = st.columns([3, 2])
with left:
    st.subheader("Conversation")
    st.info("Phase 7 shell — parallel specialists, aggregation, and synthesis are implemented.")
    st.chat_input("Route a request…", disabled=True)
with right:
    st.subheader("Live decisions")
    for label in ("Primary intent", "Web", "RAG", "Code", "Complexity", "Risk"):
        st.metric(label, "Awaiting request")
