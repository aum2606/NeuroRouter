"""Knowledge Base shell backed by the Phase 5 ingestion components."""

import streamlit as st

st.set_page_config(page_title="Knowledge Base · NeuroRouter", page_icon="📚", layout="wide")
st.title("📚 Knowledge Base")
st.caption("PDF, TXT, and Markdown ingestion with persistent local ChromaDB retrieval is ready.")
st.file_uploader("Upload PDF, TXT, or Markdown", type=["pdf", "txt", "md"], disabled=True)
st.info(
    "The upload workflow and document catalog UI arrive in Phase 10; Phase 5 provides the backend."
)
