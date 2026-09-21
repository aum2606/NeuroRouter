"""Knowledge Base shell for later document ingestion."""

import streamlit as st

st.set_page_config(page_title="Knowledge Base · NeuroRouter", page_icon="📚", layout="wide")
st.title("📚 Knowledge Base")
st.caption("Local document ingestion and ChromaDB indexing are scheduled for Phase 5.")
st.file_uploader("Upload PDF, TXT, or Markdown", type=["pdf", "txt", "md"], disabled=True)
st.info("No vector collection has been initialized.")
