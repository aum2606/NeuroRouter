"""Local document ingestion and vector collection operations."""

import streamlit as st

from neurorouter.rag.service import KnowledgeBaseService, build_knowledge_base_service
from neurorouter.utils.config import load_settings


@st.cache_resource
def _knowledge_base() -> KnowledgeBaseService:
    return build_knowledge_base_service(load_settings().rag)


st.set_page_config(page_title="Knowledge Base - NeuroRouter", page_icon="📚", layout="wide")
st.title("📚 Knowledge Base")
st.caption("Extract, chunk, embed, and persist local PDF, TXT, and Markdown evidence.")

try:
    service = _knowledge_base()
except Exception as error:
    st.error(f"Knowledge base is unavailable: {type(error).__name__}: {error}")
    st.info('Install the local RAG dependencies with `pip install -e ".[rag]"`.')
    st.stop()

if "kb_index_results" not in st.session_state:
    st.session_state.kb_index_results = []

info = service.collection_info()
metric_columns = st.columns(4)
metric_columns[0].metric("Collection", info.name)
metric_columns[1].metric("Documents", info.document_count)
metric_columns[2].metric("Chunks", info.chunk_count)
metric_columns[3].metric("Embedding", "Local hashing")

st.subheader("Ingest documents")
uploads = st.file_uploader(
    "Upload PDF, TXT, or Markdown",
    type=["pdf", "txt", "md", "markdown"],
    accept_multiple_files=True,
    help=f"Each file is limited to {load_settings().rag.max_file_size_mb} MB.",
)

if st.button(
    "Index selected documents",
    type="primary",
    disabled=not uploads,
    use_container_width=True,
):
    current_results = []
    with st.status("Indexing local evidence...", expanded=True) as status:
        for upload in uploads or []:
            st.write(f"Extracting and indexing {upload.name}")
            try:
                indexed = service.index_upload(upload.name, upload.getvalue())
            except Exception as error:
                current_results.append(
                    {
                        "document": upload.name,
                        "status": "failed",
                        "chunks": 0,
                        "detail": f"{type(error).__name__}: {error}",
                    }
                )
            else:
                current_results.append(
                    {
                        "document": indexed.document,
                        "status": indexed.status,
                        "chunks": indexed.chunks_indexed,
                        "detail": indexed.document_type,
                    }
                )
        failures = sum(item["status"] == "failed" for item in current_results)
        status.update(
            label=(f"Indexed {len(current_results) - failures} document(s); {failures} failed"),
            state="error" if failures else "complete",
            expanded=bool(failures),
        )
    st.session_state.kb_index_results = current_results
    st.cache_resource.clear()
    st.rerun()

if st.session_state.kb_index_results:
    st.subheader("Latest indexing operation")
    st.dataframe(st.session_state.kb_index_results, use_container_width=True, hide_index=True)

st.subheader("Document catalog")
documents = service.list_documents()
if documents:
    st.dataframe(
        [
            {
                "document": document.name,
                "type": document.document_type,
                "chunks": document.chunk_count,
                "pages": document.page_count or "—",
                "size_kb": round(document.size_bytes / 1024, 1),
                "indexed_at": document.indexed_at,
                "document_id": document.document_id,
                "status": "indexed",
            }
            for document in documents
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("The collection is empty. Upload a document to make RAG routing operational.")

with st.expander("Vector collection details"):
    st.json(
        {
            **info.model_dump(mode="json"),
            "persist_directory": str(load_settings().rag.persist_directory),
            "embedding_dimensions": load_settings().rag.embedding_dimensions,
            "chunk_size": load_settings().rag.chunk_size,
            "chunk_overlap": load_settings().rag.chunk_overlap,
        }
    )
