from pathlib import Path

import pytest

from neurorouter.rag.chunker import Chunker
from neurorouter.rag.embeddings import HashingEmbeddingProvider
from neurorouter.rag.index import ChromaVectorStore
from neurorouter.rag.retriever import Retriever
from neurorouter.rag.service import KnowledgeBaseService
from neurorouter.tools.document_loader import DocumentLoader, DocumentLoadError


def _service(tmp_path: Path) -> KnowledgeBaseService:
    store = ChromaVectorStore(tmp_path / "chroma", "knowledge_service")
    return KnowledgeBaseService(
        loader=DocumentLoader(max_file_size_mb=1),
        retriever=Retriever(
            chunker=Chunker(chunk_size=100, overlap=10),
            embeddings=HashingEmbeddingProvider(64),
            vector_store=store,
        ),
        vector_store=store,
    )


def test_indexes_upload_and_exposes_document_catalog(tmp_path: Path) -> None:
    service = _service(tmp_path)
    content = b"NeuroRouter keeps probabilistic judgment separate from deterministic policy."

    first = service.index_upload("architecture.txt", content)
    second = service.index_upload("architecture.txt", content)
    documents = service.list_documents()

    assert first.status == "indexed"
    assert second.status == "updated"
    assert first.chunks_indexed == 1
    assert service.collection_info().document_count == 1
    assert documents[0].document_type == "txt"
    assert documents[0].size_bytes == len(content)
    assert documents[0].indexed_at is not None


def test_rejects_upload_without_extractable_text(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(DocumentLoadError, match="no extractable text"):
        service.index_upload("empty.txt", b"   \n")

    assert service.collection_info().chunk_count == 0
