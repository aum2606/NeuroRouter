from neurorouter.rag.chunker import Chunker
from neurorouter.rag.embeddings import HashingEmbeddingProvider
from neurorouter.rag.models import (
    CollectionInfo,
    DocumentSection,
    DocumentType,
    LoadedDocument,
    RetrievedChunk,
    TextChunk,
)
from neurorouter.rag.retriever import Retriever


class MemoryVectorStore:
    def __init__(self) -> None:
        self.chunks: list[TextChunk] = []

    def upsert(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> int:
        assert len(chunks) == len(embeddings)
        self.chunks = chunks
        return len(chunks)

    def query(self, embedding: list[float], *, top_k: int) -> list[RetrievedChunk]:
        del embedding
        return [
            RetrievedChunk(**chunk.model_dump(), relevance_score=0.8 if index == 0 else 0.1)
            for index, chunk in enumerate(self.chunks[:top_k])
        ]

    def collection_info(self) -> CollectionInfo:
        return CollectionInfo(
            name="memory",
            chunk_count=len(self.chunks),
            document_count=len({chunk.document_id for chunk in self.chunks}),
        )


def test_retriever_indexes_and_filters_by_relevance() -> None:
    store = MemoryVectorStore()
    retriever = Retriever(
        chunker=Chunker(chunk_size=25, overlap=5),
        embeddings=HashingEmbeddingProvider(64),
        vector_store=store,
        minimum_relevance=0.5,
    )
    result = retriever.index(
        LoadedDocument(
            document_id="doc",
            name="guide.txt",
            document_type=DocumentType.TEXT,
            sections=[DocumentSection(text="Routing evidence. Additional policy evidence.")],
        )
    )

    matches = retriever.retrieve("routing")

    assert result.chunks_indexed >= 2
    assert len(matches) == 1
    assert matches[0].relevance_score == 0.8
    assert retriever.retrieve("  ") == []
