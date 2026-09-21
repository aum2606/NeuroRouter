"""Semantic retrieval service composed from provider and storage boundaries."""

from neurorouter.rag.chunker import Chunker
from neurorouter.rag.embeddings import EmbeddingProvider
from neurorouter.rag.index import VectorStore
from neurorouter.rag.models import IndexingResult, LoadedDocument, RetrievedChunk


class Retriever:
    """Index documents and retrieve relevant chunks without provider coupling."""

    def __init__(
        self,
        *,
        chunker: Chunker,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        top_k: int = 5,
        minimum_relevance: float = 0.15,
    ) -> None:
        self.chunker = chunker
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.top_k = top_k
        self.minimum_relevance = minimum_relevance

    def index(self, document: LoadedDocument) -> IndexingResult:
        chunks = self.chunker.chunk(document)
        vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
        indexed = self.vector_store.upsert(chunks, vectors)
        return IndexingResult(
            document_id=document.document_id,
            document=document.name,
            chunks_indexed=indexed,
            collection_name=self.vector_store.collection_info().name,
        )

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        matches = self.vector_store.query(
            self.embeddings.embed_query(query), top_k=top_k or self.top_k
        )
        return [match for match in matches if match.relevance_score >= self.minimum_relevance]
