"""Vector-store boundary and persistent Chroma implementation."""

from pathlib import Path
from typing import Protocol

from neurorouter.rag.models import CollectionInfo, RetrievedChunk, TextChunk


class VectorStore(Protocol):
    def upsert(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> int: ...

    def query(self, embedding: list[float], *, top_k: int) -> list[RetrievedChunk]: ...

    def collection_info(self) -> CollectionInfo: ...


class ChromaVectorStore:
    """Persist caller-supplied embeddings in a cosine-distance Chroma collection."""

    def __init__(self, persist_directory: Path | str, collection_name: str) -> None:
        try:
            import chromadb
        except ImportError as error:  # pragma: no cover - installation guidance
            raise RuntimeError(
                "Install NeuroRouter with the 'rag' extra to use ChromaDB"
            ) from error
        path = Path(persist_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"application": "NeuroRouter"},
        )

    def upsert(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[self._metadata(chunk) for chunk in chunks],
        )
        return len(chunks)

    def query(self, embedding: list[float], *, top_k: int) -> list[RetrievedChunk]:
        if top_k < 1 or self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            data = dict(metadata or {})
            page_value = data.pop("page", None)
            chunk_index = int(data.pop("chunk_index", 0))
            document_id = str(data.pop("document_id"))
            document = str(data.pop("document"))
            relevance = max(0.0, min(1.0, 1.0 - float(distance)))
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document=document,
                    text=text,
                    chunk_index=chunk_index,
                    page=int(page_value) if page_value is not None else None,
                    relevance_score=relevance,
                    metadata=data,
                )
            )
        return retrieved

    def collection_info(self) -> CollectionInfo:
        result = self._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        documents = sorted(
            {
                str(metadata["document"])
                for metadata in metadatas
                if metadata and "document" in metadata
            }
        )
        return CollectionInfo(
            name=self.collection_name,
            chunk_count=self._collection.count(),
            document_count=len(documents),
            documents=documents,
        )

    @staticmethod
    def _metadata(chunk: TextChunk) -> dict[str, str | int | float | bool]:
        metadata: dict[str, str | int | float | bool] = {
            "document_id": chunk.document_id,
            "document": chunk.document,
            "chunk_index": chunk.chunk_index,
        }
        if chunk.page is not None:
            metadata["page"] = chunk.page
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata.setdefault(key, value)
        return metadata
