"""Application service for safe local knowledge-base ingestion and inspection."""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from neurorouter.rag.chunker import Chunker
from neurorouter.rag.embeddings import HashingEmbeddingProvider
from neurorouter.rag.index import ChromaVectorStore, VectorStore
from neurorouter.rag.models import CollectionInfo, DocumentRecord
from neurorouter.rag.retriever import Retriever
from neurorouter.tools.document_loader import DocumentLoader, DocumentLoadError
from neurorouter.utils.config import PROJECT_ROOT, RAGSettings


class UploadIndexResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    document: str
    document_type: str
    chunks_indexed: int = Field(ge=0)
    collection_name: str
    status: str


class KnowledgeBaseService:
    """Coordinate extraction, chunking, embeddings, storage, and catalog queries."""

    def __init__(
        self,
        *,
        loader: DocumentLoader,
        retriever: Retriever,
        vector_store: VectorStore,
    ) -> None:
        self.loader = loader
        self.retriever = retriever
        self.vector_store = vector_store

    def index_upload(self, filename: str, content: bytes) -> UploadIndexResult:
        existing_ids = {record.document_id for record in self.vector_store.list_documents()}
        document = self.loader.load_bytes(filename, content)
        if not any(section.text.strip() for section in document.sections):
            raise DocumentLoadError(f"{document.name} contains no extractable text")
        document = document.model_copy(
            update={
                "metadata": {
                    **document.metadata,
                    "indexed_at": datetime.now(UTC).isoformat(),
                }
            }
        )
        indexed = self.retriever.index(document)
        return UploadIndexResult(
            document_id=document.document_id,
            document=document.name,
            document_type=document.document_type.value,
            chunks_indexed=indexed.chunks_indexed,
            collection_name=indexed.collection_name,
            status="updated" if document.document_id in existing_ids else "indexed",
        )

    def collection_info(self) -> CollectionInfo:
        return self.vector_store.collection_info()

    def list_documents(self) -> list[DocumentRecord]:
        return self.vector_store.list_documents()


def build_knowledge_base_service(
    settings: RAGSettings,
    *,
    project_root: Path = PROJECT_ROOT,
) -> KnowledgeBaseService:
    """Build the configured local RAG service for UI and scripts."""
    store = ChromaVectorStore(
        project_root / settings.persist_directory,
        settings.collection_name,
    )
    retriever = Retriever(
        chunker=Chunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap),
        embeddings=HashingEmbeddingProvider(settings.embedding_dimensions),
        vector_store=store,
        top_k=settings.retrieval_top_k,
        minimum_relevance=settings.minimum_relevance,
    )
    return KnowledgeBaseService(
        loader=DocumentLoader(max_file_size_mb=settings.max_file_size_mb),
        retriever=retriever,
        vector_store=store,
    )
