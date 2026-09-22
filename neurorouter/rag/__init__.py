"""Local document indexing and retrieval components."""

from neurorouter.rag.chunker import Chunker
from neurorouter.rag.embeddings import EmbeddingProvider, HashingEmbeddingProvider
from neurorouter.rag.index import ChromaVectorStore, VectorStore
from neurorouter.rag.models import LoadedDocument, RetrievedChunk, TextChunk
from neurorouter.rag.retriever import Retriever
from neurorouter.rag.service import KnowledgeBaseService, build_knowledge_base_service

__all__ = [
    "ChromaVectorStore",
    "Chunker",
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    "KnowledgeBaseService",
    "LoadedDocument",
    "RetrievedChunk",
    "Retriever",
    "TextChunk",
    "VectorStore",
    "build_knowledge_base_service",
]
