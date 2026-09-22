"""Validated contracts shared by the document, vector, and agent layers."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RAGModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentType(StrEnum):
    PDF = "pdf"
    TEXT = "txt"
    MARKDOWN = "markdown"


class DocumentSection(RAGModel):
    text: str
    page: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoadedDocument(RAGModel):
    document_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    document_type: DocumentType
    sections: list[DocumentSection]
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextChunk(RAGModel):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document: str = Field(min_length=1)
    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    page: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(TextChunk):
    relevance_score: float = Field(ge=0, le=1)


class IndexingResult(RAGModel):
    document_id: str
    document: str
    chunks_indexed: int = Field(ge=0)
    collection_name: str


class CollectionInfo(RAGModel):
    name: str
    chunk_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    documents: list[str] = Field(default_factory=list)


class DocumentRecord(RAGModel):
    """One indexed document summarized from its persisted chunk metadata."""

    document_id: str
    name: str
    document_type: str
    chunk_count: int = Field(ge=0)
    page_count: int = Field(ge=0)
    size_bytes: int = Field(ge=0)
    indexed_at: datetime | None = None
