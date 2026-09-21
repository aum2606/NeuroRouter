"""Validated, secret-free state passed to the routing layer."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base schema that rejects accidental, untracked fields."""

    model_config = ConfigDict(extra="forbid")


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class RequestState(StrictModel):
    text: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationMessage(StrictModel):
    role: MessageRole
    content: str = Field(min_length=1)
    timestamp: datetime | None = None


class ConversationState(StrictModel):
    recent_messages: list[ConversationMessage] = Field(default_factory=list)


class AttachmentState(StrictModel):
    count: int = Field(default=0, ge=0)
    available_document_types: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def count_matches_presence(self) -> "AttachmentState":
        if self.count == 0 and self.available_document_types:
            raise ValueError("document types cannot be present when attachment count is zero")
        return self


class CapabilityState(StrictModel):
    web_search_available: bool = False
    rag_available: bool = False
    code_available: bool = False
    finance_agent_available: bool = False


class KnowledgeBaseState(StrictModel):
    has_indexed_documents: bool = False
    collection_metadata: dict[str, Any] = Field(default_factory=dict)


class SystemState(StrictModel):
    environment: str = "development"
    app_version: str = "0.1.0"
    relevant_configuration: dict[str, Any] = Field(default_factory=dict)


class RouterState(StrictModel):
    """Complete state supplied to Jev; credentials must never be included."""

    request: RequestState
    conversation: ConversationState = Field(default_factory=ConversationState)
    attachments: AttachmentState = Field(default_factory=AttachmentState)
    capabilities: CapabilityState = Field(default_factory=CapabilityState)
    knowledge_base: KnowledgeBaseState = Field(default_factory=KnowledgeBaseState)
    system: SystemState = Field(default_factory=SystemState)
