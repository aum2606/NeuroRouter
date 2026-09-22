"""Normalized, budgeted evidence contracts supplied to synthesis."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from neurorouter.schemas.execution import AgentName


class ContextModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContextSource(ContextModel):
    source_id: str
    citation_label: str
    title: str
    url: str
    provider: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextItem(ContextModel):
    item_id: str
    content: str
    agent_name: AgentName
    relevance_score: float = Field(ge=0, le=1)
    source_id: str | None = None
    citation_label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentContext(ContextModel):
    agent_name: AgentName
    success: bool
    output: str = ""
    error: str | None = None


class EvidencePacket(ContextModel):
    items: list[ContextItem] = Field(default_factory=list)
    sources: list[ContextSource] = Field(default_factory=list)
    agent_outputs: list[AgentContext] = Field(default_factory=list)
    total_characters: int = Field(ge=0)
    dropped_items: int = Field(ge=0)
    duplicate_items: int = Field(ge=0)
