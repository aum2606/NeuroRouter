"""Bounded local knowledge-base retrieval agent."""

import asyncio
import hashlib
from urllib.parse import quote

from neurorouter.agents.base import AgentInput, AgentPayload, BaseAgent, Evidence, Source
from neurorouter.rag.retriever import Retriever
from neurorouter.schemas.execution import AgentName


class RAGAgent(BaseAgent):
    """Retrieve local chunks and expose them as inspectable agent evidence."""

    name = AgentName.RAG

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    async def execute(self, agent_input: AgentInput) -> AgentPayload:
        chunks = await asyncio.to_thread(self.retriever.retrieve, agent_input.request_text)
        if not chunks:
            return AgentPayload(
                output="No relevant local knowledge-base evidence was available.",
                metadata={"result_count": 0, "no_local_evidence": True},
            )

        sources: dict[str, Source] = {}
        evidence: list[Evidence] = []
        for chunk in chunks:
            location = f"#page={chunk.page}" if chunk.page is not None else ""
            source_id = hashlib.sha256(chunk.document_id.encode()).hexdigest()[:16]
            sources.setdefault(
                source_id,
                Source(
                    source_id=source_id,
                    title=chunk.document,
                    url=f"local://document/{quote(chunk.document_id)}{location}",
                    provider="local_knowledge_base",
                    metadata={"document_id": chunk.document_id},
                ),
            )
            evidence.append(
                Evidence(
                    evidence_id=chunk.chunk_id,
                    content=chunk.text,
                    source_id=source_id,
                    relevance_score=chunk.relevance_score,
                    metadata={
                        "document": chunk.document,
                        "page": chunk.page,
                        "chunk_id": chunk.chunk_id,
                        **chunk.metadata,
                    },
                )
            )
        return AgentPayload(
            output=f"Retrieved {len(evidence)} relevant chunks from {len(sources)} documents.",
            evidence=evidence,
            sources=list(sources.values()),
            metadata={"result_count": len(evidence), "no_local_evidence": False},
        )
