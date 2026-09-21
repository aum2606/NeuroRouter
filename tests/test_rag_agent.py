import asyncio

from neurorouter.agents.rag import RAGAgent
from neurorouter.rag.models import RetrievedChunk
from tests.test_agents import _agent_input


class StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        del query, top_k
        return self.chunks


def test_rag_agent_returns_citable_local_evidence() -> None:
    retriever = StubRetriever(
        [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                document="architecture.pdf",
                text="Jev judgments are converted by deterministic policy.",
                chunk_index=0,
                page=4,
                relevance_score=0.91,
            )
        ]
    )

    result = asyncio.run(RAGAgent(retriever).run(_agent_input()))

    assert result.success is True
    assert result.evidence[0].metadata["page"] == 4
    assert result.sources[0].provider == "local_knowledge_base"
    assert result.metadata["no_local_evidence"] is False


def test_rag_agent_degrades_gracefully_without_documents() -> None:
    result = asyncio.run(RAGAgent(StubRetriever([])).run(_agent_input()))

    assert result.success is True
    assert result.evidence == []
    assert result.metadata["no_local_evidence"] is True
