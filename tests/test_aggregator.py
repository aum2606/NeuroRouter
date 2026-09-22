from datetime import UTC, datetime

from neurorouter.agents.base import AgentResult, Evidence, Source
from neurorouter.core.aggregator import ContextAggregator
from neurorouter.schemas.execution import AgentName


def _result(
    agent: AgentName,
    *,
    evidence: list[Evidence],
    sources: list[Source],
    output: str = "agent summary",
) -> AgentResult:
    now = datetime.now(UTC)
    return AgentResult(
        agent_name=agent,
        output=output,
        evidence=evidence,
        sources=sources,
        started_at=now,
        ended_at=now,
        latency_ms=1,
        success=True,
    )


def test_aggregator_deduplicates_ranks_and_preserves_sources() -> None:
    source = Source(
        source_id="source-1",
        title="Architecture",
        url="https://example.com/architecture",
        provider="test",
    )
    duplicate_low = Evidence(
        evidence_id="low",
        content="Jev routes probabilistically.",
        source_id="source-1",
        relevance_score=0.4,
    )
    duplicate_high = Evidence(
        evidence_id="high",
        content="  Jev routes   probabilistically. ",
        source_id="source-1",
        relevance_score=0.9,
    )
    other = Evidence(
        evidence_id="other",
        content="Policy remains deterministic.",
        relevance_score=0.7,
    )

    packet = ContextAggregator(context_budget_characters=1000).aggregate(
        [
            _result(AgentName.WEB_RESEARCH, evidence=[duplicate_low], sources=[source]),
            _result(AgentName.RAG, evidence=[duplicate_high, other], sources=[source]),
        ]
    )

    assert [item.item_id for item in packet.items] == ["high", "other"]
    assert packet.items[0].citation_label == "S1"
    assert packet.sources[0].citation_label == "S1"
    assert packet.duplicate_items == 1


def test_aggregator_enforces_combined_budget_and_reports_drops() -> None:
    evidence = [
        Evidence(
            evidence_id=str(index),
            content=(str(index) + "x" * 299),
            relevance_score=1 - index / 10,
        )
        for index in range(4)
    ]
    result = _result(
        AgentName.RAG,
        evidence=evidence,
        sources=[],
        output="summary " * 100,
    )

    packet = ContextAggregator(
        context_budget_characters=1000,
        max_agent_output_characters=200,
    ).aggregate([result])

    assert packet.total_characters <= 1000
    assert len(packet.agent_outputs[0].output) <= 200
    assert packet.dropped_items >= 1
