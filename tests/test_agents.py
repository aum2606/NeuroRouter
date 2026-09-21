import asyncio
from uuid import uuid4

from neurorouter.agents.base import AgentInput, AgentPayload, BaseAgent
from neurorouter.agents.general import GeneralAgent
from neurorouter.agents.research import WebResearchAgent
from neurorouter.core.policy_engine import PolicyEngine
from neurorouter.schemas.execution import AgentName
from neurorouter.schemas.routing import Intent
from neurorouter.tools.web_search import StaticWebSearchProvider, WebSearchResult
from neurorouter.utils.config import load_thresholds
from tests.test_policy_engine import _decision, _state


def _agent_input(*, web: bool = False, context: dict | None = None) -> AgentInput:
    state = _state(web=web)
    plan = PolicyEngine(load_thresholds()).create_plan(
        _decision(
            intent=Intent.RESEARCH if web else Intent.GENERAL_QA,
            intent_probabilities={Intent.RESEARCH if web else Intent.GENERAL_QA: 1.0},
            needs_web=0.9 if web else 0.0,
        ),
        state,
    )
    return AgentInput(
        trace_id=uuid4(),
        request_text="current battery storage technology",
        state=state,
        plan=plan,
        context=context or {},
    )


def test_general_agent_returns_structured_handoff() -> None:
    result = asyncio.run(GeneralAgent().run(_agent_input()))

    assert result.success is True
    assert result.agent_name is AgentName.GENERAL
    assert result.metadata["external_evidence"] is False
    assert result.evidence == []


def test_web_agent_deduplicates_and_ranks_evidence() -> None:
    duplicate = WebSearchResult(
        title="Battery storage",
        url="https://example.com/battery",
        snippet="Current battery storage technology overview",
        provider="static",
        score=0.8,
    )
    other = WebSearchResult(
        title="Unrelated",
        url="https://example.com/other",
        snippet="Different subject",
        provider="static",
        score=0.9,
    )
    provider = StaticWebSearchProvider({"battery": [duplicate], "storage": [duplicate, other]})
    agent = WebResearchAgent(provider, max_queries=2)

    result = asyncio.run(
        agent.run(_agent_input(web=True, context={"search_queries": ["battery", "storage"]}))
    )

    assert result.success is True
    assert len(result.evidence) == 2
    assert len(result.sources) == 2
    assert result.sources[0].title == "Battery storage"
    assert result.metadata["queries"] == ["battery", "storage"]


class PartiallyFailingProvider:
    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        del max_results
        if query == "fails":
            raise ConnectionError("temporary failure")
        return [
            WebSearchResult(
                title="Working result",
                url="https://example.com/working",
                snippet="current battery storage technology",
                provider="partial",
                score=1.0,
            )
        ]


def test_web_agent_keeps_evidence_when_one_query_fails() -> None:
    agent = WebResearchAgent(PartiallyFailingProvider(), max_queries=2)

    result = asyncio.run(
        agent.run(_agent_input(web=True, context={"search_queries": ["fails", "works"]}))
    )

    assert result.success is True
    assert len(result.evidence) == 1
    assert result.metadata["query_error_types"] == ["ConnectionError"]


class FailingAgent(BaseAgent):
    name = AgentName.GENERAL

    async def execute(self, agent_input: AgentInput) -> AgentPayload:
        del agent_input
        raise ConnectionError("provider unavailable")


def test_base_agent_isolates_provider_failure() -> None:
    result = asyncio.run(FailingAgent().run(_agent_input()))

    assert result.success is False
    assert result.error is not None
    assert result.error.retriable is True
    assert result.error.error_type == "ConnectionError"
