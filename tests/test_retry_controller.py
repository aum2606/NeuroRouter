import asyncio
from datetime import UTC, datetime

from neurorouter.agents.base import AgentResult, Evidence, Source
from neurorouter.core.aggregator import ContextAggregator
from neurorouter.core.quality_policy import QualityPolicy
from neurorouter.core.retry_controller import RetryController
from neurorouter.core.synthesizer import SynthesisResult, Synthesizer
from neurorouter.llm.model_router import ModelRouter
from neurorouter.llm.prompts import SynthesisPromptRenderer, load_prompts
from neurorouter.llm.providers import MockLLMProvider
from neurorouter.schemas.execution import AgentName, ExecutionPlan, ModelTier
from neurorouter.schemas.quality import (
    JevQualityResult,
    QualityAction,
    QualityDecision,
    QualityStatus,
)
from neurorouter.utils.config import load_settings, load_thresholds


def _quality(**overrides) -> JevQualityResult:
    values = {
        "answers_request": 0.9,
        "supported_by_evidence": 0.9,
        "contains_unsupported_claims": 0.1,
        "missing_important_information": 0.1,
        "contradicts_evidence": 0.05,
        "needs_additional_retrieval": 0.1,
        "jev_latency_ms": 1,
    }
    values.update(overrides)
    return JevQualityResult(decision=QualityDecision(**values))


class SequenceQualityGate:
    def __init__(self, results: list[JevQualityResult]) -> None:
        self.results = results
        self.states = []

    async def evaluate(self, state):
        self.states.append(state)
        return self.results[min(len(self.states) - 1, len(self.results) - 1)]


def _agent_result(*, evidence: list[Evidence] | None = None) -> AgentResult:
    now = datetime.now(UTC)
    sources = (
        [
            Source(
                source_id="fresh-source",
                title="Fresh source",
                url="https://example.com/fresh",
                provider="test",
            )
        ]
        if evidence
        else []
    )
    return AgentResult(
        agent_name=AgentName.WEB_RESEARCH if evidence else AgentName.GENERAL,
        output="retrieved evidence" if evidence else "general handoff",
        evidence=evidence or [],
        sources=sources,
        started_at=now,
        ended_at=now,
        latency_ms=1,
        success=True,
    )


def _initial(response: str = "Initial candidate") -> SynthesisResult:
    return SynthesisResult(
        response=response,
        provider="mock",
        model="mock-fast",
        model_tier=ModelTier.FAST,
        latency_ms=1,
        evidence_items_used=0,
        citations_required=False,
    )


def _controller(gate, responses, *, max_retries=2, retrieval=None):
    settings = load_settings()
    response_iter = iter(responses)
    synthesizer = Synthesizer(
        provider=MockLLMProvider(lambda request: next(response_iter)),
        model_router=ModelRouter(settings.llm),
        prompt_renderer=SynthesisPromptRenderer(load_prompts().synthesis),
        settings=settings.llm,
    )
    return RetryController(
        quality_gate=gate,
        quality_policy=QualityPolicy(load_thresholds().quality),
        synthesizer=synthesizer,
        aggregator=ContextAggregator(context_budget_characters=2000),
        synthesis_prompt=load_prompts().synthesis,
        max_retries=max_retries,
        additional_retrieval=retrieval,
    )


def test_retry_controller_regenerates_then_accepts() -> None:
    gate = SequenceQualityGate([_quality(contains_unsupported_claims=0.8), _quality()])
    controller = _controller(gate, ["Stricter candidate"])

    outcome = asyncio.run(
        controller.run(
            request_text="Explain routing",
            plan=ExecutionPlan(agents=[AgentName.GENERAL], quality_gate_required=True),
            initial_synthesis=_initial(),
            agent_results=[_agent_result()],
        )
    )

    assert outcome.status is QualityStatus.ACCEPTED
    assert outcome.final_response == "Stricter candidate"
    assert outcome.retry_count == 1
    assert len(outcome.attempts) == 2


def test_retry_controller_adds_retrieval_before_regeneration() -> None:
    gate = SequenceQualityGate(
        [
            _quality(supported_by_evidence=0.2, needs_additional_retrieval=0.9),
            _quality(),
        ]
    )
    calls = []

    async def retrieve(request, candidate, action, retry_number):
        calls.append((request, candidate, action, retry_number))
        return [
            _agent_result(
                evidence=[
                    Evidence(
                        evidence_id="fresh-evidence",
                        content="Fresh supported fact",
                        source_id="fresh-source",
                        relevance_score=0.95,
                    )
                ]
            )
        ]

    controller = _controller(gate, ["Evidence-backed candidate"], retrieval=retrieve)
    plan = ExecutionPlan(
        agents=[AgentName.WEB_RESEARCH],
        web_allowed=True,
        use_web=True,
        citations_required=True,
    )

    outcome = asyncio.run(
        controller.run(
            request_text="Give current facts",
            plan=plan,
            initial_synthesis=_initial(),
            agent_results=[],
        )
    )

    assert outcome.status is QualityStatus.ACCEPTED
    assert calls[0][2] is QualityAction.ADDITIONAL_RETRIEVAL
    assert len(gate.states[1].supporting_evidence.items) == 1


def test_retry_controller_stops_at_configured_limit_with_best_response() -> None:
    gate = SequenceQualityGate([_quality(contains_unsupported_claims=0.9)])
    controller = _controller(gate, ["Retry one", "Retry two"], max_retries=2)

    outcome = asyncio.run(
        controller.run(
            request_text="Answer",
            plan=ExecutionPlan(agents=[AgentName.GENERAL]),
            initial_synthesis=_initial(),
            agent_results=[_agent_result()],
        )
    )

    assert outcome.status is QualityStatus.REVIEW
    assert outcome.retry_count == 2
    assert len(outcome.attempts) == 3
    assert "Quality review" in outcome.final_response


def test_retry_controller_does_not_loop_when_quality_gate_fails() -> None:
    failure = _quality().model_copy(
        update={"evaluation_succeeded": False, "error_type": "ConnectionError"}
    )
    controller = _controller(SequenceQualityGate([failure]), [])

    outcome = asyncio.run(
        controller.run(
            request_text="Answer",
            plan=ExecutionPlan(agents=[AgentName.GENERAL]),
            initial_synthesis=_initial(),
            agent_results=[_agent_result()],
        )
    )

    assert outcome.status is QualityStatus.REVIEW
    assert outcome.retry_count == 0
    assert len(outcome.attempts) == 1
    assert "quality gate was unavailable" in (outcome.uncertainty_note or "")


def test_retry_controller_returns_review_when_regeneration_fails() -> None:
    gate = SequenceQualityGate([_quality(contains_unsupported_claims=0.9)])
    controller = _controller(gate, [])

    outcome = asyncio.run(
        controller.run(
            request_text="Answer",
            plan=ExecutionPlan(agents=[AgentName.GENERAL]),
            initial_synthesis=_initial(),
            agent_results=[_agent_result()],
        )
    )

    assert outcome.status is QualityStatus.REVIEW
    assert outcome.retry_count == 1
    assert outcome.controller_errors == ["regeneration failed: RuntimeError"]
    assert "regeneration failed" in (outcome.uncertainty_note or "").lower()
