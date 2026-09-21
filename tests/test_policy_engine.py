import pytest

from neurorouter.core.policy_engine import PolicyEngine
from neurorouter.core.state_builder import StateBuilder
from neurorouter.schemas.execution import AgentName, ModelTier
from neurorouter.schemas.routing import Intent, RoutingDecision
from neurorouter.schemas.state import CapabilityState
from neurorouter.utils.config import load_settings, load_thresholds


def _decision(**overrides: object) -> RoutingDecision:
    values: dict[str, object] = {
        "intent": Intent.GENERAL_QA,
        "intent_confidence": 0.9,
        "intent_probabilities": {Intent.GENERAL_QA: 1.0},
        "needs_web": 0.0,
        "needs_rag": 0.0,
        "needs_code": 0.0,
        "needs_data_analysis": 0.0,
        "needs_current_information": 0.0,
        "needs_citations": 0.0,
        "needs_multi_source_research": 0.0,
        "complexity_score": 0.2,
        "complexity_confidence": 0.9,
        "complexity_probabilities": {0: 0.8, 1: 0.2},
        "risk_score": 0.1,
        "risk_confidence": 0.9,
        "risk_probabilities": {0: 0.9, 1: 0.1},
        "jev_latency_ms": 10.0,
        "jev_model": "jev-test",
    }
    values.update(overrides)
    return RoutingDecision.model_validate(values)


def _state(*, web: bool = False, rag: bool = False, code: bool = False, finance: bool = False):
    settings = load_settings()
    state = StateBuilder(settings).build(
        "test request",
        indexed_document_count=1 if rag else 0,
    )
    return state.model_copy(
        update={
            "capabilities": CapabilityState(
                web_search_available=web,
                rag_available=rag,
                code_available=code,
                finance_agent_available=finance,
            )
        }
    )


def test_simple_request_uses_general_agent_and_fast_tier() -> None:
    plan = PolicyEngine(load_thresholds()).create_plan(_decision(), _state())

    assert plan.agents == [AgentName.GENERAL]
    assert plan.model_tier is ModelTier.FAST
    assert plan.quality_gate_required is False
    assert plan.requires_review is False


def test_independent_specialists_are_grouped_for_parallel_execution() -> None:
    decision = _decision(
        intent=Intent.MIXED,
        intent_probabilities={Intent.MIXED: 1.0},
        needs_web=0.9,
        needs_rag=0.8,
        needs_code=0.75,
        needs_data_analysis=0.8,
        needs_citations=0.9,
        complexity_score=2.5,
        complexity_probabilities={2: 0.5, 3: 0.5},
    )

    plan = PolicyEngine(load_thresholds()).create_plan(
        decision, _state(web=True, rag=True, code=True)
    )

    expected = [AgentName.WEB_RESEARCH, AgentName.RAG, AgentName.CODE]
    assert plan.agents == expected
    assert plan.parallel_agents == [expected]
    assert plan.model_tier is ModelTier.REASONING
    assert plan.use_data_analysis is True
    assert plan.citations_required is True


def test_finance_depends_on_fresh_web_evidence() -> None:
    decision = _decision(
        intent=Intent.FINANCE,
        intent_probabilities={Intent.FINANCE: 1.0},
        needs_web=0.9,
    )

    plan = PolicyEngine(load_thresholds()).create_plan(decision, _state(web=True, finance=True))

    assert plan.agents == [AgentName.WEB_RESEARCH, AgentName.FINANCE]
    assert plan.agent_dependencies == {AgentName.FINANCE: [AgentName.WEB_RESEARCH]}
    assert plan.parallel_agents == []


def test_missing_requested_capability_requires_review() -> None:
    decision = _decision(needs_web=0.9)

    plan = PolicyEngine(load_thresholds()).create_plan(decision, _state(web=False))

    assert plan.agents == [AgentName.GENERAL]
    assert plan.use_web is False
    assert plan.web_allowed is False
    assert plan.requires_review is True
    assert any("web search capability is unavailable" in reason for reason in plan.reasoning)


def test_threshold_equality_activates_capability_and_standard_tier() -> None:
    thresholds = load_thresholds()
    decision = _decision(
        needs_web=thresholds.routing.web_threshold,
        complexity_score=thresholds.planning.standard_model_complexity,
        complexity_probabilities={1: 1.0},
    )

    plan = PolicyEngine(thresholds).create_plan(decision, _state(web=True))

    assert plan.use_web is True
    assert plan.model_tier is ModelTier.STANDARD


def test_counterfactual_threshold_changes_plan_without_new_jev_call() -> None:
    original_thresholds = load_thresholds()
    decision = _decision(needs_web=0.8)
    original = PolicyEngine(original_thresholds).create_plan(decision, _state(web=True))
    counterfactual_thresholds = original_thresholds.model_copy(
        update={"routing": original_thresholds.routing.model_copy(update={"web_threshold": 0.95})}
    )
    counterfactual = PolicyEngine(counterfactual_thresholds).create_plan(decision, _state(web=True))

    assert original.use_web is True
    assert counterfactual.use_web is False
    assert counterfactual.agents == [AgentName.GENERAL]


@pytest.mark.parametrize(
    ("risk_score", "intent_confidence", "expected"),
    [(2.0, 0.9, True), (0.1, 0.44, True), (0.1, 0.9, False)],
)
def test_review_rules(risk_score: float, intent_confidence: float, expected: bool) -> None:
    decision = _decision(
        risk_score=risk_score,
        risk_probabilities={round(risk_score): 1.0},
        intent_confidence=intent_confidence,
    )

    plan = PolicyEngine(load_thresholds()).create_plan(decision, _state())

    assert plan.requires_review is expected
