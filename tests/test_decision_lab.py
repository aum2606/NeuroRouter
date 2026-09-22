import pytest

from neurorouter.core.decision_lab import replay_policy
from neurorouter.core.policy_engine import PolicyEngine
from neurorouter.core.state_builder import StateBuilder
from neurorouter.schemas.routing import Intent, RoutingDecision
from neurorouter.schemas.trace import TraceRecord
from neurorouter.utils.config import load_settings, load_thresholds


def _trace() -> TraceRecord:
    decision = RoutingDecision(
        intent=Intent.RESEARCH,
        intent_confidence=0.9,
        intent_probabilities={Intent.RESEARCH: 1.0},
        needs_web=0.8,
        needs_rag=0.0,
        needs_code=0.0,
        needs_data_analysis=0.0,
        needs_current_information=0.0,
        needs_citations=0.0,
        needs_multi_source_research=0.0,
        complexity_score=1.0,
        complexity_confidence=0.9,
        complexity_probabilities={1: 1.0},
        risk_score=0.0,
        risk_confidence=0.9,
        risk_probabilities={0: 1.0},
        jev_latency_ms=5,
    )
    state = StateBuilder(load_settings()).build("Research routing policy")
    original = PolicyEngine(load_thresholds()).create_plan(decision, state)
    return TraceRecord(
        request_text=state.request.text,
        state=state.model_dump(mode="json"),
        routing_decision=decision.model_dump(mode="json"),
        execution_plan=original.model_dump(mode="json"),
    )


def test_replays_policy_without_mutating_stored_decision() -> None:
    trace = _trace()
    original_decision = trace.routing_decision.copy()

    result = replay_policy(
        trace,
        load_thresholds(),
        web_threshold=0.9,
        rag_threshold=0.65,
        code_threshold=0.7,
        data_threshold=0.7,
    )

    assert result.original_plan.use_web is True
    assert result.counterfactual_plan.use_web is False
    assert result.counterfactual_plan.agents[0].value == "GeneralAgent"
    assert any(change.field == "agents" for change in result.changes)
    assert trace.routing_decision == original_decision


def test_replay_requires_complete_trace() -> None:
    with pytest.raises(ValueError, match="does not contain"):
        replay_policy(
            TraceRecord(request_text="legacy"),
            load_thresholds(),
            web_threshold=0.5,
            rag_threshold=0.5,
            code_threshold=0.5,
            data_threshold=0.5,
        )
