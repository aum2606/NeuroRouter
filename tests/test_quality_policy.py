import pytest

from neurorouter.core.quality_policy import QualityPolicy
from neurorouter.schemas.execution import AgentName, ExecutionPlan
from neurorouter.schemas.quality import (
    JevQualityResult,
    QualityAction,
    QualityDecision,
)
from neurorouter.utils.config import load_thresholds


def _result(**overrides) -> JevQualityResult:
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


@pytest.mark.parametrize(
    ("overrides", "retrieval", "expected"),
    [
        ({"contradicts_evidence": 0.8}, True, QualityAction.RECONCILE),
        ({"contains_unsupported_claims": 0.8}, True, QualityAction.REGENERATE),
        ({"answers_request": 0.2}, True, QualityAction.REGENERATE),
        ({"missing_important_information": 0.8}, True, QualityAction.ADDITIONAL_RETRIEVAL),
        ({"needs_additional_retrieval": 0.8}, False, QualityAction.REGENERATE),
    ],
)
def test_quality_policy_composes_atomic_probabilities(
    overrides: dict, retrieval: bool, expected: QualityAction
) -> None:
    policy = QualityPolicy(load_thresholds().quality)
    plan = ExecutionPlan(agents=[AgentName.GENERAL], citations_required=True)

    decision = policy.decide(_result(**overrides), plan, retrieval_available=retrieval)

    assert decision.action is expected
    assert decision.accepted is False


def test_quality_policy_accepts_only_when_all_thresholds_pass() -> None:
    decision = QualityPolicy(load_thresholds().quality).decide(
        _result(),
        ExecutionPlan(agents=[AgentName.GENERAL]),
        retrieval_available=False,
    )

    assert decision.action is QualityAction.ACCEPT
    assert decision.accepted is True


def test_existing_evidence_makes_support_threshold_relevant_without_citations() -> None:
    decision = QualityPolicy(load_thresholds().quality).decide(
        _result(supported_by_evidence=0.2),
        ExecutionPlan(agents=[AgentName.RAG], rag_allowed=True, use_rag=True),
        retrieval_available=True,
        evidence_available=True,
    )

    assert decision.action is QualityAction.ADDITIONAL_RETRIEVAL


def test_quality_policy_reviews_gate_failure_and_plan_escalation() -> None:
    policy = QualityPolicy(load_thresholds().quality)
    failure = _result().model_copy(
        update={"evaluation_succeeded": False, "error_type": "ConnectionError"}
    )
    failed = policy.decide(
        failure, ExecutionPlan(agents=[AgentName.GENERAL]), retrieval_available=True
    )
    planned = policy.decide(
        _result(),
        ExecutionPlan(agents=[AgentName.GENERAL], requires_review=True),
        retrieval_available=True,
    )

    assert failed.action is QualityAction.REVIEW
    assert planned.action is QualityAction.REVIEW
