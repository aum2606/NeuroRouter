import pytest

from neurorouter.evals.metrics import evaluate_routing_predictions
from neurorouter.evals.schemas import RoutingEvalCase, RoutingPrediction
from neurorouter.schemas.routing import Intent, RoutingDecision


def _decision(intent: Intent, *, web: float, code: float) -> RoutingDecision:
    return RoutingDecision(
        intent=intent,
        intent_confidence=0.9,
        intent_probabilities={intent: 1.0},
        needs_web=web,
        needs_rag=0.1,
        needs_code=code,
        needs_data_analysis=0.1,
        needs_current_information=web,
        needs_citations=web,
        needs_multi_source_research=0.1,
        complexity_score=1,
        complexity_confidence=0.9,
        complexity_probabilities={1: 1.0},
        risk_score=0,
        risk_confidence=0.9,
        risk_probabilities={0: 1.0},
        jev_latency_ms=10,
    )


def _case(identifier: str, intent: Intent, *, web: bool, code: bool) -> RoutingEvalCase:
    return RoutingEvalCase(
        id=identifier,
        query=identifier,
        expected_intent=intent,
        expected_needs_web=web,
        expected_needs_rag=False,
        expected_needs_code=code,
        expected_needs_data=False,
        expected_complexity=1,
    )


def test_computes_classification_brier_and_complexity_metrics() -> None:
    cases = [
        _case("research", Intent.RESEARCH, web=True, code=False),
        _case("coding", Intent.CODING, web=False, code=True),
    ]
    predictions = [
        RoutingPrediction(
            example_id="research",
            decision=_decision(Intent.RESEARCH, web=0.9, code=0.6),
            observed_latency_ms=10,
        ),
        RoutingPrediction(
            example_id="coding",
            decision=_decision(Intent.GENERAL_QA, web=0.4, code=0.8),
            observed_latency_ms=30,
        ),
    ]

    metrics = evaluate_routing_predictions(cases, predictions, calibration_bins=5)

    assert metrics.intent_accuracy == 0.5
    assert metrics.intent_confusion_matrix["coding"]["general_qa"] == 1
    assert metrics.capabilities["needs_web"].brier_score == pytest.approx(0.085)
    assert metrics.capabilities["needs_code"].precision == 0.5
    assert metrics.unnecessary_tool_activations == 1
    assert metrics.complexity_mae == 0
    assert metrics.complexity_accuracy == 1
    assert metrics.mean_latency_ms == 20
    assert metrics.estimated_cost_usd is None


def test_failed_predictions_reduce_coverage_without_fabricating_scores() -> None:
    cases = [_case("failed", Intent.RESEARCH, web=True, code=False)]
    predictions = [RoutingPrediction(example_id="failed", observed_latency_ms=4, error="timeout")]

    metrics = evaluate_routing_predictions(cases, predictions)

    assert metrics.evaluated_examples == 0
    assert metrics.failed_examples == 1
    assert metrics.intent_accuracy == 0
    assert metrics.mean_latency_ms == 4


def test_rejects_predictions_for_unknown_cases() -> None:
    prediction = RoutingPrediction(
        example_id="unknown",
        decision=_decision(Intent.GENERAL_QA, web=0, code=0),
        observed_latency_ms=1,
    )

    with pytest.raises(ValueError, match="unknown example"):
        evaluate_routing_predictions(
            [_case("known", Intent.GENERAL_QA, web=False, code=False)],
            [prediction],
        )
