from datetime import UTC, datetime

import pytest

from neurorouter.evals.comparison import compare_reports
from neurorouter.evals.schemas import EvaluationReport, RouterMetrics


def _report(name: str, *, accuracy: float, latency: float, dataset: str = "same.jsonl"):
    return EvaluationReport(
        router_name=name,
        dataset_name=dataset,
        completed_at=datetime.now(UTC),
        predictions=[],
        metrics=RouterMetrics(
            evaluated_examples=10,
            failed_examples=0,
            intent_accuracy=accuracy,
            intent_confusion_matrix={},
            capabilities={},
            macro_precision=accuracy,
            macro_recall=accuracy,
            macro_f1=accuracy,
            mean_brier_score=1 - accuracy,
            calibration={},
            mean_latency_ms=latency,
            total_llm_calls=10 if name == "baseline" else 0,
            unnecessary_tool_activations=2,
        ),
    )


def test_compares_only_measured_report_values() -> None:
    comparison = compare_reports(
        _report("baseline", accuracy=0.7, latency=100),
        _report("NeuroRouter", accuracy=0.9, latency=40),
    )
    metrics = {metric.metric: metric for metric in comparison.metrics}

    assert metrics["intent_accuracy"].delta == pytest.approx(0.2)
    assert metrics["mean_latency_ms"].delta == -60
    assert metrics["total_llm_calls"].candidate == 0
    assert metrics["estimated_cost_usd"].delta is None


def test_rejects_incomparable_datasets() -> None:
    with pytest.raises(ValueError, match="same dataset"):
        compare_reports(
            _report("baseline", accuracy=0.7, latency=100, dataset="one"),
            _report("candidate", accuracy=0.8, latency=90, dataset="two"),
        )
