from datetime import UTC, datetime

import pytest

from neurorouter.evals.repository import EvaluationRepository
from neurorouter.evals.schemas import EvaluationReport, RouterMetrics


def _report(name: str = "router") -> EvaluationReport:
    return EvaluationReport(
        router_name=name,
        dataset_name="dataset.jsonl",
        completed_at=datetime.now(UTC),
        predictions=[],
        metrics=RouterMetrics(
            evaluated_examples=0,
            failed_examples=0,
            intent_accuracy=0,
            intent_confusion_matrix={},
            capabilities={},
            macro_precision=0,
            macro_recall=0,
            macro_f1=0,
            mean_brier_score=0,
            calibration={},
            mean_latency_ms=0,
            total_llm_calls=0,
            unnecessary_tool_activations=0,
        ),
    )


def test_evaluation_repository_round_trip(tmp_path) -> None:
    repository = EvaluationRepository(tmp_path / "evaluations.db")
    report = _report()

    repository.save(report)

    assert repository.get(report.run_id) == report
    assert repository.list_reports() == [report]


def test_repository_rejects_invalid_limit(tmp_path) -> None:
    repository = EvaluationRepository(tmp_path / "evaluations.db")

    with pytest.raises(ValueError, match="between"):
        repository.list_reports(limit=0)
