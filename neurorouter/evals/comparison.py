"""Fair comparisons between two persisted, measured router runs."""

from neurorouter.evals.schemas import (
    ComparisonMetric,
    EvaluationReport,
    RouterComparison,
)


def compare_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
) -> RouterComparison:
    if baseline.dataset_name != candidate.dataset_name:
        raise ValueError("reports must use the same dataset")
    if baseline.metrics.evaluated_examples != candidate.metrics.evaluated_examples:
        raise ValueError("reports must contain the same number of evaluated examples")
    definitions = (
        ("intent_accuracy", False),
        ("macro_f1", False),
        ("mean_brier_score", True),
        ("mean_latency_ms", True),
        ("total_llm_calls", True),
        ("unnecessary_tool_activations", True),
        ("estimated_cost_usd", True),
    )
    metrics: list[ComparisonMetric] = []
    for name, lower_is_better in definitions:
        baseline_value = getattr(baseline.metrics, name)
        candidate_value = getattr(candidate.metrics, name)
        delta = (
            candidate_value - baseline_value
            if baseline_value is not None and candidate_value is not None
            else None
        )
        metrics.append(
            ComparisonMetric(
                metric=name,
                baseline=baseline_value,
                candidate=candidate_value,
                delta=delta,
                lower_is_better=lower_is_better,
            )
        )
    return RouterComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_name=baseline.router_name,
        candidate_name=candidate.router_name,
        metrics=metrics,
    )
