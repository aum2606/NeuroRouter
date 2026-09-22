"""Dependency-free classification and calibration metrics for router evaluations."""

from statistics import fmean

from neurorouter.evals.schemas import (
    BinaryMetrics,
    CalibrationBin,
    RouterMetrics,
    RoutingEvalCase,
    RoutingPrediction,
)
from neurorouter.schemas.routing import Intent

CAPABILITIES = {
    "needs_web": ("needs_web", "expected_needs_web"),
    "needs_rag": ("needs_rag", "expected_needs_rag"),
    "needs_code": ("needs_code", "expected_needs_code"),
    "needs_data": ("needs_data_analysis", "expected_needs_data"),
}


def evaluate_routing_predictions(
    cases: list[RoutingEvalCase],
    predictions: list[RoutingPrediction],
    *,
    probability_threshold: float = 0.5,
    calibration_bins: int = 10,
) -> RouterMetrics:
    """Calculate metrics from observed predictions without external ML dependencies."""
    if not 0 <= probability_threshold <= 1:
        raise ValueError("probability_threshold must be between 0 and 1")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least 2")
    case_by_id = {case.id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("evaluation cases must have unique ids")
    prediction_ids = [prediction.example_id for prediction in predictions]
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("predictions must have unique example ids")
    unknown = set(prediction_ids) - case_by_id.keys()
    if unknown:
        raise ValueError(f"prediction references unknown example ids: {sorted(unknown)}")
    successful = [
        (case_by_id[prediction.example_id], prediction)
        for prediction in predictions
        if prediction.decision is not None
    ]
    failed_examples = len(cases) - len(successful)

    labels = [intent.value for intent in Intent]
    confusion = {expected: {predicted: 0 for predicted in labels} for expected in labels}
    intent_correct = 0
    for case, prediction in successful:
        predicted = prediction.decision.intent.value  # type: ignore[union-attr]
        confusion[case.expected_intent.value][predicted] += 1
        intent_correct += predicted == case.expected_intent.value

    capability_metrics: dict[str, BinaryMetrics] = {}
    calibration: dict[str, list[CalibrationBin]] = {}
    unnecessary = 0
    for label, (prediction_field, expected_field) in CAPABILITIES.items():
        probabilities = [
            float(getattr(prediction.decision, prediction_field))
            for _, prediction in successful
            if prediction.decision is not None
        ]
        expected = [bool(getattr(case, expected_field)) for case, _ in successful]
        metrics = _binary_metrics(expected, probabilities, probability_threshold)
        capability_metrics[label] = metrics
        unnecessary += metrics.false_positive
        calibration[label] = _calibration(expected, probabilities, calibration_bins)

    complexity_pairs = [
        (case.expected_complexity, prediction.decision.complexity_score)
        for case, prediction in successful
        if case.expected_complexity is not None and prediction.decision is not None
    ]
    complexity_mae = (
        fmean(abs(expected - predicted) for expected, predicted in complexity_pairs)
        if complexity_pairs
        else None
    )
    complexity_accuracy = (
        fmean(float(round(predicted) == expected) for expected, predicted in complexity_pairs)
        if complexity_pairs
        else None
    )
    binary_values = list(capability_metrics.values())
    costs = [
        prediction.estimated_cost_usd
        for prediction in predictions
        if prediction.estimated_cost_usd is not None
    ]
    return RouterMetrics(
        evaluated_examples=len(successful),
        failed_examples=failed_examples,
        intent_accuracy=_divide(intent_correct, len(successful)),
        intent_confusion_matrix=confusion,
        capabilities=capability_metrics,
        macro_precision=_mean([metric.precision for metric in binary_values]),
        macro_recall=_mean([metric.recall for metric in binary_values]),
        macro_f1=_mean([metric.f1 for metric in binary_values]),
        mean_brier_score=_mean([metric.brier_score for metric in binary_values]),
        calibration=calibration,
        complexity_mae=complexity_mae,
        complexity_accuracy=complexity_accuracy,
        mean_latency_ms=_mean([prediction.observed_latency_ms for prediction in predictions]),
        total_llm_calls=sum(prediction.llm_calls for prediction in predictions),
        estimated_cost_usd=(sum(costs) if predictions and len(costs) == len(predictions) else None),
        unnecessary_tool_activations=unnecessary,
    )


def _binary_metrics(
    expected: list[bool], probabilities: list[float], threshold: float
) -> BinaryMetrics:
    predicted = [probability >= threshold for probability in probabilities]
    true_positive = sum(
        actual and estimate for actual, estimate in zip(expected, predicted, strict=True)
    )
    false_positive = sum(
        not actual and estimate for actual, estimate in zip(expected, predicted, strict=True)
    )
    true_negative = sum(
        not actual and not estimate for actual, estimate in zip(expected, predicted, strict=True)
    )
    false_negative = sum(
        actual and not estimate for actual, estimate in zip(expected, predicted, strict=True)
    )
    precision = _divide(true_positive, true_positive + false_positive)
    recall = _divide(true_positive, true_positive + false_negative)
    return BinaryMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=_divide(2 * precision * recall, precision + recall),
        accuracy=_divide(true_positive + true_negative, len(expected)),
        brier_score=_mean(
            [
                (probability - float(actual)) ** 2
                for actual, probability in zip(expected, probabilities, strict=True)
            ]
        ),
    )


def _calibration(
    expected: list[bool], probabilities: list[float], bin_count: int
) -> list[CalibrationBin]:
    buckets: list[list[tuple[bool, float]]] = [[] for _ in range(bin_count)]
    for actual, probability in zip(expected, probabilities, strict=True):
        buckets[min(int(probability * bin_count), bin_count - 1)].append((actual, probability))
    result: list[CalibrationBin] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        result.append(
            CalibrationBin(
                lower_bound=index / bin_count,
                upper_bound=(index + 1) / bin_count,
                mean_probability=fmean(probability for _, probability in bucket),
                observed_rate=fmean(float(actual) for actual, _ in bucket),
                count=len(bucket),
            )
        )
    return result


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0
