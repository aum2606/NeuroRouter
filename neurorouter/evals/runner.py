"""Async evaluation runner with an injectable router prediction boundary."""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol

from neurorouter.core.router import JevRouter
from neurorouter.core.state_builder import StateBuilder
from neurorouter.evals.metrics import evaluate_routing_predictions
from neurorouter.evals.schemas import EvaluationReport, RoutingEvalCase, RoutingPrediction


class RoutingPredictor(Protocol):
    async def predict(self, case: RoutingEvalCase) -> RoutingPrediction: ...


class JevRoutingPredictor:
    """Adapt the production state builder and Jev router to the evaluation boundary."""

    def __init__(self, state_builder: StateBuilder, router: JevRouter) -> None:
        self.state_builder = state_builder
        self.router = router

    async def predict(self, case: RoutingEvalCase) -> RoutingPrediction:
        started = perf_counter()
        state = self.state_builder.build(
            case.query,
            attachment_types=case.attachment_types,
            indexed_document_count=case.indexed_document_count,
            collection_metadata={
                "evaluation_fixture": True,
                "document_count": case.indexed_document_count,
            },
        )
        result = await self.router.route(state)
        return RoutingPrediction(
            example_id=case.id,
            decision=result.decision,
            observed_latency_ms=(perf_counter() - started) * 1000,
            llm_calls=0,
            metadata={
                "jev_model": result.decision.jev_model,
                "fallback_applied": bool(result.raw_response.get("fallback_applied")),
            },
        )


class EvaluationRunner:
    """Execute labeled cases with bounded concurrency and isolate case failures."""

    def __init__(
        self,
        *,
        probability_threshold: float = 0.5,
        calibration_bins: int = 10,
        max_concurrency: int = 4,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self.probability_threshold = probability_threshold
        self.calibration_bins = calibration_bins
        self.max_concurrency = max_concurrency

    async def run(
        self,
        cases: Sequence[RoutingEvalCase],
        predictor: RoutingPredictor,
        *,
        router_name: str,
        dataset_name: str,
    ) -> EvaluationReport:
        if not cases:
            raise ValueError("at least one evaluation case is required")
        started_at = datetime.now(UTC)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def execute(case: RoutingEvalCase) -> RoutingPrediction:
            started = perf_counter()
            try:
                async with semaphore:
                    prediction = await predictor.predict(case)
                if prediction.example_id != case.id:
                    raise ValueError("predictor returned a mismatched example id")
                return prediction
            except Exception as error:
                return RoutingPrediction(
                    example_id=case.id,
                    observed_latency_ms=(perf_counter() - started) * 1000,
                    error=f"{type(error).__name__}: {error}",
                )

        predictions = list(await asyncio.gather(*(execute(case) for case in cases)))
        metrics = evaluate_routing_predictions(
            list(cases),
            predictions,
            probability_threshold=self.probability_threshold,
            calibration_bins=self.calibration_bins,
        )
        return EvaluationReport(
            router_name=router_name,
            dataset_name=dataset_name,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            predictions=predictions,
            metrics=metrics,
        )
