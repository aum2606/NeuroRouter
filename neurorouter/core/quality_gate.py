"""Second-stage Jev evaluation over candidate response quality."""

import logging
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from neurorouter.jev.client import JevClient
from neurorouter.jev.quality_parsers import parse_quality_response
from neurorouter.jev.quality_questions import build_quality_questions
from neurorouter.schemas.quality import JevQualityResult, QualityDecision, QualityState
from neurorouter.schemas.trace import StageStatus
from neurorouter.telemetry.tracer import RequestTracer

logger = logging.getLogger(__name__)


class JevQualityGate:
    """Evaluate atomic quality propositions and preserve the raw provider response."""

    def __init__(self, client: JevClient, *, tracer: RequestTracer | None = None) -> None:
        self.client = client
        self.tracer = tracer

    async def evaluate(self, state: QualityState) -> JevQualityResult:
        started_at = datetime.now(UTC)
        started = perf_counter()
        raw_response: dict[str, Any] = {}
        try:
            raw = await self.client.evaluate(
                state=state.model_dump(mode="json"),
                questions=build_quality_questions(),
            )
            raw_response = dict(raw)
            decision = parse_quality_response(raw, latency_ms=(perf_counter() - started) * 1000)
            result = JevQualityResult(decision=decision, raw_response=raw_response)
            self._record(
                result,
                started_at,
                retry_index=state.execution_metadata.get("retry_index"),
            )
            return result
        except Exception as error:
            logger.warning(
                "Jev quality evaluation failed",
                extra={"error_type": type(error).__name__},
            )
            result = JevQualityResult(
                decision=QualityDecision(
                    answers_request=0.0,
                    supported_by_evidence=0.0,
                    contains_unsupported_claims=1.0,
                    missing_important_information=1.0,
                    contradicts_evidence=0.0,
                    needs_additional_retrieval=1.0,
                    jev_latency_ms=(perf_counter() - started) * 1000,
                ),
                raw_response={
                    "quality_gate_failed": True,
                    "provider_response": raw_response,
                },
                evaluation_succeeded=False,
                error_type=type(error).__name__,
            )
            self._record(
                result,
                started_at,
                retry_index=state.execution_metadata.get("retry_index"),
                error=str(error),
            )
            return result

    def _record(
        self,
        result: JevQualityResult,
        started_at: datetime,
        *,
        retry_index: object,
        error: str | None = None,
    ) -> None:
        if not self.tracer:
            return
        self.tracer.record_stage(
            stage="quality_gate",
            component="JevQualityGate",
            status=StageStatus.SUCCEEDED if result.evaluation_succeeded else StageStatus.FAILED,
            started_at=started_at,
            dependencies=["synthesis"],
            metadata={
                "retry_index": retry_index,
                "decision": result.decision.model_dump(mode="json"),
                "evaluation_succeeded": result.evaluation_succeeded,
                "error_type": result.error_type,
            },
            error=error,
        )
