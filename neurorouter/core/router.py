"""Jev routing service with a conservative deterministic failure path."""

import logging
import math
from time import perf_counter

from neurorouter.jev.client import JevClient
from neurorouter.jev.parsers import parse_routing_response
from neurorouter.jev.routing_questions import build_routing_questions
from neurorouter.schemas.routing import Intent, JevRoutingResult, RoutingDecision
from neurorouter.schemas.state import RouterState
from neurorouter.utils.config import JevFallbackSettings

logger = logging.getLogger(__name__)


class JevRouter:
    """Batch atomic judgments and return a provider-neutral routing result."""

    def __init__(self, client: JevClient, fallback: JevFallbackSettings) -> None:
        self.client = client
        self.fallback = fallback

    async def route(self, state: RouterState) -> JevRoutingResult:
        started = perf_counter()
        try:
            raw = await self.client.evaluate(
                state=state.model_dump(mode="json"),
                questions=build_routing_questions(),
            )
            latency_ms = (perf_counter() - started) * 1000
            decision = parse_routing_response(raw, latency_ms=latency_ms)
            return JevRoutingResult(decision=decision, raw_response=dict(raw))
        except Exception as error:
            latency_ms = (perf_counter() - started) * 1000
            if not self.fallback.enabled:
                raise
            logger.warning(
                "Jev routing failed; applying configured fallback",
                extra={"error_type": type(error).__name__},
            )
            return self._fallback_result(state, latency_ms, error)

    def _fallback_result(
        self, state: RouterState, latency_ms: float, error: Exception
    ) -> JevRoutingResult:
        configured = self.fallback
        intent = Intent(configured.intent)
        decision = RoutingDecision(
            intent=intent,
            intent_confidence=configured.intent_confidence,
            intent_probabilities={intent: 1.0},
            needs_web=(configured.needs_web if state.capabilities.web_search_available else 0.0),
            needs_rag=configured.needs_rag if state.capabilities.rag_available else 0.0,
            needs_code=configured.needs_code if state.capabilities.code_available else 0.0,
            needs_data_analysis=configured.needs_data_analysis,
            needs_current_information=configured.needs_current_information,
            needs_citations=configured.needs_citations,
            needs_multi_source_research=configured.needs_multi_source_research,
            complexity_score=configured.complexity_score,
            complexity_confidence=configured.complexity_confidence,
            complexity_probabilities=_distribution_for_score(configured.complexity_score),
            risk_score=configured.risk_score,
            risk_confidence=configured.risk_confidence,
            risk_probabilities=_distribution_for_score(configured.risk_score),
            jev_latency_ms=latency_ms,
            jev_model=None,
        )
        return JevRoutingResult(
            decision=decision,
            raw_response={
                "fallback_applied": True,
                "error_type": type(error).__name__,
            },
        )


def _distribution_for_score(score: float) -> dict[int, float]:
    lower = math.floor(score)
    upper = math.ceil(score)
    if lower == upper:
        return {lower: 1.0}
    return {lower: upper - score, upper: score - lower}
