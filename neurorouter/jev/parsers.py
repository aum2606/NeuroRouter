"""Strict normalization of raw System One responses."""

from collections.abc import Mapping
from typing import Any

from neurorouter.schemas.routing import Intent, RoutingDecision


class RoutingParseError(ValueError):
    """Raised when a Jev response is incomplete or structurally invalid."""


NOUL_NAMES = (
    "needs_web",
    "needs_rag",
    "needs_code",
    "needs_data_analysis",
    "needs_current_information",
    "needs_citations",
    "needs_multi_source_research",
)


def parse_routing_response(raw: Mapping[str, Any], *, latency_ms: float) -> RoutingDecision:
    """Convert a raw SDK-shaped response into the stable routing contract."""

    answers = raw.get("answers")
    if not isinstance(answers, Mapping):
        raise RoutingParseError("Jev response is missing the answers mapping")

    intent_answer = _answer(answers, "intent", "choice")
    complexity_answer = _answer(answers, "complexity", "score")
    risk_answer = _answer(answers, "risk", "score")

    try:
        intent = Intent(str(intent_answer["choice"]))
        intent_probabilities = {
            Intent(key): float(value)
            for key, value in _mapping(intent_answer, "probabilities").items()
        }
        nouls = {name: float(_answer(answers, name, "noul")["noul"]) for name in NOUL_NAMES}
        model = raw.get("model")
        return RoutingDecision(
            intent=intent,
            intent_confidence=float(intent_answer["confidence"]),
            intent_probabilities=intent_probabilities,
            **nouls,
            complexity_score=float(complexity_answer["score"]),
            complexity_confidence=float(complexity_answer["confidence"]),
            complexity_probabilities=_integer_probabilities(complexity_answer),
            risk_score=float(risk_answer["score"]),
            risk_confidence=float(risk_answer["confidence"]),
            risk_probabilities=_integer_probabilities(risk_answer),
            jev_latency_ms=latency_ms,
            jev_model=model if isinstance(model, str) else None,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RoutingParseError(f"Invalid Jev routing answer: {error}") from error


def _answer(answers: Mapping[str, Any], name: str, expected_type: str) -> Mapping[str, Any]:
    answer = answers.get(name)
    if not isinstance(answer, Mapping):
        raise RoutingParseError(f"Jev response is missing answer: {name}")
    if answer.get("type") != expected_type:
        raise RoutingParseError(f"Answer {name!r} must have type {expected_type!r}")
    return answer


def _mapping(answer: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = answer.get(field)
    if not isinstance(value, Mapping):
        raise RoutingParseError(f"Answer field {field!r} must be a mapping")
    return value


def _integer_probabilities(answer: Mapping[str, Any]) -> dict[int, float]:
    return {int(key): float(value) for key, value in _mapping(answer, "probabilities").items()}
