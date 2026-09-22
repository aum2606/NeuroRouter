"""Strict normalization for the Jev quality-gate response."""

from collections.abc import Mapping
from typing import Any

from neurorouter.schemas.quality import QualityDecision

QUALITY_NOUL_NAMES = (
    "answers_request",
    "supported_by_evidence",
    "contains_unsupported_claims",
    "missing_important_information",
    "contradicts_evidence",
    "needs_additional_retrieval",
)


class QualityParseError(ValueError):
    """Raised when a Jev quality response is incomplete or malformed."""


def parse_quality_response(raw: Mapping[str, Any], *, latency_ms: float) -> QualityDecision:
    answers = raw.get("answers")
    if not isinstance(answers, Mapping):
        raise QualityParseError("Jev quality response is missing the answers mapping")
    values: dict[str, float] = {}
    for name in QUALITY_NOUL_NAMES:
        answer = answers.get(name)
        if not isinstance(answer, Mapping):
            raise QualityParseError(f"Jev quality response is missing answer: {name}")
        if answer.get("type") != "noul":
            raise QualityParseError(f"Quality answer {name!r} must have type 'noul'")
        try:
            values[name] = float(answer["noul"])
        except (KeyError, TypeError, ValueError) as error:
            raise QualityParseError(f"Invalid Noul answer for {name!r}") from error
    model = raw.get("model")
    return QualityDecision(
        **values,
        jev_latency_ms=latency_ms,
        jev_model=model if isinstance(model, str) else None,
    )
