"""Atomic Jev questions for candidate-response quality control."""

from typesafe_sdk import Noul, Question


def build_quality_questions() -> dict[str, Question]:
    """Build six independent Nouls evaluated against one QualityState."""

    return {
        "answers_request": _noul(
            "The candidate_response directly answers the original_request in this state.",
            true="It addresses the user's requested task or question.",
            false="It evades, misunderstands, or fails to answer the requested task.",
        ),
        "supported_by_evidence": _noul(
            "Claims in candidate_response that require support are justified by "
            "supporting_evidence in this state.",
            true="Material factual claims are supported by the supplied evidence.",
            false="Material factual claims lack support in the supplied evidence.",
        ),
        "contains_unsupported_claims": _noul(
            "The candidate_response contains material factual claims that are not supported by "
            "supporting_evidence in this state.",
            true="At least one material claim lacks supplied support.",
            false="No material claim lacks required supplied support.",
        ),
        "missing_important_information": _noul(
            "The candidate_response omits information necessary to satisfy original_request.",
            true="Important information needed for a useful answer is missing.",
            false="No necessary information is materially missing.",
        ),
        "contradicts_evidence": _noul(
            "The candidate_response materially contradicts supporting_evidence in this state.",
            true="At least one response claim conflicts with supplied evidence.",
            false="The response does not conflict with supplied evidence.",
        ),
        "needs_additional_retrieval": _noul(
            "Additional external or knowledge-base retrieval is required before "
            "original_request can be answered reliably.",
            true="The available evidence is insufficient and more retrieval is needed.",
            false="The available evidence is sufficient for the requested answer.",
        ),
    }


def _noul(proposition: str, *, true: str, false: str) -> Noul:
    return Noul(instructions=proposition, criteria={"true": true, "false": false})
