"""Atomic TypeSafe questions used by the initial routing call."""

from typesafe_sdk import Choice, Noul, Question, Score

INTENT_CRITERIA = {
    "general_qa": "A direct explanatory or factual request that does not require a specialist.",
    "research": (
        "A request to investigate external information, compare sources, or produce research."
    ),
    "document_qa": "A request whose answer should come from attached or indexed documents.",
    "coding": "A programming, software design, code generation, or debugging request.",
    "data_analysis": "A request to inspect, transform, calculate, or draw conclusions from data.",
    "finance": "A request about companies, financial statements, earnings, ratios, or markets.",
    "mixed": "A request that materially combines two or more specialist intent categories.",
}


def build_routing_questions() -> dict[str, Question]:
    """Build one batch of independent questions over the same RouterState."""

    return {
        "intent": Choice(
            instructions=(
                "Considering state.request.text and only the context in this state, which single "
                "intent best characterizes the user's primary request?"
            ),
            criteria=INTENT_CRITERIA,
        ),
        "needs_web": _noul(
            "The request in state.request.text requires retrieving information from the public web "
            "to answer correctly."
        ),
        "needs_rag": _noul(
            "The request in state.request.text requires evidence from the attachments or indexed "
            "knowledge base described in this state."
        ),
        "needs_code": _noul(
            "The request in state.request.text requires programming expertise, code generation, "
            "code inspection, or debugging."
        ),
        "needs_data_analysis": _noul(
            "The request in state.request.text requires calculations, structured data processing, "
            "statistical analysis, or interpretation of a dataset."
        ),
        "needs_current_information": _noul(
            "A correct answer to state.request.text depends on information that may have changed "
            "recently or must be current at request time."
        ),
        "needs_citations": _noul(
            "A trustworthy answer to state.request.text should cite inspectable supporting sources."
        ),
        "needs_multi_source_research": _noul(
            "Answering state.request.text well requires comparing or combining evidence from more "
            "than one independent source."
        ),
        "complexity": Score(
            instructions=(
                "Rate the execution complexity of answering state.request.text, independent of "
                "topic risk and writing length."
            ),
            criteria=[
                "Simple factual or direct request with no specialist operation.",
                "Moderate reasoning or one bounded specialist operation.",
                "Complex multi-step reasoning or coordination across capabilities.",
                "Highly complex research or multi-agent task with substantial coordination.",
            ],
        ),
        "risk": Score(
            instructions=(
                "Rate the consequence risk of acting on the answer to state.request.text, "
                "independent of task complexity."
            ),
            criteria=[
                "Harmless or read-only with negligible consequence.",
                "Low consequence; mistakes are easy to notice and reverse.",
                "Meaningful consequence requiring care or verification.",
                "Potentially high consequence involving safety, legal, financial, or "
                "irreversible impact.",
            ],
        ),
    }


def _noul(proposition: str) -> Noul:
    return Noul(
        instructions=proposition,
        criteria={
            "true": "The proposition is true for the request and supplied state.",
            "false": "The proposition is false for the request and supplied state.",
        },
    )
