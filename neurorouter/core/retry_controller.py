"""Bounded deterministic response-quality retry lifecycle."""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Protocol

from neurorouter.agents.base import AgentResult
from neurorouter.core.aggregator import ContextAggregator
from neurorouter.core.quality_policy import QualityPolicy
from neurorouter.core.synthesizer import SynthesisResult, Synthesizer
from neurorouter.llm.prompts import SynthesisPrompt
from neurorouter.schemas.execution import ExecutionPlan
from neurorouter.schemas.quality import (
    JevQualityResult,
    QualityAction,
    QualityAttempt,
    QualityOutcome,
    QualityState,
    QualityStatus,
)
from neurorouter.schemas.trace import StageStatus
from neurorouter.telemetry.tracer import RequestTracer


class QualityEvaluator(Protocol):
    async def evaluate(self, state: QualityState) -> JevQualityResult: ...


AdditionalRetrieval = Callable[[str, str, QualityAction, int], Awaitable[list[AgentResult]]]


class RetryController:
    """Compose quality decisions with at most ``max_retries`` new generations."""

    def __init__(
        self,
        *,
        quality_gate: QualityEvaluator,
        quality_policy: QualityPolicy,
        synthesizer: Synthesizer,
        aggregator: ContextAggregator,
        synthesis_prompt: SynthesisPrompt,
        max_retries: int = 2,
        additional_retrieval: AdditionalRetrieval | None = None,
        tracer: RequestTracer | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.quality_gate = quality_gate
        self.quality_policy = quality_policy
        self.synthesizer = synthesizer
        self.aggregator = aggregator
        self.synthesis_prompt = synthesis_prompt
        self.max_retries = max_retries
        self.additional_retrieval = additional_retrieval
        self.tracer = tracer

    async def run(
        self,
        *,
        request_text: str,
        plan: ExecutionPlan,
        initial_synthesis: SynthesisResult,
        agent_results: list[AgentResult],
    ) -> QualityOutcome:
        results = list(agent_results)
        evidence = self.aggregator.aggregate(results)
        synthesis = initial_synthesis
        attempts: list[QualityAttempt] = []
        errors: list[str] = []
        best_response = synthesis.response
        best_score = float("-inf")
        retry_count = 0

        while True:
            quality = await self.quality_gate.evaluate(
                QualityState(
                    original_request=request_text,
                    candidate_response=synthesis.response,
                    supporting_evidence=evidence,
                    agents_used=list(dict.fromkeys(result.agent_name for result in results)),
                    execution_metadata={
                        "retry_index": retry_count,
                        "model_tier": plan.model_tier.value,
                        "citations_required": plan.citations_required,
                        "agents_succeeded": sum(result.success for result in results),
                    },
                )
            )
            policy = self.quality_policy.decide(
                quality,
                plan,
                retrieval_available=self.additional_retrieval is not None,
                evidence_available=bool(evidence.items),
            )
            attempt = QualityAttempt(
                attempt_number=retry_count,
                candidate_response=synthesis.response,
                quality=quality,
                policy=policy,
            )
            attempts.append(attempt)
            score = self._quality_score(quality, evidence_required=plan.citations_required)
            if score > best_score:
                best_score = score
                best_response = synthesis.response
            self._record_policy(attempt)

            if policy.action is QualityAction.ACCEPT:
                return QualityOutcome(
                    final_response=synthesis.response,
                    status=QualityStatus.ACCEPTED,
                    retry_count=retry_count,
                    attempts=attempts,
                    controller_errors=errors,
                )
            if policy.action is QualityAction.REVIEW:
                note = policy.reasoning[0]
                return self._review_outcome(
                    best_response,
                    note,
                    retry_count,
                    attempts,
                    errors,
                )
            if retry_count >= self.max_retries:
                note = (
                    f"Automated quality checks did not pass after {retry_count} retries; "
                    "verify uncertain or unsupported claims."
                )
                return self._review_outcome(
                    best_response,
                    note,
                    retry_count,
                    attempts,
                    errors,
                )

            retry_count += 1
            action = policy.action
            effective_action = action
            if action in {QualityAction.ADDITIONAL_RETRIEVAL, QualityAction.RECONCILE}:
                try:
                    if self.additional_retrieval is not None:
                        added = await self.additional_retrieval(
                            request_text,
                            synthesis.response,
                            action,
                            retry_count,
                        )
                        results.extend(added)
                        evidence = self.aggregator.aggregate(results)
                        if action is QualityAction.ADDITIONAL_RETRIEVAL and not added:
                            effective_action = QualityAction.REGENERATE
                except Exception as error:
                    errors.append(f"additional retrieval failed: {type(error).__name__}")
                    effective_action = QualityAction.REGENERATE

            instruction = self.synthesis_prompt.retry_instructions.get(effective_action.value)
            if not instruction:
                instruction = self.synthesis_prompt.retry_instructions.get("regenerate")
            try:
                synthesis = await self.synthesizer.synthesize(
                    request_text=request_text,
                    plan=plan,
                    evidence=evidence,
                    retry_instruction=instruction,
                )
            except Exception as error:
                errors.append(f"regeneration failed: {type(error).__name__}")
                note = "Response regeneration failed; verify the best available candidate."
                return self._review_outcome(
                    best_response,
                    note,
                    retry_count,
                    attempts,
                    errors,
                )

    @staticmethod
    def _quality_score(result: JevQualityResult, *, evidence_required: bool) -> float:
        if not result.evaluation_succeeded:
            return -10.0
        quality = result.decision
        support = quality.supported_by_evidence if evidence_required else 1.0
        return (
            quality.answers_request
            + support
            - quality.contains_unsupported_claims
            - quality.missing_important_information
            - quality.contradicts_evidence
            - quality.needs_additional_retrieval
        )

    @staticmethod
    def _review_outcome(
        response: str,
        note: str,
        retry_count: int,
        attempts: list[QualityAttempt],
        errors: list[str],
    ) -> QualityOutcome:
        final_response = f"{response.rstrip()}\n\n> Quality review: {note}"
        return QualityOutcome(
            final_response=final_response,
            status=QualityStatus.REVIEW,
            retry_count=retry_count,
            attempts=attempts,
            uncertainty_note=note,
            controller_errors=errors,
        )

    def _record_policy(self, attempt: QualityAttempt) -> None:
        if not self.tracer:
            return
        started_at = datetime.now(UTC)
        self.tracer.record_stage(
            stage="retry_control",
            component=type(self).__name__,
            status=StageStatus.SUCCEEDED,
            started_at=started_at,
            dependencies=["quality_gate"],
            metadata={
                "attempt_number": attempt.attempt_number,
                "action": attempt.policy.action.value,
                "accepted": attempt.policy.accepted,
                "requires_review": attempt.policy.requires_review,
                "reasoning": attempt.policy.reasoning,
            },
        )
