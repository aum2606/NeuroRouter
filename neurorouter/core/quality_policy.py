"""Deterministic quality policy composed from independent Jev probabilities."""

from neurorouter.schemas.execution import ExecutionPlan
from neurorouter.schemas.quality import (
    JevQualityResult,
    QualityAction,
    QualityPolicyDecision,
)
from neurorouter.utils.config import QualityThresholds


class QualityPolicy:
    """Convert atomic quality probabilities into one bounded controller action."""

    def __init__(self, thresholds: QualityThresholds) -> None:
        self.thresholds = thresholds

    def decide(
        self,
        result: JevQualityResult,
        plan: ExecutionPlan,
        *,
        retrieval_available: bool,
        evidence_available: bool = False,
    ) -> QualityPolicyDecision:
        if not result.evaluation_succeeded:
            return self._decision(
                QualityAction.REVIEW,
                "Review required because the Jev quality gate was unavailable.",
                review=True,
            )
        quality = result.decision
        thresholds = self.thresholds
        if quality.contradicts_evidence > thresholds.contradiction_maximum:
            action = QualityAction.RECONCILE if retrieval_available else QualityAction.REGENERATE
            return self._decision(
                action,
                "Evidence contradiction probability "
                f"{quality.contradicts_evidence:.3f} exceeded maximum "
                f"{thresholds.contradiction_maximum:.3f}.",
            )
        if quality.contains_unsupported_claims > thresholds.unsupported_claims_maximum:
            return self._decision(
                QualityAction.REGENERATE,
                "Unsupported-claim probability "
                f"{quality.contains_unsupported_claims:.3f} exceeded maximum "
                f"{thresholds.unsupported_claims_maximum:.3f}.",
            )
        if quality.answers_request < thresholds.answers_request_minimum:
            return self._decision(
                QualityAction.REGENERATE,
                f"answers_request={quality.answers_request:.3f} was below minimum "
                f"{thresholds.answers_request_minimum:.3f}.",
            )
        evidence_weak = (
            plan.citations_required or evidence_available
        ) and quality.supported_by_evidence < thresholds.evidence_support_minimum
        retrieval_needed = (
            quality.needs_additional_retrieval >= thresholds.additional_retrieval_threshold
            or quality.missing_important_information > thresholds.missing_information_maximum
            or evidence_weak
        )
        if retrieval_needed:
            action = (
                QualityAction.ADDITIONAL_RETRIEVAL
                if retrieval_available
                else QualityAction.REGENERATE
            )
            return self._decision(
                action,
                "Additional evidence or regeneration required by missing-information, "
                "support, or retrieval thresholds.",
            )
        if plan.requires_review:
            return self._decision(
                QualityAction.REVIEW,
                "Execution plan requires human review even though quality thresholds passed.",
                review=True,
            )
        return QualityPolicyDecision(
            action=QualityAction.ACCEPT,
            accepted=True,
            reasoning=["All configured deterministic quality thresholds passed."],
        )

    @staticmethod
    def _decision(
        action: QualityAction,
        reason: str,
        *,
        review: bool = False,
    ) -> QualityPolicyDecision:
        return QualityPolicyDecision(
            action=action,
            requires_review=review,
            reasoning=[reason],
        )
