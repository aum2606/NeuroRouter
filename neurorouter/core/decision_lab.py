"""Counterfactual policy replay over persisted Jev routing probabilities."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from neurorouter.core.policy_engine import PolicyEngine
from neurorouter.schemas.execution import ExecutionPlan
from neurorouter.schemas.routing import RoutingDecision
from neurorouter.schemas.state import RouterState
from neurorouter.schemas.trace import TraceRecord
from neurorouter.utils.config import PolicyThresholds


class DecisionLabModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RouteChange(DecisionLabModel):
    field: str
    original: Any
    counterfactual: Any


class CounterfactualResult(DecisionLabModel):
    trace_id: str
    original_plan: ExecutionPlan
    counterfactual_plan: ExecutionPlan
    applied_thresholds: dict[str, float]
    changes: list[RouteChange] = Field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.changes)


def replay_policy(
    trace: TraceRecord,
    thresholds: PolicyThresholds,
    *,
    web_threshold: float,
    rag_threshold: float,
    code_threshold: float,
    data_threshold: float,
) -> CounterfactualResult:
    """Recompute policy from persisted state and probabilities without a Jev call."""
    if trace.routing_decision is None or trace.state is None or trace.execution_plan is None:
        raise ValueError("Trace does not contain the state, routing decision, and execution plan")
    decision = RoutingDecision.model_validate(trace.routing_decision)
    state = RouterState.model_validate(trace.state)
    original = ExecutionPlan.model_validate(trace.execution_plan)
    overrides = {
        "web_threshold": web_threshold,
        "rag_threshold": rag_threshold,
        "code_threshold": code_threshold,
        "data_threshold": data_threshold,
    }
    counterfactual_thresholds = thresholds.model_copy(
        update={"routing": thresholds.routing.model_copy(update=overrides)}
    )
    counterfactual = PolicyEngine(counterfactual_thresholds).create_plan(decision, state)
    changes = _changes(original, counterfactual)
    return CounterfactualResult(
        trace_id=str(trace.trace_id),
        original_plan=original,
        counterfactual_plan=counterfactual,
        applied_thresholds=overrides,
        changes=changes,
    )


def _changes(original: ExecutionPlan, counterfactual: ExecutionPlan) -> list[RouteChange]:
    fields = (
        "agents",
        "parallel_agents",
        "model_tier",
        "use_web",
        "use_rag",
        "use_code",
        "use_data_analysis",
        "citations_required",
        "quality_gate_required",
        "requires_review",
    )
    changes: list[RouteChange] = []
    for field in fields:
        before = getattr(original, field)
        after = getattr(counterfactual, field)
        if before != after:
            changes.append(
                RouteChange(
                    field=field,
                    original=_display_value(before),
                    counterfactual=_display_value(after),
                )
            )
    return changes


def _display_value(value: Any) -> Any:
    if isinstance(value, list):
        return [
            [_display_value(item) for item in child] if isinstance(child, list) else str(child)
            for child in value
        ]
    return value.value if hasattr(value, "value") else value
