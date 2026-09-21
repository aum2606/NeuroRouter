"""Deterministic conversion from probabilistic routing to an execution plan."""

from neurorouter.schemas.execution import AgentName, ExecutionPlan, ModelTier
from neurorouter.schemas.routing import Intent, RoutingDecision
from neurorouter.schemas.state import RouterState
from neurorouter.utils.config import PolicyThresholds


class PolicyEngine:
    """Apply configured business rules without making model calls."""

    def __init__(self, thresholds: PolicyThresholds) -> None:
        self.thresholds = thresholds

    def create_plan(self, decision: RoutingDecision, state: RouterState) -> ExecutionPlan:
        routing = self.thresholds.routing
        planning = self.thresholds.planning
        reasons: list[str] = []
        missing_capabilities: list[str] = []

        intent_is_confident = (
            decision.intent_confidence >= planning.intent_specialist_minimum_confidence
        )
        web_probability_triggered = decision.needs_web >= routing.web_threshold
        current_information_triggered = decision.needs_current_information >= routing.web_threshold
        web_requested = web_probability_triggered or current_information_triggered
        rag_probability_triggered = decision.needs_rag >= routing.rag_threshold
        rag_intent_triggered = intent_is_confident and decision.intent is Intent.DOCUMENT_QA
        rag_requested = rag_probability_triggered or rag_intent_triggered
        code_probability_triggered = decision.needs_code >= routing.code_threshold
        code_intent_triggered = intent_is_confident and decision.intent is Intent.CODING
        code_requested = code_probability_triggered or code_intent_triggered
        data_probability_triggered = decision.needs_data_analysis >= routing.data_threshold
        data_intent_triggered = intent_is_confident and decision.intent is Intent.DATA_ANALYSIS
        data_requested = data_probability_triggered or data_intent_triggered
        finance_requested = intent_is_confident and decision.intent is Intent.FINANCE

        web_allowed = state.capabilities.web_search_available
        rag_allowed = (
            state.capabilities.rag_available and state.knowledge_base.has_indexed_documents
        )
        code_allowed = state.capabilities.code_available
        finance_allowed = state.capabilities.finance_agent_available

        use_web = web_requested and web_allowed
        use_rag = rag_requested and rag_allowed
        use_data_analysis = data_requested and code_allowed
        use_code = (code_requested or data_requested) and code_allowed
        use_finance = finance_requested and finance_allowed

        self._explain_probability_trigger(
            reasons,
            "WebResearchAgent",
            "needs_web",
            decision.needs_web,
            routing.web_threshold,
            web_probability_triggered,
        )
        if current_information_triggered:
            reasons.append(
                "WebResearchAgent requested because "
                f"needs_current_information={decision.needs_current_information:.3f} met "
                f"web_threshold={routing.web_threshold:.3f}."
            )
        if rag_intent_triggered:
            reasons.append(
                "RAGAgent requested because intent=document_qa and "
                f"intent_confidence={decision.intent_confidence:.3f} met "
                f"threshold={planning.intent_specialist_minimum_confidence:.3f}."
            )
        elif rag_probability_triggered:
            reasons.append(
                "RAGAgent requested because "
                f"needs_rag={decision.needs_rag:.3f} met threshold={routing.rag_threshold:.3f}."
            )
        if code_intent_triggered:
            reasons.append(
                "CodeAgent requested because intent=coding and "
                f"intent_confidence={decision.intent_confidence:.3f} met "
                f"threshold={planning.intent_specialist_minimum_confidence:.3f}."
            )
        elif code_probability_triggered:
            reasons.append(
                "CodeAgent requested because "
                f"needs_code={decision.needs_code:.3f} met threshold={routing.code_threshold:.3f}."
            )
        if data_requested:
            trigger = "intent=data_analysis" if data_intent_triggered else "needs_data_analysis"
            reasons.append(f"Data-analysis capability requested by {trigger}.")
        if finance_requested:
            reasons.append(
                "FinanceAgent requested because intent=finance and "
                f"intent_confidence={decision.intent_confidence:.3f} met "
                f"threshold={planning.intent_specialist_minimum_confidence:.3f}."
            )

        for requested, allowed, capability, agent in (
            (web_requested, web_allowed, "web search", AgentName.WEB_RESEARCH),
            (rag_requested, rag_allowed, "indexed RAG", AgentName.RAG),
            (code_requested or data_requested, code_allowed, "code", AgentName.CODE),
            (finance_requested, finance_allowed, "finance", AgentName.FINANCE),
        ):
            if requested and not allowed:
                missing_capabilities.append(capability)
                reasons.append(
                    f"{agent.value} blocked because {capability} capability is unavailable."
                )

        agents: list[AgentName] = []
        if use_web:
            agents.append(AgentName.WEB_RESEARCH)
        if use_rag:
            agents.append(AgentName.RAG)
        if use_code:
            agents.append(AgentName.CODE)
        if use_finance:
            agents.append(AgentName.FINANCE)
        if not agents:
            agents.append(AgentName.GENERAL)
            reasons.append("GeneralAgent selected because no available specialist was activated.")

        dependencies: dict[AgentName, list[AgentName]] = {}
        if use_finance and use_web:
            dependencies[AgentName.FINANCE] = [AgentName.WEB_RESEARCH]
            reasons.append(
                "FinanceAgent scheduled after WebResearchAgent to consume fresh evidence."
            )
        independent_agents = [agent for agent in agents if agent not in dependencies]
        parallel_agents = [independent_agents] if len(independent_agents) > 1 else []

        model_tier = self._model_tier(decision.complexity_score)
        reasons.append(self._model_tier_reason(model_tier, decision.complexity_score))

        citations_required = (
            decision.needs_citations >= routing.citations_threshold
            or use_web
            or use_rag
            or use_finance
        )
        if citations_required:
            reasons.append(
                "Citations required by citation probability or an evidence-producing specialist."
            )

        quality_gate_required = (
            decision.complexity_score >= planning.quality_gate_complexity_threshold
            or decision.risk_score >= planning.quality_gate_risk_threshold
            or citations_required
            or any(agent is not AgentName.GENERAL for agent in agents)
        )
        if quality_gate_required:
            reasons.append(
                "Quality gate required by configured complexity, risk, or evidence rules."
            )

        confidences = (
            decision.intent_confidence,
            decision.complexity_confidence,
            decision.risk_confidence,
        )
        low_confidence = min(confidences) < planning.review_confidence_threshold
        high_risk = decision.risk_score >= planning.review_risk_threshold
        missing_requires_review = (
            bool(missing_capabilities) and planning.require_review_on_missing_capability
        )
        requires_review = high_risk or low_confidence or missing_requires_review
        if high_risk:
            reasons.append(
                f"Review required because risk_score={decision.risk_score:.3f} met "
                f"threshold={planning.review_risk_threshold:.3f}."
            )
        if low_confidence:
            reasons.append(
                "Review required because at least one routing confidence was below "
                f"threshold={planning.review_confidence_threshold:.3f}."
            )
        if missing_requires_review:
            reasons.append(
                "Review required because requested capabilities were unavailable: "
                + ", ".join(missing_capabilities)
                + "."
            )

        return ExecutionPlan(
            agents=agents,
            parallel_agents=parallel_agents,
            agent_dependencies=dependencies,
            model_tier=model_tier,
            web_allowed=web_allowed,
            rag_allowed=rag_allowed,
            code_allowed=code_allowed,
            use_web=use_web,
            use_rag=use_rag,
            use_code=use_code,
            use_data_analysis=use_data_analysis,
            citations_required=citations_required,
            quality_gate_required=quality_gate_required,
            requires_review=requires_review,
            reasoning=reasons,
        )

    @staticmethod
    def _explain_probability_trigger(
        reasons: list[str],
        agent: str,
        signal: str,
        probability: float,
        threshold: float,
        triggered: bool,
    ) -> None:
        if triggered:
            reasons.append(
                f"{agent} requested because {signal}={probability:.3f} met "
                f"threshold={threshold:.3f}."
            )

    def _model_tier(self, complexity_score: float) -> ModelTier:
        planning = self.thresholds.planning
        if complexity_score >= planning.reasoning_model_complexity:
            return ModelTier.REASONING
        if complexity_score >= planning.standard_model_complexity:
            return ModelTier.STANDARD
        return ModelTier.FAST

    def _model_tier_reason(self, tier: ModelTier, complexity_score: float) -> str:
        planning = self.thresholds.planning
        if tier is ModelTier.REASONING:
            threshold = planning.reasoning_model_complexity
            comparison = "met"
        elif tier is ModelTier.STANDARD:
            threshold = planning.standard_model_complexity
            comparison = "met"
        else:
            threshold = planning.standard_model_complexity
            comparison = "was below"
        return (
            f"{tier.value.capitalize()} model tier selected because "
            f"complexity_score={complexity_score:.3f} {comparison} threshold={threshold:.3f}."
        )
