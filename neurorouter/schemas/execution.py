"""Deterministic execution-plan contract used from Phase 3 onward."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentName(StrEnum):
    GENERAL = "GeneralAgent"
    WEB_RESEARCH = "WebResearchAgent"
    RAG = "RAGAgent"
    CODE = "CodeAgent"
    FINANCE = "FinanceAgent"


class ModelTier(StrEnum):
    FAST = "fast"
    STANDARD = "standard"
    REASONING = "reasoning"


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agents: list[AgentName] = Field(default_factory=list)
    parallel_agents: list[list[AgentName]] = Field(default_factory=list)
    agent_dependencies: dict[AgentName, list[AgentName]] = Field(default_factory=dict)
    model_tier: ModelTier = ModelTier.FAST
    web_allowed: bool = False
    rag_allowed: bool = False
    code_allowed: bool = False
    use_web: bool = False
    use_rag: bool = False
    use_code: bool = False
    use_data_analysis: bool = False
    citations_required: bool = False
    quality_gate_required: bool = True
    requires_review: bool = False
    reasoning: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_execution_graph(self) -> "ExecutionPlan":
        if len(self.agents) != len(set(self.agents)):
            raise ValueError("agents cannot contain duplicates")
        known = set(self.agents)
        parallel_members = [agent for group in self.parallel_agents for agent in group]
        if len(parallel_members) != len(set(parallel_members)):
            raise ValueError("an agent cannot appear in multiple parallel groups")
        if not set(parallel_members) <= known:
            raise ValueError("parallel agents must be present in agents")
        for agent, dependencies in self.agent_dependencies.items():
            if agent not in known or not set(dependencies) <= known:
                raise ValueError("agent dependencies must reference planned agents")
            if agent in dependencies:
                raise ValueError("an agent cannot depend on itself")
        if self.use_web and not self.web_allowed:
            raise ValueError("use_web requires web_allowed")
        if self.use_rag and not self.rag_allowed:
            raise ValueError("use_rag requires rag_allowed")
        if (self.use_code or self.use_data_analysis) and not self.code_allowed:
            raise ValueError("code or data analysis use requires code_allowed")
        return self
