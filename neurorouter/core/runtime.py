"""Dependency wiring for one fully traced NeuroRouter request."""

from dataclasses import dataclass, field
from typing import Any

from neurorouter.agents import CodeAgent, FinanceAgent, GeneralAgent, RAGAgent, WebResearchAgent
from neurorouter.agents.base import BaseAgent
from neurorouter.core.aggregator import ContextAggregator
from neurorouter.core.orchestrator import Orchestrator
from neurorouter.core.pipeline import ExecutionPipeline
from neurorouter.core.policy_engine import PolicyEngine
from neurorouter.core.quality_gate import JevQualityGate
from neurorouter.core.quality_policy import QualityPolicy
from neurorouter.core.retry_controller import RetryController
from neurorouter.core.router import JevRouter
from neurorouter.core.state_builder import StateBuilder
from neurorouter.core.synthesizer import Synthesizer
from neurorouter.jev.client import JevClient, TypeSafeJevClient
from neurorouter.llm.model_router import ModelRouter
from neurorouter.llm.prompts import SynthesisPromptRenderer, load_prompts
from neurorouter.llm.providers import build_llm_provider
from neurorouter.rag.chunker import Chunker
from neurorouter.rag.embeddings import HashingEmbeddingProvider
from neurorouter.rag.index import ChromaVectorStore
from neurorouter.rag.retriever import Retriever
from neurorouter.schemas.execution import AgentName
from neurorouter.telemetry.tracer import RequestTracer
from neurorouter.tools.code_runner import build_code_execution_tool
from neurorouter.tools.web_search import WikipediaSearchProvider
from neurorouter.utils.config import (
    PROJECT_ROOT,
    PolicyThresholds,
    RuntimeSettings,
    SecretSettings,
    load_settings,
    load_thresholds,
)


class _UnavailableJevClient:
    """Fail immediately so configured fallback policy can run without a secret."""

    async def evaluate(self, *, state: Any, questions: Any) -> dict[str, Any]:
        del state, questions
        raise RuntimeError("TYPESAFE_API_KEY is not configured")


@dataclass(slots=True)
class RuntimeComponents:
    """Request-scoped services sharing one tracer."""

    settings: RuntimeSettings
    thresholds: PolicyThresholds
    state_builder: StateBuilder
    router: JevRouter
    policy_engine: PolicyEngine
    pipeline: ExecutionPipeline
    indexed_document_count: int = 0
    collection_metadata: dict[str, Any] = field(default_factory=dict)


def build_runtime(
    tracer: RequestTracer,
    *,
    settings: RuntimeSettings | None = None,
    thresholds: PolicyThresholds | None = None,
    secrets: SecretSettings | None = None,
    jev_client: JevClient | None = None,
) -> RuntimeComponents:
    """Build the local runtime without global mutable service instances."""
    runtime_settings = settings or load_settings()
    policy_thresholds = thresholds or load_thresholds()
    secret_values = secrets or SecretSettings()
    client = jev_client or _build_jev_client(runtime_settings, secret_values)

    indexed_documents = 0
    collection_metadata: dict[str, Any] = {}
    agents: dict[AgentName, BaseAgent] = {AgentName.GENERAL: GeneralAgent()}

    if runtime_settings.capabilities.web_search_available:
        agents[AgentName.WEB_RESEARCH] = WebResearchAgent(
            WikipediaSearchProvider(timeout_seconds=runtime_settings.web_search.timeout_seconds),
            max_queries=runtime_settings.web_search.max_queries,
            results_per_query=runtime_settings.web_search.results_per_query,
        )
    if runtime_settings.capabilities.code_available:
        agents[AgentName.CODE] = CodeAgent(
            build_code_execution_tool(runtime_settings.code_execution)
        )
    if runtime_settings.capabilities.finance_agent_available:
        agents[AgentName.FINANCE] = FinanceAgent(
            require_web_for_current_data=runtime_settings.finance.require_web_for_current_data,
            ratio_precision=runtime_settings.finance.ratio_precision,
        )

    if runtime_settings.capabilities.rag_available:
        try:
            store = ChromaVectorStore(
                PROJECT_ROOT / runtime_settings.rag.persist_directory,
                runtime_settings.rag.collection_name,
            )
            info = store.collection_info()
            indexed_documents = info.document_count
            collection_metadata = info.model_dump(mode="json")
            agents[AgentName.RAG] = RAGAgent(
                Retriever(
                    chunker=Chunker(
                        chunk_size=runtime_settings.rag.chunk_size,
                        overlap=runtime_settings.rag.chunk_overlap,
                    ),
                    embeddings=HashingEmbeddingProvider(runtime_settings.rag.embedding_dimensions),
                    vector_store=store,
                    top_k=runtime_settings.rag.retrieval_top_k,
                    minimum_relevance=runtime_settings.rag.minimum_relevance,
                )
            )
        except Exception as error:  # optional dependency/storage failure is observable state
            collection_metadata = {
                "name": runtime_settings.rag.collection_name,
                "status": "unavailable",
                "error_type": type(error).__name__,
            }

    provider = build_llm_provider(runtime_settings.llm, secret_values)
    prompts = load_prompts()
    aggregator = ContextAggregator(
        context_budget_characters=runtime_settings.aggregation.context_budget_characters,
        max_agent_output_characters=runtime_settings.aggregation.max_agent_output_characters,
    )
    synthesizer = Synthesizer(
        provider=provider,
        model_router=ModelRouter(runtime_settings.llm),
        prompt_renderer=SynthesisPromptRenderer(prompts.synthesis),
        settings=runtime_settings.llm,
        tracer=tracer,
    )
    quality_gate = JevQualityGate(client, tracer=tracer)
    retry_controller = RetryController(
        quality_gate=quality_gate,
        quality_policy=QualityPolicy(policy_thresholds.quality),
        synthesizer=synthesizer,
        aggregator=aggregator,
        synthesis_prompt=prompts.synthesis,
        max_retries=policy_thresholds.quality.max_retries,
        tracer=tracer,
    )
    return RuntimeComponents(
        settings=runtime_settings,
        thresholds=policy_thresholds,
        state_builder=StateBuilder(runtime_settings),
        router=JevRouter(client, runtime_settings.jev.fallback, tracer=tracer),
        policy_engine=PolicyEngine(policy_thresholds),
        pipeline=ExecutionPipeline(
            orchestrator=Orchestrator(agents, tracer=tracer),
            aggregator=aggregator,
            synthesizer=synthesizer,
            retry_controller=retry_controller,
            tracer=tracer,
        ),
        indexed_document_count=indexed_documents,
        collection_metadata=collection_metadata,
    )


def _build_jev_client(settings: RuntimeSettings, secrets: SecretSettings) -> JevClient:
    if not secrets.typesafe_api_key:
        return _UnavailableJevClient()
    return TypeSafeJevClient(
        api_key=secrets.typesafe_api_key,
        model=settings.jev.model,
        timeout_seconds=settings.jev.timeout_seconds,
    )
