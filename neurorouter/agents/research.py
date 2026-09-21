"""Bounded web research with normalized, deduplicated evidence."""

import asyncio
import hashlib
import re
from urllib.parse import urlsplit, urlunsplit

from neurorouter.agents.base import AgentInput, AgentPayload, BaseAgent, Evidence, Source
from neurorouter.schemas.execution import AgentName
from neurorouter.tools.web_search import WebSearchProvider, WebSearchResult


class WebResearchAgent(BaseAgent):
    """Execute a small query set and return ranked evidence rather than prose."""

    name = AgentName.WEB_RESEARCH

    def __init__(
        self,
        provider: WebSearchProvider,
        *,
        max_queries: int = 3,
        results_per_query: int = 5,
    ) -> None:
        self.provider = provider
        self.max_queries = max_queries
        self.results_per_query = results_per_query

    async def execute(self, agent_input: AgentInput) -> AgentPayload:
        queries = self._queries(agent_input)
        collected: list[tuple[str, WebSearchResult]] = []
        responses = await asyncio.gather(
            *(self.provider.search(query, max_results=self.results_per_query) for query in queries),
            return_exceptions=True,
        )
        query_errors: list[str] = []
        for query, response in zip(queries, responses, strict=True):
            if isinstance(response, Exception):
                query_errors.append(type(response).__name__)
                continue
            collected.extend((query, result) for result in response)
        if query_errors and not collected:
            raise RuntimeError(
                "All web-search queries failed: " + ", ".join(sorted(set(query_errors)))
            )

        ranked = self._deduplicate_and_rank(agent_input.request_text, collected)
        sources: list[Source] = []
        evidence: list[Evidence] = []
        for query, result, relevance in ranked:
            source_id = _stable_id(result.url)
            sources.append(
                Source(
                    source_id=source_id,
                    title=result.title,
                    url=result.url,
                    provider=result.provider,
                    metadata=result.metadata,
                )
            )
            content = result.snippet or result.title
            evidence.append(
                Evidence(
                    evidence_id=_stable_id(f"{query}:{result.url}"),
                    content=content,
                    source_id=source_id,
                    relevance_score=relevance,
                    metadata={"query": query, "provider_score": result.score},
                )
            )

        return AgentPayload(
            output=f"Retrieved {len(evidence)} unique evidence items from {len(queries)} queries.",
            evidence=evidence,
            sources=sources,
            metadata={
                "queries": queries,
                "provider": type(self.provider).__name__,
                "result_count": len(evidence),
                "query_error_types": query_errors,
            },
        )

    def _queries(self, agent_input: AgentInput) -> list[str]:
        supplied = agent_input.context.get("search_queries")
        if isinstance(supplied, list):
            queries = [str(query).strip() for query in supplied if str(query).strip()]
        else:
            queries = [agent_input.request_text.strip()]
        return list(dict.fromkeys(queries))[: self.max_queries]

    @classmethod
    def _deduplicate_and_rank(
        cls,
        request_text: str,
        collected: list[tuple[str, WebSearchResult]],
    ) -> list[tuple[str, WebSearchResult, float]]:
        request_tokens = cls._tokens(request_text)
        unique: dict[str, tuple[str, WebSearchResult, float]] = {}
        for query, result in collected:
            key = cls._canonical_url(result.url)
            content_tokens = cls._tokens(f"{result.title} {result.snippet}")
            overlap = len(request_tokens & content_tokens) / max(len(request_tokens), 1)
            relevance = min(1.0, (0.6 * overlap) + (0.4 * result.score))
            candidate = (query, result, relevance)
            if key not in unique or relevance > unique[key][2]:
                unique[key] = candidate
        return sorted(unique.values(), key=lambda item: (-item[2], item[1].url))

    @staticmethod
    def _canonical_url(url: str) -> str:
        parts = urlsplit(url)
        return urlunsplit(
            (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", "")
        )

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) > 2}


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
