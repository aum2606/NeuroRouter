"""Normalize, deduplicate, rank, and trim specialist context."""

import hashlib
import re

from neurorouter.agents.base import AgentResult, Source
from neurorouter.schemas.context import (
    AgentContext,
    ContextItem,
    ContextSource,
    EvidencePacket,
)


class ContextAggregator:
    """Build a source-preserving packet under a deterministic character budget."""

    def __init__(
        self,
        *,
        context_budget_characters: int = 24000,
        max_agent_output_characters: int = 2000,
    ) -> None:
        if context_budget_characters < 1 or max_agent_output_characters < 1:
            raise ValueError("context and agent-output limits must be positive")
        self.context_budget_characters = context_budget_characters
        self.max_agent_output_characters = max_agent_output_characters

    def aggregate(self, results: list[AgentResult]) -> EvidencePacket:
        output_budget = min(
            self.context_budget_characters // 4,
            len(results) * self.max_agent_output_characters,
        )
        outputs: list[AgentContext] = []
        output_characters = 0
        for result in results:
            remaining = output_budget - output_characters
            normalized_output = self._normalize(result.output)
            rendered_output = normalized_output[: min(self.max_agent_output_characters, remaining)]
            outputs.append(
                AgentContext(
                    agent_name=result.agent_name,
                    success=result.success,
                    output=rendered_output,
                    error=result.error.message if result.error else None,
                )
            )
            output_characters += len(rendered_output)

        source_lookup: dict[str, Source] = {}
        candidates: dict[str, ContextItem] = {}
        duplicates = 0
        for result in results:
            if not result.success:
                continue
            source_lookup.update({source.source_id: source for source in result.sources})
            for evidence in result.evidence:
                normalized = self._normalize(evidence.content)
                if not normalized:
                    continue
                key = hashlib.sha256(normalized.casefold().encode()).hexdigest()
                item = ContextItem(
                    item_id=evidence.evidence_id,
                    content=normalized,
                    agent_name=result.agent_name,
                    relevance_score=evidence.relevance_score,
                    source_id=evidence.source_id,
                    metadata=evidence.metadata,
                )
                existing = candidates.get(key)
                if existing is not None:
                    duplicates += 1
                    if item.relevance_score <= existing.relevance_score:
                        continue
                candidates[key] = item

        ranked = sorted(
            candidates.values(),
            key=lambda item: (-item.relevance_score, item.agent_name.value, item.item_id),
        )
        included: list[ContextItem] = []
        consumed = 0
        evidence_budget = self.context_budget_characters - output_characters
        for item in ranked:
            remaining = evidence_budget - consumed
            if remaining <= 0:
                break
            if len(item.content) > remaining:
                if remaining < 80:
                    break
                item = item.model_copy(update={"content": item.content[: remaining - 1] + "…"})
            included.append(item)
            consumed += len(item.content)

        citation_labels: dict[str, str] = {}
        for item in included:
            if item.source_id and item.source_id in source_lookup:
                citation_labels.setdefault(item.source_id, f"S{len(citation_labels) + 1}")
        included = [
            item.model_copy(update={"citation_label": citation_labels.get(item.source_id or "")})
            for item in included
        ]
        sources = [
            ContextSource(
                **source_lookup[source_id].model_dump(),
                citation_label=label,
            )
            for source_id, label in citation_labels.items()
        ]
        return EvidencePacket(
            items=included,
            sources=sources,
            agent_outputs=outputs,
            total_characters=consumed + output_characters,
            dropped_items=len(ranked) - len(included),
            duplicate_items=duplicates,
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()
