"""Evidence-bounded financial analysis specialist."""

from collections.abc import Mapping
from typing import Any

from neurorouter.agents.base import AgentInput, AgentPayload, BaseAgent, Evidence, Source
from neurorouter.schemas.execution import AgentName


class FinanceAgent(BaseAgent):
    """Calculate transparent ratios and preserve upstream financial evidence."""

    name = AgentName.FINANCE

    def __init__(self, *, require_web_for_current_data: bool = True, ratio_precision: int = 4):
        self.require_web_for_current_data = require_web_for_current_data
        self.ratio_precision = ratio_precision

    async def execute(self, agent_input: AgentInput) -> AgentPayload:
        web_evidence, web_sources = self._web_context(agent_input)
        if self.require_web_for_current_data and agent_input.plan.use_web and not web_evidence:
            return AgentPayload(
                output=(
                    "Fresh financial information was requested, but WebResearchAgent supplied "
                    "no evidence. Current-market claims must not be generated."
                ),
                metadata={
                    "grounded": False,
                    "requires_review": True,
                    "fresh_evidence_missing": True,
                    "ratios": {},
                },
            )

        financial_data = agent_input.context.get("financial_data")
        analyses = self._analyze(financial_data)
        evidence = list(web_evidence)
        sources = list(web_sources)
        if analyses:
            source_id = "user-financial-data"
            sources.append(
                Source(
                    source_id=source_id,
                    title="User-provided financial data",
                    url="input://financial-data",
                    provider="user_input",
                )
            )
            evidence.extend(
                Evidence(
                    evidence_id=f"calculated-ratios-{company}",
                    content=self._format_ratios(company, ratios),
                    source_id=source_id,
                    relevance_score=1.0,
                    metadata={"company": company, "calculation": "deterministic_ratios"},
                )
                for company, ratios in analyses.items()
            )

        if analyses:
            summary = "; ".join(
                self._format_ratios(company, ratios) for company, ratios in analyses.items()
            )
            output = f"Calculated financial metrics from supplied values: {summary}"
        elif web_evidence:
            output = (
                f"Prepared {len(web_evidence)} fresh financial evidence items for synthesis. "
                "No structured statement values were supplied for ratio calculation."
            )
        else:
            output = (
                "No grounded financial evidence or structured statement values were supplied. "
                "The synthesis stage must state this limitation and avoid invented figures."
            )

        return AgentPayload(
            output=output,
            evidence=self._deduplicate_evidence(evidence),
            sources=self._deduplicate_sources(sources),
            metadata={
                "grounded": bool(evidence),
                "requires_review": not bool(evidence),
                "fresh_evidence_missing": False,
                "ratios": analyses,
                "web_evidence_count": len(web_evidence),
            },
        )

    def _analyze(self, value: Any) -> dict[str, dict[str, float]]:
        if not isinstance(value, Mapping) or not value:
            return {}
        if all(isinstance(item, (int, float)) for item in value.values()):
            datasets: Mapping[str, Any] = {"company": value}
        else:
            datasets = value
        analyses: dict[str, dict[str, float]] = {}
        for company, raw_data in datasets.items():
            if not isinstance(raw_data, Mapping):
                continue
            data = {
                str(key): float(number)
                for key, number in raw_data.items()
                if isinstance(number, (int, float)) and not isinstance(number, bool)
            }
            ratios = self._ratios(data)
            if ratios:
                analyses[str(company)] = ratios
        return analyses

    def _ratios(self, data: Mapping[str, float]) -> dict[str, float]:
        ratios: dict[str, float] = {}
        self._add_ratio(ratios, "gross_margin", data, "gross_profit", "revenue")
        if "gross_margin" not in ratios and {"revenue", "cost_of_revenue"} <= data.keys():
            revenue = data["revenue"]
            if revenue:
                ratios["gross_margin"] = self._round((revenue - data["cost_of_revenue"]) / revenue)
        self._add_ratio(ratios, "operating_margin", data, "operating_income", "revenue")
        self._add_ratio(ratios, "net_margin", data, "net_income", "revenue")
        self._add_ratio(ratios, "current_ratio", data, "current_assets", "current_liabilities")
        self._add_ratio(ratios, "debt_to_equity", data, "total_debt", "shareholders_equity")
        if {"revenue", "previous_revenue"} <= data.keys() and data["previous_revenue"]:
            ratios["revenue_growth"] = self._round(
                (data["revenue"] - data["previous_revenue"]) / data["previous_revenue"]
            )
        return ratios

    def _add_ratio(
        self,
        ratios: dict[str, float],
        name: str,
        data: Mapping[str, float],
        numerator: str,
        denominator: str,
    ) -> None:
        if {numerator, denominator} <= data.keys() and data[denominator]:
            ratios[name] = self._round(data[numerator] / data[denominator])

    def _round(self, value: float) -> float:
        return round(value, self.ratio_precision)

    @staticmethod
    def _web_context(agent_input: AgentInput) -> tuple[list[Evidence], list[Source]]:
        dependency = agent_input.context.get("dependency_results", {}).get(
            AgentName.WEB_RESEARCH.value, {}
        )
        if not isinstance(dependency, Mapping) or not dependency.get("success"):
            return [], []
        evidence = [Evidence.model_validate(item) for item in dependency.get("evidence", [])]
        sources = [Source.model_validate(item) for item in dependency.get("sources", [])]
        return evidence, sources

    @staticmethod
    def _format_ratios(company: str, ratios: Mapping[str, float]) -> str:
        rendered = ", ".join(f"{name}={value}" for name, value in sorted(ratios.items()))
        return f"{company}: {rendered}"

    @staticmethod
    def _deduplicate_evidence(items: list[Evidence]) -> list[Evidence]:
        return list({item.evidence_id: item for item in items}.values())

    @staticmethod
    def _deduplicate_sources(items: list[Source]) -> list[Source]:
        return list({item.source_id: item for item in items}.values())
