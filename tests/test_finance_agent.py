import asyncio
from uuid import uuid4

from neurorouter.agents.base import AgentInput
from neurorouter.agents.finance import FinanceAgent
from neurorouter.core.state_builder import StateBuilder
from neurorouter.schemas.execution import AgentName, ExecutionPlan
from neurorouter.utils.config import load_settings


def _finance_input(*, use_web: bool, context: dict | None = None) -> AgentInput:
    plan = ExecutionPlan(
        agents=[AgentName.WEB_RESEARCH, AgentName.FINANCE] if use_web else [AgentName.FINANCE],
        agent_dependencies={AgentName.FINANCE: [AgentName.WEB_RESEARCH]} if use_web else {},
        web_allowed=use_web,
        use_web=use_web,
    )
    return AgentInput(
        trace_id=uuid4(),
        request_text="Compare the companies' financial performance",
        state=StateBuilder(load_settings()).build("Compare financial performance"),
        plan=plan,
        context=context or {},
    )


def _web_dependency() -> dict:
    return {
        "success": True,
        "evidence": [
            {
                "evidence_id": "earnings-1",
                "content": "Company reported quarterly revenue.",
                "source_id": "filing-1",
                "relevance_score": 0.9,
                "metadata": {"period": "Q2"},
            }
        ],
        "sources": [
            {
                "source_id": "filing-1",
                "title": "Quarterly filing",
                "url": "https://example.com/filing",
                "provider": "web",
                "metadata": {},
            }
        ],
    }


def test_finance_agent_calculates_transparent_ratios() -> None:
    context = {
        "financial_data": {
            "Alpha": {
                "revenue": 200,
                "previous_revenue": 160,
                "cost_of_revenue": 80,
                "net_income": 30,
                "current_assets": 90,
                "current_liabilities": 45,
                "total_debt": 50,
                "shareholders_equity": 100,
            }
        }
    }

    result = asyncio.run(FinanceAgent().run(_finance_input(use_web=False, context=context)))

    ratios = result.metadata["ratios"]["Alpha"]
    assert result.success is True
    assert ratios["gross_margin"] == 0.6
    assert ratios["net_margin"] == 0.15
    assert ratios["current_ratio"] == 2.0
    assert ratios["debt_to_equity"] == 0.5
    assert ratios["revenue_growth"] == 0.25
    assert result.sources[0].provider == "user_input"


def test_finance_agent_preserves_fresh_web_evidence() -> None:
    context = {"dependency_results": {AgentName.WEB_RESEARCH.value: _web_dependency()}}

    result = asyncio.run(FinanceAgent().run(_finance_input(use_web=True, context=context)))

    assert result.success is True
    assert result.metadata["grounded"] is True
    assert result.metadata["web_evidence_count"] == 1
    assert result.evidence[0].evidence_id == "earnings-1"


def test_finance_agent_blocks_current_claims_without_web_evidence() -> None:
    context = {
        "dependency_results": {
            AgentName.WEB_RESEARCH.value: {"success": True, "evidence": [], "sources": []}
        }
    }

    result = asyncio.run(FinanceAgent().run(_finance_input(use_web=True, context=context)))

    assert result.success is True
    assert result.metadata["grounded"] is False
    assert result.metadata["fresh_evidence_missing"] is True
    assert result.metadata["requires_review"] is True
    assert "must not be generated" in result.output


def test_finance_agent_is_explicit_when_no_evidence_exists() -> None:
    result = asyncio.run(FinanceAgent().run(_finance_input(use_web=False)))

    assert result.success is True
    assert result.evidence == []
    assert result.metadata["requires_review"] is True
    assert "avoid invented figures" in result.output
