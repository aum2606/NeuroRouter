"""Measured routing evaluation, calibration, and comparison dashboard."""

import asyncio

import plotly.graph_objects as go
import streamlit as st

from neurorouter.core.router import JevRouter
from neurorouter.core.state_builder import StateBuilder
from neurorouter.evals.comparison import compare_reports
from neurorouter.evals.dataset import load_routing_dataset
from neurorouter.evals.repository import EvaluationRepository
from neurorouter.evals.runner import EvaluationRunner, JevRoutingPredictor
from neurorouter.jev.client import TypeSafeJevClient
from neurorouter.schemas.routing import Intent
from neurorouter.utils.config import PROJECT_ROOT, SecretSettings, load_settings


def _confusion_chart(matrix: dict[str, dict[str, int]]) -> go.Figure:
    labels = [intent.value for intent in Intent]
    figure = go.Figure(
        go.Heatmap(
            z=[[matrix[row][column] for column in labels] for row in labels],
            x=labels,
            y=labels,
            colorscale=[[0, "#0d1728"], [1, "#55e6c1"]],
            texttemplate="%{z}",
            hovertemplate="Expected %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>",
        )
    )
    figure.update_layout(
        height=470,
        xaxis_title="Predicted intent",
        yaxis_title="Expected intent",
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
    )
    return figure


def _calibration_chart(bins: list) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(color="#64748b", dash="dash"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[item.mean_probability for item in bins],
            y=[item.observed_rate for item in bins],
            mode="lines+markers",
            name="Measured",
            marker=dict(
                color="#55e6c1",
                size=[max(8, min(24, item.count * 2)) for item in bins],
            ),
            hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<extra></extra>",
        )
    )
    figure.update_layout(
        height=390,
        xaxis=dict(title="Mean predicted probability", range=[0, 1], tickformat=".0%"),
        yaxis=dict(title="Observed positive rate", range=[0, 1], tickformat=".0%"),
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
    )
    return figure


st.set_page_config(page_title="Evaluations - NeuroRouter", page_icon="📈", layout="wide")
st.title("📈 Router Evaluations")
st.caption("Measured classification, calibration, latency, and baseline-comparison results only.")

settings = load_settings()
dataset_path = PROJECT_ROOT / settings.evaluation.dataset_path
database_path = PROJECT_ROOT / settings.evaluation.database_path
cases = load_routing_dataset(dataset_path)
repository = EvaluationRepository(database_path)
secrets = SecretSettings()
reports = repository.list_reports(limit=100)

top = st.columns(4)
top[0].metric("Dataset examples", len(cases))
top[1].metric("Measured runs", len(reports))
top[2].metric("Probability threshold", settings.evaluation.probability_threshold)
top[3].metric("Fabricated results", "Never")

with st.expander("Run a measured Jev evaluation"):
    sample_count = st.slider("Examples to evaluate", 1, len(cases), len(cases))
    concurrency = st.slider("Concurrent requests", 1, 8, 4)
    if not secrets.typesafe_api_key:
        st.info("Set TYPESAFE_API_KEY in .env to enable real evaluation runs.")
    if st.button(
        "Run Jev evaluation",
        type="primary",
        disabled=not bool(secrets.typesafe_api_key),
    ):
        client = TypeSafeJevClient(
            api_key=secrets.typesafe_api_key,
            model=settings.jev.model,
            timeout_seconds=settings.jev.timeout_seconds,
        )
        fallback = settings.jev.fallback.model_copy(update={"enabled": False})
        predictor = JevRoutingPredictor(StateBuilder(settings), JevRouter(client, fallback))
        with st.status("Running measured router evaluation...", expanded=True) as status:
            report = asyncio.run(
                EvaluationRunner(
                    probability_threshold=settings.evaluation.probability_threshold,
                    calibration_bins=settings.evaluation.calibration_bins,
                    max_concurrency=concurrency,
                ).run(
                    cases[:sample_count],
                    predictor,
                    router_name=f"NeuroRouter Jev ({settings.jev.model})",
                    dataset_name=f"{dataset_path.name}:{sample_count}",
                )
            )
            repository.save(report)
            status.update(label="Evaluation persisted", state="complete", expanded=False)
        st.rerun()

if not reports:
    st.warning(
        "No measured evaluation runs exist yet. Configure TypeSafe and run the dataset here, "
        "or use `python evals/evaluate_router.py`."
    )
    st.stop()

st.subheader("Run history")
st.dataframe(
    [
        {
            "run_id": str(report.run_id),
            "router": report.router_name,
            "dataset": report.dataset_name,
            "examples": report.metrics.evaluated_examples,
            "failures": report.metrics.failed_examples,
            "intent_accuracy": report.metrics.intent_accuracy,
            "macro_f1": report.metrics.macro_f1,
            "mean_brier": report.metrics.mean_brier_score,
            "latency_ms": report.metrics.mean_latency_ms,
            "completed": report.completed_at,
        }
        for report in reports
    ],
    use_container_width=True,
    hide_index=True,
)

selected = st.selectbox(
    "Inspect measured run",
    reports,
    format_func=lambda report: (
        f"{str(report.run_id)[:8]} · {report.router_name} · {report.dataset_name}"
    ),
)
metrics = selected.metrics

summary = st.columns(6)
summary[0].metric("Intent accuracy", f"{metrics.intent_accuracy:.1%}")
summary[1].metric("Macro F1", f"{metrics.macro_f1:.1%}")
summary[2].metric("Mean Brier", f"{metrics.mean_brier_score:.3f}")
summary[3].metric("Mean latency", f"{metrics.mean_latency_ms:,.0f} ms")
summary[4].metric("LLM calls", metrics.total_llm_calls)
summary[5].metric(
    "Estimated cost",
    f"${metrics.estimated_cost_usd:.4f}" if metrics.estimated_cost_usd is not None else "n/a",
)

st.subheader("Capability classification")
st.dataframe(
    [
        {"capability": name, **values.model_dump(mode="json")}
        for name, values in metrics.capabilities.items()
    ],
    use_container_width=True,
    hide_index=True,
)

confusion_col, calibration_col = st.columns(2, gap="large")
with confusion_col:
    st.subheader("Intent confusion matrix")
    st.plotly_chart(_confusion_chart(metrics.intent_confusion_matrix), use_container_width=True)
with calibration_col:
    st.subheader("Probability calibration")
    capability = st.selectbox("Noul signal", list(metrics.calibration))
    st.plotly_chart(_calibration_chart(metrics.calibration[capability]), use_container_width=True)

failed = [prediction for prediction in selected.predictions if prediction.error]
with st.expander(f"Prediction details · {len(failed)} failures"):
    st.json([prediction.model_dump(mode="json") for prediction in selected.predictions])

if len(reports) >= 2:
    st.divider()
    st.subheader("Measured baseline comparison")
    baseline_col, candidate_col = st.columns(2)
    baseline = baseline_col.selectbox(
        "Baseline run",
        reports,
        index=1,
        format_func=lambda report: f"{report.router_name} · {str(report.run_id)[:8]}",
    )
    candidate = candidate_col.selectbox(
        "Candidate run",
        reports,
        index=0,
        format_func=lambda report: f"{report.router_name} · {str(report.run_id)[:8]}",
    )
    try:
        comparison = compare_reports(baseline, candidate)
    except ValueError as error:
        st.info(str(error))
    else:
        st.dataframe(
            [metric.model_dump(mode="json") for metric in comparison.metrics],
            use_container_width=True,
            hide_index=True,
        )
