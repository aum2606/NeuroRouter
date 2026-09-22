"""Measured router evaluation and comparison infrastructure."""

from neurorouter.evals.comparison import compare_reports
from neurorouter.evals.dataset import load_routing_dataset
from neurorouter.evals.metrics import evaluate_routing_predictions
from neurorouter.evals.repository import EvaluationRepository
from neurorouter.evals.runner import EvaluationRunner, JevRoutingPredictor

__all__ = [
    "EvaluationRepository",
    "EvaluationRunner",
    "JevRoutingPredictor",
    "compare_reports",
    "evaluate_routing_predictions",
    "load_routing_dataset",
]
