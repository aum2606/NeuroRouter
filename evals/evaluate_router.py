"""Run a measured Jev routing evaluation and persist its report."""

import argparse
import asyncio
from pathlib import Path

from neurorouter.core.router import JevRouter
from neurorouter.core.state_builder import StateBuilder
from neurorouter.evals.dataset import load_routing_dataset
from neurorouter.evals.repository import EvaluationRepository
from neurorouter.evals.runner import EvaluationRunner, JevRoutingPredictor
from neurorouter.jev.client import TypeSafeJevClient
from neurorouter.utils.config import PROJECT_ROOT, SecretSettings, load_settings


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--router-name", default="NeuroRouter Jev")
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args()


async def _run(arguments: argparse.Namespace) -> None:
    settings = load_settings()
    secrets = SecretSettings()
    if not secrets.typesafe_api_key:
        raise SystemExit("TYPESAFE_API_KEY is required for a measured Jev evaluation")
    dataset_path = arguments.dataset or PROJECT_ROOT / settings.evaluation.dataset_path
    cases = load_routing_dataset(dataset_path)
    if arguments.limit is not None:
        if arguments.limit < 1:
            raise SystemExit("--limit must be positive")
        cases = cases[: arguments.limit]
    client = TypeSafeJevClient(
        api_key=secrets.typesafe_api_key,
        model=settings.jev.model,
        timeout_seconds=settings.jev.timeout_seconds,
    )
    fallback = settings.jev.fallback.model_copy(update={"enabled": False})
    predictor = JevRoutingPredictor(
        StateBuilder(settings),
        JevRouter(client, fallback),
    )
    report = await EvaluationRunner(
        probability_threshold=settings.evaluation.probability_threshold,
        calibration_bins=settings.evaluation.calibration_bins,
        max_concurrency=arguments.concurrency,
    ).run(
        cases,
        predictor,
        router_name=arguments.router_name,
        dataset_name=dataset_path.name,
    )
    repository = EvaluationRepository(PROJECT_ROOT / settings.evaluation.database_path)
    repository.save(report)
    print(report.metrics.model_dump_json(indent=2))
    print(f"Saved evaluation run {report.run_id}")


if __name__ == "__main__":
    asyncio.run(_run(_arguments()))
