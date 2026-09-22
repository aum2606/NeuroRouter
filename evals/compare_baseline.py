"""Compare two persisted measured router evaluation runs."""

import argparse

from neurorouter.evals.comparison import compare_reports
from neurorouter.evals.repository import EvaluationRepository
from neurorouter.utils.config import PROJECT_ROOT, load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_run_id")
    parser.add_argument("candidate_run_id")
    arguments = parser.parse_args()
    settings = load_settings()
    repository = EvaluationRepository(PROJECT_ROOT / settings.evaluation.database_path)
    baseline = repository.get(arguments.baseline_run_id)
    candidate = repository.get(arguments.candidate_run_id)
    if baseline is None or candidate is None:
        raise SystemExit("Both evaluation run IDs must exist")
    print(compare_reports(baseline, candidate).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
