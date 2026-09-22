"""Strict JSONL loading for versionable router evaluation datasets."""

import json
from pathlib import Path

from pydantic import ValidationError

from neurorouter.evals.schemas import RoutingEvalCase


class DatasetError(ValueError):
    """Raised when a JSONL evaluation dataset is malformed."""


def load_routing_dataset(path: Path | str) -> list[RoutingEvalCase]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
    cases: list[RoutingEvalCase] = []
    identifiers: set[str] = set()
    with dataset_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                case = RoutingEvalCase.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as error:
                raise DatasetError(
                    f"Invalid evaluation case at line {line_number}: {error}"
                ) from error
            if case.id in identifiers:
                raise DatasetError(f"Duplicate evaluation case id at line {line_number}: {case.id}")
            identifiers.add(case.id)
            cases.append(case)
    if not cases:
        raise DatasetError("Evaluation dataset contains no cases")
    return cases
