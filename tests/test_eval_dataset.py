from pathlib import Path

import pytest

from neurorouter.evals.dataset import DatasetError, load_routing_dataset
from neurorouter.schemas.routing import Intent
from neurorouter.utils.config import PROJECT_ROOT


def test_repository_dataset_is_valid_and_balanced() -> None:
    cases = load_routing_dataset(PROJECT_ROOT / "evals" / "routing_dataset.jsonl")

    counts = {intent: sum(case.expected_intent is intent for case in cases) for intent in Intent}
    assert len(cases) == 28
    assert set(counts.values()) == {4}


def test_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    case = (
        '{"id":"same","query":"q","expected_intent":"general_qa",'
        '"expected_needs_web":false,"expected_needs_rag":false,'
        '"expected_needs_code":false,"expected_needs_data":false}'
    )
    path = tmp_path / "duplicate.jsonl"
    path.write_text(f"{case}\n{case}\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="Duplicate"):
        load_routing_dataset(path)
