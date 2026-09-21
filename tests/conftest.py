"""Test fixtures and import configuration."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from neurorouter.telemetry.database import TraceRepository


@pytest.fixture
def repository(tmp_path: Path) -> Iterator[TraceRepository]:
    yield TraceRepository(tmp_path / "traces.db")
