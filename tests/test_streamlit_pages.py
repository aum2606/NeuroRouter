from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE_PATHS = [Path("app.py"), *sorted(Path("pages").glob("*.py"))]


@pytest.mark.parametrize("relative_path", PAGE_PATHS, ids=str)
def test_streamlit_page_renders_without_exception(relative_path: Path) -> None:
    app = AppTest.from_file(str(ROOT / relative_path), default_timeout=20).run()

    assert not app.exception
