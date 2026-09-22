from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_release_assets_exist() -> None:
    expected = {
        ".github/workflows/ci.yml",
        ".streamlit/config.toml",
        ".dockerignore",
        "Dockerfile",
        "LICENSE",
        "SECURITY.md",
        "docs/architecture.md",
        "docs/demo.md",
        "docs/deployment.md",
        "requirements.txt",
    }

    assert not [path for path in expected if not (ROOT / path).is_file()]


def test_provider_models_and_zero_billing_defaults_are_explicit() -> None:
    settings = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    llm = settings["llm"]

    assert llm["provider"] == "mock"
    assert llm["allow_paid_models"] is False
    assert llm["providers"]["openrouter"]["model_tiers"] == {
        "fast": "openrouter/free",
        "standard": "openrouter/free",
        "reasoning": "openrouter/free",
    }
    assert all(provider["free_tier_only"] for provider in llm["providers"].values())


def test_streamlit_theme_and_package_metadata_parse() -> None:
    theme = tomllib.loads((ROOT / ".streamlit/config.toml").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert theme["theme"]["base"] == "dark"
    assert project["requires-python"] == ">=3.11"
    assert project["urls"]["Repository"].endswith("/NeuroRouter")


def test_container_runs_as_non_root_and_omits_secrets() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "USER neurorouter" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert ".env" in ignored
    assert ".streamlit/secrets.toml" in ignored
    assert "data" in ignored


def test_readme_links_core_release_documents() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for link in ("docs/architecture.md", "docs/demo.md", "docs/deployment.md", "SECURITY.md"):
        assert f"]({link})" in readme
