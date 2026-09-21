from pathlib import Path

import pytest

from neurorouter.utils.config import RuntimeSettings, load_settings, load_thresholds


def test_project_configuration_loads() -> None:
    settings = load_settings()
    thresholds = load_thresholds()

    assert settings.app.name == "NeuroRouter"
    assert settings.telemetry.database_path == Path("data/neurorouter.db")
    assert thresholds.routing.web_threshold == pytest.approx(0.65)
    assert thresholds.quality.max_retries == 2


def test_invalid_yaml_shape_is_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "settings.yaml"
    invalid.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a YAML mapping"):
        load_settings(invalid)


def test_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUROROUTER_APP__ENVIRONMENT", "test")

    overridden = RuntimeSettings(**load_settings().model_dump())

    assert overridden.app.environment == "test"
