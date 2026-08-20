from __future__ import annotations

import stat
from pathlib import Path

from quanta.settings import RuntimeSettings

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def test_settings_persist_secrets_without_exposing_them(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    settings = RuntimeSettings(path, "default/model", DEFAULT_BASE_URL)

    public = settings.update(
        llm_api_key="llm-secret",
        kaggle_api_token="kaggle-secret",
        llm_model="provider/model",
        llm_base_url="https://api.openai.com/v1",
    )

    assert public["llm_configured"] is True
    assert public["kaggle_configured"] is True
    assert public["llm_base_url"] == "https://api.openai.com/v1"
    assert "secret" not in str(public)
    assert RuntimeSettings(path, "other/model", DEFAULT_BASE_URL).llm_api_key() == "llm-secret"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_empty_values_remove_only_saved_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "environment-key")
    settings = RuntimeSettings(tmp_path / "settings.json", "default/model", DEFAULT_BASE_URL)
    settings.update(llm_api_key="saved-key")

    result = settings.update(llm_api_key="")

    assert settings.llm_api_key() == "environment-key"
    assert result["llm_source"] == "environment"


def test_base_url_falls_back_to_env_then_default(tmp_path: Path, monkeypatch) -> None:
    settings = RuntimeSettings(tmp_path / "settings.json", "default/model", DEFAULT_BASE_URL)
    assert settings.llm_base_url() == DEFAULT_BASE_URL

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    assert settings.llm_base_url() == "http://localhost:11434/v1"

    settings.update(llm_base_url="https://api.openai.com/v1")
    assert settings.llm_base_url() == "https://api.openai.com/v1"
