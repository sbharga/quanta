from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class RuntimeSettings:
    """Thread-safe local settings with environment-variable fallbacks."""

    def __init__(
        self,
        path: Path,
        default_model: str,
        default_base_url: str,
        default_api_key: str | None = None,
        default_max_iterations: int = 12,
    ):
        self.path = path
        self.default_model = default_model
        self.default_base_url = default_base_url
        self.default_api_key = default_api_key
        self.default_max_iterations = default_max_iterations
        self._lock = threading.RLock()

    def llm_api_key(self) -> str | None:
        return self._value("llm_api_key", "LLM_API_KEY") or self.default_api_key

    def kaggle_api_token(self) -> str | None:
        return self._value("kaggle_api_token", "KAGGLE_API_TOKEN")

    def llm_model(self) -> str:
        with self._lock:
            stored = self._read().get("llm_model")
        return str(stored or os.environ.get("LLM_MODEL") or self.default_model)

    def llm_base_url(self) -> str:
        with self._lock:
            stored = self._read().get("llm_base_url")
        return str(stored or os.environ.get("LLM_BASE_URL") or self.default_base_url)

    def max_iterations(self) -> int:
        with self._lock:
            stored = self._read().get("max_iterations")
        value = stored if stored is not None else os.environ.get("QUANTA_MAX_ITERATIONS")
        try:
            parsed = int(value) if value is not None else None
        except (TypeError, ValueError):
            parsed = None
        return parsed if parsed and parsed > 0 else self.default_max_iterations

    def public(self) -> dict[str, Any]:
        stored = self._read()
        llm_source = _source(stored, "llm_api_key", "LLM_API_KEY")
        if llm_source == "not configured" and self.default_api_key:
            llm_source = "application configuration"
        kaggle_file = any(
            path.exists()
            for path in (
                Path.home() / ".kaggle" / "access_token",
                Path.home() / ".kaggle" / "kaggle.json",
            )
        )
        return {
            "llm_model": self.llm_model(),
            "llm_base_url": self.llm_base_url(),
            "llm_configured": bool(self.llm_api_key()),
            "llm_source": llm_source,
            "kaggle_configured": bool(self.kaggle_api_token()) or kaggle_file,
            "kaggle_source": (
                _source(stored, "kaggle_api_token", "KAGGLE_API_TOKEN")
                if self.kaggle_api_token()
                else "credential file"
                if kaggle_file
                else "not configured"
            ),
            "max_iterations": self.max_iterations(),
        }

    def update(
        self,
        *,
        llm_api_key: str | None = None,
        kaggle_api_token: str | None = None,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        max_iterations: int | None = None,
        clear_max_iterations: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            values = self._read()
            for key, value in (
                ("llm_api_key", llm_api_key),
                ("kaggle_api_token", kaggle_api_token),
                ("llm_model", llm_model),
                ("llm_base_url", llm_base_url),
            ):
                if value is None:
                    continue
                cleaned = value.strip()
                if cleaned:
                    values[key] = cleaned
                else:
                    values.pop(key, None)
            if clear_max_iterations:
                values.pop("max_iterations", None)
            elif max_iterations is not None:
                values["max_iterations"] = max_iterations
            self._write(values)
        return self.public()

    def _value(self, key: str, environment_key: str) -> str | None:
        with self._lock:
            stored = self._read().get(key)
        return str(stored) if stored else os.environ.get(environment_key)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write(self, values: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(self.path)
        self.path.chmod(0o600)


def _source(stored: dict[str, Any], key: str, environment_key: str) -> str:
    if stored.get(key):
        return "app settings"
    if os.environ.get(environment_key):
        return "environment"
    return "not configured"
