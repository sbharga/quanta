from __future__ import annotations

from pathlib import Path

import pytest
from httpx2 import Request
from openai import APIConnectionError

import quanta.analysis as analysis_module
from quanta.analysis import AnalysisEngine, _code_generation_prompt, _is_retryable_model_error, _strip_code_fence
from quanta.config import Config
from quanta.db import Store
from quanta.events import EventSink
from quanta.settings import RuntimeSettings


def make_engine(tmp_path: Path) -> tuple[AnalysisEngine, Store]:
    config = Config(
        data_dir=tmp_path,
        database_path=tmp_path / "test.sqlite3",
        llm_api_key="test-key",
        llm_model="test/model",
        llm_base_url="https://example.test/v1",
        execution_timeout=10,
        max_iterations=3,
        max_download_bytes=1000,
        max_output_chars=1000,
        model_min_interval=0,
        model_max_retries=2,
    )
    store = Store(config.database_path)
    store.initialize()
    events = EventSink(store)
    settings = RuntimeSettings(tmp_path / "settings.json", config.llm_model, config.llm_base_url, config.llm_api_key)
    engine = AnalysisEngine(config, store, events, object(), settings)  # type: ignore[arg-type]
    return engine, store


def test_strip_code_fence_handles_bare_code() -> None:
    assert _strip_code_fence("df.head()") == "df.head()"


def test_strip_code_fence_handles_fence_at_start() -> None:
    source = "```python\ndf = pd.read_csv(DATASET_DIR / 'a.csv')\ndf.head()\n```"
    assert _strip_code_fence(source) == "df = pd.read_csv(DATASET_DIR / 'a.csv')\ndf.head()"


def test_strip_code_fence_handles_preamble_before_fence() -> None:
    source = "Here's the code:\n```python\ndf.head()\n```"
    assert _strip_code_fence(source) == "df.head()"


def test_strip_code_fence_empty_block_returns_empty_string() -> None:
    # This is the exact shape that used to slip through as an empty cell.
    assert _strip_code_fence("```python\n```") == ""


def test_strip_code_fence_empty_block_with_trailing_text() -> None:
    assert _strip_code_fence("```python\n```\nLet me know if you need more.") == ""


def test_connection_errors_are_retryable() -> None:
    error = APIConnectionError(request=Request("POST", "https://example.test/v1/chat/completions"))
    assert _is_retryable_model_error(error) is True
    assert _is_retryable_model_error(ValueError("bad request")) is False


def test_model_invocation_retries_and_emits_activity(tmp_path: Path, monkeypatch) -> None:
    engine, store = make_engine(tmp_path)
    session = store.create_session("owner/dataset")
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary")
        return "ok"

    monkeypatch.setattr(analysis_module, "_is_retryable_model_error", lambda _exc: True)
    monkeypatch.setattr(
        engine,
        "_retry_model",
        lambda session_id, _exc, attempt: engine.events.emit(
            session_id,
            "model_retry",
            attempt=attempt,
            max_retries=2,
            delay=0,
            message="retrying",
        ),
    )

    assert engine._invoke_model(session["id"], "Waiting for model", operation) == "ok"
    assert attempts == 2
    records = store.events_after(session["id"], 0)
    assert [record["type"] for record in records] == ["status", "model_retry", "status"]


def test_code_prompt_puts_failure_before_required_fix() -> None:
    prompt = _code_generation_prompt(
        instruction="Replace seaborn with matplotlib",
        manifest={"files": []},
        columns="",
        target={"source": "import seaborn as sns"},
        recent_result={"status": "error", "outputs": [{"message": "ModuleNotFoundError: seaborn"}]},
    )

    assert prompt.index("import seaborn as sns") < prompt.index("ModuleNotFoundError: seaborn")
    assert prompt.index("ModuleNotFoundError: seaborn") < prompt.index("Replace seaborn with matplotlib")
    assert prompt.endswith("Replace seaborn with matplotlib\n")


def test_unchanged_failing_edit_is_not_saved_or_executed(tmp_path: Path, monkeypatch) -> None:
    engine, store = make_engine(tmp_path)
    session = store.create_session("owner/dataset")
    cell = store.create_cell(session["id"], "code", "import seaborn as sns")
    monkeypatch.setattr(engine, "_stream_model", lambda *_args, **_kwargs: cell["source"])
    state = {
        "session_id": session["id"],
        "manifest": {"files": []},
        "decision": {
            "action": "edit",
            "instruction": "Replace seaborn with matplotlib",
            "target_cell_id": cell["id"],
        },
        "recent_result": {"status": "error", "outputs": [{"message": "ModuleNotFoundError: seaborn"}]},
    }

    result = engine._generate_code(state)  # type: ignore[arg-type]

    assert store.get_cell(cell["id"])["revision"] == 1
    assert result["decision"]["target_cell_id"] is None
    assert result["unchanged_generations"] == 1
    assert "identical" in result["recent_result"]["model_feedback"]

    with pytest.raises(RuntimeError, match="same failing code twice"):
        engine._generate_code({**state, "unchanged_generations": 1})  # type: ignore[arg-type]
    assert store.get_cell(cell["id"])["revision"] == 1
