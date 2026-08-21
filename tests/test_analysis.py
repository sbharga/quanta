from __future__ import annotations

from pathlib import Path

import pytest
from httpx2 import Request
from openai import APIConnectionError

import quanta.analysis as analysis_module
from quanta.analysis import (
    AnalysisEngine,
    AnalystDecision,
    _code_generation_prompt,
    _installed_packages_brief,
    _is_retryable_model_error,
    _strip_code_fence,
    _truncate_execution_for_prompt,
)
from quanta.config import Config
from quanta.db import Store
from quanta.events import EventSink
from quanta.packages import PackageManager
from quanta.settings import RuntimeSettings


def make_engine(tmp_path: Path) -> tuple[AnalysisEngine, Store]:
    config = Config(
        data_dir=tmp_path,
        database_path=tmp_path / "test.sqlite3",
        packages_dir=tmp_path / "packages",
        llm_api_key="test-key",
        llm_model="test/model",
        llm_base_url="https://example.test/v1",
        execution_timeout=10,
        max_iterations=3,
        max_download_bytes=1000,
        max_output_chars=1000,
        model_timeout=60,
        model_max_retries=2,
    )
    store = Store(config.database_path)
    store.initialize()
    events = EventSink(store)
    settings = RuntimeSettings(
        tmp_path / "settings.json", config.llm_model, config.llm_base_url, config.llm_api_key, config.max_iterations
    )
    packages = PackageManager(config.packages_dir)
    engine = AnalysisEngine(config, store, events, object(), settings, packages)  # type: ignore[arg-type]
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


def test_truncate_execution_for_prompt_keeps_traceback_tail() -> None:
    traceback_text = ("frame " * 200) + "ValueError: the actual problem"
    execution = {"status": "error", "outputs": [{"type": "error", "message": traceback_text}]}

    result = _truncate_execution_for_prompt(execution, limit=200)

    assert "ValueError: the actual problem" in result
    assert len(result) <= 200


def test_generate_code_receives_every_successful_active_cell(tmp_path: Path, monkeypatch) -> None:
    engine, store = make_engine(tmp_path)
    session = store.create_session("owner/dataset")
    first = store.create_cell(session["id"], "code", "df = pd.read_csv(DATASET_DIR / 'data.csv')")
    first_execution = store.start_execution(session["id"], first)
    store.finish_execution(first_execution["id"], "ok", [])
    second = store.create_cell(session["id"], "code", "summary = df.groupby('group').value.mean()")
    second_execution = store.start_execution(session["id"], second)
    store.finish_execution(second_execution["id"], "ok", [])
    failed = store.create_cell(session["id"], "code", "broken = missing_name")
    failed_execution = store.start_execution(session["id"], failed)
    store.finish_execution(failed_execution["id"], "error", [{"type": "error", "message": "NameError"}])
    monkeypatch.setattr(engine, "notebook", _FakeNotebook(store))
    prompts = []

    def capture_prompt(_session_id, _kind, prompt, **_kwargs):
        prompts.append(prompt)
        return "summary"

    monkeypatch.setattr(engine, "_stream_model", capture_prompt)

    engine._generate_code(  # type: ignore[arg-type]
        {
            "session_id": session["id"],
            "manifest": {"files": []},
            "decision": {"action": "create", "instruction": "Show the summary"},
        }
    )

    prompt = prompts[0]
    assert first["source"] in prompt
    assert second["source"] in prompt
    assert prompt.index(first["source"]) < prompt.index(second["source"])
    assert failed["source"] not in prompt
    assert "Every variable, import, function, and model they defined remains available" in prompt


def test_regenerate_cell_rewrites_target_and_excludes_it_from_context(tmp_path: Path, monkeypatch) -> None:
    engine, store = make_engine(tmp_path)
    session = store.create_session("owner/dataset")
    other = store.create_cell(session["id"], "code", "df = pd.read_csv('x.csv')")
    other_execution = store.start_execution(session["id"], other)
    store.finish_execution(other_execution["id"], "ok", [])
    target = store.create_cell(session["id"], "code", "old_broken_code")
    monkeypatch.setattr(engine, "notebook", _FakeNotebook(store))
    prompts = []

    def capture_prompt(_session_id, _kind, prompt, **_kwargs):
        prompts.append(prompt)
        return "new_code = 1"

    monkeypatch.setattr(engine, "_stream_model", capture_prompt)

    updated = engine.regenerate_cell(session["id"], target["id"])

    assert updated["source"] == "new_code = 1"
    assert other["source"] in prompts[0]
    assert "old_broken_code" in prompts[0]  # still shown as the existing cell to replace


def test_regenerate_cell_rejects_non_code_cells(tmp_path: Path) -> None:
    engine, store = make_engine(tmp_path)
    session = store.create_session("owner/dataset")
    markdown = store.create_cell(session["id"], "markdown", "Some notes")

    with pytest.raises(ValueError, match="code cells"):
        engine.regenerate_cell(session["id"], markdown["id"])


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


def test_edit_targeting_markdown_creates_code_instead(tmp_path: Path, monkeypatch) -> None:
    engine, store = make_engine(tmp_path)
    session = store.create_session("owner/dataset")
    markdown = store.create_cell(session["id"], "markdown", "Explained the previous result.")
    monkeypatch.setattr(engine, "notebook", _FakeNotebook(store))
    monkeypatch.setattr(engine, "_stream_model", lambda *_args, **_kwargs: "value = 42")

    result = engine._generate_code(  # type: ignore[arg-type]
        {
            "session_id": session["id"],
            "manifest": {"files": []},
            "decision": {
                "action": "edit",
                "instruction": "Define the missing value",
                "target_cell_id": markdown["id"],
            },
        }
    )

    assert store.get_cell(markdown["id"])["source"] == "Explained the previous result."
    created = store.get_cell(result["decision"]["target_cell_id"])
    assert created["kind"] == "code"
    assert created["source"] == "value = 42"


def test_execute_rejects_markdown_target_without_terminating_run(tmp_path: Path, monkeypatch) -> None:
    engine, store = make_engine(tmp_path)
    session = store.create_session("owner/dataset")
    markdown = store.create_cell(session["id"], "markdown", "A summary")
    monkeypatch.setattr(engine, "notebook", _FakeNotebook(store))

    result = engine._execute(  # type: ignore[arg-type]
        {
            "session_id": session["id"],
            "decision": {"action": "rerun", "target_cell_id": markdown["id"]},
        }
    )

    assert result["recent_result"]["status"] == "error"
    assert "Only code cells" in result["recent_result"]["outputs"][0]["message"]
    assert [event["type"] for event in store.events_after(session["id"], 0)] == ["note"]


def test_record_usage_persists_totals_and_emits_event(tmp_path: Path) -> None:
    engine, store = make_engine(tmp_path)
    session = store.create_session("owner/dataset")

    engine._record_usage(session["id"], {"test/model": {"input_tokens": 120, "output_tokens": 30}})

    assert store.get_session(session["id"])["prompt_tokens"] == 120
    assert store.get_session(session["id"])["completion_tokens"] == 30
    events = store.events_after(session["id"], 0)
    assert [event["type"] for event in events] == ["usage"]
    assert events[0]["payload"]["total_prompt_tokens"] == 120


def test_record_usage_is_a_noop_for_empty_usage(tmp_path: Path) -> None:
    engine, store = make_engine(tmp_path)
    session = store.create_session("owner/dataset")

    engine._record_usage(session["id"], {})

    assert store.get_session(session["id"])["prompt_tokens"] == 0
    assert store.events_after(session["id"], 0) == []


def test_route_decision_maps_install_action(tmp_path: Path) -> None:
    engine, _store = make_engine(tmp_path)
    assert engine._route_decision({"decision": {"action": "install"}}) == "install"  # type: ignore[arg-type]


class _FakeKernels:
    def __init__(self) -> None:
        self.marked_stale = 0

    def mark_all_stale(self) -> None:
        self.marked_stale += 1


class _FakeNotebook:
    def __init__(self, store: Store | None = None) -> None:
        self.kernels = _FakeKernels()
        self.store = store

    def create_cell(self, session_id: str, kind: str, source: str) -> dict:
        assert self.store is not None
        return self.store.create_cell(session_id, kind, source)

    def revise_cell(self, session_id: str, cell_id: str, source: str, author: str) -> dict:
        assert self.store is not None
        return self.store.revise_cell(cell_id, source, author)


class _FakeDecisionModel:
    def __init__(self, decision: AnalystDecision) -> None:
        self.decision = decision

    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, _messages):
        return self.decision


def test_decision_writes_at_most_one_narration_for_an_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, store = make_engine(tmp_path)
    session = store.create_session("owner/dataset")
    code = store.create_cell(session["id"], "code", "1 + 1")
    execution = store.start_execution(session["id"], code)
    store.finish_execution(execution["id"], "ok", [{"type": "result", "text": "2"}])
    monkeypatch.setattr(engine, "notebook", _FakeNotebook(store))
    decision = AnalystDecision(
        action="finish",
        rationale="The answer is complete.",
        instruction="Analysis complete",
        narration="The result is **2**.",
    )
    monkeypatch.setattr(engine, "_model", lambda: _FakeDecisionModel(decision))
    state = {"session_id": session["id"], "summary": "A test", "user_request": "Calculate it"}

    engine._decide(state)  # type: ignore[arg-type]
    engine._decide(state)  # type: ignore[arg-type]

    markdown = [cell for cell in store.list_cells(session["id"]) if cell["kind"] == "markdown"]
    assert [cell["source"] for cell in markdown] == ["The result is **2**."]


def test_decision_turns_markdown_create_into_narration_and_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, store = make_engine(tmp_path)
    session = store.create_session("owner/dataset")
    code = store.create_cell(session["id"], "code", "result = 42")
    execution = store.start_execution(session["id"], code)
    store.finish_execution(execution["id"], "ok", [{"type": "result", "text": "42"}])
    monkeypatch.setattr(engine, "notebook", _FakeNotebook(store))
    decision = AnalystDecision(
        action="create",
        rationale="Provide the final summary.",
        instruction="markdown\n# Summary\n\nThe result is **42**.",
        narration="Summarized the result.",
    )
    monkeypatch.setattr(engine, "_model", lambda: _FakeDecisionModel(decision))

    result = engine._decide(  # type: ignore[arg-type]
        {"session_id": session["id"], "summary": "A test", "user_request": "Summarize it"}
    )

    assert result["decision"]["action"] == "finish"
    markdown = [cell for cell in store.list_cells(session["id"]) if cell["kind"] == "markdown"]
    assert [cell["source"] for cell in markdown] == ["# Summary\n\nThe result is **42**."]


def test_installed_packages_brief_does_not_hide_names_after_forty(tmp_path: Path, monkeypatch) -> None:
    engine, _store = make_engine(tmp_path)
    packages = [{"name": f"package-{index:02}", "version": "1"} for index in range(45)]
    packages.extend(
        [
            {"name": "torch", "version": "2"},
            {"name": "torchvision", "version": "1"},
        ]
    )
    monkeypatch.setattr(engine.packages, "list_installed", lambda: packages)

    brief = _installed_packages_brief(engine.packages)

    assert "package-44" in brief
    assert "torch" in brief
    assert "torchvision" in brief


def test_install_node_records_package_and_marks_kernels_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, store = make_engine(tmp_path)
    session = store.create_session("owner/dataset")
    notebook = _FakeNotebook()
    monkeypatch.setattr(engine, "notebook", notebook)
    monkeypatch.setattr(
        engine.packages,
        "install",
        lambda specs, progress: {"status": "ok", "packages": [{"name": "seaborn", "version": "0.13.2"}], "log": "ok"},
    )
    state = {
        "session_id": session["id"],
        "decision": {"action": "install", "packages": ["seaborn"], "instruction": "need seaborn"},
    }

    result = engine._install(state)  # type: ignore[arg-type]

    assert result["recent_install_result"]["status"] == "ok"
    assert notebook.kernels.marked_stale == 1
    recorded = store.list_packages()
    assert len(recorded) == 1
    assert recorded[0]["name"] == "seaborn"
    events = [record["type"] for record in store.events_after(session["id"], 0)]
    assert events == ["package_install_started", "package_install_finished"]


def test_install_node_refuses_to_repeat_a_failed_install_in_the_same_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, store = make_engine(tmp_path)
    session = store.create_session("owner/dataset")
    monkeypatch.setattr(engine, "notebook", _FakeNotebook())
    calls = []

    def fake_install(specs, progress):
        calls.append(specs)
        raise ValueError("No solution found")

    monkeypatch.setattr(engine.packages, "install", fake_install)
    state = {
        "session_id": session["id"],
        "decision": {"action": "install", "packages": ["nonexistent-package"], "instruction": "need it"},
    }

    first = engine._install(state)  # type: ignore[arg-type]
    assert first["recent_install_result"]["status"] == "error"
    assert first["failed_installs"] == ["nonexistent-package"]

    second = engine._install({**state, "failed_installs": first["failed_installs"]})  # type: ignore[arg-type]

    assert len(calls) == 1  # the second attempt never called packages.install again
    assert "already failed" in second["recent_install_result"]["outputs"][0]["message"]


def test_install_node_skips_packages_that_are_already_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, store = make_engine(tmp_path)
    session = store.create_session("owner/dataset")
    monkeypatch.setattr(engine.packages, "list_installed", lambda: [{"name": "scikit-learn", "version": "1.7"}])
    monkeypatch.setattr(
        engine.packages,
        "install",
        lambda *_args: pytest.fail("install subprocess should not run"),
    )

    result = engine._install(  # type: ignore[arg-type]
        {
            "session_id": session["id"],
            "decision": {"action": "install", "packages": ["Scikit_Learn"], "instruction": "install it"},
        }
    )

    assert result["recent_install_result"]["status"] == "ok"
    assert [event["type"] for event in store.events_after(session["id"], 0)] == ["note"]


def test_model_uses_configured_request_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _store = make_engine(tmp_path)
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(analysis_module, "ChatOpenAI", fake_chat_openai)

    assert engine._model() is not None
    assert captured["timeout"] == 60
