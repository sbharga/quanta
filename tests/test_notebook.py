from __future__ import annotations

from pathlib import Path

from quanta.config import Config
from quanta.db import Store
from quanta.events import EventSink
from quanta.notebook import NotebookService


class FakeKernels:
    def __init__(self) -> None:
        self.sources: list[str] = []

    def has_worker(self, _session_id: str) -> bool:
        return False

    def execute(self, _session_id, source, _data, _artifacts, _callback):
        self.sources.append(source)
        return "ok", []


def make_config(tmp_path: Path) -> Config:
    return Config(
        data_dir=tmp_path,
        database_path=tmp_path / "test.sqlite3",
        llm_api_key=None,
        llm_model="test/model",
        llm_base_url="https://openrouter.ai/api/v1",
        execution_timeout=10,
        max_iterations=3,
        max_download_bytes=1000,
        max_output_chars=1000,
        model_min_interval=0,
        model_max_retries=0,
    )


def test_restore_does_not_run_an_untested_revision(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    session = store.create_session("owner/dataset")
    cell = store.create_cell(session["id"], "code", "x = 1")
    execution = store.start_execution(session["id"], cell)
    store.finish_execution(execution["id"], "ok", [])
    store.revise_cell(cell["id"], "x = dangerous_change()", "user")
    kernels = FakeKernels()
    notebook = NotebookService(make_config(tmp_path), store, EventSink(store), kernels)  # type: ignore[arg-type]

    notebook.restore_kernel(session["id"])

    assert kernels.sources == []


def test_export_notebook_covers_every_output_type(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    config = make_config(tmp_path)
    session = store.create_session("owner/dataset")

    markdown_cell = store.create_cell(session["id"], "markdown", "## Summary\nSome text.")

    code_cell = store.create_cell(session["id"], "code", "df.head()")
    execution = store.start_execution(session["id"], code_cell)
    artifacts_dir = config.session_dir(session["id"]) / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000a49444154789c6360000002000155a3b4990000000049454e44ae426082"
    )
    (artifacts_dir / "plot-1.png").write_bytes(png_bytes)
    store.finish_execution(
        execution["id"],
        "ok",
        [
            {"type": "stdout", "text": "loading...\n"},
            {"type": "result", "text": "42"},
            {
                "type": "table",
                "columns": ["a", "b"],
                "rows": [[1, None], [2, "x"]],
                "total_rows": 2,
                "truncated": False,
            },
            {"type": "image", "path": "plot-1.png", "alt": "Generated plot"},
            {"type": "error", "message": "Traceback (most recent call last):\nKeyError: 'x'"},
        ],
    )

    notebook = NotebookService(config, store, EventSink(store), FakeKernels())  # type: ignore[arg-type]
    document = notebook.export_notebook(session["id"])

    assert document["nbformat"] == 4
    assert [cell["cell_type"] for cell in document["cells"]] == ["markdown", "code"]
    assert document["cells"][0]["source"] == ["## Summary\n", "Some text."]

    outputs = {o["output_type"] for o in document["cells"][1]["outputs"]}
    assert outputs == {"stream", "execute_result", "display_data", "error"}

    image_output = next(o for o in document["cells"][1]["outputs"] if o["output_type"] == "display_data")
    assert "image/png" in image_output["data"]

    error_output = next(o for o in document["cells"][1]["outputs"] if o["output_type"] == "error")
    assert error_output["ename"] == "KeyError"

    assert markdown_cell["kind"] == "markdown"  # sanity check on fixture setup


def test_build_timeline_folds_events_and_skips_draft_deltas(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    config = make_config(tmp_path)
    events = EventSink(store)
    session = store.create_session("owner/dataset")

    notebook = NotebookService(config, store, events, FakeKernels())  # type: ignore[arg-type]
    events.emit(session["id"], "user_request", content="Explore trends")
    events.emit(session["id"], "agent_decision", action="create", rationale="need data", instruction="load csv")
    cell = notebook.create_cell(session["id"], "code", "df = pd.read_csv(...)")
    events.emit(session["id"], "draft_started", draft_id="d1", kind="code")
    events.emit(session["id"], "draft_delta", draft_id="d1", kind="code", delta="df")
    events.emit(session["id"], "draft_finished", draft_id="d1", kind="code")
    events.emit(session["id"], "complete", message="done")

    timeline = notebook.build_timeline(session["id"])

    types = [entry["type"] for entry in timeline["entries"]]
    assert types == ["request", "cell", "complete"]
    cell_entry = next(entry for entry in timeline["entries"] if entry["type"] == "cell")
    assert cell_entry["cell"]["id"] == cell["id"]
    assert timeline["event_cursor"] > 0


def test_build_timeline_restores_only_an_active_draft(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    config = make_config(tmp_path)
    events = EventSink(store)
    session = store.create_session("owner/dataset")
    notebook = NotebookService(config, store, events, FakeKernels())  # type: ignore[arg-type]

    events.emit(
        session["id"],
        "draft_started",
        draft_id="draft-1",
        kind="code",
        action="edit",
        target_cell_id="cell-1",
    )
    events.emit(session["id"], "draft_delta", draft_id="draft-1", kind="code", delta="x = ")
    events.emit(session["id"], "draft_delta", draft_id="draft-1", kind="code", delta="1")

    timeline = notebook.build_timeline(session["id"])

    assert timeline["active_draft"] == {
        "draft_id": "draft-1",
        "kind": "code",
        "action": "edit",
        "target_cell_id": "cell-1",
        "text": "x = 1",
    }

    events.emit(session["id"], "draft_discarded", draft_id="draft-1", kind="code")
    assert notebook.build_timeline(session["id"])["active_draft"] is None


def test_timeline_and_export_use_latest_active_revision_execution(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    config = make_config(tmp_path)
    events = EventSink(store)
    session = store.create_session("owner/dataset")
    notebook = NotebookService(config, store, events, FakeKernels())  # type: ignore[arg-type]
    cell = notebook.create_cell(session["id"], "code", "x = 1")
    old = store.start_execution(session["id"], cell)
    store.finish_execution(old["id"], "ok", [{"type": "result", "text": "old"}])
    revised = store.revise_cell(cell["id"], "x = 2", "assistant")
    first = store.start_execution(session["id"], revised)
    store.finish_execution(first["id"], "error", [{"type": "error", "message": "first"}])
    latest = store.start_execution(session["id"], revised)
    store.finish_execution(latest["id"], "ok", [{"type": "result", "text": "latest"}])

    cell_entry = next(entry for entry in notebook.build_timeline(session["id"])["entries"] if entry["type"] == "cell")
    assert [execution["id"] for execution in cell_entry["cell"]["executions"]] == [latest["id"]]
    exported = notebook.export_notebook(session["id"])["cells"][0]
    assert exported["outputs"][0]["data"]["text/plain"] == ["latest"]
