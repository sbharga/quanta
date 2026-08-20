from __future__ import annotations

from pathlib import Path

from quanta.db import Store


def test_cell_revisions_and_executions_are_append_only(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    session = store.create_session("owner/dataset")
    cell = store.create_cell(session["id"], "code", "x = 1")
    execution = store.start_execution(session["id"], cell)
    store.finish_execution(execution["id"], "ok", [{"type": "result", "text": "1"}])

    revised = store.revise_cell(cell["id"], "x = 2", "user")
    cells = store.list_cells(session["id"])

    assert revised["revision"] == 2
    assert cells[0]["source"] == "x = 2"
    assert cells[0]["executions"][0]["outputs"][0]["text"] == "1"


def test_events_can_be_replayed_from_cursor(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    session = store.create_session("owner/dataset")
    first = store.append_event(session["id"], "status", {"status": "running"})
    store.append_event(session["id"], "complete", {"message": "done"})

    events = store.events_after(session["id"], first)

    assert [event["type"] for event in events] == ["complete"]
    assert events[0]["payload"] == {"message": "done"}


def test_reconciles_sessions_interrupted_by_restart(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    session = store.create_session("owner/dataset")
    store.update_session(session["id"], status="running")

    store.reconcile_interrupted_sessions()

    restored = store.get_session(session["id"])
    assert restored["status"] == "interrupted"
    assert "stopped" in restored["error"]


def test_reconciles_executions_interrupted_by_restart(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    session = store.create_session("owner/dataset")
    cell = store.create_cell(session["id"], "code", "1 + 1")
    execution = store.start_execution(session["id"], cell)

    store.reconcile_interrupted_executions()

    restored = store.list_cells(session["id"])[0]["executions"][0]
    assert restored["id"] == execution["id"]
    assert restored["status"] == "interrupted"
    assert restored["finished_at"] is not None
    assert "interrupted" in restored["outputs"][0]["message"].lower()


def test_session_title_round_trips(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    session = store.create_session("owner/dataset")
    assert session["title"] is None

    store.update_session(session["id"], title="Streaming trends")

    assert store.get_session(session["id"])["title"] == "Streaming trends"
    listed = store.list_sessions()
    assert listed[0]["title"] == "Streaming trends"


def test_delete_session_cascades_and_removes_checkpoints(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    session = store.create_session("owner/dataset")
    other = store.create_session("owner/other-dataset")
    cell = store.create_cell(session["id"], "code", "x = 1")
    store.append_event(session["id"], "status", {"status": "running"})

    # LangGraph's SqliteSaver creates and owns these tables independently;
    # simulate them here rather than depending on langgraph internals.
    with store.connect() as db:
        db.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_id TEXT)")
        db.execute("CREATE TABLE writes (thread_id TEXT, idx INTEGER)")
        db.execute("INSERT INTO checkpoints VALUES (?, 'c1')", (session["id"],))
        db.execute("INSERT INTO checkpoints VALUES (?, 'c2')", (other["id"],))
        db.execute("INSERT INTO writes VALUES (?, 0)", (session["id"],))

    store.delete_session(session["id"])

    with store.connect() as db:
        remaining_checkpoints = db.execute("SELECT thread_id FROM checkpoints").fetchall()
        remaining_writes = db.execute("SELECT thread_id FROM writes").fetchall()
    assert [row[0] for row in remaining_checkpoints] == [other["id"]]
    assert remaining_writes == []
    assert store.list_cells(session["id"]) == []
    try:
        store.get_cell(cell["id"])
    except KeyError:
        pass
    else:
        raise AssertionError("cell should have been cascade-deleted")


def test_delete_session_raises_for_unknown_id(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    try:
        store.delete_session("missing")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
