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


def test_list_cells_groups_executions_by_cell_across_a_multi_cell_session(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    session = store.create_session("owner/dataset")
    first = store.create_cell(session["id"], "code", "a = 1")
    second = store.create_cell(session["id"], "code", "b = 2")

    first_execution_1 = store.start_execution(session["id"], first)
    store.finish_execution(first_execution_1["id"], "ok", [{"type": "result", "text": "1"}])
    second_execution_1 = store.start_execution(session["id"], second)
    store.finish_execution(second_execution_1["id"], "error", [{"type": "error", "message": "boom"}])
    first_execution_2 = store.start_execution(session["id"], first)
    store.finish_execution(first_execution_2["id"], "ok", [{"type": "result", "text": "1 again"}])

    cells = store.list_cells(session["id"])
    by_id = {cell["id"]: cell for cell in cells}

    assert [execution["outputs"][0]["text"] for execution in by_id[first["id"]]["executions"]] == ["1", "1 again"]
    assert len(by_id[second["id"]]["executions"]) == 1
    assert by_id[second["id"]]["executions"][0]["status"] == "error"
    assert all("outputs_json" not in execution for cell in cells for execution in cell["executions"])


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


def test_record_usage_accumulates_tokens_across_calls(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    session = store.create_session("owner/dataset")

    first = store.record_usage(session["id"], 100, 20)
    second = store.record_usage(session["id"], 50, 10)

    assert first == {"prompt_tokens": 100, "completion_tokens": 20}
    assert second == {"prompt_tokens": 150, "completion_tokens": 30}
    assert store.get_session(session["id"])["prompt_tokens"] == 150
    assert store.get_session(session["id"])["completion_tokens"] == 30


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


def test_package_rows_survive_session_delete(tmp_path: Path) -> None:
    # The shared package environment outlives any single session, so deleting
    # the session that happened to trigger an install must not delete the
    # record of a package still installed on disk — hence ON DELETE SET NULL
    # on packages.session_id, unlike every other table's ON DELETE CASCADE.
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    session = store.create_session("owner/dataset")
    store.record_package("seaborn", "seaborn==0.13.2", "0.13.2", "user", session["id"])

    store.delete_session(session["id"])

    packages = store.list_packages()
    assert len(packages) == 1
    assert packages[0]["name"] == "seaborn"
    assert packages[0]["session_id"] is None


def test_record_package_upserts_on_reinstall(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    session = store.create_session("owner/dataset")
    store.record_package("seaborn", "seaborn==0.12.0", "0.12.0", "user", session["id"])

    store.record_package("seaborn", "seaborn==0.13.2", "0.13.2", "assistant", session["id"])

    packages = store.list_packages()
    assert len(packages) == 1
    assert packages[0]["requirement"] == "seaborn==0.13.2"
    assert packages[0]["version"] == "0.13.2"
    assert packages[0]["author"] == "assistant"


def test_forget_package_removes_row(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    session = store.create_session("owner/dataset")
    store.record_package("seaborn", "seaborn==0.13.2", "0.13.2", "user", session["id"])

    store.forget_package("seaborn")

    assert store.list_packages() == []


def test_revise_cell_raises_key_error_for_nonexistent_cell(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    try:
        store.revise_cell("nonexistent-cell-id", "x = 1", "user")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")


def test_session_exists(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    session = store.create_session("owner/dataset")
    assert store.session_exists(session["id"]) is True
    assert store.session_exists("nonexistent-session") is False


def test_connect_rolls_back_on_error(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    session = store.create_session("owner/dataset")

    try:
        with store.connect() as db:
            db.execute("UPDATE sessions SET title = 'Failed Update' WHERE id = ?", (session["id"],))
            raise RuntimeError("something went wrong")
    except RuntimeError:
        pass

    assert store.get_session(session["id"])["title"] is None

