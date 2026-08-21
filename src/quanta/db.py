from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    dataset_ref TEXT NOT NULL,
                    title TEXT,
                    status TEXT NOT NULL,
                    summary TEXT,
                    manifest_json TEXT,
                    suggestions_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cells (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    active_revision_id TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(session_id, position)
                );
                CREATE TABLE IF NOT EXISTS cell_revisions (
                    id TEXT PRIMARY KEY,
                    cell_id TEXT NOT NULL REFERENCES cells(id) ON DELETE CASCADE,
                    revision INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    author TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(cell_id, revision)
                );
                CREATE TABLE IF NOT EXISTS executions (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    cell_id TEXT NOT NULL REFERENCES cells(id) ON DELETE CASCADE,
                    revision_id TEXT NOT NULL REFERENCES cell_revisions(id),
                    execution_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    outputs_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_session_seq
                    ON events(session_id, seq);
                CREATE INDEX IF NOT EXISTS idx_cells_session_position
                    ON cells(session_id, position);
                CREATE INDEX IF NOT EXISTS idx_executions_session
                    ON executions(session_id);
                CREATE INDEX IF NOT EXISTS idx_cell_revisions_cell
                    ON cell_revisions(cell_id);
                CREATE TABLE IF NOT EXISTS packages (
                    name TEXT PRIMARY KEY,
                    requirement TEXT NOT NULL,
                    version TEXT,
                    author TEXT NOT NULL,
                    session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
                    installed_at TEXT NOT NULL
                );
                """
            )
            # No migration framework exists yet, so patch older databases
            # in place. CREATE TABLE IF NOT EXISTS above never adds columns
            # to a table that already exists.
            columns = {row[1] for row in db.execute("PRAGMA table_info(sessions)")}
            if "title" not in columns:
                db.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
            if "prompt_tokens" not in columns:
                db.execute("ALTER TABLE sessions ADD COLUMN prompt_tokens INTEGER NOT NULL DEFAULT 0")
            if "completion_tokens" not in columns:
                db.execute("ALTER TABLE sessions ADD COLUMN completion_tokens INTEGER NOT NULL DEFAULT 0")

    def create_session(self, dataset_ref: str) -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        now = utc_now()
        with self.connect() as db:
            db.execute(
                "INSERT INTO sessions "
                "(id, dataset_ref, title, status, summary, manifest_json, suggestions_json, "
                "error, created_at, updated_at) "
                "VALUES (?, ?, NULL, ?, NULL, NULL, NULL, NULL, ?, ?)",
                (session_id, dataset_ref, "created", now, now),
            )
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(session_id)
        result = dict(row)
        result["manifest"] = _loads(result.pop("manifest_json"))
        result["suggestions"] = _loads(result.pop("suggestions_json")) or []
        return result

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT id, dataset_ref, title, status, summary, created_at, updated_at "
                "FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def reconcile_interrupted_sessions(self) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE sessions SET status = 'interrupted', "
                "error = 'The application stopped during this run. Submit a request to resume.', "
                "updated_at = ? WHERE status IN ('running', 'downloading', 'inspecting')",
                (utc_now(),),
            )

    def reconcile_interrupted_executions(self) -> None:
        now = utc_now()
        outputs = json.dumps([{"type": "error", "message": "Execution interrupted when Quanta stopped."}])
        with self.connect() as db:
            db.execute(
                "UPDATE executions SET status = 'interrupted', outputs_json = ?, finished_at = ? "
                "WHERE status = 'running'",
                (outputs, now),
            )

    def update_session(self, session_id: str, **values: Any) -> None:
        allowed = {"title", "status", "summary", "manifest", "suggestions", "error"}
        invalid = values.keys() - allowed
        if invalid:
            raise ValueError(f"Unsupported session fields: {sorted(invalid)}")
        encoded = {
            (f"{key}_json" if key in {"manifest", "suggestions"} else key): (
                json.dumps(value) if key in {"manifest", "suggestions"} else value
            )
            for key, value in values.items()
        }
        encoded["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in encoded)
        with self.connect() as db:
            db.execute(
                f"UPDATE sessions SET {assignments} WHERE id = ?",
                (*encoded.values(), session_id),
            )

    def record_usage(self, session_id: str, prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
        with self.connect() as db:
            db.execute(
                "UPDATE sessions SET prompt_tokens = prompt_tokens + ?, completion_tokens = completion_tokens + ? "
                "WHERE id = ?",
                (prompt_tokens, completion_tokens, session_id),
            )
            row = db.execute(
                "SELECT prompt_tokens, completion_tokens FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return {"prompt_tokens": row["prompt_tokens"], "completion_tokens": row["completion_tokens"]}

    def delete_session(self, session_id: str) -> None:
        with self.connect() as db:
            cursor = db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            if cursor.rowcount == 0:
                raise KeyError(session_id)
            # requests/cells/cell_revisions/executions/events cascade via FK.
            # LangGraph's checkpointer (SqliteSaver) shares this same database
            # file and keys its own tables by thread_id == session_id, but it
            # has no FK relationship to `sessions`, so it never cascades.
            for table in ("checkpoints", "writes"):
                exists = db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (table,),
                ).fetchone()
                if exists:
                    db.execute(f"DELETE FROM {table} WHERE thread_id = ?", (session_id,))

    def add_request(self, session_id: str, content: str) -> str:
        request_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute(
                "INSERT INTO requests VALUES (?, ?, ?, ?)",
                (request_id, session_id, content, utc_now()),
            )
        return request_id

    def list_requests(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT id, content, created_at FROM requests WHERE session_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def create_cell(self, session_id: str, kind: str, source: str, author: str = "assistant") -> dict[str, Any]:
        cell_id = str(uuid.uuid4())
        revision_id = str(uuid.uuid4())
        now = utc_now()
        with self.connect() as db:
            position = db.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 FROM cells WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            db.execute(
                "INSERT INTO cells VALUES (?, ?, ?, ?, ?, ?)",
                (cell_id, session_id, position, kind, revision_id, now),
            )
            db.execute(
                "INSERT INTO cell_revisions VALUES (?, ?, 1, ?, ?, ?)",
                (revision_id, cell_id, source, author, now),
            )
        return self.get_cell(cell_id)

    def session_exists(self, session_id: str) -> bool:
        with self.connect() as db:
            row = db.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return row is not None

    def revise_cell(self, cell_id: str, source: str, author: str) -> dict[str, Any]:
        revision_id = str(uuid.uuid4())
        with self.connect() as db:
            cell_row = db.execute("SELECT 1 FROM cells WHERE id = ?", (cell_id,)).fetchone()
            if cell_row is None:
                raise KeyError(cell_id)
            row = db.execute(
                "SELECT COALESCE(MAX(revision), 0) + 1 FROM cell_revisions WHERE cell_id = ?",
                (cell_id,),
            ).fetchone()
            revision = row[0]
            db.execute(
                "INSERT INTO cell_revisions VALUES (?, ?, ?, ?, ?, ?)",
                (revision_id, cell_id, revision, source, author, utc_now()),
            )
            db.execute(
                "UPDATE cells SET active_revision_id = ? WHERE id = ?",
                (revision_id, cell_id),
            )
        return self.get_cell(cell_id)

    def get_cell(self, cell_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                """
                SELECT c.*, r.revision, r.source, r.author, r.created_at AS revised_at
                FROM cells c JOIN cell_revisions r ON r.id = c.active_revision_id
                WHERE c.id = ?
                """,
                (cell_id,),
            ).fetchone()
        if row is None:
            raise KeyError(cell_id)
        return dict(row)

    def list_cells(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT c.*, r.revision, r.source, r.author, r.created_at AS revised_at
                FROM cells c JOIN cell_revisions r ON r.id = c.active_revision_id
                WHERE c.session_id = ? ORDER BY c.position
                """,
                (session_id,),
            ).fetchall()
            cells = [dict(row) for row in rows]
            executions_by_cell: dict[str, list[dict[str, Any]]] = {cell["id"]: [] for cell in cells}
            execution_rows = db.execute(
                "SELECT * FROM executions WHERE session_id = ? ORDER BY execution_count",
                (session_id,),
            ).fetchall()
            for item in execution_rows:
                execution = {**dict(item), "outputs": _loads(item["outputs_json"]) or []}
                execution.pop("outputs_json", None)
                executions_by_cell.setdefault(execution["cell_id"], []).append(execution)
            for cell in cells:
                cell["executions"] = executions_by_cell.get(cell["id"], [])
        return cells

    def start_execution(self, session_id: str, cell: dict[str, Any]) -> dict[str, Any]:
        execution_id = str(uuid.uuid4())
        with self.connect() as db:
            count = db.execute(
                "SELECT COALESCE(MAX(execution_count), 0) + 1 FROM executions WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            db.execute(
                "INSERT INTO executions VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    execution_id,
                    session_id,
                    cell["id"],
                    cell["active_revision_id"],
                    count,
                    "running",
                    "[]",
                    utc_now(),
                ),
            )
        return {"id": execution_id, "execution_count": count}

    def finish_execution(self, execution_id: str, status: str, outputs: list[dict[str, Any]]) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE executions SET status = ?, outputs_json = ?, finished_at = ? WHERE id = ?",
                (status, json.dumps(outputs), utc_now(), execution_id),
            )

    def append_event(self, session_id: str, event_type: str, payload: Any) -> int:
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO events(session_id, type, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (session_id, event_type, json.dumps(payload), utc_now()),
            )
            assert cursor.lastrowid is not None  # always set after a successful INSERT
            return cursor.lastrowid

    def events_after(self, session_id: str, seq: int, limit: int = 250) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM events WHERE session_id = ? AND seq > ? ORDER BY seq LIMIT ?",
                (session_id, seq, limit),
            ).fetchall()
        return [{**dict(row), "payload": _loads(row["payload_json"])} for row in rows]

    def latest_event(self, session_id: str, event_type: str | None = None) -> dict[str, Any] | None:
        query = "SELECT * FROM events WHERE session_id = ?"
        parameters: list[Any] = [session_id]
        if event_type:
            query += " AND type = ?"
            parameters.append(event_type)
        query += " ORDER BY seq DESC LIMIT 1"
        with self.connect() as db:
            row = db.execute(query, parameters).fetchone()
        if row is None:
            return None
        return {**dict(row), "payload": _loads(row["payload_json"])}

    def record_package(
        self, name: str, requirement: str, version: str | None, author: str, session_id: str | None
    ) -> None:
        with self.connect() as db:
            db.execute(
                """
                INSERT INTO packages(name, requirement, version, author, session_id, installed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    requirement = excluded.requirement,
                    version = excluded.version,
                    author = excluded.author,
                    session_id = excluded.session_id,
                    installed_at = excluded.installed_at
                """,
                (name, requirement, version, author, session_id, utc_now()),
            )

    def forget_package(self, name: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM packages WHERE name = ?", (name,))

    def list_packages(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM packages ORDER BY name").fetchall()
        return [dict(row) for row in rows]


def _loads(value: str | None) -> Any:
    return json.loads(value) if value else None
