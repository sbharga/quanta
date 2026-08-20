from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

from quanta.config import Config
from quanta.db import Store
from quanta.events import EventSink
from quanta.kernel import KernelManager


class NotebookService:
    def __init__(self, config: Config, store: Store, events: EventSink, kernels: KernelManager):
        self.config = config
        self.store = store
        self.events = events
        self.kernels = kernels

    def create_cell(self, session_id: str, kind: str, source: str, author: str = "assistant") -> dict[str, Any]:
        cell = self.store.create_cell(session_id, kind, source, author)
        self.events.emit(session_id, "cell_created", cell=_public_cell(cell))
        return cell

    def revise_cell(self, session_id: str, cell_id: str, source: str, author: str) -> dict[str, Any]:
        cell = self.store.get_cell(cell_id)
        if cell["session_id"] != session_id:
            raise KeyError(cell_id)
        revised = self.store.revise_cell(cell_id, source, author)
        self.events.emit(session_id, "cell_revised", cell=_public_cell(revised))
        self.events.emit(
            session_id,
            "cells_stale",
            after_position=revised["position"],
            message="Later cells may depend on this revision.",
        )
        return revised

    def run_cell(self, session_id: str, cell_id: str) -> dict[str, Any]:
        cell = self.store.get_cell(cell_id)
        if cell["session_id"] != session_id or cell["kind"] != "code":
            raise ValueError("Only code cells in this session can be executed")
        execution = self.store.start_execution(session_id, cell)
        self.events.emit(
            session_id,
            "execution_started",
            cell_id=cell_id,
            execution_id=execution["id"],
            execution_count=execution["execution_count"],
        )

        def stream(output: dict[str, Any]) -> None:
            self.events.emit(
                session_id,
                "execution_output",
                cell_id=cell_id,
                execution_id=execution["id"],
                output=output,
            )

        data_dir = self.config.session_dir(session_id) / "data"
        artifacts = self.config.session_dir(session_id) / "artifacts"
        status, outputs = self.kernels.execute(session_id, cell["source"], data_dir, artifacts, stream)
        self.store.finish_execution(execution["id"], status, outputs)
        self.events.emit(
            session_id,
            "execution_finished",
            cell_id=cell_id,
            execution_id=execution["id"],
            status=status,
            outputs=outputs,
        )
        return {"status": status, "outputs": outputs, **execution}

    def restore_kernel(self, session_id: str) -> None:
        if self.kernels.has_worker(session_id):
            return
        cells = self.store.list_cells(session_id)
        successful = []
        for cell in cells:
            active_executions = [
                execution for execution in cell["executions"] if execution["revision_id"] == cell["active_revision_id"]
            ]
            if cell["kind"] == "code" and active_executions and active_executions[-1]["status"] == "ok":
                successful.append((active_executions[-1]["execution_count"], cell))
        successful.sort(key=lambda item: item[0])
        if not successful:
            return
        self.events.emit(session_id, "kernel_replay_started", count=len(successful))
        data_dir = self.config.session_dir(session_id) / "data"
        artifacts = self.config.session_dir(session_id) / "artifacts"
        for index, (_count, cell) in enumerate(successful, 1):
            status, _ = self.kernels.execute(session_id, cell["source"], data_dir, artifacts, lambda _output: None)
            self.events.emit(
                session_id,
                "kernel_replay_progress",
                current=index,
                total=len(successful),
                cell_id=cell["id"],
                status=status,
            )
            if status != "ok":
                break
        self.events.emit(session_id, "kernel_replay_finished")

    def build_timeline(self, session_id: str) -> dict[str, Any]:
        """Fold the durable event log into an ordered, replay-safe narrative.

        Unlike the old client behaviour of nuking and rebuilding the whole
        timeline on every refresh, this is meant to be called exactly once
        when a session is opened. From then on the client applies live SSE
        events incrementally. Deliberately skips `draft_delta` (there can be
        tens of thousands of these in a long session) and treats `cell`
        entries as pointers into the current, authoritative cell state from
        `list_cells` rather than replaying every revision.
        """
        cells = {cell["id"]: _public_cell_full(cell) for cell in self.store.list_cells(session_id)}
        seen_cells: set[str] = set()
        entries: list[dict[str, Any]] = []
        active_draft: dict[str, Any] | None = None
        cursor = 0
        while True:
            batch = self.store.events_after(session_id, cursor, limit=500)
            if not batch:
                break
            for record in batch:
                cursor = record["seq"]
                event_type = record["type"]
                payload = record["payload"]
                if event_type == "user_request":
                    entries.append({"type": "request", "seq": cursor, "created_at": record["created_at"], **payload})
                elif event_type in ("cell_created", "cell_revised"):
                    cell_id = (payload.get("cell") or {}).get("id")
                    if cell_id and cell_id not in seen_cells:
                        seen_cells.add(cell_id)
                        entries.append({"type": "cell", "seq": cursor, "cell_id": cell_id})
                elif event_type in _NOTE_EVENT_TYPES:
                    entries.append(
                        {
                            "type": "note",
                            "seq": cursor,
                            "created_at": record["created_at"],
                            "event": event_type,
                            **payload,
                        }
                    )
                elif event_type in ("error", "complete"):
                    entries.append({"type": event_type, "seq": cursor, "created_at": record["created_at"], **payload})
                elif event_type == "draft_started":
                    active_draft = {**payload, "text": ""}
                elif event_type == "draft_delta" and active_draft:
                    if payload.get("draft_id") == active_draft.get("draft_id"):
                        active_draft["text"] += str(payload.get("delta", ""))
                elif (
                    event_type in ("draft_finished", "draft_discarded")
                    and active_draft
                    and payload.get("draft_id") == active_draft.get("draft_id")
                ):
                    active_draft = None
                # Skipped as timeline entries: draft events (tracked above only
                # to restore an in-progress stream), execution_started/output/
                # execution_finished (already folded into the cell snapshot),
                # dataset_manifest/dataset_summary/session_created/question
                # (served separately by GET /api/sessions/<id>).
            if len(batch) < 500:
                break
        for entry in entries:
            if entry["type"] == "cell":
                cell = cells.get(entry["cell_id"])
                if cell:
                    entry["cell"] = cell
        return {"entries": entries, "event_cursor": cursor, "active_draft": active_draft}

    def export_notebook(self, session_id: str) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        cells = self.store.list_cells(session_id)
        artifacts_dir = self.config.session_dir(session_id) / "artifacts"
        return {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "pygments_lexer": "ipython3"},
                "quanta": {
                    "session_id": session_id,
                    "dataset_ref": session["dataset_ref"],
                    "title": session.get("title"),
                },
            },
            "cells": [_export_cell(cell, artifacts_dir) for cell in cells],
        }


def _export_cell(cell: dict[str, Any], artifacts_dir: Path) -> dict[str, Any]:
    if cell["kind"] == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": _split_source(cell["source"]),
        }
    active = [item for item in cell["executions"] if item["revision_id"] == cell["active_revision_id"]]
    execution = active[-1] if active else None
    outputs: list[dict[str, Any]] = []
    execution_count = None
    if execution:
        execution_count = execution.get("execution_count")
        for output in execution["outputs"]:
            exported = _export_output(output, artifacts_dir)
            if exported is not None:
                outputs.append(exported)
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": execution_count,
        "source": _split_source(cell["source"]),
        "outputs": outputs,
    }


def _export_output(output: dict[str, Any], artifacts_dir: Path) -> dict[str, Any] | None:
    kind = output.get("type")
    if kind in ("stdout", "stderr"):
        return {
            "output_type": "stream",
            "name": kind,
            "text": _split_source(output.get("text", "")),
        }
    if kind == "error":
        message = output.get("message", "")
        lines = message.splitlines() or [message]
        # Kernel errors are a pre-formatted traceback.format_exc() string, not a
        # structured (type, value) pair, so ename/evalue are best-effort summaries.
        last_line = lines[-1] if lines else ""
        ename, _, evalue = last_line.partition(": ")
        return {
            "output_type": "error",
            "ename": ename or "Error",
            "evalue": evalue or last_line,
            "traceback": lines,
        }
    if kind == "result":
        text = output.get("text", "")
        return {
            "output_type": "execute_result",
            "execution_count": None,
            "metadata": {},
            "data": {"text/plain": _split_source(text)},
        }
    if kind == "table":
        return {
            "output_type": "execute_result",
            "execution_count": None,
            "metadata": {},
            "data": {
                "text/plain": _split_source(_table_text(output)),
                "text/html": _split_source(_table_html(output)),
            },
        }
    if kind == "image":
        image_path = artifacts_dir / output.get("path", "")
        try:
            data = image_path.read_bytes()
        except OSError:
            return None
        return {
            "output_type": "display_data",
            "metadata": {},
            "data": {"image/png": base64.b64encode(data).decode("ascii")},
        }
    return None


def _split_source(text: str) -> list[str]:
    """nbformat stores multi-line text as a list of lines, each keeping its trailing \\n."""
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    return lines


def _table_text(output: dict[str, Any]) -> str:
    columns = output.get("columns", [])
    rows = output.get("rows", [])
    header = "\t".join(str(column) for column in columns)
    body = "\n".join("\t".join("" if value is None else str(value) for value in row) for row in rows)
    return f"{header}\n{body}" if body else header


def _table_html(output: dict[str, Any]) -> str:
    columns = output.get("columns", [])
    rows = output.get("rows", [])
    head = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape('' if value is None else str(value))}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f'<table border="1"><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def _public_cell(cell: dict[str, Any]) -> dict[str, Any]:
    keys = {"id", "session_id", "position", "kind", "revision", "source", "author", "created_at", "revised_at"}
    return {key: value for key, value in cell.items() if key in keys}


def _public_cell_full(cell: dict[str, Any]) -> dict[str, Any]:
    base = _public_cell(cell)
    active = [
        execution
        for execution in cell.get("executions", [])
        if execution["revision_id"] == cell["active_revision_id"]
    ]
    base["executions"] = [
        {
            "id": execution["id"],
            "execution_count": execution["execution_count"],
            "status": execution["status"],
            "outputs": execution["outputs"],
            "started_at": execution["started_at"],
            "finished_at": execution["finished_at"],
        }
        for execution in active[-1:]
    ]
    return base


_NOTE_EVENT_TYPES = {
    "progress",
    "kernel_replay_started",
    "kernel_replay_progress",
    "kernel_replay_finished",
    "cells_stale",
    "note",
    "package_install_started",
    "package_install_finished",
}
# package_install_output is deliberately excluded: it streams one event per
# line of uv's output (potentially dozens per install) and is only ever
# rendered as a transient status-line update in the UI, not a timeline entry
# (see applyEvent in app.js) — folding it here would make a session refresh
# mid-install show a flood of notes that never appeared during the live run.
