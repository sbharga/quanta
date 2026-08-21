from __future__ import annotations

import ast
import io
import multiprocessing as mp
import os
import queue
import reprlib
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any


class KernelManager:
    def __init__(self, timeout: int = 120, max_output_chars: int = 200_000, packages_dir: Path | None = None):
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.packages_dir = packages_dir
        self._workers: dict[str, KernelProcess] = {}
        self._lock = threading.Lock()

    def has_worker(self, session_id: str) -> bool:
        with self._lock:
            worker = self._workers.get(session_id)
            return bool(worker and worker.alive)

    def execute(
        self,
        session_id: str,
        source: str,
        work_dir: Path,
        artifact_dir: Path,
        on_output: Callable[[dict[str, Any]], None],
        capture_artifacts: bool = True,
    ) -> tuple[str, list[dict[str, Any]]]:
        worker = self._get_worker(session_id, work_dir, artifact_dir)
        return worker.execute(source, on_output, self.timeout, self.max_output_chars, capture_artifacts)

    def stop(self, session_id: str) -> None:
        with self._lock:
            worker = self._workers.pop(session_id, None)
        if worker:
            worker.stop()

    def close(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop()

    def mark_all_stale(self) -> None:
        """Flag every live worker to refresh its import caches before its next execute.

        Called after a package install/uninstall so sessions other than the one
        that triggered it also see the change, without losing their kernel state.
        """
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            worker.mark_stale()

    def _get_worker(self, session_id: str, work_dir: Path, artifact_dir: Path) -> KernelProcess:
        with self._lock:
            worker = self._workers.get(session_id)
            if worker is None or not worker.alive:
                worker = KernelProcess(work_dir, artifact_dir, self.packages_dir)
                self._workers[session_id] = worker
            return worker


class KernelProcess:
    def __init__(self, work_dir: Path, artifact_dir: Path, packages_dir: Path | None = None):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        context = mp.get_context("spawn")
        self.commands: Any = context.Queue()
        self.results: Any = context.Queue()
        self.process = context.Process(
            target=_worker_main,
            args=(self.commands, self.results, str(work_dir), str(artifact_dir), str(packages_dir or "")),
            daemon=True,
        )
        self.process.start()
        self.lock = threading.Lock()
        self._stale = False

    @property
    def alive(self) -> bool:
        return self.process.is_alive()

    def mark_stale(self) -> None:
        self._stale = True

    def execute(
        self,
        source: str,
        on_output: Callable[[dict[str, Any]], None],
        timeout: int,
        max_output_chars: int,
        capture_artifacts: bool = True,
    ) -> tuple[str, list[dict[str, Any]]]:
        request_id = str(uuid.uuid4())
        outputs: list[dict[str, Any]] = []
        output_chars = 0
        truncated = False
        with self.lock:
            if self._stale:
                # A package was installed/uninstalled since this worker started;
                # refresh its import caches before running the next cell. Sent
                # without an id, so the correlation check below silently ignores
                # any stray reply (the worker doesn't send one).
                self.commands.put({"type": "refresh"})
                self._stale = False
            self.commands.put(
                {"type": "execute", "id": request_id, "source": source, "capture_artifacts": capture_artifacts}
            )
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    message = self.results.get(timeout=min(0.25, max(0.01, deadline - time.monotonic())))
                except queue.Empty:
                    if not self.alive:
                        error = {"type": "error", "message": "Execution worker exited unexpectedly"}
                        on_output(error)
                        outputs.append(error)
                        return "error", outputs
                    continue
                if message.get("id") != request_id:
                    continue
                if message["type"] == "done":
                    return message["status"], outputs
                payload = {key: value for key, value in message.items() if key != "id"}
                if payload["type"] in {"stdout", "stderr"}:
                    remaining = max_output_chars - output_chars
                    if remaining <= 0:
                        if not truncated:
                            truncated = True
                            notice = {"type": payload["type"], "text": "\n…[output truncated]"}
                            outputs.append(notice)
                            on_output(notice)
                        continue
                    payload["text"] = payload["text"][:remaining]
                    output_chars += len(payload["text"])
                outputs.append(payload)
                on_output(payload)
        self.stop()
        timeout_output = {"type": "error", "message": f"Cell exceeded the {timeout}s timeout"}
        on_output(timeout_output)
        outputs.append(timeout_output)
        return "timeout", outputs

    def stop(self) -> None:
        if not self.process.is_alive():
            return
        try:
            self.commands.put({"type": "stop"})
            self.process.join(timeout=1)
        finally:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
            if self.process.is_alive() and hasattr(self.process, "kill"):
                self.process.kill()


class _QueueStream(io.TextIOBase):
    def __init__(self, result_queue: Any, request_id: str, stream_type: str):
        self.result_queue = result_queue
        self.request_id = request_id
        self.stream_type = stream_type

    def write(self, value: str) -> int:
        if value:
            self.result_queue.put({"id": self.request_id, "type": self.stream_type, "text": value})
        return len(value)

    def flush(self) -> None:
        return None


def _worker_main(commands: Any, results: Any, work_dir: str, artifact_dir: str, packages_dir: str = "") -> None:
    os.chdir(work_dir)
    _set_resource_limits()
    if packages_dir:
        # Appended, not prepended: the app's own site-packages must always
        # resolve pandas/numpy/matplotlib first. Only genuinely new top-level
        # packages installed via the in-app package manager come from here.
        import sys

        sys.path.append(packages_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    namespace: dict[str, Any] = {
        "__name__": "__quanta_notebook__",
        "DATASET_DIR": Path(work_dir),
        "pd": pd,
        "np": np,
        "plt": plt,
    }
    while True:
        command = commands.get()
        command_type = command.get("type")
        if command_type == "stop":
            return
        if command_type == "refresh":
            import importlib

            importlib.invalidate_caches()
            continue
        request_id = command["id"]
        capture_artifacts = command.get("capture_artifacts", True)
        stdout = _QueueStream(results, request_id, "stdout")
        stderr = _QueueStream(results, request_id, "stderr")
        status = "ok"
        try:
            import contextlib

            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                value = _execute_source(command["source"], namespace)
                if value is not None:
                    results.put({"id": request_id, **_serialize_value(value)})
                if capture_artifacts:
                    for figure_number in plt.get_fignums():
                        figure = plt.figure(figure_number)
                        filename = f"plot-{uuid.uuid4().hex}.png"
                        target = Path(artifact_dir) / filename
                        figure.savefig(target, dpi=140, bbox_inches="tight")
                        results.put({"id": request_id, "type": "image", "path": filename, "alt": "Generated plot"})
                plt.close("all")
        except BaseException:  # The traceback is the notebook output.
            status = "error"
            plt.close("all")
            results.put({"id": request_id, "type": "error", "message": traceback.format_exc()})
        results.put({"id": request_id, "type": "done", "status": status})


def _execute_source(source: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(source, mode="exec")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        statements = ast.Module(body=tree.body[:-1], type_ignores=[])
        expression = ast.Expression(body=tree.body[-1].value)
        if statements.body:
            exec(compile(statements, "<quanta-cell>", "exec"), namespace, namespace)
        return eval(compile(expression, "<quanta-cell>", "eval"), namespace, namespace)
    exec(compile(tree, "<quanta-cell>", "exec"), namespace, namespace)
    return None


def _serialize_value(value: Any) -> dict[str, Any]:
    import pandas as pd

    if isinstance(value, pd.Series):
        value = value.to_frame()
    if isinstance(value, pd.DataFrame):
        preview = value.head(200)
        clean = preview.astype(object).where(pd.notna(preview), None)
        rows = __import__("json").loads(clean.to_json(orient="values", date_format="iso"))
        return {
            "type": "table",
            "columns": [str(column) for column in clean.columns],
            "rows": rows,
            "total_rows": len(value),
            "truncated": len(value) > len(preview),
        }
    return {"type": "result", "text": reprlib.Repr().repr(value)}


def _set_resource_limits() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    except ImportError, OSError, ValueError:
        pass
