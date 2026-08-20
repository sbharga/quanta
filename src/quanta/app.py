from __future__ import annotations

import atexit
import json
import shutil
import time
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)

from quanta.analysis import AnalysisEngine, RunManager
from quanta.config import Config
from quanta.datasets import normalize_dataset_ref
from quanta.db import Store
from quanta.events import EventSink
from quanta.kernel import KernelManager
from quanta.notebook import NotebookService
from quanta.settings import RuntimeSettings


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    if test_config and "INSTANCE_PATH" in test_config:
        app.instance_path = str(Path(test_config["INSTANCE_PATH"]).resolve())
    config = Config.from_env(app.instance_path)
    if test_config:
        config = Config(
            data_dir=Path(test_config.get("DATA_DIR", config.data_dir)),
            database_path=Path(test_config.get("DATABASE_PATH", config.database_path)),
            llm_api_key=test_config.get("LLM_API_KEY", config.llm_api_key),
            llm_model=test_config.get("LLM_MODEL", config.llm_model),
            llm_base_url=test_config.get("LLM_BASE_URL", config.llm_base_url),
            execution_timeout=test_config.get("EXECUTION_TIMEOUT", config.execution_timeout),
            max_iterations=test_config.get("MAX_ITERATIONS", config.max_iterations),
            max_download_bytes=test_config.get("MAX_DOWNLOAD_BYTES", config.max_download_bytes),
            max_output_chars=test_config.get("MAX_OUTPUT_CHARS", config.max_output_chars),
            model_min_interval=test_config.get("MODEL_MIN_INTERVAL", config.model_min_interval),
            model_max_retries=test_config.get("MODEL_MAX_RETRIES", config.model_max_retries),
        )
        config.data_dir.mkdir(parents=True, exist_ok=True)
        config.database_path.parent.mkdir(parents=True, exist_ok=True)
    store = Store(config.database_path)
    store.initialize()
    store.reconcile_interrupted_sessions()
    store.reconcile_interrupted_executions()
    events = EventSink(store)
    settings = RuntimeSettings(
        config.data_dir / "settings.json",
        config.llm_model,
        config.llm_base_url,
        config.llm_api_key,
    )
    kernels = KernelManager(config.execution_timeout, config.max_output_chars)
    notebook = NotebookService(config, store, events, kernels)
    engine = AnalysisEngine(config, store, events, notebook, settings)
    runs = RunManager(engine, store, events)
    app.extensions["quanta"] = {
        "config": config,
        "store": store,
        "events": events,
        "kernels": kernels,
        "notebook": notebook,
        "settings": settings,
        "engine": engine,
        "runs": runs,
    }
    atexit.register(kernels.close)

    @app.get("/")
    def index() -> str:
        return render_template("index.html", model=settings.llm_model())

    @app.get("/api/settings")
    def get_settings() -> Response:
        return jsonify(settings.public())

    @app.put("/api/settings")
    def update_settings() -> tuple[Response, int] | Response:
        body = request.get_json(silent=True) or {}
        allowed = {"llm_api_key", "kaggle_api_token", "llm_model", "llm_base_url"}
        unknown = set(body) - allowed
        if unknown:
            return jsonify({"error": f"Unsupported settings: {', '.join(sorted(unknown))}"}), 400
        invalid = [key for key, value in body.items() if value is not None and not isinstance(value, str)]
        if invalid:
            return jsonify({"error": f"Settings must be strings: {', '.join(sorted(invalid))}"}), 400
        base_url = body.get("llm_base_url")
        if base_url is not None and base_url.strip() and not base_url.strip().startswith(("http://", "https://")):
            return jsonify({"error": "Base URL must start with http:// or https://"}), 400
        updated = settings.update(
            llm_api_key=body.get("llm_api_key") if "llm_api_key" in body else None,
            kaggle_api_token=body.get("kaggle_api_token") if "kaggle_api_token" in body else None,
            llm_model=body.get("llm_model") if "llm_model" in body else None,
            llm_base_url=base_url if "llm_base_url" in body else None,
        )
        return jsonify(updated)

    @app.get("/api/sessions")
    def list_sessions() -> Response:
        return jsonify({"sessions": store.list_sessions()})

    @app.post("/api/sessions")
    def create_session() -> tuple[Response, int] | Response:
        body = request.get_json(silent=True) or {}
        try:
            dataset_ref = normalize_dataset_ref(str(body.get("dataset", "")))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        session = store.create_session(dataset_ref)
        config.session_dir(session["id"])
        events.emit(session["id"], "session_created", session=session)
        runs.start(session["id"])
        return jsonify(session), 201

    @app.get("/api/sessions/<session_id>")
    def get_session(session_id: str) -> Response:
        try:
            session = store.get_session(session_id)
        except KeyError:
            abort(404)
        session["cells"] = store.list_cells(session_id)
        session["running"] = runs.is_running(session_id)
        timeline = notebook.build_timeline(session_id)
        session["timeline"] = timeline["entries"]
        session["event_cursor"] = timeline["event_cursor"]
        session["active_draft"] = timeline["active_draft"]
        if session["status"] == "waiting":
            question = store.latest_event(session_id, "question")
            session["pending_prompt"] = question["payload"].get("prompt") if question else None
        return jsonify(session)

    @app.patch("/api/sessions/<session_id>")
    def update_session(session_id: str) -> tuple[Response, int] | Response:
        try:
            store.get_session(session_id)
        except KeyError:
            abort(404)
        body = request.get_json(silent=True) or {}
        if "title" not in body:
            return jsonify({"error": "title is required"}), 400
        title = str(body["title"]).strip()
        if not title:
            return jsonify({"error": "title cannot be empty"}), 400
        store.update_session(session_id, title=title[:200])
        return jsonify(store.get_session(session_id))

    @app.delete("/api/sessions/<session_id>")
    def delete_session(session_id: str) -> tuple[Response, int] | Response:
        try:
            store.get_session(session_id)
        except KeyError:
            abort(404)
        runs.cancel(session_id)
        kernels.stop(session_id)
        store.delete_session(session_id)
        shutil.rmtree(config.session_dir(session_id), ignore_errors=True)
        return jsonify({"deleted": True})

    @app.get("/api/sessions/<session_id>/notebook.ipynb")
    def download_notebook(session_id: str) -> tuple[Response, int] | Response:
        try:
            store.get_session(session_id)
        except KeyError:
            abort(404)
        document = notebook.export_notebook(session_id)
        response = jsonify(document)
        response.headers["Content-Disposition"] = f'attachment; filename="quanta-{session_id[:8]}.ipynb"'
        response.mimetype = "application/x-ipynb+json"
        return response

    @app.post("/api/sessions/<session_id>/responses")
    def respond(session_id: str) -> tuple[Response, int] | Response:
        try:
            session = store.get_session(session_id)
        except KeyError:
            abort(404)
        body = request.get_json(silent=True) or {}
        content = str(body.get("content", "")).strip()
        if not content:
            return jsonify({"error": "A response is required"}), 400
        if not session.get("title"):
            store.update_session(session_id, title=_derive_title(content))
        started = (
            runs.start(session_id, resume=content)
            if session["status"] == "waiting"
            else runs.start(session_id, followup=content)
        )
        if not started:
            return jsonify({"error": "This session is already running"}), 409
        return jsonify({"started": True})

    @app.patch("/api/sessions/<session_id>/cells/<cell_id>")
    def edit_cell(session_id: str, cell_id: str) -> tuple[Response, int] | Response:
        source = str((request.get_json(silent=True) or {}).get("source", ""))
        if not source.strip():
            return jsonify({"error": "Cell source cannot be empty"}), 400
        try:
            cell = notebook.revise_cell(session_id, cell_id, source, "user")
        except KeyError:
            abort(404)
        return jsonify(cell)

    @app.post("/api/sessions/<session_id>/cells/<cell_id>/run")
    def run_cell(session_id: str, cell_id: str) -> tuple[Response, int] | Response:
        try:
            cell = store.get_cell(cell_id)
        except KeyError:
            abort(404)
        if cell["session_id"] != session_id:
            abort(404)
        if cell["kind"] != "code":
            return jsonify({"error": "Only code cells can be executed"}), 400

        def execute() -> None:
            try:
                notebook.restore_kernel(session_id)
                notebook.run_cell(session_id, cell_id)
            except Exception as exc:
                events.emit(session_id, "error", message=f"{type(exc).__name__}: {exc}")

        if not runs.run_manual(session_id, execute):
            return jsonify({"error": "Wait for the current run to pause or finish"}), 409
        return jsonify({"started": True}), 202

    @app.post("/api/sessions/<session_id>/cancel")
    def cancel(session_id: str) -> Response:
        try:
            store.get_session(session_id)
        except KeyError:
            abort(404)
        runs.cancel(session_id)
        kernels.stop(session_id)
        return jsonify({"cancelling": runs.is_running(session_id)})

    @app.get("/api/sessions/<session_id>/events")
    def event_stream(session_id: str) -> Response:
        try:
            store.get_session(session_id)
        except KeyError:
            abort(404)
        last_id = request.headers.get("Last-Event-ID") or request.args.get("after", "0")
        try:
            cursor = int(last_id)
        except ValueError:
            cursor = 0

        @stream_with_context
        def generate():
            nonlocal cursor
            heartbeat = time.monotonic()
            while True:
                # Drain everything available before sleeping, instead of
                # yielding at most one 250-row page per 0.25s tick — a
                # session that fell behind (e.g. after a long replay) used
                # to trickle out for seconds even though the rows were
                # already sitting in the database.
                records = store.events_after(session_id, cursor)
                while records:
                    for record in records:
                        cursor = record["seq"]
                        payload = json.dumps(record["payload"], separators=(",", ":"))
                        yield f"id: {cursor}\nevent: {record['type']}\ndata: {payload}\n\n"
                    records = store.events_after(session_id, cursor)
                if time.monotonic() - heartbeat >= 15:
                    yield ": heartbeat\n\n"
                    heartbeat = time.monotonic()
                time.sleep(0.25)

        # Flask's stream_with_context() stub loses generate()'s real (argument-less)
        # signature, so mypy thinks it still needs the wrapped generator's args.
        response = Response(generate(), mimetype="text/event-stream")  # type: ignore[call-arg]
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @app.get("/api/sessions/<session_id>/artifacts/<path:filename>")
    def artifact(session_id: str, filename: str):
        try:
            store.get_session(session_id)
        except KeyError:
            abort(404)
        directory = config.session_dir(session_id) / "artifacts"
        return send_from_directory(directory, filename)

    return app


def _derive_title(content: str) -> str:
    first_line = content.strip().splitlines()[0] if content.strip() else content
    if len(first_line) <= 60:
        return first_line
    return first_line[:57].rstrip() + "..."
