from __future__ import annotations

import json
import random
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any, Literal, TypedDict, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field, SecretStr

from quanta.config import Config
from quanta.datasets import (
    column_brief,
    compact_manifest,
    download_dataset,
    inspect_dataset,
)
from quanta.db import Store
from quanta.events import EventSink
from quanta.notebook import NotebookService
from quanta.packages import PackageBackendUnavailable, PackageManager, canonical_name, package_name
from quanta.settings import RuntimeSettings


class AnalysisState(TypedDict, total=False):
    session_id: str
    user_request: str
    manifest: dict[str, Any]
    summary: str
    suggestions: list[str]
    decision: dict[str, Any]
    recent_result: dict[str, Any]
    recent_install_result: dict[str, Any]
    iterations: int
    completion_message: str
    unchanged_generations: int
    failed_installs: list[str]


class DatasetBrief(BaseModel):
    summary: str = Field(description="A concise description of the available data")
    suggestions: list[str] = Field(
        min_length=3, max_length=6, description="Specific analysis questions for this dataset"
    )


class AnalystDecision(BaseModel):
    action: Literal["create", "edit", "rerun", "ask", "install", "finish"]
    rationale: str = Field(description="One or two sentences on why this action follows from the notebook so far")
    instruction: str = Field(description="What to write, change, run, explain, or ask")
    narration: str = Field(
        default="",
        description=(
            "Optional concise Markdown interpretation of the latest executed code cell. Leave empty if the latest "
            "notebook cell is already Markdown or no new result needs explanation."
        ),
    )
    target_cell_id: str | None = Field(default=None, description="Existing code cell ID for edit or rerun")
    packages: list[str] = Field(
        default_factory=list,
        description="Package requirements to install (e.g. 'seaborn', 'scikit-learn==1.5.0'), only when action is install",
    )


class AnalysisCancelled(RuntimeError):
    pass


T = TypeVar("T")


class AnalysisEngine:
    def __init__(
        self,
        config: Config,
        store: Store,
        events: EventSink,
        notebook: NotebookService,
        settings: RuntimeSettings,
        packages: PackageManager,
    ):
        self.config = config
        self.store = store
        self.events = events
        self.notebook = notebook
        self.settings = settings
        self.packages = packages
        self._model_instance: ChatOpenAI | None = None
        self._model_signature: tuple[str, str, str] | None = None
        self._model_lock = threading.Lock()
        self._cancelled: set[str] = set()
        self._cancel_lock = threading.Lock()
        connection = sqlite3.connect(config.database_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(connection)
        self.graph = self._build_graph().compile(checkpointer=self.checkpointer)

    def _build_graph(self):
        graph = StateGraph(AnalysisState)
        graph.add_node("load", self._load)
        graph.add_node("ingest", self._ingest)
        graph.add_node("summarize", self._summarize)
        graph.add_node("collect_request", self._collect_request)
        graph.add_node("decide", self._decide)
        graph.add_node("generate_code", self._generate_code)
        graph.add_node("execute", self._execute)
        graph.add_node("ask", self._ask)
        graph.add_node("install", self._install)
        graph.add_node("finish", self._finish)
        graph.add_edge(START, "load")
        graph.add_conditional_edges("load", self._route_load, ["ingest", "collect_request", "decide"])
        graph.add_edge("ingest", "summarize")
        graph.add_conditional_edges("summarize", self._route_summary, ["collect_request", "decide"])
        graph.add_edge("collect_request", "decide")
        graph.add_conditional_edges(
            "decide",
            self._route_decision,
            ["generate_code", "execute", "ask", "install", "finish"],
        )
        graph.add_conditional_edges("generate_code", self._route_after_generate, ["execute", "decide"])
        graph.add_conditional_edges("execute", self._route_after_step, ["decide", "finish"])
        graph.add_conditional_edges("ask", self._route_after_step, ["decide", "finish"])
        graph.add_conditional_edges("install", self._route_after_step, ["decide", "finish"])
        graph.add_edge("finish", END)
        return graph

    def _load(self, state: AnalysisState) -> AnalysisState:
        self._check_cancelled(state["session_id"])
        session = self.store.get_session(state["session_id"])
        return {
            "manifest": session["manifest"] or {},
            "summary": session["summary"] or "",
            "suggestions": session["suggestions"],
            "iterations": state.get("iterations", 0),
        }

    def _route_load(self, state: AnalysisState) -> str:
        if not state.get("manifest"):
            return "ingest"
        return "decide" if state.get("user_request") else "collect_request"

    def _ingest(self, state: AnalysisState) -> AnalysisState:
        session_id = state["session_id"]
        session = self.store.get_session(session_id)
        data_dir = self.config.session_dir(session_id) / "data"
        self.store.update_session(session_id, status="downloading", error=None)
        self.events.emit(session_id, "status", status="downloading", message="Downloading from Kaggle")

        def report_progress(line: str) -> None:
            self.events.emit(session_id, "progress", message=line)

        try:
            download_dataset(
                session["dataset_ref"],
                data_dir,
                report_progress,
                self.config.max_download_bytes,
                lambda: self.is_cancelled(session_id),
                self.settings.kaggle_api_token(),
            )
        except RuntimeError:
            self._check_cancelled(session_id)
            raise
        self.store.update_session(session_id, status="inspecting")
        self.events.emit(session_id, "status", status="inspecting", message="Inspecting every dataset file")
        manifest = inspect_dataset(data_dir)
        self.store.update_session(session_id, manifest=manifest)
        self.events.emit(session_id, "dataset_manifest", manifest=manifest)
        return {"manifest": manifest}

    def _summarize(self, state: AnalysisState) -> AnalysisState:
        session_id = state["session_id"]
        self._check_cancelled(session_id)
        model = self._model()
        if model is None:
            summary, suggestions = _fallback_brief(state["manifest"])
        else:
            structured = model.with_structured_output(DatasetBrief, method="function_calling")
            brief = self._invoke_model(
                session_id,
                "Waiting for the model to summarize the dataset",
                lambda: structured.invoke(
                    [
                        SystemMessage(
                            content="Summarize a dataset inventory for an analyst. Be precise, mention important files and fields, and suggest concrete analyses."
                        ),
                        HumanMessage(content=compact_manifest(state["manifest"])),
                    ]
                ),
            )
            assert isinstance(brief, DatasetBrief)  # guaranteed by with_structured_output's schema
            summary, suggestions = brief.summary, brief.suggestions
        self.store.update_session(session_id, status="waiting", summary=summary, suggestions=suggestions)
        self.events.emit(session_id, "dataset_summary", summary=summary, suggestions=suggestions)
        return {"summary": summary, "suggestions": suggestions}

    def _route_summary(self, state: AnalysisState) -> str:
        return "decide" if state.get("user_request") else "collect_request"

    def _collect_request(self, state: AnalysisState) -> AnalysisState:
        answer = interrupt(
            {
                "kind": "analysis_request",
                "summary": state.get("summary", ""),
                "suggestions": state.get("suggestions", []),
            }
        )
        request = str(answer).strip()
        if not request:
            request = "Explore the dataset and identify the most useful insights."
        self.store.add_request(state["session_id"], request)
        self.events.emit(state["session_id"], "user_request", content=request)
        return {"user_request": request, "iterations": 0}

    def _decide(self, state: AnalysisState) -> AnalysisState:
        session_id = state["session_id"]
        self._check_cancelled(session_id)
        model = self._model()
        if model is None:
            raise RuntimeError("Configure a model API key in Settings to run an AI analysis")
        cells = self.store.list_cells(session_id)
        context = _cell_context(cells)
        requests = self.store.list_requests(session_id)
        if requests and requests[-1]["content"] == state.get("user_request"):
            requests = requests[:-1]
        history = "\n".join(f"- {item['content']}" for item in requests)
        installed = _installed_packages_brief(self.packages)
        structured = model.with_structured_output(AnalystDecision, method="function_calling")
        decision = self._invoke_model(
            session_id,
            "Waiting for the model to choose the next step",
            lambda: structured.invoke(
                [
                    SystemMessage(content=_ANALYST_SYSTEM),
                    HumanMessage(
                        content=(
                            f"Dataset summary:\n{state.get('summary', '')}\n\n"
                            f"Conversation requests:\n{history or '(none)'}\n\n"
                            f"Current user request:\n{state.get('user_request', '')}\n\n"
                            f"Notebook so far:\n{context}\n\n"
                            f"Installed packages beyond the standard library: {installed}\n\n"
                            f"Most recent package-install result: "
                            f"{json.dumps(state.get('recent_install_result', {}), default=str)[:2000]}"
                        )
                    ),
                ]
            ),
        )
        assert isinstance(decision, AnalystDecision)  # guaranteed by with_structured_output's schema
        payload = decision.model_dump()
        markdown = _markdown_instruction(payload["instruction"]) if payload["action"] == "create" else None
        if markdown:
            payload["action"] = "finish"
            payload["instruction"] = "The requested analysis is complete."
            payload["narration"] = markdown
        self.events.emit(session_id, "agent_decision", **payload)
        narration = payload["narration"].strip()
        latest_cell = cells[-1] if cells else None
        if narration and (
            markdown or (latest_cell and latest_cell["kind"] == "code" and latest_cell["executions"])
        ):
            self.notebook.create_cell(session_id, "markdown", narration)
        return {"decision": payload, "iterations": state.get("iterations", 0) + 1}

    def _route_decision(self, state: AnalysisState) -> str:
        action = state["decision"]["action"]
        if action in {"create", "edit"}:
            return "generate_code"
        if action == "rerun":
            return "execute"
        if action == "ask":
            return "ask"
        if action == "install":
            return "install"
        return "finish"

    def _generate_code(self, state: AnalysisState) -> AnalysisState:
        session_id = state["session_id"]
        self._check_cancelled(session_id)
        decision = state["decision"]
        target = None
        if decision["action"] == "edit" and decision.get("target_cell_id"):
            try:
                target = self.store.get_cell(decision["target_cell_id"])
            except KeyError:
                target = None
            if target and (target["session_id"] != session_id or target["kind"] != "code"):
                self.events.emit(
                    session_id,
                    "note",
                    message="The requested edit target was not a code cell; creating a new code cell instead.",
                )
                target = None
        successful_cells = _successful_code_context(self.store.list_cells(session_id))
        columns = column_brief(state["manifest"])
        prompt = _code_generation_prompt(
            instruction=decision["instruction"],
            manifest=state["manifest"],
            columns=columns,
            target=target,
            recent_result=state.get("recent_result", {}),
            installed=_installed_packages_brief(self.packages),
            successful_cells=successful_cells,
        )
        source = self._stream_model(
            session_id,
            "code",
            prompt,
            system=_CODE_SYSTEM,
            action=decision["action"],
            target_cell_id=target["id"] if target else None,
        )
        source = _strip_code_fence(source)
        if not source:
            self.events.emit(
                session_id,
                "note",
                message="The model returned an empty code cell; asking it to try again.",
            )
            return {"decision": {**decision, "target_cell_id": None}}
        if target and source.strip() == target["source"].strip():
            unchanged = state.get("unchanged_generations", 0) + 1
            message = (
                "The generated replacement was identical to the failing cell, so Quanta did not save or execute it."
            )
            self.events.emit(session_id, "note", message=message)
            if unchanged >= 2:
                raise RuntimeError(
                    "The model generated the same failing code twice. Submit a follow-up request with a different approach."
                )
            return {
                "decision": {**decision, "target_cell_id": None},
                "recent_result": {**state.get("recent_result", {}), "model_feedback": message},
                "unchanged_generations": unchanged,
            }
        if target:
            cell = self.notebook.revise_cell(session_id, target["id"], source, "assistant")
        else:
            cell = self.notebook.create_cell(session_id, "code", source)
        return {"decision": {**decision, "target_cell_id": cell["id"]}, "unchanged_generations": 0}

    def _route_after_generate(self, state: AnalysisState) -> str:
        return "execute" if state["decision"].get("target_cell_id") else "decide"

    def _execute(self, state: AnalysisState) -> AnalysisState:
        session_id = state["session_id"]
        self._check_cancelled(session_id)
        cell_id = state["decision"].get("target_cell_id")
        if not cell_id:
            return {
                "recent_result": {
                    "status": "error",
                    "outputs": [{"type": "error", "message": "No target cell was selected"}],
                }
            }
        try:
            target = self.store.get_cell(cell_id)
        except KeyError:
            target = None
        if not target or target["session_id"] != session_id or target["kind"] != "code":
            message = "Only code cells in this session can be executed. Choose a code cell or create a new one."
            self.events.emit(session_id, "note", message=message)
            return {
                "recent_result": {
                    "status": "error",
                    "outputs": [{"type": "error", "message": message}],
                }
            }
        self.notebook.restore_kernel(session_id)
        result = self.notebook.run_cell(session_id, cell_id)
        self._check_cancelled(session_id)
        return {"recent_result": {"cell_id": cell_id, **result}}

    def _ask(self, state: AnalysisState) -> AnalysisState:
        session_id = state["session_id"]
        question = state["decision"]["instruction"]
        answer = interrupt({"kind": "followup", "question": question})
        self._check_cancelled(session_id)
        request = f"{state.get('user_request', '')}\nUser clarification: {answer}"
        self.store.add_request(session_id, str(answer))
        self.events.emit(session_id, "user_request", content=str(answer))
        return {"user_request": request}

    def _install(self, state: AnalysisState) -> AnalysisState:
        session_id = state["session_id"]
        self._check_cancelled(session_id)
        decision = state["decision"]
        already_failed = set(state.get("failed_installs", []))
        specs = [spec for spec in decision.get("packages", []) if spec not in already_failed]
        if not specs:
            message = "The requested package already failed to install earlier this run; not retrying it."
            self.events.emit(session_id, "note", message=message)
            return {
                "recent_install_result": {
                    "status": "error",
                    "outputs": [{"type": "error", "message": message}],
                }
            }
        installed_names = {
            canonical_name(package["name"])
            for package in self.packages.list_installed()
        }
        if all(canonical_name(package_name(spec)) in installed_names for spec in specs):
            message = f"Already installed: {', '.join(specs)}. Skipping the redundant install."
            self.events.emit(session_id, "note", message=message)
            return {
                "recent_install_result": {
                    "status": "ok",
                    "outputs": [{"type": "result", "text": message}],
                }
            }
        self.events.emit(session_id, "package_install_started", specs=specs)

        def progress(line: str) -> None:
            self.events.emit(session_id, "package_install_output", line=line)

        try:
            result = self.packages.install(specs, progress)
        except (ValueError, PackageBackendUnavailable) as exc:
            message = str(exc)
            self.events.emit(session_id, "package_install_finished", status="error", packages=[], message=message)
            return {
                "recent_install_result": {
                    "status": "error",
                    "outputs": [{"type": "error", "message": message}],
                },
                "failed_installs": [*state.get("failed_installs", []), *specs],
            }
        status = result["status"]
        message = (
            f"Installed {', '.join(specs)}" if status == "ok" else f"Failed to install {', '.join(specs)}"
        )
        if status == "ok":
            installed_by_name = {
                canonical_name(package["name"]): package["version"] for package in result["packages"]
            }
            for spec in specs:
                name = canonical_name(package_name(spec))
                self.store.record_package(name, spec, installed_by_name.get(name), "assistant", session_id)
            self.notebook.kernels.mark_all_stale()
        self.events.emit(
            session_id, "package_install_finished", status=status, packages=result["packages"], message=message
        )
        outputs = [{"type": "error" if status == "error" else "result", "text": result["log"] or message}]
        update: AnalysisState = {"recent_install_result": {"status": status, "outputs": outputs}}
        if status == "error":
            update["failed_installs"] = [*state.get("failed_installs", []), *specs]
        return update

    def _finish(self, state: AnalysisState) -> AnalysisState:
        self._check_cancelled(state["session_id"])
        self.store.update_session(state["session_id"], status="complete", error=None)
        return {"completion_message": state.get("decision", {}).get("instruction", "Analysis complete")}

    def _route_after_step(self, state: AnalysisState) -> str:
        return "finish" if state.get("iterations", 0) >= self.config.max_iterations else "decide"

    def _stream_model(
        self,
        session_id: str,
        kind: str,
        prompt: str,
        system: str | None = None,
        action: str = "create",
        target_cell_id: str | None = None,
    ) -> str:
        model = self._model()
        if model is None:
            raise RuntimeError("Configure a model API key in Settings")
        messages: list[Any] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        attempts = self.config.model_max_retries + 1
        for attempt in range(1, attempts + 1):
            self.events.emit(
                session_id,
                "status",
                status="running",
                phase="model_waiting",
                message=f"Waiting for the model to write {kind}",
            )
            draft_id = uuid.uuid4().hex
            self.events.emit(
                session_id,
                "draft_started",
                draft_id=draft_id,
                kind=kind,
                action=action,
                target_cell_id=target_cell_id,
            )
            chunks: list[str] = []
            streaming = False
            try:
                for chunk in model.stream(messages):
                    self._check_cancelled(session_id)
                    text = chunk.text
                    if text:
                        if not streaming:
                            streaming = True
                            self.events.emit(
                                session_id,
                                "status",
                                status="running",
                                phase="model_streaming",
                                message=f"Streaming {kind}",
                            )
                        chunks.append(text)
                        self.events.emit(session_id, "draft_delta", draft_id=draft_id, kind=kind, delta=text)
            except Exception as exc:
                self.events.emit(session_id, "draft_discarded", draft_id=draft_id, kind=kind)
                if attempt >= attempts or not _is_retryable_model_error(exc):
                    raise
                self._retry_model(session_id, exc, attempt)
                continue
            self.events.emit(session_id, "draft_finished", draft_id=draft_id, kind=kind)
            return "".join(chunks).strip()
        raise RuntimeError("Model generation failed")

    def _invoke_model(self, session_id: str, message: str, operation: Callable[[], T]) -> T:
        attempts = self.config.model_max_retries + 1
        for attempt in range(1, attempts + 1):
            self.events.emit(
                session_id,
                "status",
                status="running",
                phase="model_waiting",
                message=message,
            )
            try:
                return operation()
            except Exception as exc:
                if attempt >= attempts or not _is_retryable_model_error(exc):
                    raise
                self._retry_model(session_id, exc, attempt)
        raise RuntimeError("Model request failed")

    def _retry_model(self, session_id: str, exc: Exception, attempt: int) -> None:
        retry_after = _retry_after(exc)
        delay = retry_after if retry_after is not None else min(30.0, 2**attempt + random.uniform(0, 0.5))
        self.events.emit(
            session_id,
            "model_retry",
            attempt=attempt,
            max_retries=self.config.model_max_retries,
            delay=round(delay, 1),
            message=f"Model request was rate limited or unavailable; retrying in {delay:.1f}s",
        )
        deadline = time.monotonic() + delay
        while (remaining := deadline - time.monotonic()) > 0:
            self._check_cancelled(session_id)
            time.sleep(min(0.25, remaining))

    def _model(self) -> ChatOpenAI | None:
        api_key = self.settings.llm_api_key()
        if not api_key:
            return None
        model_name = self.settings.llm_model()
        base_url = self.settings.llm_base_url()
        signature = (api_key, model_name, base_url)
        with self._model_lock:
            if signature != self._model_signature:
                # ChatOpenAI talks to any OpenAI-compatible /chat/completions
                # endpoint via base_url — OpenRouter (the default), OpenAI
                # itself, a local Ollama/LM Studio/vLLM server, Groq,
                # Together, etc. all work without provider-specific code.
                self._model_instance = ChatOpenAI(
                    model=model_name,
                    api_key=SecretStr(api_key),
                    base_url=base_url,
                    temperature=0,
                    timeout=self.config.model_timeout,
                    max_retries=0,
                )
                self._model_signature = signature
            return self._model_instance

    def cancel(self, session_id: str) -> None:
        with self._cancel_lock:
            self._cancelled.add(session_id)

    def clear_cancelled(self, session_id: str) -> None:
        with self._cancel_lock:
            self._cancelled.discard(session_id)

    def is_cancelled(self, session_id: str) -> bool:
        with self._cancel_lock:
            return session_id in self._cancelled

    def _check_cancelled(self, session_id: str) -> None:
        if self.is_cancelled(session_id):
            raise AnalysisCancelled("Analysis cancelled")


class RunManager:
    def __init__(self, engine: AnalysisEngine, store: Store, events: EventSink):
        self.engine = engine
        self.store = store
        self.events = events
        self._running: set[str] = set()
        self._lock = threading.Lock()

    def start(self, session_id: str, resume: str | None = None, followup: str | None = None) -> bool:
        with self._lock:
            if session_id in self._running:
                return False
            self.engine.clear_cancelled(session_id)
            self._running.add(session_id)
        thread = threading.Thread(target=self._run, args=(session_id, resume, followup), daemon=True)
        thread.start()
        return True

    def run_manual(self, session_id: str, task: Any, message: str = "Executing code") -> bool:
        with self._lock:
            if session_id in self._running:
                return False
            self._running.add(session_id)
        previous_status = self.store.get_session(session_id)["status"]
        self.engine.clear_cancelled(session_id)

        def run() -> None:
            final_status: tuple[str, str] | None = None
            try:
                self.store.update_session(session_id, status="running", error=None)
                self.events.emit(session_id, "status", status="running", message=message)
                task()
            finally:
                if self.engine.is_cancelled(session_id):
                    self.store.update_session(session_id, status="cancelled", error=None)
                    final_status = ("cancelled", "Current work stopped")
                else:
                    self.store.update_session(session_id, status=previous_status, error=None)
                    final_status = (previous_status, "Ready for another request")
                with self._lock:
                    self._running.discard(session_id)
                self.events.emit(session_id, "status", status=final_status[0], message=final_status[1])

        threading.Thread(target=run, daemon=True).start()
        return True

    def cancel(self, session_id: str) -> None:
        with self._lock:
            running = session_id in self._running
        self.engine.cancel(session_id)
        if running:
            self.store.update_session(session_id, status="cancelling", error=None)
            self.events.emit(session_id, "status", status="cancelling", message="Stopping current work")

    def is_running(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._running

    def _run(self, session_id: str, resume: str | None, followup: str | None) -> None:
        final_event: tuple[str, dict[str, Any]] | None = None
        try:
            self.store.update_session(session_id, status="running", error=None)
            self.events.emit(session_id, "status", status="running", message="AI analysis is running")
            config = {"configurable": {"thread_id": session_id}, "recursion_limit": 100}
            if resume is not None:
                graph_input: Any = Command(resume=resume)
            else:
                graph_input = {
                    "session_id": session_id,
                    "iterations": 0,
                    "user_request": followup or "",
                }
                if followup:
                    self.store.add_request(session_id, followup)
                    self.events.emit(session_id, "user_request", content=followup)
            result = self.engine.graph.invoke(graph_input, config=config)
            interrupts = result.get("__interrupt__", ())
            if interrupts:
                payload = interrupts[0].value
                self.store.update_session(session_id, status="waiting")
                final_event = ("question", {"prompt": payload})
            else:
                final_event = (
                    "complete",
                    {"message": result.get("completion_message", "Analysis complete")},
                )
        except AnalysisCancelled:
            self.store.update_session(session_id, status="cancelled", error=None)
            final_event = ("status", {"status": "cancelled", "message": "Current work stopped"})
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.store.update_session(session_id, status="error", error=message)
            final_event = ("error", {"message": message})
        finally:
            with self._lock:
                self._running.discard(session_id)
        if final_event:
            self.events.emit(session_id, final_event[0], **final_event[1])


def _is_retryable_model_error(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {})
    value = headers.get("retry-after") if headers else None
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _fallback_brief(manifest: dict[str, Any]) -> tuple[str, list[str]]:
    table_files = [item for item in manifest["files"] if item.get("inspection", {}).get("kind") == "table"]
    names = ", ".join(item["path"] for item in manifest["files"][:8])
    summary = (
        f"The dataset contains {manifest['file_count']} files ({manifest['total_bytes']:,} bytes). "
        f"Detected {len(table_files)} tabular files. Available files include {names}."
    )
    return summary, [
        "Profile data quality, missing values, and unusual records",
        "Explore distributions and relationships among important variables",
        "Identify trends, segments, and the strongest actionable insights",
    ]


def _cell_context(cells: list[dict[str, Any]]) -> str:
    if not cells:
        return "No cells yet. Begin with loading the relevant files and inspecting their structure."
    blocks = []
    for cell in cells[-12:]:
        execution = cell["executions"][-1] if cell["executions"] else None
        output = json.dumps(execution, default=str)[:6000] if execution else "not executed"
        blocks.append(
            f"Cell {cell['id']} ({cell['kind']}, revision {cell['revision']}):\n"
            f"{cell['source'][:6000]}\nOutput: {output}"
        )
    return "\n\n".join(blocks)


def _installed_packages_brief(packages: PackageManager) -> str:
    """Return every installed distribution so the agent never re-requests a hidden package."""
    try:
        names = sorted(package["name"] for package in packages.list_installed())
    except OSError:
        return "(none)"
    if not names:
        return "(none)"
    return ", ".join(names)


def _successful_code_context(cells: list[dict[str, Any]]) -> str:
    """Render active code revisions whose latest execution succeeded, in execution order."""
    successful: list[tuple[int, dict[str, Any]]] = []
    for cell in cells:
        if cell["kind"] != "code":
            continue
        active_executions = [
            execution
            for execution in cell["executions"]
            if execution["revision_id"] == cell["active_revision_id"]
        ]
        if active_executions and active_executions[-1]["status"] == "ok":
            successful.append((active_executions[-1]["execution_count"], cell))
    successful.sort(key=lambda item: item[0])
    return "\n\n".join(
        f"Successful cell {cell['id']} (revision {cell['revision']}):\n{cell['source']}"
        for _execution_count, cell in successful
    )


def _markdown_instruction(instruction: str) -> str | None:
    """Extract a Markdown document mistakenly requested through the code-only create action."""
    stripped = instruction.strip()
    if stripped.lower().startswith("markdown\n"):
        return stripped.split("\n", 1)[1].strip() or None
    match = re.fullmatch(r"```(?:markdown|md)\s*\n(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match and match.group(1).strip() else None


def _code_generation_prompt(
    *,
    instruction: str,
    manifest: dict[str, Any],
    columns: str,
    target: dict[str, Any] | None,
    recent_result: dict[str, Any],
    installed: str = "(none)",
    successful_cells: str = "",
) -> str:
    prompt = (
        "Write one executable Python notebook cell for the requested data-analysis step. "
        "Use DATASET_DIR (a pathlib.Path) for files; pandas as pd, numpy as np, and pyplot as plt are already available. "
        "End with the key DataFrame/value expression when a table should be displayed. Return code only, without fences.\n\n"
        f"Dataset inventory:\n{compact_manifest(manifest)}\n"
        f"\nInstalled packages beyond the standard library, pandas, NumPy, and Matplotlib: {installed}\n"
    )
    if columns:
        prompt += f"\n{columns}\n"
    if successful_cells:
        prompt += (
            "\nPreviously successful cells, in execution order:\n"
            "These cells already ran in this persistent kernel. Every variable, import, function, and model they "
            "defined remains available. Reuse that state instead of reloading data or repeating completed work.\n\n"
            f"{successful_cells}\n"
        )
    if target:
        prompt += (
            f"\nExisting failing cell to replace:\n{target['source']}\n"
            f"\nLatest execution result and traceback:\n{json.dumps(recent_result, default=str)[:12000]}\n"
            "\nThe replacement must fix the execution error and must not reproduce the existing source unchanged.\n"
        )
    prompt += f"\nRequired change (follow this over the existing cell):\n{instruction}\n"
    return prompt


_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _strip_code_fence(source: str) -> str:
    """Extract the first fenced code block anywhere in the response.

    The model sometimes writes prose before the fence ("Here's the code:\n```python\n...")
    even when told not to. The old startswith("```") check only handled a fence at
    position 0, so any preamble left the fence markers embedded in the source and the
    cell failed with a SyntaxError. This also naturally handles the case of an empty
    fenced block (```python\n```), which now correctly returns "".
    """
    match = _FENCE_RE.search(source)
    if match:
        return match.group(1).strip()
    return source.strip()


_ANALYST_SYSTEM = """You control an iterative data-analysis notebook. Choose exactly one next action.
Use create to add and run a new code cell. Use edit when an earlier failed or flawed cell must be corrected, and provide its exact cell ID. Use rerun to execute an existing correct cell again. Use ask only when essential ambiguity prevents progress. Use install when a cell needs a third-party package that is not already installed — list the exact requirement(s) (e.g. "seaborn", "scikit-learn==1.5.0") in packages, and prefer the standard library, pandas, NumPy, or Matplotlib whenever they already cover the need. Use finish once the user's request has been answered.
The target_cell_id for edit or rerun must be the ID of a code cell, never a Markdown narration cell. If the missing work belongs in a new step rather than replacing existing code, choose create instead of edit.
The create and edit actions are exclusively for executable Python that performs computation. Never use them for prose, Markdown, a summary, an interpretation, or a conclusion, and never instruct a code cell to print narrative text. Put all user-facing prose in narration. When the analysis is complete, put the polished final answer itself in narration and choose finish; do not write a short meta-description saying that you summarized it.
When the latest notebook cell is executed code with useful output that has not yet been explained, put one concise, evidence-backed Markdown interpretation in narration while choosing the next action. Leave narration empty when the latest cell is already Markdown, when there is no new result, or when the result needs correction rather than explanation. Never repeat an interpretation.
Inspect outputs before drawing conclusions. Recover from errors by editing the responsible cell. Prefer small, readable cells, bounded displays, and evidence-backed analysis. Never invent results. Do not access files outside DATASET_DIR.
Before choosing create, check the notebook so far: never regenerate a cell whose output already answers the same question. If useful analysis already ran and was explained in a text cell, treat that as done and move the request forward instead of repeating it.
Before choosing install, check the installed-packages list given to you: never request a package that is already installed, and never request install again for a package whose install already failed earlier this run.
Visualize when it helps: prefer a chart over a text-only answer for a distribution, comparison, trend, or relationship. A dataset with numeric or categorical columns almost always has at least one chart worth showing before you finish.
Column names may contain leading/trailing whitespace or unusual characters — the code-generation step will show you their exact repr(); always index columns exactly as given there, never by a cleaned-up guess."""

_CODE_SYSTEM = """You write single, executable Python cells for a persistent notebook kernel.
DATASET_DIR (pathlib.Path), pd, np, and plt are already imported and available; do not re-import or redefine them. Read dataset files only through DATASET_DIR; never use Kaggle notebook paths such as /kaggle/input.
The prompt includes every active code cell whose latest execution succeeded. Those cells already ran in this kernel, so all variables, imports, functions, and models they defined remain available. Reuse them instead of reloading data or repeating completed computation.
You may import the standard library, pandas, NumPy, and Matplotlib, plus any package explicitly listed as installed below — nothing else. If a cell genuinely needs a package that isn't listed, do not import it; the calling agent installs packages via a separate action before code is generated.
This is a plain Python exec kernel, not Jupyter or IPython: do not use display(), IPython imports, or notebook magics. End with a value expression or use print() for text output.
Code cells are for computation and computed outputs, not prose presentation. Never convert a requested Markdown summary, interpretation, or conclusion into print(...); user-facing narrative belongs in a Markdown cell created by the analyst.
The kernel worker is a daemonic process and cannot create child processes. Use num_workers=0 and n_jobs=1, and disable multiprocessing/joblib parallelism in any similar API.
Column names are given to you verbatim via repr() — including any leading/trailing whitespace — and must be indexed exactly as shown, never guessed or stripped unless the task explicitly asks you to clean them.
When a chart is appropriate: create exactly one figure with plt.figure(figsize=(9, 5)) or an equivalent axes call, always set a title and axis labels, and do not call plt.show() — every open figure is captured and displayed automatically after the cell runs. In Matplotlib boxplot(), use tick_labels= rather than the removed labels= argument.
Return only the Python source, with no prose before or after it and no markdown fence."""
