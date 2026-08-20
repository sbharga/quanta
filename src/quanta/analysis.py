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
from quanta.settings import RuntimeSettings


class AnalysisState(TypedDict, total=False):
    session_id: str
    user_request: str
    manifest: dict[str, Any]
    summary: str
    suggestions: list[str]
    decision: dict[str, Any]
    recent_result: dict[str, Any]
    last_text_cell: dict[str, Any]
    iterations: int
    completion_message: str
    unchanged_generations: int


class DatasetBrief(BaseModel):
    summary: str = Field(description="A concise description of the available data")
    suggestions: list[str] = Field(
        min_length=3, max_length=6, description="Specific analysis questions for this dataset"
    )


class AnalystDecision(BaseModel):
    action: Literal["create", "edit", "rerun", "text", "ask", "finish"]
    rationale: str = Field(description="One or two sentences on why this action follows from the notebook so far")
    instruction: str = Field(description="What to write, change, run, explain, or ask")
    target_cell_id: str | None = Field(default=None, description="Existing cell ID for edit or rerun")


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
    ):
        self.config = config
        self.store = store
        self.events = events
        self.notebook = notebook
        self.settings = settings
        self._model_instance: ChatOpenAI | None = None
        self._model_signature: tuple[str, str, str] | None = None
        self._model_lock = threading.Lock()
        self._cancelled: set[str] = set()
        self._cancel_lock = threading.Lock()
        self._model_schedule_lock = threading.Lock()
        self._next_model_call = 0.0
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
        graph.add_node("generate_text", self._generate_text)
        graph.add_node("ask", self._ask)
        graph.add_node("finish", self._finish)
        graph.add_edge(START, "load")
        graph.add_conditional_edges("load", self._route_load, ["ingest", "collect_request", "decide"])
        graph.add_edge("ingest", "summarize")
        graph.add_conditional_edges("summarize", self._route_summary, ["collect_request", "decide"])
        graph.add_edge("collect_request", "decide")
        graph.add_conditional_edges(
            "decide",
            self._route_decision,
            ["generate_code", "execute", "generate_text", "ask", "finish"],
        )
        graph.add_conditional_edges("generate_code", self._route_after_generate, ["execute", "decide"])
        graph.add_conditional_edges("execute", self._route_after_step, ["decide", "finish"])
        graph.add_conditional_edges("generate_text", self._route_after_step, ["decide", "finish"])
        graph.add_conditional_edges("ask", self._route_after_step, ["decide", "finish"])
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
                            f"Most recent execution:\n{json.dumps(state.get('recent_result', {}), default=str)[:12000]}\n\n"
                            f"Most recent narrative cell:\n{json.dumps(state.get('last_text_cell', {}), default=str)[:2000]}"
                        )
                    ),
                ]
            ),
        )
        assert isinstance(decision, AnalystDecision)  # guaranteed by with_structured_output's schema
        payload = decision.model_dump()
        self.events.emit(session_id, "agent_decision", **payload)
        return {"decision": payload, "iterations": state.get("iterations", 0) + 1}

    def _route_decision(self, state: AnalysisState) -> str:
        action = state["decision"]["action"]
        if action in {"create", "edit"}:
            return "generate_code"
        if action == "rerun":
            return "execute"
        if action == "text":
            return "generate_text"
        if action == "ask":
            return "ask"
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
        columns = column_brief(state["manifest"])
        prompt = _code_generation_prompt(
            instruction=decision["instruction"],
            manifest=state["manifest"],
            columns=columns,
            target=target,
            recent_result=state.get("recent_result", {}),
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
        self.notebook.restore_kernel(session_id)
        result = self.notebook.run_cell(session_id, cell_id)
        self._check_cancelled(session_id)
        return {"recent_result": {"cell_id": cell_id, **result}}

    def _generate_text(self, state: AnalysisState) -> AnalysisState:
        session_id = state["session_id"]
        self._check_cancelled(session_id)
        prompt = (
            "Write a concise notebook Markdown cell explaining the analysis result. "
            "Do not claim facts not present in the execution output.\n\n"
            f"Instruction: {state['decision']['instruction']}\n"
            f"Latest output: {json.dumps(state.get('recent_result', {}), default=str)[:12000]}"
        )
        source = self._stream_model(session_id, "text", prompt, action="create")
        cell = self.notebook.create_cell(session_id, "markdown", source)
        # Deliberately leave `recent_result` untouched: it still holds the last
        # execution's real output, which `_decide` needs on the next turn.
        # Clobbering it with the narrative cell was why the agent kept
        # re-running analysis it had just explained.
        return {"last_text_cell": {"cell_id": cell["id"], "source": source}}

    def _ask(self, state: AnalysisState) -> AnalysisState:
        session_id = state["session_id"]
        question = state["decision"]["instruction"]
        answer = interrupt({"kind": "followup", "question": question})
        self._check_cancelled(session_id)
        request = f"{state.get('user_request', '')}\nUser clarification: {answer}"
        self.store.add_request(session_id, str(answer))
        self.events.emit(session_id, "user_request", content=str(answer))
        return {"user_request": request}

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
            self._wait_for_model_slot(session_id)
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
            self._wait_for_model_slot(session_id)
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

    def _wait_for_model_slot(self, session_id: str) -> None:
        with self._model_schedule_lock:
            now = time.monotonic()
            scheduled = max(now, self._next_model_call)
            self._next_model_call = scheduled + self.config.model_min_interval
        remaining = scheduled - time.monotonic()
        if remaining > 0:
            self.events.emit(
                session_id,
                "status",
                status="running",
                phase="model_pacing",
                message=f"Pacing model requests for {remaining:.1f}s",
            )
        while remaining > 0:
            self._check_cancelled(session_id)
            time.sleep(min(remaining, 0.25))
            remaining = scheduled - time.monotonic()

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

    def run_manual(self, session_id: str, task: Any) -> bool:
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
                self.events.emit(session_id, "status", status="running", message="Executing code")
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


def _code_generation_prompt(
    *,
    instruction: str,
    manifest: dict[str, Any],
    columns: str,
    target: dict[str, Any] | None,
    recent_result: dict[str, Any],
) -> str:
    prompt = (
        "Write one executable Python notebook cell for the requested data-analysis step. "
        "Use DATASET_DIR (a pathlib.Path) for files; pandas as pd, numpy as np, and pyplot as plt are already available. "
        "End with the key DataFrame/value expression when a table should be displayed. Return code only, without fences.\n\n"
        f"Dataset inventory:\n{compact_manifest(manifest)}\n"
    )
    if columns:
        prompt += f"\n{columns}\n"
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
Use create to add and run a new code cell. Use edit when an earlier failed or flawed cell must be corrected, and provide its exact cell ID. Use rerun to execute an existing correct cell again. Use text to add a concise interpretation after useful output. Use ask only when essential ambiguity prevents progress. Use finish once the user's request has been answered.
Inspect outputs before drawing conclusions. Recover from errors by editing the responsible cell. Prefer small, readable cells, bounded displays, and evidence-backed analysis. Never invent results. Do not access files outside DATASET_DIR and do not install packages.
Before choosing create, check the notebook so far: never regenerate a cell whose output already answers the same question. If useful analysis already ran and was explained in a text cell, treat that as done and move the request forward instead of repeating it.
Visualize when it helps: prefer a chart over a text-only answer for a distribution, comparison, trend, or relationship. A dataset with numeric or categorical columns almost always has at least one chart worth showing before you finish.
Column names may contain leading/trailing whitespace or unusual characters — the code-generation step will show you their exact repr(); always index columns exactly as given there, never by a cleaned-up guess."""

_CODE_SYSTEM = """You write single, executable Python cells for a persistent notebook kernel.
DATASET_DIR (pathlib.Path), pd, np, and plt are already imported and available; do not re-import or redefine them.
Use only the standard library, pandas, NumPy, and Matplotlib. Do not import optional packages such as seaborn.
Column names are given to you verbatim via repr() — including any leading/trailing whitespace — and must be indexed exactly as shown, never guessed or stripped unless the task explicitly asks you to clean them.
When a chart is appropriate: create exactly one figure with plt.figure(figsize=(9, 5)) or an equivalent axes call, always set a title and axis labels, and do not call plt.show() — every open figure is captured and displayed automatically after the cell runs.
Return only the Python source, with no prose before or after it and no markdown fence."""
