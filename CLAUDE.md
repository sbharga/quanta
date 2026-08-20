Quanta is a local Flask + LangGraph application that turns a Kaggle dataset into a live, AI-authored analysis notebook: it downloads and inventories a dataset, proposes analysis questions, and streams code, narrative, tables, plots, progress, and errors while a language model iteratively works through the request.

## Commands

- Use Python 3.14 and `uv`; install with `uv sync --group dev` and run with `uv run quanta`.
- Run the full checks with `uv run ruff check .`, `uv run mypy src`, and `uv run pytest -q`.
- Run one test with `uv run pytest tests/test_notebook.py::test_name -vv`; the suite needs neither Kaggle nor an LLM key.
- The browser serves committed `src/quanta/static/tailwind.css`, not the source stylesheet. After changing CSS or Tailwind classes, rebuild it with `./.tools/tailwindcss -i src/quanta/static/tailwind.src.css -o src/quanta/static/tailwind.css --minify`. The CLI is local and gitignored; see README for how to fetch it if absent.

## Runtime Map

- `src/quanta/__init__.py` is the CLI entrypoint; `app.py:create_app` wires the Flask routes and the process-local `Store`, `AnalysisEngine`, `RunManager`, `NotebookService`, `KernelManager`, and `PackageManager` singletons.
- `analysis.py` owns the LangGraph workflow, model pacing/retries, interrupt/resume behavior, and background run lifecycle. LangGraph checkpoints share the application SQLite file and use the session ID as `thread_id`.
- `db.py` is the durable source of truth. `notebook.py` projects its append-only cells, revisions, executions, and events into the restored timeline and `.ipynb` export.
- `kernel.py` runs one persistent spawned Python worker per session. Worker state survives cells and is reconstructed by replaying successful active revisions after process loss.
- Code-generation prompts include every successful active code revision in execution order. Keep this selection aligned with kernel replay so generated cells can safely reuse existing variables after process loss.
- `packages.py` installs/uninstalls third-party packages via `uv pip install --target <packages_dir>` into a shared directory the kernel worker appends to `sys.path`, outside the app's own `uv`-managed `.venv`.
- The frontend is one vanilla-JS controller in `static/app.js` over `templates/index.html`; there is no npm package or frontend framework.

## Contracts To Preserve

- SSE is durable, not ephemeral: emitters persist events first, `/events` replays by sequence, and `build_timeline` folds authoritative state on reopen. When adding an event, check its emitter, `KNOWN_EVENTS`/`applyEvent` in `app.js`, and whether `notebook.py` must restore or deliberately suppress it.
- Draft deltas are omitted as timeline entries, but `build_timeline` reconstructs an unfinished draft so refreshes during generation do not lose the stream.
- Internal `agent_decision` events remain persisted for workflow history but are intentionally not rendered; users see direct cell creation/edit streams instead of raw CREATE/EDIT commands.
- A terminal `complete`, `question`, `error`, or cancelled-status event must be emitted only after `RunManager` removes the session from `_running`; otherwise the newly enabled composer races into a 409.
- Cells and revisions are append-only. The UI and export show only the newest execution for the active revision, while SQLite retains older executions. Keep execution payloads revision-aware when changing projections.
- A generated edit identical to a failing source must not create or execute another revision. Code-generation prompts must include the latest traceback and place the required correction after the old source.
- Runtime Settings values override environment variables and are stored owner-only under the configured data directory. API responses must never return saved secrets.
- `instance/`, SQLite databases, downloaded datasets, generated artifacts, `.tools/`, and local settings are runtime data and are gitignored; tests should use `tmp_path` rather than the live database.
- Model-generated Python is intentionally executed with host permissions. Do not describe the worker limits as a security sandbox.
- The `packages` table is the one deliberate exception to append-only/cascade-delete: its `session_id` uses `ON DELETE SET NULL`, not `ON DELETE CASCADE`, because the shared package directory outlives any single session. The kernel worker appends (never prepends) `packages_dir` to `sys.path`, and `packages.py` refuses to install any name the app's own venv already provides — both exist so a notebook package can never shadow Quanta's own pandas/NumPy/Matplotlib. A new install only needs `importlib.invalidate_caches()` in the worker (`KernelManager.mark_all_stale`); only an uninstall needs the worker restarted, since an already-imported module can't be unloaded from a live process.

## Ruff Ignores

`BLE001` (blind except) and `S102` (exec use) are disabled project-wide, not per-file — see `pyproject.toml` for the rationale (deliberate top-level failure boundaries in `analysis.py`/`app.py`/`datasets.py`, and `kernel.py`'s `exec()`/`eval()` being the product itself, not incidental).
