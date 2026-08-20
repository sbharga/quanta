# Quanta

Quanta is a local Flask and LangGraph application that turns a Kaggle dataset into a live, AI-authored analysis notebook. It downloads and inventories the dataset, proposes analysis questions, and streams code, narrative, tables, plots, progress, and errors while a language model iteratively works through the request.

## Features

- Kaggle dataset URLs and `owner/dataset` slugs
- Bounded inspection of CSV, TSV, JSON, JSONL, Parquet, Excel, SQLite, text, and image files
- Searchable dataset explorer with schemas, row samples, file metadata, and inspection warnings
- Dataset-specific suggested questions plus a custom request
- Durable LangGraph checkpoints and complete SQLite session history
- Live Server-Sent Events with replay after browser reconnects
- Persistent per-session Python state with stdout, stderr, DataFrames, values, and Matplotlib figures; generated code receives every successful active cell and can reuse its variables
- Immutable cell revisions, manual editing, reruns, and AI-directed error correction
- Session restoration after application restarts
- Session rename and delete from the sidebar
- One-click `.ipynb` export of the full notebook, including outputs and plots
- Click a generated plot to enlarge it
- In-app Python package manager: install third-party packages (seaborn, scikit-learn, ...) from the UI, or let the AI analysis agent install what a cell needs, without restarting Quanta

## Setup

Python 3.14 and [uv](https://docs.astral.sh/uv/) are expected.

```bash
uv sync --group dev
export LLM_API_KEY="your-key"
export LLM_MODEL="openai/gpt-4o-mini"
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export KAGGLE_API_TOKEN="your-kaggle-token"
uv run quanta
```

Open <http://127.0.0.1:5000>. Kaggle also supports credentials in `~/.kaggle/access_token` or the legacy `~/.kaggle/kaggle.json` file.

The model is reached through any **OpenAI-compatible** `/chat/completions` endpoint, configured by `LLM_BASE_URL` plus `LLM_API_KEY`/`LLM_MODEL`. The default (`https://openrouter.ai/api/v1`) routes through [OpenRouter](https://openrouter.ai), so `LLM_MODEL` follows OpenRouter's `provider/model` naming (e.g. `openai/gpt-4o-mini`) by default — but pointing `LLM_BASE_URL` elsewhere works just as well:

| Provider | `LLM_BASE_URL` | Example `LLM_MODEL` |
| --- | --- | --- |
| OpenRouter (default) | `https://openrouter.ai/api/v1` | `openai/gpt-4o-mini` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Ollama (local) | `http://localhost:11434/v1` | `llama3.1` |
| LM Studio / vLLM (local) | whatever the server prints on startup | the loaded model's name |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Together | `https://api.together.xyz/v1` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |

Dataset inspection works without a model key, but AI analysis requires one.

Open **Settings** (the gear icon at the bottom of the sidebar) to configure or replace the model API key, model, base URL, and Kaggle API token without restarting Quanta. Values entered in the app override environment variables. Secrets are never returned to the browser and are stored in `instance/settings.json` with owner-only permissions. Clearing a saved override returns that setting to its environment-variable or Kaggle credential-file fallback.

## Frontend

The UI is server-rendered HTML plus vanilla JavaScript (`src/quanta/static/app.js`) styled with [Tailwind CSS](https://tailwindcss.com) — no Node.js or npm is required to run Quanta. Tailwind's standalone CLI is used only to rebuild the compiled stylesheet after editing styles:

```bash
mkdir -p .tools
curl -sL -o .tools/tailwindcss \
  https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-linux-x64
chmod +x .tools/tailwindcss

.tools/tailwindcss -i src/quanta/static/tailwind.src.css \
                   -o src/quanta/static/tailwind.css --minify
```

`src/quanta/static/tailwind.css` is the compiled, committed stylesheet the app actually serves; `tailwind.src.css` is the source (theme tokens plus a small set of hand-written rules for markdown rendering and code highlighting) and only needs rebuilding when you change styles. Pick the CLI binary matching your platform from the [Tailwind releases page](https://github.com/tailwindlabs/tailwindcss/releases/latest) if you're not on `linux-x64`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_API_KEY` | none | Authentication for the model endpoint |
| `LLM_MODEL` | `openai/gpt-4o-mini` | Analysis model |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible `/chat/completions` endpoint |
| `QUANTA_DATA_DIR` | Flask instance directory | Database, datasets, and artifacts |
| `QUANTA_PACKAGES_DIR` | `<data dir>/packages` | Shared directory for packages installed via the Packages UI or the AI agent |
| `QUANTA_EXECUTION_TIMEOUT` | `120` | Maximum seconds per cell |
| `QUANTA_MAX_ITERATIONS` | `12` | Agent steps per request |
| `QUANTA_MAX_DOWNLOAD_BYTES` | `5368709120` | Compressed and extracted dataset limit |
| `QUANTA_MAX_OUTPUT_CHARS` | `200000` | Captured text per execution |
| `QUANTA_MODEL_TIMEOUT` | `60` | Timeout in seconds for each model API request |
| `QUANTA_MODEL_MAX_RETRIES` | `3` | Retries for rate limits and transient model failures |

## Packages

Open **Packages** (in the session header) to install third-party Python packages — seaborn, scikit-learn, statsmodels, and so on — or let the AI analysis agent install what a cell needs on its own. Packages install into a shared directory (`QUANTA_PACKAGES_DIR`, default `instance/packages/`) outside Quanta's own `uv`-managed virtual environment via `uv pip install`, so a bad install can never break Quanta itself and `uv sync` never wipes what you installed. The standard library, pandas, NumPy, and Matplotlib are always available and cannot be reinstalled or upgraded through this UI. Newly installed packages become importable in already-open sessions without restarting Quanta or losing kernel state; removing a package restarts that session's kernel (replaying its successful cells) since an already-imported module can't be unloaded from a running process. Installing a package requires [uv](https://docs.astral.sh/uv/) on `PATH`.

## Persistence

Session metadata, events, cells, revisions, executions, and LangGraph checkpoints are stored in `instance/quanta.sqlite3`. Downloaded data and generated figures are stored under `instance/sessions/<session-id>/`. Installed packages live under `instance/packages/` and are shared by every session. On restart, the visible notebook and all previous outputs are restored. Before the next execution, successful active cells are replayed to reconstruct Python state. Code-generation prompts include those same successful cells in execution order, allowing later cells to reuse their variables rather than reload data or repeat work.

## Security

Quanta executes model-generated Python in a local subprocess. Time, file-descriptor, and output limits reduce accidental damage, but **this is not a security sandbox**. Generated code can access the host with the permissions of the Quanta process. Installing a package runs its maintainer's code (including build/install-time scripts) with the same host permissions — the package manager validates requirement syntax and refuses to shadow packages Quanta already ships, but it does not vet package contents. Run this application only locally with trusted users and use container or external sandbox isolation before exposing it to a network.

## Tests

```bash
uv run pytest
```

The test suite uses local fixture data and does not require Kaggle or a model API key.
