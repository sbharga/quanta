from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path
    database_path: Path
    packages_dir: Path
    llm_api_key: str | None
    llm_model: str
    llm_base_url: str
    execution_timeout: int
    max_iterations: int
    max_download_bytes: int
    max_output_chars: int
    model_timeout: float
    model_max_retries: int
    model_max_output_tokens: int = 4096

    @classmethod
    def from_env(cls, instance_path: str | Path) -> Config:
        root = Path(os.environ.get("QUANTA_DATA_DIR", instance_path)).resolve()
        root.mkdir(parents=True, exist_ok=True)
        packages_dir = Path(os.environ.get("QUANTA_PACKAGES_DIR", root / "packages")).resolve()
        packages_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            data_dir=root,
            database_path=root / "quanta.sqlite3",
            packages_dir=packages_dir,
            llm_api_key=os.environ.get("LLM_API_KEY"),
            llm_model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
            llm_base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            execution_timeout=int(os.environ.get("QUANTA_EXECUTION_TIMEOUT", "120")),
            max_iterations=int(os.environ.get("QUANTA_MAX_ITERATIONS", "12")),
            max_download_bytes=int(os.environ.get("QUANTA_MAX_DOWNLOAD_BYTES", str(5 * 1024**3))),
            max_output_chars=int(os.environ.get("QUANTA_MAX_OUTPUT_CHARS", "200000")),
            model_timeout=float(os.environ.get("QUANTA_MODEL_TIMEOUT", "60")),
            model_max_retries=int(os.environ.get("QUANTA_MODEL_MAX_RETRIES", "3")),
            model_max_output_tokens=int(os.environ.get("QUANTA_MODEL_MAX_OUTPUT_TOKENS", "4096")),
        )

    def session_dir(self, session_id: str) -> Path:
        path = self.data_dir / "sessions" / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path
