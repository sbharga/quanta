from __future__ import annotations

from pathlib import Path
from typing import Any

from quanta.kernel import KernelManager


def test_kernel_preserves_state_and_captures_rich_outputs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    artifacts = tmp_path / "artifacts"
    data.mkdir()
    manager = KernelManager(timeout=15)
    streamed: list[dict[str, Any]] = []
    try:
        status, first = manager.execute("session", "x = 40\nprint('loaded')", data, artifacts, streamed.append)
        second_status, second = manager.execute(
            "session",
            "pd.DataFrame({'answer': [x + 2]})",
            data,
            artifacts,
            streamed.append,
        )
    finally:
        manager.close()

    assert status == "ok"
    assert second_status == "ok"
    assert any(output.get("text") == "loaded" for output in first)
    table = next(output for output in second if output["type"] == "table")
    assert table["columns"] == ["answer"]
    assert table["rows"] == [[42]]


def test_kernel_returns_traceback(tmp_path: Path) -> None:
    data = tmp_path / "data"
    artifacts = tmp_path / "artifacts"
    data.mkdir()
    manager = KernelManager(timeout=15)
    try:
        status, outputs = manager.execute("session", "raise ValueError('bad value')", data, artifacts, lambda _: None)
    finally:
        manager.close()

    assert status == "error"
    assert "ValueError: bad value" in outputs[-1]["message"]
