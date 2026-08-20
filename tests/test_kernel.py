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


def test_worker_imports_from_packages_dir_after_app_site_packages(tmp_path: Path) -> None:
    data = tmp_path / "data"
    artifacts = tmp_path / "artifacts"
    packages_dir = tmp_path / "packages"
    data.mkdir()
    packages_dir.mkdir()
    (packages_dir / "quanta_fake_pkg.py").write_text("VALUE = 99\n", encoding="utf-8")

    manager = KernelManager(timeout=15, packages_dir=packages_dir)
    try:
        status, outputs = manager.execute(
            "session",
            "import sys\nimport quanta_fake_pkg\n(quanta_fake_pkg.VALUE, sys.path.index(str(sys.path[-1])) >= 0)",
            data,
            artifacts,
            lambda _: None,
        )
    finally:
        manager.close()

    assert status == "ok"
    assert any(output.get("type") == "result" and "99" in output.get("text", "") for output in outputs)


def test_worker_picks_up_a_package_installed_after_it_started(tmp_path: Path) -> None:
    data = tmp_path / "data"
    artifacts = tmp_path / "artifacts"
    packages_dir = tmp_path / "packages"
    data.mkdir()
    # Deliberately do NOT create packages_dir yet: the worker starts before
    # any package has ever been installed, mirroring a fresh session.
    manager = KernelManager(timeout=15, packages_dir=packages_dir)
    try:
        first_status, _ = manager.execute("session", "x = 1", data, artifacts, lambda _: None)
        assert first_status == "ok"

        # Simulate an install landing after the worker was already spawned.
        packages_dir.mkdir()
        (packages_dir / "quanta_fake_pkg.py").write_text("VALUE = 7\n", encoding="utf-8")
        manager.mark_all_stale()

        status, outputs = manager.execute(
            "session", "import quanta_fake_pkg\nquanta_fake_pkg.VALUE", data, artifacts, lambda _: None
        )
    finally:
        manager.close()

    assert status == "ok"
    assert any(output.get("type") == "result" and "7" in output.get("text", "") for output in outputs)
