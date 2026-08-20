from __future__ import annotations

import io
from pathlib import Path

import pytest

from quanta import packages as packages_module
from quanta.packages import PackageBackendUnavailable, PackageManager, canonical_name, normalize_spec


class _FakeProcess:
    def __init__(self, returncode: int, output: str = ""):
        self.returncode = returncode
        self.stdout = io.StringIO(output)

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def kill(self) -> None:
        return None


def test_normalize_spec_accepts_typical_requirements() -> None:
    assert normalize_spec("seaborn") == "seaborn"
    assert normalize_spec(" scikit-learn==1.5.0 ") == "scikit-learn==1.5.0"
    assert normalize_spec("seaborn[stats]") == "seaborn[stats]"


@pytest.mark.parametrize(
    "spec",
    [
        "--index-url=https://evil.example",
        "-r requirements.txt",
        "; rm -rf /",
        "../escape",
        "",
        "   ",
    ],
)
def test_normalize_spec_rejects_flag_injection_and_garbage(spec: str) -> None:
    with pytest.raises(ValueError):
        normalize_spec(spec)


def test_normalize_spec_rejects_distributions_the_app_already_provides() -> None:
    # Flask is a real dependency of the running app environment.
    with pytest.raises(ValueError, match="already provided"):
        normalize_spec("flask")
    with pytest.raises(ValueError, match="already provided"):
        normalize_spec("Flask==3.1.2")


def test_canonical_name_unifies_separators_and_case() -> None:
    assert canonical_name("Scikit_Learn") == canonical_name("scikit-learn") == "scikit-learn"


def test_install_rejects_when_uv_is_not_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(packages_module.shutil, "which", lambda _name: None)
    manager = PackageManager(tmp_path / "packages")
    with pytest.raises(PackageBackendUnavailable):
        manager.install(["seaborn"], lambda _line: None)


def test_install_writes_constraints_and_records_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packages_dir = tmp_path / "packages"
    manager = PackageManager(packages_dir)
    calls: list[list[str]] = []

    def fake_popen(command: list[str], **_kwargs: object) -> _FakeProcess:
        calls.append(command)
        return _FakeProcess(0, "Resolved 1 package\nInstalled 1 package\n")

    monkeypatch.setattr(packages_module.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(packages_module.subprocess, "Popen", fake_popen)

    lines: list[str] = []
    result = manager.install(["seaborn"], lines.append)

    assert result["status"] == "ok"
    assert result["packages"] == []  # nothing was actually installed by the fake process
    assert "Installed 1 package" in result["log"]
    assert lines  # progress callback received streamed lines

    assert len(calls) == 1
    command = calls[0]
    assert "--target" in command and str(packages_dir) in command
    assert "--constraint" in command
    assert command[-1] == "seaborn"
    constraints_path = Path(command[command.index("--constraint") + 1])
    assert constraints_path.exists()
    assert constraints_path.read_text(encoding="utf-8").strip()


def test_install_reports_error_on_nonzero_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = PackageManager(tmp_path / "packages")
    monkeypatch.setattr(packages_module.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(
        packages_module.subprocess, "Popen", lambda *_a, **_k: _FakeProcess(1, "error: no solution found\n")
    )

    result = manager.install(["some-nonexistent-package"], lambda _line: None)

    assert result["status"] == "error"
    assert "no solution found" in result["log"]


def test_uninstall_invokes_uv_pip_uninstall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packages_dir = tmp_path / "packages"
    manager = PackageManager(packages_dir)
    calls: list[list[str]] = []

    def fake_popen(command: list[str], **_kwargs: object) -> _FakeProcess:
        calls.append(command)
        return _FakeProcess(0, "Uninstalled 1 package\n")

    monkeypatch.setattr(packages_module.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(packages_module.subprocess, "Popen", fake_popen)

    result = manager.uninstall("seaborn", lambda _line: None)

    assert result["status"] == "ok"
    assert len(calls) == 1
    assert "uninstall" in calls[0]
    assert calls[0][-1] == "seaborn"


def test_list_installed_is_empty_for_a_fresh_directory(tmp_path: Path) -> None:
    manager = PackageManager(tmp_path / "packages")
    assert manager.list_installed() == []


def test_list_installed_is_cached_until_a_package_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = PackageManager(tmp_path / "packages")
    scans = 0

    class FakeDistribution:
        def __init__(self) -> None:
            self.metadata = {"Name": "torch"}
            self.version = "2.8"

    def fake_distributions(*, path=None):
        nonlocal scans
        if path is None:
            return []
        scans += 1
        return [FakeDistribution()]

    monkeypatch.setattr(packages_module, "distributions", fake_distributions)
    monkeypatch.setattr(manager, "_write_constraints", lambda: tmp_path / "constraints.txt")
    monkeypatch.setattr(manager, "_run", lambda _command, _progress: ("ok", "unchanged"))

    assert manager.list_installed() == [{"name": "torch", "version": "2.8"}]
    assert manager.list_installed() == [{"name": "torch", "version": "2.8"}]
    assert scans == 1

    manager.install(["new-package"], lambda _line: None)

    assert scans == 2
