from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from functools import cache
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

# A PEP 508-ish package requirement: name, optional extras, optional version
# specifier. Deliberately conservative — this string is passed as a single
# argv element to `uv pip install`, never through a shell, but a spec
# starting with "-" could still be interpreted as a flag (e.g. "--index-url
# https://evil"), so that is rejected explicitly before the regex even runs.
PACKAGE_SPEC = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?([<>=!~]=?[A-Za-z0-9._*+!-]+)?$"
)

_LOG_TAIL_LINES = 20


def normalize_spec(value: str) -> str:
    """Validate and normalize a package requirement string.

    Mirrors `datasets.normalize_dataset_ref`: strip whitespace, reject
    anything that isn't a plausible `name[extras]==version` requirement, and
    raise ValueError (not some other exception) so callers can turn it
    directly into a 400 response.
    """
    value = value.strip()
    if not value or value.startswith("-"):
        raise ValueError(f"Invalid package requirement: {value!r}")
    if not PACKAGE_SPEC.fullmatch(value):
        raise ValueError(f"Invalid package requirement: {value!r}")
    name = canonical_name(package_name(value))
    if name in _protected_names():
        raise ValueError(f"{package_name(value)} is already provided by Quanta and cannot be installed")
    return value


def package_name(spec: str) -> str:
    """Extract the bare distribution name from a validated spec, for display/lookup."""
    return re.split(r"[\[<>=!~]", spec, maxsplit=1)[0]


def canonical_name(name: str) -> str:
    """PEP 503 canonicalization, so "Scikit_Learn" and "scikit-learn" are the same package."""
    return re.sub(r"[-_.]+", "-", name).lower()


@cache
def _protected_names() -> set[str]:
    """Distributions the app's own venv already provides.

    Installing over one of these is refused outright rather than routed through
    the constraints file: the worker appends the shared package directory to
    sys.path *after* the app's own site-packages, so a shadowing copy here
    would just be dead weight at best — but a same-named package with a
    slightly different API (a typo-squat, or an unrelated project that reused
    a name) would silently do nothing, which is worse than a clear error.
    """
    return {canonical_name(dist.metadata["Name"]) for dist in distributions() if dist.metadata.get("Name")}


class PackageBackendUnavailable(RuntimeError):
    pass


class PackageManager:
    """Installs third-party packages into a shared directory outside the app's own venv.

    Packages are installed with `uv pip install --target <packages_dir>`, constrained
    to the app's own dependency versions so a notebook package cannot silently shadow
    the pandas/numpy/matplotlib the kernel worker relies on. The worker appends
    `packages_dir` to sys.path (after its own site-packages), so the app's pinned
    versions always win; only genuinely new top-level packages resolve from here.
    """

    def __init__(self, packages_dir: Path, timeout: int = 300):
        self.packages_dir = packages_dir
        self.timeout = timeout
        self._lock = threading.RLock()
        self._installed_cache: list[dict[str, str]] | None = None

    def backend_available(self) -> bool:
        return shutil.which("uv") is not None

    def list_installed(self) -> list[dict[str, str]]:
        with self._lock:
            if self._installed_cache is None:
                self.packages_dir.mkdir(parents=True, exist_ok=True)
                found = []
                for dist in distributions(path=[str(self.packages_dir)]):
                    name = dist.metadata.get("Name")
                    if name:
                        found.append({"name": name, "version": dist.version or ""})
                found.sort(key=lambda item: item["name"].lower())
                self._installed_cache = found
            return [package.copy() for package in self._installed_cache]

    def install(self, specs: list[str], progress: Callable[[str], None]) -> dict[str, Any]:
        specs = [normalize_spec(spec) for spec in specs]
        if not specs:
            raise ValueError("At least one package is required")
        with self._lock:
            self.packages_dir.mkdir(parents=True, exist_ok=True)
            constraints = self._write_constraints()
            command = [
                shutil.which("uv") or "uv",
                "pip",
                "install",
                "--target",
                str(self.packages_dir),
                "--python",
                sys.executable,
                "--constraint",
                str(constraints),
                *specs,
            ]
            try:
                status, log = self._run(command, progress)
            finally:
                self._installed_cache = None
            return {"status": status, "packages": self.list_installed(), "log": log}

    def uninstall(self, name: str, progress: Callable[[str], None]) -> dict[str, Any]:
        name = normalize_spec(name)
        with self._lock:
            command = [
                shutil.which("uv") or "uv",
                "pip",
                "uninstall",
                "--target",
                str(self.packages_dir),
                "--python",
                sys.executable,
                name,
            ]
            try:
                status, log = self._run(command, progress)
            finally:
                self._installed_cache = None
            return {"status": status, "packages": self.list_installed(), "log": log}

    def _write_constraints(self) -> Path:
        # Pin the app's own environment so uv cannot resolve a different
        # numpy/pandas/matplotlib into the shared target directory. Any
        # overlapping distribution that does land there is then guaranteed to
        # be the same version already imported by the worker.
        lines = sorted(
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in distributions()
            if dist.metadata.get("Name") and dist.version
        )
        path = self.packages_dir / ".constraints.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _run(self, command: list[str], progress: Callable[[str], None]) -> tuple[str, str]:
        if not self.backend_available():
            raise PackageBackendUnavailable(
                "The uv package manager was not found on PATH. Install uv to manage packages."
            )
        tail: list[str] = []
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise PackageBackendUnavailable(f"Could not start uv: {exc}") from exc

        def consume_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                tail.append(line)
                del tail[:-_LOG_TAIL_LINES]
                progress(line)

        reader = threading.Thread(target=consume_output, daemon=True)
        reader.start()
        try:
            process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            tail.append(f"Timed out after {self.timeout}s")
        reader.join(timeout=2)
        status = "ok" if process.returncode == 0 else "error"
        return status, "\n".join(tail)
