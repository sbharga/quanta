from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

DATASET_SLUG = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
TABULAR_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".parquet", ".xlsx", ".xls"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
_MANIFEST_CHAR_LIMIT = 60_000


def normalize_dataset_ref(value: str) -> str:
    value = value.strip().rstrip("/")
    if value.startswith(("https://", "http://")):
        parsed = urlparse(value)
        if parsed.netloc.lower() not in {"kaggle.com", "www.kaggle.com"}:
            raise ValueError("Only kaggle.com dataset URLs are supported")
        parts = [part for part in parsed.path.split("/") if part]
        try:
            marker = parts.index("datasets")
            value = f"{parts[marker + 1]}/{parts[marker + 2]}"
        except (ValueError, IndexError) as exc:
            raise ValueError("Expected a Kaggle URL like /datasets/owner/dataset") from exc
    if not DATASET_SLUG.fullmatch(value):
        raise ValueError("Use a Kaggle dataset URL or an owner/dataset slug")
    return value


def download_dataset(
    slug: str,
    destination: Path,
    progress: Callable[[str], None] | None = None,
    max_bytes: int = 5 * 1024**3,
    cancelled: Callable[[], bool] | None = None,
    kaggle_api_token: str | None = None,
) -> None:
    marker = destination / ".quanta-complete"
    if marker.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f"{destination.name}.partial-{uuid.uuid4().hex}")
    staging.mkdir()
    command = [
        shutil.which("kaggle") or "kaggle",
        "datasets",
        "download",
        slug,
        "--path",
        str(staging),
        "--force",
    ]
    try:
        environment = os.environ.copy()
        if kaggle_api_token:
            environment["KAGGLE_API_TOKEN"] = kaggle_api_token
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )

        def consume_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                if progress and line.strip():
                    progress(line.strip())

        reader = threading.Thread(target=consume_output, daemon=True)
        reader.start()
        while process.poll() is None:
            if cancelled and cancelled():
                process.terminate()
                raise RuntimeError("Dataset download cancelled")
            if _directory_size(staging) > max_bytes:
                process.terminate()
                raise ValueError("Downloaded dataset exceeds QUANTA_MAX_DOWNLOAD_BYTES")
            time.sleep(0.25)
        reader.join(timeout=2)
        if process.returncode:
            raise RuntimeError(
                "Kaggle download failed. Configure a Kaggle token in Settings or use local Kaggle credentials."
            )
        archives = list(staging.glob("*.zip"))
        if not archives:
            raise RuntimeError("Kaggle did not produce a dataset archive")
        compressed = sum(archive.stat().st_size for archive in archives)
        uncompressed = sum(_zip_size(archive) for archive in archives)
        if compressed > max_bytes or uncompressed > max_bytes:
            raise ValueError("Dataset exceeds QUANTA_MAX_DOWNLOAD_BYTES")
        for archive in archives:
            _safe_extract_zip(archive, staging, max_bytes)
            archive.unlink()
        (staging / ".quanta-complete").touch()
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
    except Exception:
        if "process" in locals() and process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _safe_extract_zip(archive: Path, destination: Path, max_bytes: int) -> None:
    root = destination.resolve()
    total = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            total += member.file_size
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if member.is_dir():
                continue
            if total > max_bytes:
                raise ValueError("Uncompressed dataset exceeds QUANTA_MAX_DOWNLOAD_BYTES")
        bundle.extractall(destination)


def inspect_dataset(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and not item.name.startswith(".quanta-")):
        relative = path.relative_to(root).as_posix()
        item: dict[str, Any] = {
            "path": relative,
            "size": path.stat().st_size,
            "type": path.suffix.lower().lstrip(".") or "file",
        }
        try:
            item["inspection"] = inspect_file(path)
        except Exception as exc:  # Keep inspecting the rest of a mixed dataset.
            message = f"{relative}: {type(exc).__name__}: {exc}"
            item["error"] = message
            errors.append(message)
        files.append(item)
    if not files:
        raise ValueError("The downloaded dataset is empty")
    return {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "files": files,
        "errors": errors,
    }


def inspect_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        with Image.open(path) as image:
            return {"kind": "image", "width": image.width, "height": image.height, "mode": image.mode}
    if suffix in {".sqlite", ".sqlite3", ".db"}:
        return _inspect_sqlite(path)
    if suffix in TABULAR_SUFFIXES:
        return _inspect_table(path)
    if suffix in {".txt", ".md", ".sql", ".xml", ".yaml", ".yml"}:
        text = path.read_text(encoding="utf-8", errors="replace")[:4000]
        return {"kind": "text", "preview": text, "truncated": path.stat().st_size > len(text.encode())}
    return {"kind": "binary"}


def _inspect_table(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    rows: int | None
    if suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else _sniff_separator(path)
        frame = pd.read_csv(path, sep=separator, nrows=500)
        rows = _count_csv_records(path, separator)
        sheets = None
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(path, lines=True, nrows=500)
        rows = _count_lines(path)
        sheets = None
    elif suffix == ".json":
        frame, rows = _read_json_sample(path)
        sheets = None
    elif suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        batches = parquet.iter_batches(batch_size=500)
        first_batch = next(batches, None)
        frame = first_batch.to_pandas() if first_batch is not None else pd.DataFrame()
        sheets = None
    else:
        workbook = pd.ExcelFile(path)
        sheets = workbook.sheet_names
        frame = pd.read_excel(path, sheet_name=sheets[0], nrows=500)
        rows = None
    return {
        "kind": "table",
        "rows": rows,
        "sample_rows": len(frame),
        "columns": [
            {
                "name": str(column),
                "dtype": str(frame[column].dtype),
                "nulls_in_sample": int(frame[column].isna().sum()),
                "unique_in_sample": int(frame[column].nunique(dropna=True)),
            }
            for column in frame.columns
        ],
        "preview": _records(frame.head(5)),
        "sheets": sheets,
    }


def _inspect_sqlite(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = []
        names = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for (name,) in names:
            escaped = name.replace('"', '""')
            columns = connection.execute(f'PRAGMA table_info("{escaped}")').fetchall()
            count = connection.execute(f'SELECT COUNT(*) FROM "{escaped}"').fetchone()[0]
            tables.append(
                {
                    "name": name,
                    "rows": count,
                    "columns": [{"name": col[1], "dtype": col[2]} for col in columns],
                }
            )
        return {"kind": "database", "tables": tables}
    finally:
        connection.close()


def compact_manifest(manifest: dict[str, Any]) -> str:
    compact = {**manifest, "root": "DATASET_DIR"}
    serialized = _serialize_manifest(compact)
    if len(serialized) <= _MANIFEST_CHAR_LIMIT:
        return serialized

    files = manifest.get("files", [])
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for file in files:
        path = Path(str(file.get("path", "")))
        key = (path.parent.as_posix(), str(file.get("type", "file")))
        grouped.setdefault(key, []).append(file)

    group_summaries = [
        {
            "directory": directory,
            "type": file_type,
            "file_count": len(items),
            "total_bytes": sum(int(item.get("size", 0)) for item in items),
        }
        for (directory, file_type), items in grouped.items()
    ]
    reduced: dict[str, Any] = {
        "root": "DATASET_DIR",
        "file_count": manifest.get("file_count", len(files)),
        "total_bytes": manifest.get("total_bytes", 0),
        "errors": [],
        "omitted_error_count": len(manifest.get("errors", [])),
        "file_groups": group_summaries[:200],
        "omitted_file_group_count": max(0, len(group_summaries) - 200),
        "files": [],
        "omitted_file_count": len(files),
    }
    while len(_serialize_manifest(reduced)) > _MANIFEST_CHAR_LIMIT and reduced["file_groups"]:
        reduced["file_groups"].pop()
        reduced["omitted_file_group_count"] += 1

    for error in manifest.get("errors", []):
        candidate = {**reduced, "errors": [*reduced["errors"], error]}
        candidate["omitted_error_count"] -= 1
        if len(_serialize_manifest(candidate)) > _MANIFEST_CHAR_LIMIT:
            break
        reduced = candidate

    representative_files: list[dict[str, Any]] = []
    for index in range(max((len(items) for items in grouped.values()), default=0)):
        representative_files.extend(items[index] for items in grouped.values() if index < len(items))

    for file in representative_files:
        detail = file
        candidate = {**reduced, "files": [*reduced["files"], detail]}
        candidate["omitted_file_count"] -= 1
        if len(_serialize_manifest(candidate)) > _MANIFEST_CHAR_LIMIT:
            detail = _manifest_file_brief(file)
            candidate = {**reduced, "files": [*reduced["files"], detail]}
            candidate["omitted_file_count"] -= 1
        if len(_serialize_manifest(candidate)) > _MANIFEST_CHAR_LIMIT:
            break
        reduced = candidate
    return _serialize_manifest(reduced)


def _serialize_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=True)


def _manifest_file_brief(file: dict[str, Any]) -> dict[str, Any]:
    brief: dict[str, Any] = {key: file[key] for key in ("path", "size", "type", "error") if key in file}
    inspection = file.get("inspection") or {}
    kind = inspection.get("kind")
    if kind == "table":
        brief["inspection"] = {
            key: inspection[key]
            for key in ("kind", "rows", "sample_rows", "columns", "sheets")
            if key in inspection
        }
    elif kind == "text":
        brief["inspection"] = {
            key: inspection[key] for key in ("kind", "truncated") if key in inspection
        }
    elif inspection:
        brief["inspection"] = inspection
    return brief


def column_brief(manifest: dict[str, Any]) -> str:
    """Render every table's column names as repr() with whitespace flagged.

    JSON pretty-printing makes a stray leading/trailing space in a column name
    (e.g. ' Artist Type') visually indistinguishable from ordinary indentation,
    which is how the model ends up writing df['Artist Type'] and hitting a
    KeyError. repr() plus an explicit marker makes it impossible to miss.
    """
    lines: list[str] = []
    for file in manifest.get("files", []):
        inspection = file.get("inspection") or {}
        columns = inspection.get("columns")
        if not columns:
            continue
        lines.append(f"{file['path']}:")
        for column in columns:
            name = column.get("name", "")
            dtype = column.get("dtype", "")
            flags = []
            if name != name.strip():
                flags.append("LEADING/TRAILING WHITESPACE")
            if name != str(name):
                flags.append("NON-STRING NAME")
            marker = f"  <-- {', '.join(flags)}" if flags else ""
            lines.append(f"  {name!r:40s} {dtype}{marker}")
    if not lines:
        return ""
    return "Column names (verbatim, index EXACTLY as shown — never guess or strip):\n" + "\n".join(lines)


def _sniff_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _count_csv_records(path: Path, separator: str) -> int:
    if path.stat().st_size > 100 * 1024**2:
        with path.open("rb") as handle:
            sample = handle.read(1024**2)
        sample_lines = max(1, sample.count(b"\n"))
        return max(0, round(path.stat().st_size / len(sample) * sample_lines) - 1)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return max(0, sum(1 for _ in csv.reader(handle, delimiter=separator)) - 1)


def _read_json_sample(path: Path) -> tuple[pd.DataFrame, int | None]:
    if path.stat().st_size <= 10 * 1024**2:
        frame = pd.read_json(path)
        return frame.head(500), len(frame)
    import ijson

    records = []
    with path.open("rb") as handle:
        for record in ijson.items(handle, "item"):
            records.append(record)
            if len(records) >= 500:
                break
    if not records:
        raise ValueError("Large JSON files must contain a top-level array of records")
    return pd.DataFrame.from_records(records), None


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _zip_size(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(member.file_size for member in archive.infolist())


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = frame.astype(object).where(pd.notna(frame), None)
    return json.loads(cleaned.to_json(orient="records", date_format="iso"))
