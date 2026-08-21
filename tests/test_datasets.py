from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quanta.datasets import column_brief, compact_manifest, inspect_dataset, normalize_dataset_ref


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("owner/data-set", "owner/data-set"),
        ("https://www.kaggle.com/datasets/owner/data-set", "owner/data-set"),
        ("https://kaggle.com/datasets/owner/data-set/", "owner/data-set"),
    ],
)
def test_normalize_dataset_ref(value: str, expected: str) -> None:
    assert normalize_dataset_ref(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "just-a-name", "https://example.com/datasets/a/b", "a/b/extra"],
)
def test_rejects_invalid_dataset_refs(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_dataset_ref(value)


def test_inspects_mixed_dataset(tmp_path: Path) -> None:
    pd.DataFrame({"city": ["Leeds", "Oslo", "Leeds"], "score": [2.5, None, 4.0]}).to_csv(
        tmp_path / "scores.csv", index=False
    )
    (tmp_path / "notes.txt").write_text("A useful data dictionary", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(json.dumps([{"source": "fixture"}]), encoding="utf-8")

    manifest = inspect_dataset(tmp_path)

    assert manifest["file_count"] == 3
    scores = next(item for item in manifest["files"] if item["path"] == "scores.csv")
    assert scores["inspection"]["kind"] == "table"
    assert scores["inspection"]["rows"] == 3
    assert [column["name"] for column in scores["inspection"]["columns"]] == [
        "city",
        "score",
    ]


def test_column_brief_flags_leading_whitespace(tmp_path: Path) -> None:
    # Reproduces the real-world KeyError: ' Artist Type' — a CSV whose header
    # has a stray leading space before one column name.
    (tmp_path / "artists.csv").write_text("Artist,Sex, Artist Type\nDrake,Male,Solo\n", encoding="utf-8")

    manifest = inspect_dataset(tmp_path)
    brief = column_brief(manifest)

    assert "' Artist Type'" in brief
    assert "LEADING/TRAILING WHITESPACE" in brief
    # A clean name should not be flagged.
    artist_line = next(line for line in brief.splitlines() if line.strip().startswith("'Artist'"))
    assert "WHITESPACE" not in artist_line


def test_column_brief_empty_for_manifest_without_tables(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("just text", encoding="utf-8")
    manifest = inspect_dataset(tmp_path)

    assert column_brief(manifest) == ""


def test_column_brief_degrades_under_a_tight_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from quanta import datasets as datasets_module

    monkeypatch.setattr(datasets_module, "_COLUMN_BRIEF_CHAR_LIMIT", 300)
    manifest = {
        "files": [
            {
                "path": f"table-{index}.csv",
                "inspection": {
                    "kind": "table",
                    "columns": [{"name": f"col_{n}", "dtype": "object"} for n in range(3)],
                },
            }
            for index in range(10)
        ]
    }

    brief = column_brief(manifest)

    assert "table-0.csv" in brief
    assert "table-9.csv" not in brief  # later files dropped once the budget runs out
    assert "omitted for length" in brief
    assert len(brief) < len(column_brief({**manifest, "files": manifest["files"][:1]})) * 10


def test_compact_manifest_is_valid_and_samples_every_directory() -> None:
    files = []
    for directory in ("Car/train", "Car/valid", "Truck/train", "Truck/valid"):
        files.extend(
            {
                "path": f"{directory}/{index:05}.jpg",
                "size": 1024,
                "type": "jpg",
                "inspection": {"kind": "image", "width": 256, "height": 256, "mode": "RGB"},
            }
            for index in range(2500)
        )
    manifest = {
        "root": "/tmp/dataset",
        "file_count": len(files),
        "total_bytes": sum(file["size"] for file in files),
        "files": files,
        "errors": [],
    }

    serialized = compact_manifest(manifest)
    compact = json.loads(serialized)

    assert len(serialized) <= 60_000
    assert compact["omitted_file_count"] > 0
    sampled_paths = [file["path"] for file in compact["files"]]
    assert all(any(path.startswith(directory) for path in sampled_paths) for directory in (
        "Car/train",
        "Car/valid",
        "Truck/train",
        "Truck/valid",
    ))
