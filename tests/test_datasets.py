from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quanta.datasets import column_brief, inspect_dataset, normalize_dataset_ref


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
