"""Keyed submission CSVs and run-record writers for RelBench evaluation."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["EvalSample", "atomic_json", "write_submission", "write_manifest"]


@dataclass(frozen=True)
class EvalSample:
    """One test task row and its official submission key."""

    entity_id: Any
    anchor: datetime
    key: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)


def atomic_json(path: Path, value: dict) -> None:
    """Write JSON through a temp file so readers never see a partial record."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def write_submission(path: Path, target_column: str,
                     rows: Iterable[tuple[Mapping[str, Any], float]]) -> int:
    """Write one keyed submission CSV: key columns plus the prediction.

    Every row must share the same key columns; a mismatch raises
    :class:`ValueError` because the official scorer joins on them.
    Returns the number of rows written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = iter(rows)
    try:
        first_key, first_value = next(rows)
    except StopIteration:
        raise ValueError("a submission needs at least one row") from None
    key_columns = tuple(first_key)
    written = 0
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*key_columns, target_column])
        writer.writeheader()
        for key, value in ((first_key, first_value), *rows):
            if tuple(key) != key_columns:
                raise ValueError(f"inconsistent submission key columns in {path.name}")
            writer.writerow({**key, target_column: float(value)})
            written += 1
    return written


def write_manifest(output_dir: Path, *, runner: str, tasks: list[str],
                   extra: Mapping[str, Any] | None = None) -> Path:
    """Record which tasks a runner emitted, beside its submission CSVs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"runner": runner, "tasks": list(tasks), **dict(extra or {})}
    path = output_dir / "runner_manifest.json"
    atomic_json(path, manifest)
    return path
