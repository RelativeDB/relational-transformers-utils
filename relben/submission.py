"""Submission CSVs, run records, and score reports for RelBench evaluation."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["EvalSample", "atomic_json", "write_manifest",
           "write_report", "write_submission"]


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


_HIGHER_IS_BETTER = {"roc_auc"}


def write_report(output: Path, results: list[dict], runners: list[str],
                 *, baseline: str | None = None) -> None:
    """Write ``results.json`` and a markdown score matrix to ``output``.

    Each result row carries ``task``, ``runner``, ``metric``, and ``value``,
    plus optional ``valid``. Gains against ``baseline`` use ``runner -
    baseline`` for metrics in the higher-is-better set and the reverse for
    everything else, so positive always means the runner improved on the
    baseline.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "results.json", {"runners": runners, "results": results})

    lines = ["# Benchmark results", "", "## Scores", "",
             "| Task | Metric | " + " | ".join(runners) + " |",
             "|---|---|" + "---:|" * len(runners)]
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    metrics: dict[str, str] = {}
    for row in results:
        by_task[row["task"]][row["runner"]] = row
        if row.get("metric"):
            metrics[row["task"]] = row["metric"]
    for task in sorted(by_task):
        cells = []
        for runner in runners:
            row = by_task[task].get(runner)
            cells.append("—" if not row or "value" not in row
                         else f"{row['value']:.6f}")
        lines.append(f"| {task} | {metrics.get(task, '—')} | " + " | ".join(cells) + " |")

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in results:
        if "value" in row and row.get("valid", True):
            grouped[(row["runner"], row["metric"])].append(row["value"])
    lines.extend(["", "## Macro means (valid tasks only)", "",
                  "| Runner | Metric | Mean | Tasks |", "|---|---|---:|---:|"])
    for (runner, metric), values in sorted(grouped.items()):
        lines.append(f"| {runner} | {metric} | "
                     f"{sum(values) / len(values):.6f} | {len(values)} |")

    if baseline and baseline in runners:
        others = [r for r in runners if r != baseline]
        lines.extend(["", f"## Gain versus {baseline}", "",
                      "Positive is better in every cell.", "",
                      "| Task | " + " | ".join(others) + " |",
                      "|---|" + "---:|" * len(others)])
        for task in sorted(by_task):
            base = by_task[task].get(baseline, {}).get("value")
            cells = []
            for runner in others:
                value = by_task[task].get(runner, {}).get("value")
                if base is None or value is None:
                    cells.append("—")
                elif metrics.get(task) in _HIGHER_IS_BETTER:
                    cells.append(f"{value - base:+.6f}")
                else:
                    cells.append(f"{base - value:+.6f}")
            lines.append(f"| {task} | " + " | ".join(cells) + " |")

    (output / "results.md").write_text("\n".join(lines) + "\n")
