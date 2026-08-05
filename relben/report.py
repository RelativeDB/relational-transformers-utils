"""Score matrices and gain-versus-baseline reports over benchmark results."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .submission import atomic_json

__all__ = ["write_report"]

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
