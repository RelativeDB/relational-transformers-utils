"""Normalization utilities matching the RT-J reference preprocessor.

The reference (``rustler/src/pre.rs``) normalizes each numeric column by its
whole-column mean and sample standard deviation, shares one global mean/std
across every datetime cell, replaces a zero standard deviation with 1.0, and
persists every model-valued channel as bfloat16. The functions here reproduce
those conventions bit for bit so scores do not drift between pipelines.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
from relational_transformers.constants import SEM_BOOLEAN, SEM_DATETIME, SEM_NUMBER

from .rows import Row, TemporalBound
from .schema import Schema, ValueType

__all__ = [
    "NormalizationError",
    "NormalizationMode",
    "ColumnStats",
    "bf16_as_f32",
    "days_since_epoch",
    "mean_std",
    "normalize_sequence",
]


class NormalizationError(RuntimeError):
    """Raised when reference normalization lacks required statistics."""


class NormalizationMode(str, Enum):
    """How scalar cells are normalized before entering the model.

    ``ZERO_SHOT`` derives statistics inside each context, so no dataset scan
    is needed and a context's values never depend on which other contexts
    share the batch. ``REFERENCE`` uses persisted :class:`ColumnStats`, which
    matches preprocessing-time training statistics.
    """

    ZERO_SHOT = "zero_shot"
    REFERENCE = "reference"

    @classmethod
    def coerce(cls, value: NormalizationMode | str) -> NormalizationMode:
        if isinstance(value, NormalizationMode):
            return value
        return cls(str(value))


def bf16_as_f32(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even bfloat16, widened back to float32.

    The reference persists every model-valued channel as bfloat16; this
    reproduces that storage boundary deterministically without a runtime
    dtype dependency. Apply the same rounding to text channels you
    materialize.
    """
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def days_since_epoch(t: datetime) -> float:
    """Datetime as float days; naive datetimes are treated as UTC."""
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.timestamp() / 86400.0


def mean_std(values: Sequence[float], *, ddof: int = 0) -> tuple[float, float]:
    """Finite mean/std with the reference's safe zero-variance convention."""
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    sd = float(a.std(ddof=ddof)) if a.size > ddof else 0.0
    return float(a.mean()), sd if math.isfinite(sd) and sd != 0.0 else 1.0


def _task_key(task: Any) -> str:
    key = getattr(task, "id", None)
    return str(key) if key is not None else str(task)


class ColumnStats:
    """Per-column ``(mean, std)`` for numeric cells, plus one global normalizer
    for datetimes. Fitted from the data, exactly as the reference does it.

    Numeric columns use the sample standard deviation (``ddof=1``, matching
    polars' ``std(1)`` in the reference); the single global datetime
    normalizer uses population standard deviation (``ddof=0``, matching the
    reference's Welford accumulator). Zero standard deviation is replaced by
    1.0 everywhere.

    Fit under the training temporal bound: statistics drawn from rows after
    the anchor leak the future into every scaled value.
    """

    __slots__ = ("stats", "task_stats", "dt", "bound")

    def __init__(self, stats: dict[tuple[str, str], tuple[float, float]],
                 dt: tuple[float, float] = (0.0, 1.0),
                 bound: str = "unbounded",
                 task_stats: dict[str, tuple[float, float]] | None = None):
        self.stats = dict(stats)
        self.task_stats = dict(task_stats or {})
        self.dt = dt
        self.bound = bound

    @classmethod
    def fit(cls, schema: Schema, tables: Mapping[str, Iterable[Row]],
            bound: TemporalBound | None = None) -> ColumnStats:
        """Fit statistics from caller-provided rows per table."""
        bound = TemporalBound.unbounded() if bound is None else bound
        out: dict[tuple[str, str], tuple[float, float]] = {}
        dt_vals: list[float] = []
        for table in schema.tables:
            wanted = {c.name: c.type for c in table.columns
                      if c.type in (ValueType.NUMBER, ValueType.BOOLEAN,
                                    ValueType.DATETIME)}
            wanted.update({link.fk_column: link.feature_type
                           for link in schema.links_from(table.name)
                           if link.feature_type in (ValueType.NUMBER,
                                                    ValueType.BOOLEAN,
                                                    ValueType.DATETIME)})
            if not wanted:
                continue
            acc: dict[str, list[float]] = {c: [] for c in wanted}
            for r in tables.get(table.name, ()):
                if not bound.admits_row(r):
                    continue
                for c, vt in wanted.items():
                    v = (r.parents.get(c) if c in r.parents else r.cells.get(c))
                    if v is None:
                        continue
                    if vt is ValueType.DATETIME:
                        dt_vals.append(days_since_epoch(v))
                        continue
                    if isinstance(v, bool):
                        v = 1.0 if v else 0.0
                    if isinstance(v, (int, float)) and not math.isnan(float(v)):
                        acc[c].append(float(v))
            for c, vals in acc.items():
                if not vals:
                    continue
                a = np.asarray(vals, float)
                # ddof=1 to match polars' std(1) in the reference
                sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
                out[(table.name, c)] = (float(a.mean()),
                                        sd if sd != 0.0 else 1.0)
        if len(dt_vals) > 1:
            a = np.asarray(dt_vals, float)
            # the reference's global datetime Welford accumulator divides M2
            # by N, unlike numeric columns which use sample std (N-1)
            sd = float(a.std(ddof=0))
            dt = (float(a.mean()), sd if sd != 0.0 else 1.0)
        else:
            dt = (0.0, 1.0)
        return cls(out, dt=dt, bound=repr(bound))

    def has(self, table: str, column: str) -> bool:
        return (table, column) in self.stats

    def transform(self, table: str, column: str, x: float) -> float:
        mu, sd = self.stats[(table, column)]
        return (x - mu) / sd

    def transform_datetime(self, x: float) -> float:
        mu, sd = self.dt
        return (x - mu) / sd

    def with_task_values(self, task: Any, values: Sequence[float]) -> ColumnStats:
        """Return a copy carrying preprocessing-time stats for one task.

        The task target is a real column in the reference pipeline, so its
        transform must be persisted just like every physical numeric column.
        """
        vals = np.asarray(list(values), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            raise ValueError("task statistics need at least one finite value")
        sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        task_stats = dict(self.task_stats)
        task_stats[_task_key(task)] = (float(vals.mean()), sd if sd != 0.0 else 1.0)
        return ColumnStats(self.stats, dt=self.dt, bound=self.bound,
                           task_stats=task_stats)

    def with_column_values(self, table: str, column: str,
                           values: Sequence[float]) -> ColumnStats:
        """Return a copy with reference-style numeric statistics for one column."""
        vals = np.asarray(list(values), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            raise ValueError("column statistics need at least one finite value")
        sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        stats = dict(self.stats)
        stats[(table, column)] = (float(vals.mean()), sd if sd != 0.0 else 1.0)
        return ColumnStats(stats, dt=self.dt, bound=self.bound,
                           task_stats=self.task_stats)

    def with_datetime_values(self, values: Sequence[float]) -> ColumnStats:
        """Return a copy with the reference's global datetime normalizer.

        Values use the same day units as :func:`days_since_epoch`.
        """
        vals = np.asarray(list(values), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            raise ValueError("datetime statistics need at least one finite value")
        sd = float(vals.std(ddof=0)) if vals.size > 1 else 0.0
        return ColumnStats(self.stats,
                           dt=(float(vals.mean()), sd if sd != 0.0 else 1.0),
                           bound=self.bound, task_stats=self.task_stats)

    def task(self, task: Any) -> tuple[float, float]:
        key = _task_key(task)
        try:
            return self.task_stats[key]
        except KeyError as e:
            raise NormalizationError(
                f"reference normalization has no target statistics for task "
                f"{key!r}; fit them with ColumnStats.with_task_values()") from e

    def to_dict(self) -> dict:
        return {"bound": self.bound, "datetime": list(self.dt),
                "tasks": {k: list(v) for k, v in self.task_stats.items()},
                "stats": {f"{t}.{c}": list(v) for (t, c), v in self.stats.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> ColumnStats:
        stats = {}
        for k, v in (d.get("stats") or {}).items():
            t, _, c = k.partition(".")
            stats[(t, c)] = (float(v[0]), float(v[1]))
        dt = tuple(d.get("datetime") or (0.0, 1.0))
        tasks = {str(k): (float(v[0]), float(v[1]))
                 for k, v in (d.get("tasks") or {}).items()}
        return cls(stats, dt=(float(dt[0]), float(dt[1])),
                   bound=d.get("bound", "unbounded"), task_stats=tasks)

    def __len__(self) -> int:
        return len(self.stats)

    def __repr__(self) -> str:
        return (f"<ColumnStats {len(self.stats)} columns "
                f"{len(self.task_stats)} tasks "
                f"datetime={self.dt[0]:.1f}/{self.dt[1]:.1f} bound={self.bound}>")


def normalize_sequence(
    columns: Sequence[tuple[str, str]],
    sem_types: Sequence[int],
    values: Sequence[Any],
    is_target: Sequence[bool],
    *,
    mode: NormalizationMode | str = NormalizationMode.ZERO_SHOT,
    column_stats: ColumnStats | None = None,
    label_stats: tuple[float, float] | None = None,
    target_key: tuple[str, str] | None = None,
) -> list[float]:
    """Normalize one context's scalar cells into model-ready floats.

    Cells are parallel sequences: a ``(table, column)`` key, a semantic type
    from :mod:`relational_transformers.constants`, the raw value (numbers,
    bools, or datetimes), and the target mask. Target cells and missing values
    become ``0.0``. Text cells become ``0.0`` in the scalar channel; their
    content travels through embeddings.

    Zero-shot mode derives each column's statistics from the non-target cells
    in this sequence, so a value never depends on other contexts in the batch.
    Reference mode reads persisted ``column_stats`` and raises
    :class:`NormalizationError` for a column without statistics. Cells whose
    key equals ``target_key`` use ``label_stats`` when provided.
    """
    mode = NormalizationMode.coerce(mode)
    scalar_sems = (SEM_NUMBER, SEM_BOOLEAN)
    num_vals: dict[tuple[str, str], list[float]] = {}
    dt_vals: list[float] = []
    for key, sem, value, target in zip(columns, sem_types, values, is_target, strict=True):
        if target or value is None:
            continue
        if sem == SEM_DATETIME:
            dt_vals.append(value if isinstance(value, float)
                           else days_since_epoch(value))
        elif sem in scalar_sems:
            x = (1.0 if value else 0.0) if isinstance(value, bool) else float(value)
            num_vals.setdefault(key, []).append(x)

    stats: dict[tuple[str, str], tuple[float, float]] = {}
    for key, vals in num_vals.items():
        if target_key is not None and key == target_key and label_stats is not None:
            stats[key] = label_stats
        elif mode is NormalizationMode.REFERENCE:
            if column_stats is None or not column_stats.has(key[0], key[1]):
                raise NormalizationError(
                    f"reference normalization has no statistics for "
                    f"{key[0]}.{key[1]}")
            stats[key] = column_stats.stats[key]
        else:
            stats[key] = mean_std(vals)

    dt_stats = (column_stats.dt
                if mode is NormalizationMode.REFERENCE and column_stats is not None
                else mean_std(dt_vals))

    out: list[float] = []
    for key, sem, value, target in zip(columns, sem_types, values, is_target, strict=True):
        if target or value is None:
            out.append(0.0)
            continue
        if sem == SEM_DATETIME:
            mu, sd = dt_stats
            days = value if isinstance(value, float) else days_since_epoch(value)
            out.append((days - mu) / sd)
        elif sem in scalar_sems:
            x = (1.0 if value else 0.0) if isinstance(value, bool) else float(value)
            mu, sd = stats[key]
            out.append((x - mu) / sd)
        else:
            out.append(0.0)
    return out
