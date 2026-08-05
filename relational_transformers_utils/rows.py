"""Row and temporal-bound value types shared by the context utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

__all__ = ["TemporalBound", "Row"]


def _to_utc(t: datetime) -> datetime:
    if t.tzinfo is None:
        return t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


@dataclass(frozen=True)
class TemporalBound:
    """"Nothing newer than this" — the temporal-leakage guard.

    ``as_of is None`` means unbounded (static tables without time).
    """

    as_of: datetime | None = None

    @staticmethod
    def at_or_before(t: datetime) -> TemporalBound:
        return TemporalBound(_to_utc(t))

    @staticmethod
    def unbounded() -> TemporalBound:
        return TemporalBound(None)

    @property
    def is_unbounded(self) -> bool:
        return self.as_of is None

    def admits(self, timestamp: datetime | None) -> bool:
        """A row with no timestamp is static and always admitted."""
        if self.as_of is None or timestamp is None:
            return True
        return _to_utc(timestamp) <= self.as_of

    def admits_row(self, row: Row) -> bool:
        return self.admits(row.timestamp)


@dataclass(frozen=True)
class Row:
    """One row's typed feature cells.

    FK values are reported via ``parents``. Primary keys are identity only and
    never emit feature tokens. Missing/null values: simply omit the cell —
    nulls emit no token.
    """

    table: str
    id: Any
    cells: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime | None = None
    parents: dict[str, Any] = field(default_factory=dict)  # fk column -> parent id

    def __post_init__(self) -> None:
        if self.timestamp is not None:
            object.__setattr__(self, "timestamp", _to_utc(self.timestamp))
        object.__setattr__(self, "key", (self.table, self.id))

    key: tuple[str, Any] = field(init=False, repr=False, compare=False)

    def to_json_dict(self) -> dict:
        cells = {}
        for k, v in self.cells.items():
            cells[k] = v.isoformat() if isinstance(v, datetime) else v
        return {
            "table": self.table,
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "cells": cells,
            "parents": dict(self.parents),
        }

    @staticmethod
    def from_json_dict(d: dict) -> Row:
        ts = d.get("timestamp")
        return Row(
            table=d["table"],
            id=d["id"],
            cells=dict(d.get("cells") or {}),
            timestamp=datetime.fromisoformat(ts) if ts else None,
            parents=dict(d.get("parents") or {}),
        )
