"""Schema declaration and row values: the data contract for context tools.

Only *shape* and *values* live here — no URLs, no credentials, no
connectors. ``Row`` carries one row's typed cells; ``TemporalBound`` is the
temporal-leakage guard every lookup honors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

__all__ = ["ValueType", "ColumnDef", "TableDef", "LinkDef", "Schema",
           "SchemaError", "Row", "TemporalBound"]


class ValueType(Enum):
    """Semantic value types — exactly RT's sem types (F10–F13)."""

    NUMBER = "number"
    TEXT = "text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"


class SchemaError(ValueError):
    """Raised when a schema is internally inconsistent."""


@dataclass(frozen=True)
class ColumnDef:
    """A typed feature column.

    FK columns are graph edges unless their link opts into a feature token.
    Primary keys are always identity-only; see :class:`TableDef`.
    """

    name: str
    type: ValueType

    @staticmethod
    def of(name: str, type: ValueType) -> ColumnDef:
        return ColumnDef(name, type)


@dataclass(frozen=True)
class LinkDef:
    """A foreign-key edge with an optional, non-targetable feature token.

    ``feature_type=None`` is reference behavior: the FK is graph structure
    only.  When set, the raw FK value is additionally emitted as a feature;
    the edge is retained in either case. Primary keys are never features.
    """

    from_table: str
    fk_column: str
    to_table: str
    feature_type: ValueType | None = None

    @staticmethod
    def link(from_table: str, fk_column: str, to_table: str,
             feature_type: ValueType | None = None) -> LinkDef:
        return LinkDef(from_table, fk_column, to_table, feature_type)


@dataclass(frozen=True)
class TableDef:
    """A table: typed feature columns + identity (PK) + optional row time.

    The primary key names rows and resolves links. It is always identity-only,
    matching reference preprocessing. Declaring it as a feature is rejected.

    ``time_column`` drives temporal filtering (F24) and windows.
    """

    name: str
    columns: tuple[ColumnDef, ...] = ()
    primary_key: str | None = None
    time_column: str | None = None

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for c in self.columns:
            if c.name in seen:
                raise SchemaError(
                    f"table {self.name!r}: duplicate column {c.name!r}")
            seen.add(c.name)
        if self.primary_key is not None and self.primary_key in seen:
            raise SchemaError(
                f"table {self.name!r}: primary key {self.primary_key!r} "
                f"cannot also be a feature column")
        if self.time_column is not None and self.time_column not in seen:
            raise SchemaError(
                f"table {self.name!r}: time_column {self.time_column!r} "
                f"is not a declared column")

    @staticmethod
    def new_table(name: str) -> TableDef.Builder:
        return TableDef.Builder(name)

    def column(self, name: str) -> ColumnDef | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    class Builder:
        def __init__(self, name: str) -> None:
            self._name = name
            self._columns: list[ColumnDef] = []
            self._pk: str | None = None
            self._time: str | None = None

        def column(self, name_or_def, type: ValueType | None = None) -> TableDef.Builder:
            if isinstance(name_or_def, ColumnDef):
                self._columns.append(name_or_def)
            else:
                if type is None:
                    raise SchemaError("column(name, type): type is required")
                self._columns.append(ColumnDef(name_or_def, type))
            return self

        def primary_key(self, column: str) -> TableDef.Builder:
            self._pk = column
            return self

        def time_column(self, column: str) -> TableDef.Builder:
            self._time = column
            return self

        def build(self) -> TableDef:
            return TableDef(self._name, tuple(self._columns), self._pk, self._time)


@dataclass(frozen=True)
class Schema:
    """The declared relational graph. Validates on construction."""

    tables: tuple[TableDef, ...] = ()
    links: tuple[LinkDef, ...] = ()
    _by_name: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_name: dict[str, TableDef] = {}
        for t in self.tables:
            if t.name in by_name:
                raise SchemaError(f"duplicate table {t.name!r}")
            by_name[t.name] = t
        for link in self.links:
            if link.from_table not in by_name:
                raise SchemaError(
                    f"link {link}: unknown from_table {link.from_table!r}")
            if link.to_table not in by_name:
                raise SchemaError(
                    f"link {link}: unknown to_table {link.to_table!r}")
            if by_name[link.to_table].primary_key is None:
                raise SchemaError(
                    f"link {link}: to_table {link.to_table!r} has no primary key")
        object.__setattr__(self, "_by_name", by_name)

    @staticmethod
    def new_schema() -> Schema.Builder:
        return Schema.Builder()

    def table(self, name: str) -> TableDef | None:
        return self._by_name.get(name)

    def require_table(self, name: str) -> TableDef:
        t = self._by_name.get(name)
        if t is None:
            raise SchemaError(f"unknown table {name!r}")
        return t

    def links_from(self, table: str) -> list[LinkDef]:
        """F→P links whose *from* side is ``table`` (its parents)."""
        return [link for link in self.links if link.from_table == table]

    def links_to(self, table: str) -> list[LinkDef]:
        """P→F links whose *to* side is ``table`` (its children edges)."""
        return [link for link in self.links if link.to_table == table]

    def to_json_dict(self) -> dict:
        """JSON-friendly form."""
        return {
            "tables": [
                {
                    "name": t.name,
                    "columns": [{"name": c.name, "type": c.type.value}
                                for c in t.columns],
                    "primary_key": t.primary_key,
                    "time_column": t.time_column,
                }
                for t in self.tables
            ],
            "links": [
                {"from_table": link.from_table, "fk_column": link.fk_column,
                 "to_table": link.to_table,
                 "feature_type": (None if link.feature_type is None
                                  else link.feature_type.value)}
                for link in self.links
            ],
        }

    class Builder:
        def __init__(self) -> None:
            self._tables: list[TableDef] = []
            self._links: list[LinkDef] = []

        def table(self, table: TableDef) -> Schema.Builder:
            self._tables.append(table)
            return self

        def link(self, link_or_from, fk_column: str | None = None,
                 to_table: str | None = None,
                 feature_type: ValueType | None = None) -> Schema.Builder:
            if isinstance(link_or_from, LinkDef):
                self._links.append(link_or_from)
            else:
                self._links.append(LinkDef(link_or_from, fk_column, to_table,
                                           feature_type))
            return self

        def build(self) -> Schema:
            return Schema(tuple(self._tables), tuple(self._links))


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
