"""Columnar context population: array-backed graph store + traversal.

Tables live as numpy columns; adjacency is CSR int arrays; ``Row`` objects
materialize lazily, only for rows a context actually emits. Tens of millions
of rows fit in a few hundred MB. ``ColumnarTraversal`` implements the
shared-context scoring protocol over the store's own node numbering; derived
prediction targets enter through the same :class:`~.traversal.TaskAdapter`
seam the row traversals use.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .graph import ContextGraph
from .schema import Row, Schema
from .traversal import TaskAdapter, TraversalResult

__all__ = ["ColumnarStore", "ColumnarTraversal"]


# datetime64 ticks per second, read off the dtype (pandas >= 3 defaults to
# microseconds; assuming nanoseconds put timestamps 1000x too early).
_TICKS_PER_SEC = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9,
                  "m": 1.0 / 60, "h": 1.0 / 3600, "D": 1.0 / 86400}


def _column(frame: Any, name: str) -> np.ndarray:
    """One column as a numpy array, from a DataFrame or a dict of arrays."""
    col = frame[name]
    to_numpy = getattr(col, "to_numpy", None)
    return np.asarray(to_numpy() if to_numpy is not None else col)


def _columns_of(frame: Any) -> list:
    cols = getattr(frame, "columns", None)
    return list(cols if cols is not None else frame.keys())


def _n_rows(frame: Any) -> int:
    cols = _columns_of(frame)
    return len(_column(frame, cols[0])) if cols else 0


def _epoch_seconds(values) -> np.ndarray:
    """A datetime-ish column -> float64 epoch seconds, NaN where missing."""
    arr = np.asarray(values)
    if arr.dtype == object:
        out = np.empty(len(arr), dtype=np.float64)
        for i, v in enumerate(arr):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out[i] = np.nan
            elif isinstance(v, datetime):
                out[i] = v.timestamp()
            elif isinstance(v, np.datetime64):
                out[i] = (np.nan if np.isnat(v) else
                          np.datetime64(v, "us").astype("int64") / 1e6)
            else:
                out[i] = float(v)
        return out
    if not np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype(np.float64)
    unit = np.datetime_data(arr.dtype)[0]
    per_sec = _TICKS_PER_SEC.get(unit)
    if per_sec is None:                        # exotic resolution
        arr = arr.astype("datetime64[us]")
        per_sec = _TICKS_PER_SEC["us"]
    out = arr.astype("int64").astype(np.float64) / per_sec
    out[np.isnat(arr)] = np.nan
    return out


def _is_missing(v: Any) -> bool:
    """Scalar null test without pandas: None, NaN, or NaT."""
    if v is None:
        return True
    if isinstance(v, (list, tuple, np.ndarray)):
        return False
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, np.datetime64):
        return bool(np.isnat(v))
    if isinstance(v, np.floating):
        return bool(np.isnan(v))
    return False


def _notna_mask(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return ~np.isnat(arr)
    if np.issubdtype(arr.dtype, np.floating):
        return ~np.isnan(arr)
    if arr.dtype == object:
        return np.array([not _is_missing(v) for v in arr], dtype=bool)
    return np.ones(len(arr), dtype=bool)


def _scalar(v: Any) -> Any:
    """numpy scalar -> python scalar; datetime64 -> aware datetime."""
    if isinstance(v, np.datetime64):
        secs = np.datetime64(v, "us").astype("int64") / 1e6
        return datetime.fromtimestamp(secs, timezone.utc)
    if isinstance(v, np.generic):
        return v.item()
    return v


class _Table:
    """One table's columns plus the derived vectors the traversal needs."""

    def __init__(self, name: str, frame: Any, pkey: str | None,
                 time_col: str | None, feature_cols: list,
                 fk_cols: dict):
        self.name = name
        self.pkey = pkey
        self.time_col = time_col
        self.feature_cols = feature_cols
        self.fk_cols = fk_cols            # fk column -> parent table
        self.n = _n_rows(frame)
        present = set(_columns_of(frame))
        wanted = set(feature_cols) | set(fk_cols)
        if pkey is not None:
            wanted.add(pkey)
        if time_col is not None:
            wanted.add(time_col)
        self.cols: dict = {c: _column(frame, c) for c in wanted if c in present}
        self.ids = (self.cols[pkey] if pkey is not None and pkey in self.cols
                    else np.arange(self.n))
        self._id_pos: dict | None = None
        self._sorted: tuple | None = None
        if time_col is not None and time_col in self.cols:
            self.ts = _epoch_seconds(self.cols[time_col])
        else:
            self.ts = np.full(self.n, np.nan)
        counts = np.zeros(self.n, dtype=np.int32)
        # (name, values, non-null mask) per carried column; the masks are
        # reused at row materialization.
        self.cells_of: list = []
        for col in feature_cols:
            if col in self.cols:
                mask = _notna_mask(self.cols[col])
                counts += mask.astype(np.int32)
                self.cells_of.append((col, self.cols[col], mask))
        self.cell_counts = counts
        self.fk_items: list = [(fk, parent) for fk, parent in fk_cols.items()
                               if fk in self.cols]

    @property
    def id_pos(self) -> dict:
        if self._id_pos is None:
            pos: dict = {}
            for i, v in enumerate(self.ids):
                pos.setdefault(_scalar(v), i)
            self._id_pos = pos
        return self._id_pos

    def positions_of(self, values: np.ndarray) -> np.ndarray:
        """Ids -> row positions, -1 where absent. Vectorized."""
        values = np.asarray(values)
        if self._sorted is None:
            order = np.argsort(self.ids, kind="stable")
            self._sorted = (order, np.asarray(self.ids)[order])
        order, sorted_ids = self._sorted
        if len(sorted_ids) == 0:
            return np.full(len(values), -1, dtype=np.int64)
        idx = np.searchsorted(sorted_ids, values)
        idx_clipped = np.clip(idx, 0, len(sorted_ids) - 1)
        hit = sorted_ids[idx_clipped] == values
        return np.where(hit, order[idx_clipped], -1).astype(np.int64)


class ColumnarStore:
    """Array-backed store over a schema's tables (plus optional task tables).

    ``frames`` maps physical table name -> DataFrame. ``task_frames`` maps a
    task table name -> DataFrame whose ``entity_col`` column links it to
    ``entity_table`` (``task_links``), with every remaining column treated
    as a cell. Node ids are assigned per table in sorted-name order,
    physical tables first, then task tables.
    """

    def __init__(self, schema: Schema, frames: dict, *,
                 task_frames: dict | None = None,
                 task_links: dict[str, tuple[str, str, str]] | None = None):
        self.schema = schema
        self.tables: dict[str, _Table] = {}
        for tdef in schema.tables:
            frame = frames[tdef.name]
            fk_cols = {l.fk_column: l.to_table
                       for l in schema.links_from(tdef.name)}
            self.tables[tdef.name] = _Table(
                tdef.name, frame, tdef.primary_key, tdef.time_column,
                [c.name for c in tdef.columns
                 if c.name in set(_columns_of(frame))],
                fk_cols)
        self.task_tables: dict[str, _Table] = {}
        self.task_links = dict(task_links or {})
        for name, frame in (task_frames or {}).items():
            entity_table, entity_col, time_col = self.task_links[name]
            cells = [c for c in _columns_of(frame) if c != entity_col]
            self.task_tables[name] = _Table(
                name, frame, None, time_col, cells, {entity_col: entity_table})

        # ---- global node numbering --------------------------------------
        self.order = sorted(self.tables) + sorted(self.task_tables)
        self.base: dict[str, int] = {}
        at = 0
        for name in self.order:
            self.base[name] = at
            at += self._table(name).n
        self.n_nodes = at

        # concatenated per-node vectors
        self.node_ts = np.concatenate(
            [self._table(n).ts for n in self.order]) if self.order else \
            np.empty(0)
        self.node_cells = np.concatenate(
            [self._table(n).cell_counts for n in self.order]).astype(np.int32)
        # A timestamp emits one extra cell only where the time column is not
        # already a declared feature column.
        extra = np.concatenate([
            (np.zeros(self._table(n).n, dtype=np.int32)
             if self._table(n).time_col in self._table(n).feature_cols
             else (~np.isnan(self._table(n).ts)).astype(np.int32))
            for n in self.order])
        self.node_cells = self.node_cells + extra
        self.node_table_idx = np.concatenate(
            [np.full(self._table(n).n, i, dtype=np.int32)
             for i, n in enumerate(self.order)])

        self._build_edges()
        self._row_cache: dict[int, Row] = {}
        self._native_graph = None

    # ------------------------------------------------------------------
    def _table(self, name: str) -> _Table:
        return self.tables.get(name) or self.task_tables[name]

    def _resolve_parents(self, table: _Table, fk: str,
                         parent: _Table) -> np.ndarray:
        """fk column values -> parent node positions (-1 where unmatched)."""
        if fk not in table.cols:
            return np.full(table.n, -1, dtype=np.int64)
        values = table.cols[fk]
        pos = parent.positions_of(values)
        missing = ~_notna_mask(values)
        if missing.any():
            pos = np.where(missing, -1, pos)
        return pos

    def _build_edges(self) -> None:
        """Resolve foreign keys to (parent, child) node id pairs."""
        heads: list = []      # parent node ids
        tails: list = []      # child node ids
        for name in self.order:
            table = self._table(name)
            child_base = self.base[name]
            for fk, parent_name in table.fk_cols.items():
                parent = self._table(parent_name)
                pos = self._resolve_parents(table, fk, parent)
                ok = pos >= 0
                tails.append(np.nonzero(ok)[0] + child_base)
                heads.append(pos[ok] + self.base[parent_name])
        if heads:
            self.edge_parent = np.concatenate(heads).astype(np.int64)
            self.edge_child = np.concatenate(tails).astype(np.int64)
        else:
            self.edge_parent = np.empty(0, dtype=np.int64)
            self.edge_child = np.empty(0, dtype=np.int64)
        self._adjacency: dict = {}

    def _csr(self, children: bool):
        """The ordered adjacency, read back from the native graph and cached."""
        got = self._adjacency.get(children)
        if got is None:
            got = self.native_graph().adjacency(children=children)
            self._adjacency[children] = got
        return got

    @property
    def p2f_offsets(self) -> np.ndarray:
        return self._csr(True)[0]

    @property
    def p2f_child(self) -> np.ndarray:
        return self._csr(True)[1]

    @property
    def f2p_offsets(self) -> np.ndarray:
        return self._csr(False)[0]

    @property
    def f2p_parent(self) -> np.ndarray:
        return self._csr(False)[1]

    def native_graph(self):
        """The native graph, built once. Node ids are this store's numbering,
        and they seed the walk and BFS streams, so the numbering here decides
        the sampling."""
        if self._native_graph is None:
            is_task = np.zeros(self.n_nodes, dtype=np.uint8)
            for name in self.task_tables:
                base = self.base[name]
                is_task[base:base + self._table(name).n] = 1
            self._native_graph = ContextGraph(
                self.node_ts, self.node_cells, self.node_table_idx, is_task,
                self.edge_parent, self.edge_child)
        return self._native_graph

    # ------------------------------------------------------------------
    def children(self, node: int) -> np.ndarray:
        return self.p2f_child[self.p2f_offsets[node]:
                              self.p2f_offsets[node + 1]]

    def parents_of(self, node: int) -> np.ndarray:
        return self.f2p_parent[self.f2p_offsets[node]:
                               self.f2p_offsets[node + 1]]

    def node_of(self, table: str, entity_id: Any) -> int | None:
        t = self._table(table)
        pos = t.id_pos.get(_scalar(entity_id))
        return None if pos is None else self.base[table] + int(pos)

    def table_of(self, node: int) -> str:
        return self.order[self.node_table_idx[node]]

    def row(self, node: int) -> Row:
        """Materialize one node as a Row (cached)."""
        cached = self._row_cache.get(node)
        if cached is not None:
            return cached
        name = self.table_of(node)
        table = self._table(name)
        pos = node - self.base[name]
        cells = {}
        for col, values, present in table.cells_of:
            if present[pos]:
                cells[col] = _scalar(values[pos])
        ts = None
        stamp = self.node_ts[node]
        if not math.isnan(stamp):
            ts = datetime.fromtimestamp(float(stamp), timezone.utc)
        parents = {}
        entity = name not in self.tables
        for fk, parent_name in table.fk_items:
            v = table.cols[fk][pos]
            if not _is_missing(v):
                parents["__entity__" if entity else fk] = _scalar(v)
        rid = _scalar(table.ids[pos])
        row = Row(name, rid, cells, ts, parents)
        self._row_cache[node] = row
        return row


class ColumnarTraversal:
    """Shared-context traversal over a :class:`ColumnarStore`.

    Implements the walk-tiered context assembly and the ``cohort_targets``
    contract the engine's shared-context path expects, entirely over the
    store's arrays; only emitted context rows materialize as ``Row``
    objects. Per-entity (non-shared) execution is intentionally
    unsupported — a bare unmasked label row in context would leak the
    answer — so pair this traversal with ``shared_context=True``.
    """

    def __init__(self, store: ColumnarStore, *,
                 task_adapter: TaskAdapter | None = None,
                 fallback: Callable[[str], None] | None = None):
        self.store = store
        self.task_adapter = task_adapter
        self.fallback = fallback or (
            lambda message: warnings.warn(message, stacklevel=3))
        self._focal_lookup: dict[str, dict] = {}
        self._label_ok: dict[str, np.ndarray] = {}
        self._eligible_key: tuple | None = None
        self._eligible: np.ndarray | None = None

    def _labels_ok(self, task_spec: Any) -> np.ndarray:
        got = self._label_ok.get(task_spec.table_name)
        if got is not None:
            return got
        s = self.store
        ok = np.zeros(s.n_nodes, dtype=bool)
        t = s._table(task_spec.table_name)
        base = s.base[task_spec.table_name]
        if task_spec.target_column in t.cols:
            ok[base:base + t.n] = _notna_mask(t.cols[task_spec.target_column])
        self._label_ok[task_spec.table_name] = ok
        return ok

    def _focal(self, task_spec: Any) -> dict:
        got = self._focal_lookup.get(task_spec.table_name)
        if got is not None:
            return got
        s = self.store
        t = s._table(task_spec.table_name)
        entity_col = next(iter(t.fk_cols))
        base = s.base[task_spec.table_name]
        lookup = {}
        ents = t.cols[entity_col]
        for pos in range(t.n):
            lookup[(ents[pos].item() if isinstance(ents[pos], np.generic)
                    else ents[pos], float(t.ts[pos]))] = base + pos
        self._focal_lookup[task_spec.table_name] = lookup
        return lookup
    def cohort_targets(self, entity_table, entity_ids, anchor, task_spec,
                       *, history: int):
        s = self.store
        lookup = self._focal(task_spec)
        anchor_f = (anchor.timestamp() if anchor is not None else math.nan)
        targets, inject, extra = [], [], {}
        labels_ok = self._labels_ok(task_spec)
        task_idx = s.order.index(task_spec.table_name)
        for eid in entity_ids:
            node = lookup.get((eid, anchor_f))
            if node is None:
                return None
            row = s.row(node)
            targets.append((eid, row.key))
            inject.append(row)
            extra[row.key] = int(node)
            enode = s.node_of(entity_table, eid)
            if enode is not None:
                erow = s.row(enode)
                inject.append(erow)
                extra[erow.key] = int(enode)
                kids = s.children(enode)
                keep = ((s.node_table_idx[kids] == task_idx)
                        & labels_ok[kids] & (s.node_ts[kids] < anchor_f))
                kids = kids[keep]
                order = np.argsort(-s.node_ts[kids], kind="stable")
                hist_nodes = kids[order].tolist()
                for c in hist_nodes[:history]:
                    hrow = s.row(c)
                    inject.append(hrow)
                    extra[hrow.key] = c
        return targets, inject, extra


    def traverse(self, schema, graph, entity_table, entity_id, bound, policy,
                 *, query=None) -> TraversalResult:
        """Assemble one entity's context; only emitted nodes become Rows."""
        if query is None:
            raise RuntimeError("ColumnarTraversal requires a query")
        if self.task_adapter is None:
            raise RuntimeError("ColumnarTraversal requires a task_adapter")
        s = self.store
        task_spec = self.task_adapter.spec(query, schema)
        anchor = bound.as_of
        anchor_f = anchor.timestamp() if anchor is not None else math.nan
        target = self._focal(task_spec).get((entity_id, anchor_f))
        if target is None:
            # An empty context scores the entity as having no history — a
            # wiring or anchor mismatch, so it is reported, never silent.
            self.fallback(f"no task row for entity {entity_id!r} at anchor "
                          f"{anchor_f}; returning an EMPTY context, which "
                          f"scores the entity as having no history")
            return TraversalResult()
        cutoff_f = float(s.node_ts[target])
        if math.isnan(cutoff_f):
            # An undated focal row falls back to the DECLARED anchor, not to
            # +inf — +inf admitted every row in the store, including the
            # future the anchor exists to hide.
            cutoff_f = anchor_f if not math.isnan(anchor_f) else math.inf

        # The walk mask is shared per (task table, cutoff); only the
        # target's own bit differs, punched out around the call.
        key = (task_spec.table_name, cutoff_f)
        if self._eligible_key != key:
            self._eligible = (self._labels_ok(task_spec)
                              & (np.isnan(s.node_ts) | (s.node_ts <= cutoff_f))
                              ).astype(np.uint8)
            self._eligible_key = key
        eligible = self._eligible
        was = eligible[target]
        eligible[target] = 0

        # The fallback stage pads a short context from the task table, so the
        # native side needs that table's contiguous node range.
        t = s._table(task_spec.table_name)
        try:
            nodes, focal_flags = s.native_graph().assemble(
                int(target), cutoff_f, eligible, policy,
                fallback_base=s.base[task_spec.table_name], fallback_n=t.n,
                # Emitted ROWS are not bounded by the CELL budget (an all-null
                # row costs zero cells), so the buffer gets real slack and the
                # graph errors rather than truncating if it binds.
                max_nodes=int(min(s.n_nodes, 1 << 21)))
        finally:
            eligible[target] = was

        rows: list[Row] = []
        node_ids: list[tuple[tuple[str, Any], int]] = []
        focal_keys = set()
        for node, is_focal in zip(nodes.tolist(), focal_flags.tolist()):
            row = s.row(node)
            if node == target:
                cells_masked = dict(row.cells)
                cells_masked.pop(task_spec.target_column, None)
                row = Row(row.table, row.id, cells_masked, row.timestamp,
                          row.parents)
            rows.append(row)
            node_ids.append((row.key, node))
            if is_focal:
                focal_keys.add(row.key)
        focal_keys = frozenset(focal_keys)
        full = len(rows) > 0 and int(
            s.node_cells[nodes].sum()) >= policy.max_context_cells
        return TraversalResult(tuple(rows), focal_keys, 0, full,
                               tuple(node_ids))
