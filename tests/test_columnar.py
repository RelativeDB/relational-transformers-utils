from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from relational_transformers_utils import ColumnarStore, ColumnarTraversal, ContextPolicy
from relational_transformers_utils.schema import ColumnDef, LinkDef, Schema, TableDef, ValueType


def _dt(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=timezone.utc)


def _dt64(day: int, unit: str = "us") -> np.datetime64:
    return np.datetime64(f"2024-01-{day:02d}", unit)


def _schema() -> Schema:
    return Schema(
        tables=(
            TableDef("customers",
                     columns=(ColumnDef("age", ValueType.NUMBER),),
                     primary_key="id"),
            TableDef("orders",
                     columns=(ColumnDef("amount", ValueType.NUMBER),
                              ColumnDef("placed", ValueType.DATETIME)),
                     primary_key="id", time_column="placed"),
        ),
        links=(LinkDef("orders", "customer_id", "customers"),),
    )


def _store() -> ColumnarStore:
    frames = {
        "customers": {"id": np.asarray([1, 2]),
                      "age": np.asarray([30.0, np.nan])},
        "orders": {"id": np.asarray([10, 11, 12, 13]),
                   "amount": np.asarray([5.0, 6.0, 7.0, np.nan]),
                   "placed": np.asarray([_dt64(1), _dt64(2), _dt64(3), _dt64(4)]),
                   # order 13 dangles: customer 9 does not exist
                   "customer_id": np.asarray([1, 1, 2, 9])},
    }
    task = {"churn_task": {"customer": np.asarray([1, 2, 1]),
                           "at": np.asarray([_dt64(6), _dt64(6), _dt64(5)]),
                           "label": np.asarray([1.0, 0.0, np.nan])}}
    return ColumnarStore(_schema(), frames, task_frames=task,
                         task_links={"churn_task":
                                     ("customers", "customer", "at")})


def test_store_materializes_rows_lazily_and_exactly():
    store = _store()
    node = store.node_of("orders", 12)
    row = store.row(node)
    assert row.table == "orders" and row.id == 12
    assert row.cells["amount"] == 7.0
    assert row.timestamp == _dt(3)
    assert row.parents == {"customer_id": 2}
    # null cells emit nothing; a NaN amount is absent, never NaN
    null_row = store.row(store.node_of("orders", 13))
    assert "amount" not in null_row.cells
    # cell accounting: amount + placed for a dated order with an amount
    assert store.node_cells[node] == 2
    assert store.node_cells[store.node_of("orders", 13)] == 1


def test_store_edges_skip_dangling_foreign_keys():
    store = _store()
    kids = [store.table_of(int(c)) for c in
            store.children(store.node_of("customers", 1))]
    assert kids.count("orders") == 2       # orders 10, 11; 13 dangles
    assert store.parents_of(store.node_of("orders", 13)).size == 0


class _Adapter:
    def spec(self, query, schema):
        return SimpleNamespace(direct_target=False, table_name="churn_task",
                               target_column="label")


def _policy() -> ContextPolicy:
    return ContextPolicy(seed=3, num_walks=64, walk_length=4,
                         max_context_cells=64)


def test_columnar_traversal_masks_the_target_label():
    store = _store()
    traversal = ColumnarTraversal(store, task_adapter=_Adapter())
    from relational_transformers_utils.schema import TemporalBound

    result = traversal.traverse(_schema(), None, "customers", 1,
                                TemporalBound.at_or_before(_dt(6)),
                                _policy(), query=object())
    rows = {r.key: r for r in result.rows}
    target_key = next(k for k, r in rows.items()
                      if r.table == "churn_task" and r.timestamp == _dt(6))
    assert "label" not in rows[target_key].cells   # masked focal target
    again = traversal.traverse(_schema(), None, "customers", 1,
                               TemporalBound.at_or_before(_dt(6)),
                               _policy(), query=object())
    assert [r.key for r in result.rows] == [r.key for r in again.rows]


def test_missing_task_row_reports_and_returns_empty():
    store = _store()
    messages = []
    traversal = ColumnarTraversal(store, task_adapter=_Adapter(),
                                  fallback=messages.append)
    from relational_transformers_utils.schema import TemporalBound

    result = traversal.traverse(_schema(), None, "customers", 2,
                                TemporalBound.at_or_before(_dt(9)),
                                _policy(), query=object())
    assert result.rows == ()
    assert messages and "no task row" in messages[0]


def test_cohort_targets_injects_history_below_the_anchor():
    store = _store()
    traversal = ColumnarTraversal(store, task_adapter=_Adapter())
    spec = _Adapter().spec(None, None)
    got = traversal.cohort_targets("customers", [1], _dt(6), spec, history=3)
    assert got is not None
    targets, inject, extra = got
    assert targets[0][0] == 1
    tables = [r.table for r in inject]
    assert "churn_task" in tables and "customers" in tables
    # the labeled history row at day 5 belongs to customer 1; the NaN-label
    # row is filtered by the label mask
    history = [r for r in inject if r.table == "churn_task"
               and r.timestamp == _dt(5)]
    assert history == []   # its label is NaN, so it carries no signal
    assert all(key in extra for key, _ in [(r.key, r) for r in inject])


def test_datetime_units_and_object_columns_normalize():
    from relational_transformers_utils.columnar import _epoch_seconds, _notna_mask

    ns = np.asarray([_dt64(1, "ns")])
    us = np.asarray([_dt64(1)])
    assert _epoch_seconds(ns)[0] == _epoch_seconds(us)[0] == _dt(1).timestamp()
    objs = np.asarray([_dt(2), None, float("nan"), 7.0], dtype=object)
    secs = _epoch_seconds(objs)
    assert secs[0] == _dt(2).timestamp() and secs[3] == 7.0
    assert np.isnan(secs[1]) and np.isnan(secs[2])
    assert _notna_mask(objs).tolist() == [True, False, False, True]
