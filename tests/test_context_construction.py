from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from relational_transformers_utils import (
    BreadthFirstTraversal,
    CachedEncoder,
    ContextPolicy,
    PrecomputedEmbeddingError,
    ReferenceTraversal,
    Row,
    TemporalBound,
)
from relational_transformers_utils.schema import ColumnDef, LinkDef, Schema, TableDef, ValueType


def _dt(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=timezone.utc)


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


class _Graph:
    """A minimal GraphAccess over in-memory rows."""

    def __init__(self, customers, orders):
        self._rows = {"customers": customers, "orders": orders}
        self._link_index = {}
        for order in orders:
            self._link_index.setdefault(
                order.parents["customer_id"], []).append(order)

    def entities(self, table, ids, bound):
        wanted = set(ids)
        return [r for r in self._rows[table]
                if r.id in wanted and bound.admits_row(r)]

    def children(self, link, parent_id, bound, limit):
        rows = [r for r in self._link_index.get(parent_id, [])
                if bound.admits_row(r)]
        rows.sort(key=lambda r: r.timestamp, reverse=True)
        return rows[:limit]

    def cohort(self, table, anchor, bound, limit):
        return [r.id for r in self._rows[table]
                if r.id != anchor and bound.admits_row(r)][:limit]

    def all_ids(self, table):
        return [r.id for r in self._rows[table]]

    def all_rows(self, table):
        return list(self._rows[table])


def _graph():
    customers = [Row("customers", i, {"age": 30.0 + i}) for i in range(3)]
    orders = [Row("orders", 100 + i, {"amount": 10.0 * i},
                  timestamp=_dt(1 + i), parents={"customer_id": i % 3})
              for i in range(6)]
    return _Graph(customers, orders)


def test_breadth_first_collects_the_temporal_neighbourhood():
    result = BreadthFirstTraversal().traverse(
        _schema(), _graph(), "customers", 0,
        TemporalBound.at_or_before(_dt(3)), ContextPolicy(seed=1))
    tables = {r.table for r in result.rows}
    assert ("customers", 0) in {r.key for r in result.rows}
    assert "orders" in tables
    assert all(r.timestamp is None or r.timestamp <= _dt(3)
               for r in result.rows)


def test_reference_traversal_without_a_task_adapter_is_target_local():
    policy = ContextPolicy(seed=7, num_walks=64, walk_length=4)
    result = ReferenceTraversal().traverse(
        _schema(), _graph(), "customers", 1,
        TemporalBound.at_or_before(_dt(6)), policy)
    again = ReferenceTraversal().traverse(
        _schema(), _graph(), "customers", 1,
        TemporalBound.at_or_before(_dt(6)), policy)
    assert [r.key for r in result.rows] == [r.key for r in again.rows]
    assert ("customers", 1) in {r.key for r in result.rows}


def test_cached_encoder_deduplicates_and_enforces_strict_tables():
    calls = []

    def encode(texts, normalize):
        calls.append(list(texts))
        return np.ones((len(texts), 4), np.float32)

    encoder = CachedEncoder(encode)
    encoder.encode(["a", "b", "a"])
    encoder.encode(["a", "b"])
    assert calls == [["a", "b"]]

    strict = CachedEncoder(encode)
    strict.install_precomputed({"known": np.zeros(4, np.float32)})
    with pytest.raises(PrecomputedEmbeddingError):
        strict.encode(["unknown"])


class _DerivedAdapter:
    """COUNT(orders) OVER 2 DAYS FOLLOWING, expressed without a language."""

    def spec(self, query, schema):
        from types import SimpleNamespace
        return SimpleNamespace(id="churn", direct_target=False,
                               table_name="churn_task",
                               target_column="label", time_column=None)

    def window_span(self, query):
        from datetime import timedelta
        return timedelta(days=2)

    def aggregated_tables(self, query, entity_table):
        return {"orders"}

    def label(self, query, schema, visible, entity_cells, ts):
        from datetime import timedelta
        window_end = ts + timedelta(days=2)
        return float(sum(1 for r in visible.get("orders", ())
                         if r.timestamp is not None
                         and ts < r.timestamp <= window_end))


def test_reference_traversal_synthesizes_history_task_rows():
    policy = ContextPolicy(seed=5, num_walks=64, walk_length=4,
                           num_history_windows=2)
    traversal = ReferenceTraversal(task_adapter=_DerivedAdapter())
    result = traversal.traverse(
        _schema(), _graph(), "customers", 0,
        TemporalBound.at_or_before(_dt(6)), policy, query=object())
    task_rows = [r for r in result.rows if r.table == "churn_task"]
    assert task_rows, "derived task rows must reach the context"
    target = [r for r in task_rows if r.timestamp == _dt(6)]
    history = [r for r in task_rows if r.timestamp != _dt(6)]
    assert target and target[0].cells == {}     # unknown focal target
    assert history
    orders = _graph().all_rows("orders")
    for row in history:
        # every history window carries the label the adapter derived for its
        # own entity: that entity's orders inside (ts, ts + 2 days] — peers'
        # orders never leak into it
        assert set(row.cells) == {"label"}
        entity = row.parents["__entity__"]
        from datetime import timedelta
        expected = sum(
            1 for o in orders
            if o.parents["customer_id"] == entity
            and row.timestamp < o.timestamp <= row.timestamp + timedelta(days=2))
        assert row.cells["label"] == float(expected)

    again = traversal.traverse(
        _schema(), _graph(), "customers", 0,
        TemporalBound.at_or_before(_dt(6)), policy, query=object())
    assert [r.key for r in result.rows] == [r.key for r in again.rows]


def test_schema_builders_and_json_round_trip():
    schema = (Schema.new_schema()
              .table(TableDef.new_table("customers")
                     .column("age", ValueType.NUMBER)
                     .column(ColumnDef.of("name", ValueType.TEXT))
                     .primary_key("id").build())
              .table(TableDef.new_table("orders")
                     .column("placed", ValueType.DATETIME)
                     .primary_key("id").time_column("placed").build())
              .link("orders", "customer_id", "customers")
              .build())
    payload = schema.to_json_dict()
    assert payload["links"][0]["fk_column"] == "customer_id"
    assert schema.require_table("orders").time_column == "placed"
    assert schema.table("nope") is None
    assert [link.to_table for link in schema.links_from("orders")] == ["customers"]
    assert [link.from_table for link in schema.links_to("customers")] == ["orders"]

    import pytest as _pytest

    from relational_transformers_utils.schema import SchemaError
    with _pytest.raises(SchemaError, match="duplicate column"):
        TableDef("t", columns=(ColumnDef("a", ValueType.NUMBER),
                               ColumnDef("a", ValueType.NUMBER)))
    with _pytest.raises(SchemaError, match="unknown to_table"):
        Schema(tables=(TableDef("a", primary_key="id"),),
               links=(LinkDef("a", "fk", "missing"),))


def test_factory_supplied_task_graph_drives_the_shared_path():
    anchor = _dt(6)
    base = 3 + 6      # physical rows: 3 customers + 6 orders

    def factory(task_spec, entity_id, when):
        target = Row("churn_task", (entity_id, when, "target"), {},
                     timestamp=when, parents={"__entity__": entity_id})
        history = Row("churn_task", (entity_id, _dt(4), 1), {"label": 1.0},
                      timestamp=_dt(4), parents={"__entity__": entity_id})
        node_ids = {target.key: base, history.key: base + 1}
        return [target, history], node_ids, target.key

    policy = ContextPolicy(seed=5, num_walks=64, walk_length=4,
                           num_history_windows=1)
    traversal = ReferenceTraversal(task_adapter=_DerivedAdapter(),
                                   task_graph_factory=factory)
    result = traversal.traverse(
        _schema(), _graph(), "customers", 0,
        TemporalBound.at_or_before(anchor), policy, query=object())
    keys = {r.key for r in result.rows}
    assert ("churn_task", (0, anchor, "target")) in keys
    target_row = next(r for r in result.rows
                      if r.key == ("churn_task", (0, anchor, "target")))
    assert "label" not in target_row.cells

    again = traversal.traverse(
        _schema(), _graph(), "customers", 0,
        TemporalBound.at_or_before(anchor), policy, query=object())
    assert [r.key for r in result.rows] == [r.key for r in again.rows]
