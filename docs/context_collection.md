# Context Collection

A prediction context is the target entity's row plus the most recent related rows at or
before an anchor time. `CscAdjacency` and `CscIndex` answer that "latest children"
question quickly and reproducibly.

## CscAdjacency

One adjacency serves one foreign-key link. Build it from parallel edge arrays, then
query any parent:

```python
from relational_transformers_utils import CscAdjacency

adjacency = CscAdjacency(
    n_parents=3,
    edge_parent=[0, 0, 2],
    edge_child=[10, 11, 12],
    edge_ts=[1.0, 2.0, 1.5],
)
adjacency.children(parent_dense=0, anchor_ts=1.5, limit=8)
# => [10]  (child 11 is newer than the anchor)
```

Construction stably sorts edges by parent and timestamp, so ties keep input order and
results match the RT-J reference byte for byte. Each `children` call binary-searches
the parent's slice for "latest at or before anchor" and returns dense child ids
newest-first. Edges with out-of-range parents are dropped during construction.

## CscIndex

`CscIndex.build` wraps a whole schema: it snapshots caller-provided rows per table,
assigns dense ids, and builds one adjacency per declared link. The caller owns
retrieval; the index never fetches anything.

```python
from relational_transformers_utils import CscIndex, Row, TemporalBound

index = CscIndex.build(schema, {
    "customers": customer_rows,        # any iterable of Row
    "orders": order_rows,
})

bound = TemporalBound.at_or_before(anchor_time)
customer = index.entities("customers", [customer_id], bound)
recent = index.children(orders_link, customer_id, bound, limit=16)
cohort = index.cohort("customers", customer_id, bound, limit=32)
```

Rows without a timestamp are static and admitted under every bound. A row's foreign-key
values live in `Row.parents`, keyed by FK column name; a value may be a single id or a
list of ids.

```{eval-rst}
.. collapse:: Dangling foreign keys

   A handful of dangling FK values is data, and those edges are simply dropped. When
   every candidate FK value of a link dangles, the index emits a ``UserWarning``,
   because a fully severed link usually means a key-type mismatch (an integer primary
   key against string FK values after a CSV round-trip) and every downstream child
   count silently reads zero.
```

## Temporal Bounds

`TemporalBound.at_or_before(t)` is the leakage guard: rows newer than the anchor never
enter a context, and `CscIndex.build` can also apply a bound at snapshot time. Naive
datetimes are treated as UTC. Fit normalization statistics under the same bound, as the
[Normalization](normalization.md) page explains.
