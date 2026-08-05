# Traversal Strategies

The traversals turn a target entity plus related rows into an ordered context.
They operate on `Row` objects, a declared `Schema`, and a `ContextPolicy`;
derived prediction targets enter through a `TaskAdapter`, so a query language
can plug in without the library knowing it exists.

## ContextPolicy

One frozen dataclass carries every assembly knob: the cell budget, per-hop
fanouts, walk counts and length, cohort sizing, history windows, and the seed.
The default geometry follows the RT-J reference evaluator; pass
`max_context_cells=8192` to match the reference evaluation context size.

```python
from relational_transformers_utils import ContextPolicy

policy = ContextPolicy(max_context_cells=2048, bfs_width=32, seed=0)
```

## BreadthFirstTraversal

Pull-per-hop breadth-first expansion through a `GraphAccess` implementation:
the target's row, then its children newest-first under the temporal bound,
capped by `fanout_at(hop)` per hop. Use it when rows come from retrievers and
no peer ranking is needed.

```python
from relational_transformers_utils import BreadthFirstTraversal, TemporalBound

result = BreadthFirstTraversal().traverse(
    schema, graph, "customers", customer_id,
    TemporalBound.at_or_before(anchor), policy)
```

## ReferenceTraversal

Reference tiering: a BFS from the target, graph-walk-ranked peers, and a
random table fallback, all drawn from the deterministic sampling primitives
in [walks](context_collection.md#deterministic-sampling). Without a task
adapter it collects the target entity's own temporal neighbourhood.

With an adapter, derived targets work too: the adapter supplies the task spec,
the task window span, the aggregated tables, and the self-label value for
history windows, and the traversal synthesizes the task rows. The query object
passes through opaquely.

```python
class MyTaskAdapter:
    def spec(self, query, schema): ...
    def window_span(self, query): ...
    def aggregated_tables(self, query, entity_table): ...
    def label(self, query, schema, visible, entity_cells, ts): ...

traversal = ReferenceTraversal(task_adapter=MyTaskAdapter())
```

## Columnar Contexts

`ColumnarStore` holds tables as numpy columns with CSR adjacency and
materializes `Row` objects lazily, only for rows a context emits, so tens of
millions of rows fit in a few hundred MB. `ColumnarTraversal` runs the
shared-context walk assembly over the store through
[`ContextGraph`](context_collection.md#deterministic-sampling), behind the
same `TaskAdapter` seam.

```python
from relational_transformers_utils import ColumnarStore, ColumnarTraversal

store = ColumnarStore(schema, frames, task_frames=..., task_links=...)
traversal = ColumnarTraversal(store, task_adapter=MyTaskAdapter())
```

An entity with no task row at the anchor is reported through the injected
`fallback` callable rather than silently scored on an empty context.
