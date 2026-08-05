# Quickstart

This page walks the path from retrieved rows to a scored, measured prediction. Each step
has a dedicated page with the full contract.

## Collect Context

Declare the schema shape, hand `CscIndex` your rows, and query time-bounded children:

```python
from relational_transformers_utils import CscIndex, TemporalBound

index = CscIndex.build(schema, {"customers": customer_rows, "orders": order_rows})
bound = TemporalBound.at_or_before(anchor_time)
recent_orders = index.children(orders_link, customer_id, bound, limit=16)
```

The index answers "the latest N children at or before this anchor" with one binary
search per query. See [Context Collection](context_collection.md).

## Normalize Scalars

Fit column statistics once under the training bound, then normalize each context's
cells:

```python
from relational_transformers_utils import ColumnStats, normalize_sequence

stats = ColumnStats.fit(schema, {"customers": customer_rows, "orders": order_rows},
                        bound=bound)
values = normalize_sequence(columns, sem_types, raw_values, is_target,
                            mode="reference", column_stats=stats)
```

The normalized floats feed the `number_values` and `datetime_values` channels of a
`RelationalBatch`. See [Normalization](normalization.md).

## Predict and Measure

Run the model from the core package, then measure with the utilities here:

```python
from relational_transformers import RelationalTransformer
from relational_transformers_utils import AblationEvaluator, classification_report

model = RelationalTransformer("RelativeDB/rt-j-fp16")
scores = model.predict(batches)

print(classification_report(scores, labels))
print(AblationEvaluator(examples, {"support": support_positions})(model))
```

## Benchmark

`relben` carries the RelBench task catalog and submission tooling:

```python
from relben import select_tasks, write_submission

for task in select_tasks(["rel-f1"]):
    write_submission(out_dir / task.filename, task.target, predictions[task.id])
```
