# Ablation

`AblationEvaluator` measures how much named groups of cells move a model's predictions
across a dataset. It builds on `RelationalBatch.ablate` from the core package, which
pads out selected cells while node identities and positions stay stable.

```python
from relational_transformers import RelationalExample
from relational_transformers_utils import AblationEvaluator

examples = [RelationalExample(input=batch, label=label) for batch, label in pairs]
metrics = AblationEvaluator(
    examples,
    ablations={"support_history": [11, 12], "orders": [5, 6, 7]},
)(model)
# => {'support_history_mean_delta': ..., 'support_history_mean_absolute_delta': ...,
#     'orders_mean_delta': ..., 'orders_mean_absolute_delta': ...}
```

Deltas are computed on identity-activation outputs, so classification results live in
logit space and sigmoid saturation cannot hide movement. `mean_delta` shows direction:
a negative value for a churn model means removing those cells lowered predicted churn.
`mean_absolute_delta` shows magnitude regardless of direction.

The evaluator composes with the core package's `SequentialEvaluator`, so one call can
report task metrics and ablation deltas together.

## Choosing Positions

Positions are cell indices in each example's batch, and the grouping is yours.
Applications typically group by table, by column, by entity, or by time range. Keep the
groups aligned with a question someone asked; the evaluator measures whatever you name,
meaningful or ill-posed alike.

Single-context ablation without the evaluator stays a two-liner with the core package:

```python
without_support = batch.ablate(support_positions)
full, ablated = model.predict([batch, without_support], activation="identity")
```
