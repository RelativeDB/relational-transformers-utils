# Normalization

RT-J consumes normalized scalars. The conventions here reproduce the reference
preprocessor exactly, because a checkpoint trained under one normalization scheme reads
values from another scheme as noise.

## Conventions

Three rules carry the whole contract:

1. Each numeric column is scaled by its whole-column mean and **sample** standard
   deviation (`ddof=1`).
2. Every datetime cell in the dataset shares one global mean and **population**
   standard deviation (`ddof=0`), in day units.
3. A zero standard deviation becomes 1.0, so constant columns pass through centered.

The two ddof values differ on purpose; the reference computes them that way, and
changing either drifts every score.

## ColumnStats

Fit statistics once from your rows, under the training temporal bound:

```python
from relational_transformers_utils import ColumnStats, TemporalBound

stats = ColumnStats.fit(schema, tables, bound=TemporalBound.at_or_before(train_end))
stats.transform("customers", "age", 34.0)
stats.transform_datetime(days_value)
```

Statistics fitted without a bound include rows after the anchor, and that leaks the
future into every scaled value. `to_dict` and `from_dict` round-trip the fitted state
for persistence beside the model artifact. `with_task_values`, `with_column_values`,
and `with_datetime_values` return copies carrying additional statistics, and `task()`
raises `NormalizationError` for a task that was never fitted.

## Zero-shot and Reference Modes

`normalize_sequence` turns one context's raw cells into model-ready floats:

```python
from relational_transformers_utils import normalize_sequence

values = normalize_sequence(
    columns,       # (table, column) per cell
    sem_types,     # constants from relational_transformers.constants
    raw_values,    # numbers, bools, datetimes, strings, or None
    is_target,     # target mask
    mode="zero_shot",
)
```

Zero-shot mode derives each column's statistics from the non-target cells inside this
sequence, so a context's values never depend on which other contexts share a batch.
Reference mode passes `column_stats=` and raises `NormalizationError` for any column
without fitted statistics. In both modes, target cells, missing values, and text cells
come back as `0.0`; text content travels through embeddings.

## bfloat16 Boundary

The reference persists every model-valued channel as bfloat16. `bf16_as_f32` reproduces
that storage boundary with round-to-nearest-even, widened back to float32:

```python
from relational_transformers_utils import bf16_as_f32

number_values = bf16_as_f32(number_values)
```

Apply it to every float channel you materialize, including text and column embeddings,
so a Python pipeline and the reference produce identical inputs.
