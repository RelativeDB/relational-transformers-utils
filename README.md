# Relational Transformers Utils

Utility tooling that wires [Relational Transformers](https://relationaltransformers.com)
into a working product: context collection, normalization, ablation measurement,
metrics, checkpoint quantization, and experimental categorical causal discovery.
A second package, `relben`, holds RelBench
benchmark utilities. Everything is pure Python over numpy and torch.

Applications own retrieval and encoding. This package covers the numeric steps between
retrieved rows and a `RelationalBatch`, and the measurement steps after a prediction
comes back. It contains no query language, no connectors, and no context-builder
pipeline.

## Installation

```bash
pip install -U relational-transformers-utils
```

## What's Inside

### Context collection (`csc`)

`CscAdjacency` is a compressed-sparse-column adjacency over foreign-key edges. Build it
once from edge arrays, then answer time-bounded "latest children at or before this
anchor" queries with one binary search each. `CscIndex` wraps a whole schema of
caller-provided rows behind the same idea. Tie handling matches the RT-J reference
byte for byte.

```python
from relational_transformers_utils import CscIndex, TemporalBound

index = CscIndex.build(schema, {"customers": customer_rows, "orders": order_rows})
recent_orders = index.children(link, customer_id, TemporalBound.at_or_before(anchor), 16)
```

### Normalization

`ColumnStats` fits per-column mean/std for numeric cells and one global normalizer for
datetimes, with the reference preprocessor's exact conventions: sample std for columns,
population std for datetimes, and 1.0 in place of a zero std. `normalize_sequence`
turns one context's raw scalar cells into model-ready floats in zero-shot or reference
mode, and `bf16_as_f32` reproduces the bfloat16 storage boundary.

```python
from relational_transformers_utils import ColumnStats, normalize_sequence

stats = ColumnStats.fit(schema, tables, bound=training_bound)
values = normalize_sequence(columns, sem_types, raw_values, is_target,
                            mode="reference", column_stats=stats)
```

### Ablation

`AblationEvaluator` measures how much named groups of cells move a model's predictions
across a dataset, through `RelationalBatch.ablate` and identity-activation scores.

```python
from relational_transformers_utils import AblationEvaluator

metrics = AblationEvaluator(examples, {"support": [11, 12]})(model)
```

### Experimental causal discovery

`relational_transformers_utils.causal` provides entropy-based pairwise direction
scores, paired-row bootstrap stability, and bounded orientation of a supplied
small graph skeleton. `CausalFeatureSelector` proposes available column/join-path
groups and selects them with a caller-supplied validation evaluator. Inputs are explicitly categorical observations. Scores are
causal-model preferences under assumptions, not prediction contribution scores.
See [Causal discovery](docs/causal.md) for the API, assumptions, and upstream provenance.

```python
from relational_transformers_utils.causal import entropic_direction

x = [0, 1, 2, 3] * 50
y = [value // 2 for value in x]
result = entropic_direction(x, y, names=("X", "Y"), bootstrap=100, seed=42)
print(result.direction)    # ('X', 'Y') under the default exogenous-noise criterion
print(result.margin_bits)  # 1.0; positive favors X -> Y
```

This synthetic example illustrates the score; the direction alone does not establish
causation. Use `criterion="total"` to include the proposed cause's marginal entropy
(the example then ties), or `orient_graph` to rank orientations of a known skeleton.

### Metrics

Pure-numpy AUROC with tie-corrected rank sums, accuracy, Brier score, clamped log loss,
bootstrap AUROC intervals, MAE, R², and a direction-aware `better()` comparator for
model selection.

### Quantizers

`rt-quantize` converts an RT-J checkpoint to FP8, row-wise int8, or packed int4. Every
output loads through the standard `relational-transformers` constructor.

```bash
rt-quantize RelativeDB/rt-j-fp16 ./rt-j-int8 --format int8
```

### Benchmarks (`relben`)

The curated 21-task RelBench catalog, keyed submission-CSV writers, atomic run records,
score-matrix reports with gain-versus-baseline tables, and hurdle-gate tuning for
zero-inflated regression targets.

```python
from relben import select_tasks, write_submission

for task in select_tasks(["rel-f1"]):
    write_submission(out / task.filename, task.target, predictions[task.id])
```

## Development

```bash
python -m pip install -e '.[dev]'
pytest
```

## License

Apache License 2.0.
