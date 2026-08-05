# relational_transformers_utils API

The narrative guides live under Documentation in the sidebar; each section here links
back to its guide.

## Context Collection

Guide: [Context Collection](../context_collection.md)

```{eval-rst}
.. autoclass:: relational_transformers_utils.CscAdjacency
   :members:

.. autoclass:: relational_transformers_utils.CscIndex
   :members:

.. autoclass:: relational_transformers_utils.Row
   :members:

.. autoclass:: relational_transformers_utils.TemporalBound
   :members:
```

## Deterministic Sampling

Guide: [Context Collection](../context_collection.md)

```{eval-rst}
.. autoclass:: relational_transformers_utils.StdRng
   :members:

.. autofunction:: relational_transformers_utils.rand_sample

.. autofunction:: relational_transformers_utils.reference_walk_counts

.. autoclass:: relational_transformers_utils.ContextGraph
   :members:

.. autoclass:: relational_transformers_utils.ContextTruncated
```

## Schema

```{eval-rst}
.. autoclass:: relational_transformers_utils.Schema
   :members:

.. autoclass:: relational_transformers_utils.TableDef
   :members:

.. autoclass:: relational_transformers_utils.ColumnDef
   :members:

.. autoclass:: relational_transformers_utils.LinkDef
   :members:

.. autoclass:: relational_transformers_utils.ValueType
   :members:
```

## Traversal

Guide: [Traversal Strategies](../traversal.md)

```{eval-rst}
.. autoclass:: relational_transformers_utils.ContextPolicy
   :members:

.. autoclass:: relational_transformers_utils.BreadthFirstTraversal
   :members:

.. autoclass:: relational_transformers_utils.ReferenceTraversal
   :members:

.. autoclass:: relational_transformers_utils.TaskAdapter
   :members:

.. autoclass:: relational_transformers_utils.TraversalResult
   :members:

.. autoclass:: relational_transformers_utils.ColumnarStore
   :members:

.. autoclass:: relational_transformers_utils.ColumnarTraversal
   :members:
```

## Text Encoding

```{eval-rst}
.. autoclass:: relational_transformers_utils.CachedEncoder
   :members:
```

## Normalization

Guide: [Normalization](../normalization.md)

```{eval-rst}
.. autoclass:: relational_transformers_utils.ColumnStats
   :members:

.. autoclass:: relational_transformers_utils.NormalizationMode
   :members:

.. autofunction:: relational_transformers_utils.normalize_sequence

.. autofunction:: relational_transformers_utils.bf16_as_f32

.. autofunction:: relational_transformers_utils.days_since_epoch

.. autofunction:: relational_transformers_utils.mean_std
```

## Ablation

Guide: [Ablation](../ablation.md)

```{eval-rst}
.. autoclass:: relational_transformers_utils.AblationEvaluator
   :members:
```

## Metrics

Guide: [Metrics](../metrics.md)

```{eval-rst}
.. autofunction:: relational_transformers_utils.roc_auc

.. autofunction:: relational_transformers_utils.accuracy

.. autofunction:: relational_transformers_utils.brier_score

.. autofunction:: relational_transformers_utils.log_loss

.. autofunction:: relational_transformers_utils.bootstrap_auroc

.. autofunction:: relational_transformers_utils.mean_absolute_error

.. autofunction:: relational_transformers_utils.r2_score

.. autofunction:: relational_transformers_utils.better

.. autofunction:: relational_transformers_utils.classification_report
```

## Quantization

Guide: [Quantization](../quantization.md)

```{eval-rst}
.. autofunction:: relational_transformers_utils.quantize_model

.. autofunction:: relational_transformers_utils.quantize_checkpoint

.. autofunction:: relational_transformers_utils.quantize_state
```
