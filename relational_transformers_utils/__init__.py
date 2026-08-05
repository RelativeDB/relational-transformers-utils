"""Utilities that wire Relational Transformers into a product pipeline.

The numpy-only modules (``csc``, ``metrics``, ``normalize``, ``rows``,
``schema``) import eagerly. ``AblationEvaluator`` and the quantizers need
torch and load on first attribute access, so a metrics-only consumer never
pays the torch import.
"""

from .columnar import ColumnarStore, ColumnarTraversal
from .csc import CscAdjacency, CscIndex
from .graph import ContextGraph, ContextTruncated
from .metrics import (
    accuracy,
    better,
    bootstrap_auroc,
    brier_score,
    classification_report,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc,
)
from .normalize import (
    SEM_BOOLEAN,
    SEM_DATETIME,
    SEM_NUMBER,
    SEM_TEXT,
    ColumnStats,
    NormalizationError,
    NormalizationMode,
    bf16_as_f32,
    days_since_epoch,
    mean_std,
    normalize_sequence,
)
from .schema import ColumnDef, LinkDef, Row, Schema, SchemaError, TableDef, TemporalBound, ValueType
from .text import CachedEncoder, PrecomputedEmbeddingError
from .traversal import (
    BreadthFirstTraversal,
    ContextPolicy,
    GraphAccess,
    GraphTraversal,
    ReferenceTraversal,
    TaskAdapter,
    TraversalResult,
)
from .walks import StdRng, rand_sample, reference_walk_counts, stdrng_first_u64_batch

__version__ = "0.1.0"

_TORCH_EXPORTS = {
    "AblationEvaluator": ("ablation", "AblationEvaluator"),
    "quantize_checkpoint": ("quantization", "quantize_checkpoint"),
    "quantize_model": ("quantization", "quantize_model"),
    "quantize_state": ("quantization", "quantize_state"),
}


def __getattr__(name):
    if name in _TORCH_EXPORTS:
        import importlib

        module_name, attribute = _TORCH_EXPORTS[name]
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attribute)
    raise AttributeError(
        f"module 'relational_transformers_utils' has no attribute {name!r}")


__all__ = [
    "AblationEvaluator",
    "BreadthFirstTraversal",
    "CachedEncoder",
    "ColumnarStore",
    "ColumnarTraversal",
    "ContextPolicy",
    "CscAdjacency",
    "CscIndex",
    "ColumnDef",
    "ColumnStats",
    "ContextGraph",
    "ContextTruncated",
    "GraphAccess",
    "GraphTraversal",
    "LinkDef",
    "NormalizationError",
    "NormalizationMode",
    "PrecomputedEmbeddingError",
    "ReferenceTraversal",
    "Row",
    "SEM_BOOLEAN",
    "SEM_DATETIME",
    "SEM_NUMBER",
    "SEM_TEXT",
    "Schema",
    "SchemaError",
    "StdRng",
    "TableDef",
    "TaskAdapter",
    "TraversalResult",
    "TemporalBound",
    "ValueType",
    "accuracy",
    "better",
    "bf16_as_f32",
    "bootstrap_auroc",
    "brier_score",
    "classification_report",
    "days_since_epoch",
    "log_loss",
    "mean_absolute_error",
    "mean_std",
    "normalize_sequence",
    "quantize_checkpoint",
    "quantize_model",
    "quantize_state",
    "r2_score",
    "rand_sample",
    "reference_walk_counts",
    "roc_auc",
    "stdrng_first_u64_batch",
]
