"""Utilities that wire Relational Transformers into a product pipeline."""

from .ablation import AblationEvaluator
from .csc import CscAdjacency, CscIndex
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
    ColumnStats,
    NormalizationError,
    NormalizationMode,
    bf16_as_f32,
    days_since_epoch,
    mean_std,
    normalize_sequence,
)
from .quantization import quantize_checkpoint, quantize_model, quantize_state
from .rows import Row, TemporalBound
from .schema import ColumnDef, LinkDef, Schema, SchemaError, TableDef, ValueType

__version__ = "0.1.0"

__all__ = [
    "AblationEvaluator",
    "CscAdjacency",
    "CscIndex",
    "ColumnDef",
    "ColumnStats",
    "LinkDef",
    "NormalizationError",
    "NormalizationMode",
    "Row",
    "Schema",
    "SchemaError",
    "TableDef",
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
    "roc_auc",
]
