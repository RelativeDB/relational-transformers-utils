"""Hurdle-gate tuning for zero-inflated regression targets."""

from __future__ import annotations

import numpy as np

__all__ = ["nmae", "tune_hurdle_gate"]


def nmae(predictions, labels, *, train_std: float) -> float:
    """Mean absolute error normalized by the training-split label std."""
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if train_std <= 0:
        raise ValueError("train_std must be positive")
    return float(np.abs(predictions - labels).mean() / train_std)


def tune_hurdle_gate(regression_predictions, existence_probabilities, labels,
                     *, grid=None) -> tuple[float, float, bool]:
    """Search a probability threshold that zeroes predictions for unlikely rows.

    Zero-inflated targets often score better when a low existence probability
    forces the prediction to zero. The search compares gated MAE against the
    ungated baseline over a validation split and reports whether gating won.

    Returns ``(best_mae, best_threshold, use_gate)``.
    """
    rv = np.asarray(regression_predictions, dtype=np.float64).reshape(-1)
    pv = np.asarray(existence_probabilities, dtype=np.float64).reshape(-1)
    yv = np.asarray(labels, dtype=np.float64).reshape(-1)
    baseline = float(np.abs(rv - yv).mean())
    thresholds = (np.arange(0.20, 0.81, 0.02) if grid is None
                  else np.asarray(list(grid), dtype=np.float64))
    scored = [(float(np.abs(np.where(pv > t, rv, 0.0) - yv).mean()), round(float(t), 2))
              for t in thresholds]
    best_mae, best_threshold = min(scored)
    return best_mae, best_threshold, best_mae < baseline
