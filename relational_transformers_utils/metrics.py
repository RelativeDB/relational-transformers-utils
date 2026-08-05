"""Pure-numpy evaluation metrics for relational prediction tasks."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import numpy as np

__all__ = [
    "roc_auc",
    "accuracy",
    "brier_score",
    "log_loss",
    "bootstrap_auroc",
    "mean_absolute_error",
    "r2_score",
    "better",
    "classification_report",
]


def roc_auc(labels, scores) -> float:
    """Area under the ROC curve via tie-corrected rank sums.

    Returns ``nan`` when only one class is present. Ties in ``scores``
    receive their average rank, matching the Mann-Whitney convention.
    """
    labels = (np.asarray(labels).reshape(-1) > 0).astype(np.int8)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    if not pos or not neg:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = np.bincount(inverse, weights=ranks)[inverse] / counts[inverse]
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def accuracy(scores, labels, *, threshold: float = 0.5) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = (np.asarray(labels).reshape(-1) > 0)
    return float(((scores >= threshold) == labels).mean())


def brier_score(scores, labels) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = (np.asarray(labels).reshape(-1) > 0).astype(np.float64)
    return float(np.square(scores - labels).mean())


def log_loss(scores, labels, *, eps: float = 1e-6) -> float:
    """Negative log likelihood with probabilities clamped to ``[eps, 1-eps]``."""
    scores = np.clip(np.asarray(scores, dtype=np.float64).reshape(-1), eps, 1.0 - eps)
    labels = (np.asarray(labels).reshape(-1) > 0)
    picked = np.where(labels, scores, 1.0 - scores)
    return float(-np.log(picked).mean())


def bootstrap_auroc(scores: Sequence[float], labels: Sequence[float], *,
                    rounds: int = 400, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval for AUROC over resampled examples."""
    rng = random.Random(seed)
    n = len(labels)
    areas: list[float] = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        area = roc_auc([labels[i] for i in idx], [scores[i] for i in idx])
        if not math.isnan(area):
            areas.append(area)
    areas.sort()
    return areas[int(0.025 * len(areas))], areas[int(0.975 * len(areas))]


def mean_absolute_error(predictions, labels) -> float:
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    return float(np.abs(predictions - labels).mean())


def r2_score(predictions, labels) -> float:
    """Coefficient of determination; ``nan`` when labels have no variance."""
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    denominator = float(np.square(labels - labels.mean()).sum())
    if not denominator:
        return float("nan")
    return float(1.0 - np.square(predictions - labels).sum() / denominator)


def better(task_type: str, candidate: float, incumbent: float,
           minimum_improvement: float = 0.0) -> bool:
    """Direction-aware improvement test: higher wins for ``clf``, lower for ``reg``."""
    if task_type == "clf":
        return candidate > incumbent + minimum_improvement
    return candidate < incumbent - minimum_improvement


def classification_report(scores, labels, *, threshold: float = 0.5) -> dict[str, float]:
    """Bundle the standard binary metrics into one dictionary."""
    return {
        "n": int(np.asarray(labels).reshape(-1).size),
        "accuracy": accuracy(scores, labels, threshold=threshold),
        "auroc": roc_auc(labels, scores),
        "brier": brier_score(scores, labels),
        "log_loss": log_loss(scores, labels),
    }
