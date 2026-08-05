from __future__ import annotations

import math

import numpy as np
import pytest

from relational_transformers_utils.metrics import (
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


def _pairwise_auc(labels, scores):
    positives = [s for s, y in zip(scores, labels, strict=True) if y]
    negatives = [s for s, y in zip(scores, labels, strict=True) if not y]
    total = 0.0
    for p in positives:
        for n in negatives:
            total += 1.0 if p > n else 0.5 if p == n else 0.0
    return total / (len(positives) * len(negatives))


def test_roc_auc_matches_pairwise_oracle_with_ties():
    rng = np.random.default_rng(3)
    for _ in range(20):
        labels = rng.integers(0, 2, size=40)
        if labels.min() == labels.max():
            continue
        scores = rng.integers(0, 5, size=40).astype(float)  # heavy ties
        assert roc_auc(labels, scores) == pytest.approx(_pairwise_auc(labels, scores))


def test_roc_auc_is_nan_with_one_class():
    assert math.isnan(roc_auc([1, 1, 1], [0.2, 0.4, 0.9]))


def test_perfect_and_inverted_rankings():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_binary_metric_values():
    scores = [0.9, 0.8, 0.3, 0.1]
    labels = [1, 1, 0, 0]
    assert accuracy(scores, labels) == 1.0
    assert brier_score(scores, labels) == pytest.approx((0.01 + 0.04 + 0.09 + 0.01) / 4)
    assert log_loss(scores, labels) > 0.0
    report = classification_report(scores, labels)
    assert report["n"] == 4
    assert report["auroc"] == 1.0


def test_log_loss_clamps_extreme_probabilities():
    assert math.isfinite(log_loss([0.0, 1.0], [1, 0]))


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 2, size=200)
    scores = labels * 0.6 + rng.random(200) * 0.4
    low, high = bootstrap_auroc(list(scores), list(labels), rounds=100, seed=1)
    point = roc_auc(labels, scores)
    assert low <= point <= high


def test_regression_metrics():
    assert mean_absolute_error([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.5)
    assert r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    assert math.isnan(r2_score([1.0, 2.0], [3.0, 3.0]))


def test_better_is_direction_aware():
    assert better("clf", 0.8, 0.7)
    assert not better("clf", 0.7, 0.8)
    assert better("reg", 0.3, 0.4)
    assert not better("reg", 0.4, 0.3)
    assert not better("clf", 0.71, 0.7, 0.05)
