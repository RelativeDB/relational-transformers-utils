# Metrics

Every metric is pure numpy over labels and scores, so the same functions serve training
loops, evaluators, and benchmark harnesses.

## Classification

`roc_auc` computes the area under the ROC curve through tie-corrected rank sums. Tied
scores receive their average rank, which matches the Mann-Whitney statistic, and a
single-class input returns `nan` rather than dividing by zero.

```python
from relational_transformers_utils import classification_report, roc_auc

auc = roc_auc(labels, scores)
report = classification_report(scores, labels)
# => {'n': 400, 'accuracy': ..., 'auroc': ..., 'brier': ..., 'log_loss': ...}
```

`accuracy` takes a `threshold` (default 0.5). `log_loss` clamps probabilities to
`[eps, 1-eps]` on both sides, so a hard 0.0 or 1.0 prediction scores finitely.
`bootstrap_auroc` resamples examples and returns a 95% percentile interval:

```python
from relational_transformers_utils import bootstrap_auroc

low, high = bootstrap_auroc(scores, labels, rounds=400, seed=0)
```

## Regression

`mean_absolute_error` and `r2_score` cover the regression tasks; `r2_score` returns
`nan` when labels have no variance. For benchmark-style normalized MAE, `relben.nmae`
divides by the training-split label standard deviation.

## Model Selection

`better(task_type, candidate, incumbent, minimum_improvement)` encodes metric
direction once: higher wins for `"clf"`, lower wins for `"reg"`, and the improvement
margin guards against promoting on noise. Early-stopping and checkpoint-promotion
loops share it so no caller re-derives the sign.
