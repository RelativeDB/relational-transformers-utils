# RelBench Utilities

`relben` packages the benchmark side: the curated task catalog, submission files, run
records, reports, and hurdle-gate tuning. It never imports an engine, so any runner
that produces predictions can use it.

## Task Catalog

`EVAL_TASKS` is the curated 21-task RelBench set (12 classification, 9 regression)
used by the relational-transformer paper. Two extra rel-f1 scalar tables live in
`EXTRA_TASKS` and require explicit selection.

```python
from relben import EVAL_TASKS, select_tasks

select_tasks(None)                      # all 21
select_tasks(["rel-f1"])                # one database
select_tasks(["rel-amazon/user-churn"])  # one task
```

Each `EvalTask` carries `database`, `table`, `target`, and `task_type`, plus the
derived `id` (`rel-f1/driver-dnf`) and `filename` (`rel-f1__driver-dnf.csv`). An
unknown selector raises `ValueError` rather than silently matching nothing.

## Submissions and Run Records

The official scorer joins a submission CSV on its key columns, so `write_submission`
enforces a consistent key across rows and appends the prediction under the target
column:

```python
from relben import write_manifest, write_submission

rows = ((sample.key, prediction) for sample, prediction in zip(samples, predictions))
write_submission(out_dir / task.filename, task.target, rows)
write_manifest(out_dir, runner="ours", tasks=[task.id])
```

`EvalSample` holds one test row's entity id, anchor time, submission key, and optional
runner parameters. `atomic_json` writes records through a temp file and rename, so a
crashed run never leaves a half-written JSON behind.

## Reports

`write_report` turns result rows into `results.json` and a markdown score matrix, with
macro means over valid tasks and a gain table against a chosen baseline. Gain signs
follow metric direction, so positive always means the runner improved on the baseline:
`runner - baseline` for AUROC and `baseline - runner` for NMAE.

```python
from relben import write_report

write_report(out_dir, results, ["rt", "ours"], baseline="rt")
```

## Hurdle Gates

Zero-inflated regression targets often score better when a low existence probability
forces the prediction to zero. `tune_hurdle_gate` grid-searches the threshold on a
validation split and reports whether gating beat the ungated baseline:

```python
from relben import nmae, tune_hurdle_gate

best_mae, threshold, use_gate = tune_hurdle_gate(reg_preds, exist_probs, labels)
score = nmae(final_preds, labels, train_std=train_labels.std())
```
