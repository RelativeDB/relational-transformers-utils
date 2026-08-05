from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import pytest

from relben import (
    EVAL_TASKS,
    EXTRA_TASKS,
    EvalSample,
    nmae,
    select_tasks,
    tune_hurdle_gate,
    write_manifest,
    write_report,
    write_submission,
)


def test_catalog_is_the_curated_12_clf_9_reg_set():
    assert len(EVAL_TASKS) == 21
    assert sum(t.task_type == "clf" for t in EVAL_TASKS) == 12
    assert sum(t.task_type == "reg" for t in EVAL_TASKS) == 9
    assert select_tasks(None) == list(EVAL_TASKS)
    assert all(t.database == "rel-f1" for t in select_tasks(["rel-f1"]))
    with pytest.raises(ValueError):
        select_tasks(["rel-nonexistent"])


def test_extra_tasks_require_explicit_selection():
    default_ids = {t.id for t in select_tasks(None)}
    assert not default_ids & {t.id for t in EXTRA_TASKS}
    assert select_tasks(["rel-f1/qualifying-position"])[0].target == "position"


def test_submission_csv_round_trips(tmp_path):
    path = tmp_path / "rel-f1__driver-dnf.csv"
    rows = [
        ({"date": "2020-01-01", "driverId": 1}, 0.25),
        ({"date": "2020-01-01", "driverId": 2}, 0.75),
    ]
    assert write_submission(path, "did_not_finish", rows) == 2
    with path.open() as f:
        parsed = list(csv.DictReader(f))
    assert parsed[1] == {"date": "2020-01-01", "driverId": "2", "did_not_finish": "0.75"}


def test_submission_rejects_inconsistent_keys(tmp_path):
    rows = [({"date": "d", "driverId": 1}, 0.5), ({"driverId": 2, "date": "d"}, 0.5)]
    with pytest.raises(ValueError):
        write_submission(tmp_path / "bad.csv", "y", rows)


def test_manifest_and_sample(tmp_path):
    sample = EvalSample(11, datetime(2021, 3, 1, tzinfo=timezone.utc), {"driverId": 11})
    assert sample.params == {}
    path = write_manifest(tmp_path, runner="ours", tasks=["rel-f1/driver-dnf"])
    assert json.loads(path.read_text())["runner"] == "ours"


def test_report_scores_and_gains(tmp_path):
    results = [
        {"task": "rel-f1/driver-dnf", "runner": "rt", "metric": "roc_auc", "value": 0.70},
        {"task": "rel-f1/driver-dnf", "runner": "ours", "metric": "roc_auc", "value": 0.75},
        {"task": "rel-f1/driver-position", "runner": "rt", "metric": "nmae", "value": 0.90},
        {"task": "rel-f1/driver-position", "runner": "ours", "metric": "nmae", "value": 0.80},
    ]
    write_report(tmp_path, results, ["rt", "ours"], baseline="rt")
    markdown = (tmp_path / "results.md").read_text()
    assert "+0.050000" in markdown  # roc_auc gain: ours - rt
    assert "+0.100000" in markdown  # nmae gain: rt - ours
    assert json.loads((tmp_path / "results.json").read_text())["runners"] == ["rt", "ours"]


def test_hurdle_gate_helps_zero_inflated_targets():
    labels = [0.0, 0.0, 0.0, 5.0]
    regression = [1.0, 1.0, 1.0, 5.0]
    existence = [0.1, 0.1, 0.1, 0.9]
    best_mae, threshold, use_gate = tune_hurdle_gate(regression, existence, labels)
    assert use_gate
    assert best_mae == 0.0
    assert 0.2 <= threshold <= 0.8
    assert nmae(regression, labels, train_std=2.0) == pytest.approx(0.375)
