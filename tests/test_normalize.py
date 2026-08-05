from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from relational_transformers.constants import SEM_BOOLEAN, SEM_DATETIME, SEM_NUMBER, SEM_TEXT

from relational_transformers_utils import (
    ColumnStats,
    NormalizationError,
    NormalizationMode,
    Row,
    bf16_as_f32,
    mean_std,
    normalize_sequence,
)
from relational_transformers_utils.schema import ColumnDef, Schema, TableDef, ValueType


def _schema() -> Schema:
    return Schema(
        tables=(
            TableDef(
                "customers",
                columns=(
                    ColumnDef("age", ValueType.NUMBER),
                    ColumnDef("joined", ValueType.DATETIME),
                    ColumnDef("name", ValueType.TEXT),
                ),
                primary_key="id",
            ),
        )
    )


def _rows() -> dict:
    return {
        "customers": [
            Row("customers", 1, {"age": 30.0, "joined": datetime(2020, 1, 1, tzinfo=timezone.utc)}),
            Row("customers", 2, {"age": 40.0, "joined": datetime(2021, 1, 1, tzinfo=timezone.utc)}),
            Row("customers", 3, {"age": 50.0}),
        ]
    }


def test_bf16_rounding_matches_bit_definition():
    values = np.asarray([1.0, 1.0000001, 3.14159265, -2.7182818], dtype=np.float32)
    rounded = bf16_as_f32(values)
    bits = rounded.view(np.uint32)
    assert (bits & 0xFFFF).max() == 0
    assert np.abs(rounded - values).max() < 0.02


def test_fit_uses_sample_std_for_columns_and_population_std_for_datetimes():
    stats = ColumnStats.fit(_schema(), _rows())
    mu, sd = stats.stats[("customers", "age")]
    assert mu == pytest.approx(40.0)
    assert sd == pytest.approx(np.asarray([30.0, 40.0, 50.0]).std(ddof=1))
    dt_mu, dt_sd = stats.dt
    days = [d.timestamp() / 86400.0 for d in (
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2021, 1, 1, tzinfo=timezone.utc),
    )]
    assert dt_mu == pytest.approx(np.mean(days))
    assert dt_sd == pytest.approx(np.asarray(days).std(ddof=0))


def test_column_stats_round_trips_through_dict():
    stats = ColumnStats.fit(_schema(), _rows()).with_task_values("churn", [0.0, 1.0, 1.0])
    restored = ColumnStats.from_dict(stats.to_dict())
    assert restored.stats == stats.stats
    assert restored.dt == pytest.approx(stats.dt)
    assert restored.task_stats == stats.task_stats


def test_task_stats_raise_without_fit():
    with pytest.raises(NormalizationError):
        ColumnStats({}).task("churn")


def test_zero_shot_normalization_is_batch_invariant():
    key = ("customers", "age")
    columns = [key, key, key]
    sems = [SEM_NUMBER] * 3
    values = [30.0, 40.0, 50.0]
    targets = [False, False, False]
    first = normalize_sequence(columns, sems, values, targets)
    second = normalize_sequence(columns, sems, values, targets)
    assert first == second
    assert first[0] == pytest.approx((30.0 - 40.0) / mean_std([30.0, 40.0, 50.0])[1])


def test_reference_normalization_reads_persisted_stats():
    stats = ColumnStats.fit(_schema(), _rows())
    normalized = normalize_sequence(
        [("customers", "age")], [SEM_NUMBER], [34.0], [False],
        mode=NormalizationMode.REFERENCE, column_stats=stats,
    )
    mu, sd = stats.stats[("customers", "age")]
    assert normalized[0] == pytest.approx((34.0 - mu) / sd)


def test_reference_normalization_requires_statistics():
    with pytest.raises(NormalizationError):
        normalize_sequence(
            [("orders", "amount")], [SEM_NUMBER], [5.0], [False],
            mode="reference", column_stats=ColumnStats({}),
        )


def test_targets_texts_and_nulls_become_zero():
    normalized = normalize_sequence(
        [("t", "target"), ("t", "text"), ("t", "missing"), ("t", "flag"),
         ("t", "when")],
        [SEM_NUMBER, SEM_TEXT, SEM_NUMBER, SEM_BOOLEAN, SEM_DATETIME],
        [99.0, "hello", None, True, datetime(2022, 6, 1, tzinfo=timezone.utc)],
        [True, False, False, False, False],
    )
    assert normalized[0] == 0.0
    assert normalized[1] == 0.0
    assert normalized[2] == 0.0
    assert normalized[3] == 0.0
