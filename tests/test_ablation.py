from __future__ import annotations

import numpy as np
import pytest
import torch
from relational_transformers import (
    RelationalBatch,
    RelationalExample,
    RelationalTransformer,
    RTJModel,
)
from relational_transformers.checkpoints import save_checkpoint
from relational_transformers.constants import SEM_NUMBER, SEM_TEXT

from relational_transformers_utils import AblationEvaluator

D_TEXT = 4


def _context(seed: int) -> RelationalBatch:
    rng = np.random.default_rng(seed)
    size = 5
    text = rng.normal(size=(1, size, D_TEXT)).astype(np.float32)
    columns = rng.normal(size=(1, size, D_TEXT)).astype(np.float32)
    return RelationalBatch(
        node_idxs=[[7, 7, 7, 8, 8]],
        f2p_nbr_idxs=np.full((1, size, 5), -1, dtype=np.int64),
        col_name_idxs=[[0, 1, 2, 3, 4]],
        table_name_idxs=[[0, 0, 0, 1, 1]],
        is_padding=np.zeros((1, size), dtype=bool),
        sem_types=[[SEM_NUMBER, SEM_NUMBER, SEM_TEXT, SEM_NUMBER, SEM_TEXT]],
        is_targets=[[True, False, False, False, False]],
        number_values=rng.normal(size=(1, size)).astype(np.float32),
        datetime_values=np.zeros((1, size), dtype=np.float32),
        boolean_values=np.zeros((1, size), dtype=np.float32),
        text_values=text,
        col_name_values=columns,
    )


@pytest.fixture
def model(tmp_path) -> RelationalTransformer:
    torch.manual_seed(7)
    tiny = RTJModel(num_blocks=2, d_model=12, d_text=D_TEXT, num_heads=3, d_ff=24)
    save_checkpoint(tiny, tmp_path, {"model": {
        "num_blocks": 2, "d_model": 12, "d_text": D_TEXT, "num_heads": 3, "d_ff": 24,
    }})
    return RelationalTransformer(tmp_path, device="cpu")


def test_ablation_reports_deltas_per_named_group(model):
    examples = [RelationalExample(_context(1), 1.0), RelationalExample(_context(2), 0.0)]
    metrics = AblationEvaluator(examples, {"orders": [3, 4]})(model)
    assert set(metrics) == {"orders_mean_delta", "orders_mean_absolute_delta"}
    assert metrics["orders_mean_absolute_delta"] >= abs(metrics["orders_mean_delta"])


def test_ablating_cells_changes_the_prediction(model):
    example = RelationalExample(_context(3), 1.0)
    metrics = AblationEvaluator([example], {"all_context": [1, 2, 3, 4]})(model)
    assert metrics["all_context_mean_absolute_delta"] > 0.0


def test_requires_examples_and_ablations(model):
    with pytest.raises(ValueError):
        AblationEvaluator([], {"orders": [1]})(model)
    with pytest.raises(ValueError):
        AblationEvaluator([RelationalExample(_context(4), 1.0)], {})(model)
