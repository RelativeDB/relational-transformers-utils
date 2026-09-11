import numpy as np
import pytest

from relational_transformers_utils.causal import (
    CausalFeatureSelector,
    GraphCandidate,
    GraphResult,
)


def graph(ambiguous=False):
    candidates = [GraphCandidate((("stock", "delay"), ("delay", "cancelled")), 1.0)]
    if ambiguous:
        candidates.append(GraphCandidate((("delay", "stock"), ("stock", "cancelled")), 1.0))
    return GraphResult(
        ("stock", "delay", "cancelled", "region"), tuple(candidates), (), (), 100, 10
    )


def selector(**kwargs):
    defaults = dict(
        target="cancelled",
        feature_groups={
            "inventory_join": ["stock"],
            "order_delay": ["delay"],
            "customer": ["region"],
        },
        available_groups=["inventory_join", "order_delay", "customer"],
    )
    defaults.update(kwargs)
    return CausalFeatureSelector(graph(), **defaults)


def test_unavailable_mediator_never_reaches_evaluator():
    seen = []

    def evaluate(groups):
        seen.append(groups)
        assert "order_delay" not in groups
        return 0.9 if "inventory_join" in groups else 0.5

    result = selector(available_groups=["inventory_join", "customer"]).select(evaluate)
    assert result.selected_groups == ("inventory_join",)
    assert result.excluded_unavailable == ("order_delay",)
    assert len(seen) == len(set(seen))
    assert set(seen) == {item.groups for item in result.evaluations}


def test_validation_can_retain_noncausal_predictor():
    result = selector().select(lambda groups: 0.9 if "customer" in groups else 0.5)
    assert result.selected_groups == ("customer",)
    assert any(item.proposal == "backward_elimination" for item in result.evaluations)


def test_fewer_groups_with_loss_metric_and_global_tolerance():
    scores = {
        (): 1.0,
        ("customer", "inventory_join", "order_delay"): 0.10,
        ("order_delay",): 0.12,
        ("inventory_join", "order_delay"): 0.11,
        ("inventory_join",): 0.14,
    }
    result = selector(greater_is_better=False, score_tolerance=0.025).select(scores.__getitem__)
    assert result.selected_groups == ("order_delay",)
    assert result.validation_score == 0.12
    assert result.validation_score - min(item.score for item in result.evaluations) <= 0.025


def test_tied_graphs_union_parents_instead_of_arbitrary_orientation():
    result = CausalFeatureSelector(
        graph(True),
        target="cancelled",
        feature_groups={"stock": ["stock"], "delay": ["delay"], "region": ["region"]},
        available_groups=["stock", "delay", "region"],
    ).select(lambda groups: len(groups))
    assert any(
        item.groups == ("delay", "stock") and item.proposal == "causal_parents"
        for item in result.evaluations
    )


def test_budget_is_strict_and_explicit():
    calls = []
    result = selector(max_evaluations=4).select(lambda groups: calls.append(groups) or len(groups))
    assert len(calls) == 4
    assert result.budget_exhausted
    with pytest.raises(ValueError, match="initial proposals"):
        selector(max_evaluations=1)


def test_empty_availability_and_repeat_calls():
    obj = selector(available_groups=[])
    a = obj.select(lambda groups: 0.5)
    b = obj.select(lambda groups: 0.8)
    assert a.selected_groups == ()
    assert len(a.evaluations) == 1
    assert a.validation_score == 0.5 and b.validation_score == 0.8
    assert not a.budget_exhausted


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": "missing"},
        {"feature_groups": {"bad": ["cancelled"]}},
        {"feature_groups": {"bad": ["absent"]}},
        {"feature_groups": {"bad": "stock"}},
        {"available_groups": ["missing"]},
        {"available_groups": "inventory_join"},
        {"score_tolerance": -1},
        {"graph_tolerance": np.nan},
        {"max_evaluations": 1.5},
        {"max_evaluations": True},
        {"greater_is_better": "yes"},
    ],
)
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        selector(**kwargs)


@pytest.mark.parametrize("value", [np.nan, np.inf, [0.8], True, None])
def test_invalid_validation_scores(value):
    with pytest.raises(ValueError, match="finite scalar"):
        selector().select(lambda groups: value)


def test_callback_failure_propagates():
    def fail(groups):
        raise RuntimeError("validation failed")

    with pytest.raises(RuntimeError, match="validation failed"):
        selector().select(fail)


def test_real_rt_prediction_callback(tmp_path):
    import torch
    from relational_transformers import RelationalBatch, RelationalTransformer, RTJModel
    from relational_transformers.checkpoints import save_checkpoint

    from relational_transformers_utils.metrics import brier_score

    torch.manual_seed(7)
    config = dict(num_blocks=1, d_model=8, d_text=4, num_heads=2, d_ff=16)
    save_checkpoint(RTJModel(**config), tmp_path, config)
    model = RelationalTransformer(tmp_path, device="cpu")
    rng = np.random.default_rng(7)
    contexts = [
        RelationalBatch.from_text_cells(rng.normal(size=(4, 8)).astype(np.float32), target=0)
        for _ in range(4)
    ]
    positions = {"inventory_join": [1], "order_delay": [2], "customer": [3]}

    def evaluate(groups):
        remove = [p for name, cells in positions.items() if name not in groups for p in cells]
        inputs = [context.ablate(remove) if remove else context for context in contexts]
        return brier_score(model.predict(inputs), [0, 1, 0, 1])

    result = selector(greater_is_better=False).select(evaluate)
    assert result.validation_score == pytest.approx(evaluate(result.selected_groups))
    assert result.validation_score == min(item.score for item in result.evaluations)


def test_removal_tolerance_does_not_accumulate():
    scores = {
        (): 1.0,
        ("customer", "inventory_join", "order_delay"): 0.10,
        ("inventory_join", "order_delay"): 0.11,
        ("order_delay",): 0.50,
        ("inventory_join",): 0.13,
    }
    result = selector(greater_is_better=False, score_tolerance=0.025).select(scores.__getitem__)
    # .13 is within tolerance of the .11 intermediate subset, but outside
    # tolerance of the global .10 reference, so it must not be selected.
    assert result.selected_groups == ("inventory_join", "order_delay")
    assert any(item.score == 0.13 for item in result.evaluations)
