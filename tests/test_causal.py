"""Contract and mathematical checks; standalone experiments live outside this repo."""

import numpy as np
import pytest

from relational_transformers_utils.causal import (
    entropic_direction,
    greedy_coupling_entropy,
    orient_graph,
)


def test_coupling_known_solution_and_inputs_unchanged():
    a = np.array([0.75, 0.25])
    b = np.array([0.6, 0.3, 0.1])
    masses = np.array([0.6, 0.25, 0.1, 0.05])
    assert greedy_coupling_entropy([a, b, [1]]) == pytest.approx(-sum(masses * np.log2(masses)))
    np.testing.assert_array_equal(a, [0.75, 0.25])
    np.testing.assert_array_equal(b, [0.6, 0.3, 0.1])
    assert greedy_coupling_entropy([[0.5, 0.5]] * 100) == pytest.approx(1)


@pytest.mark.parametrize(
    "marginals",
    [[], [[]], [[-0.1, 1.1]], [[np.nan, 1]], [[np.inf]], [[0, 0]], [[0.4, 0.4]], [[[1]]]],
)
def test_invalid_marginals(marginals):
    with pytest.raises(ValueError):
        greedy_coupling_entropy(marginals)


def test_entropy_bounds_and_permutation_invariance():
    rng = np.random.default_rng(8)
    for _ in range(20):
        marginals = [rng.dirichlet(np.ones(k)) for k in [2, 4, 6]]
        entropies = [-sum(p * np.log2(p)) for p in marginals]
        score = greedy_coupling_entropy(marginals)
        assert max(entropies) - 1e-12 <= score <= sum(entropies) + 1e-12
        shuffled = [rng.permutation(p) for p in marginals[::-1]]
        assert greedy_coupling_entropy(shuffled) == pytest.approx(score)


def test_many_to_one_direction_and_score_semantics():
    x = np.tile(np.arange(4), 50)
    y = x // 2
    noise = entropic_direction(x, y, names=("cause", "effect"))
    assert noise.direction == ("cause", "effect")
    assert noise.forward_noise_bits == 0
    assert noise.reverse_noise_bits == pytest.approx(1)
    assert noise.margin_bits == pytest.approx(1)
    total = entropic_direction(x, y, criterion="total")
    assert total.forward_bits == pytest.approx(2)
    assert total.reverse_bits == pytest.approx(2)
    assert total.direction is None


def test_pair_relabel_swap_and_bootstrap_reproducibility():
    x = np.tile(np.arange(6), 20)
    y = x // 2
    a = entropic_direction(x, y, bootstrap=12, seed=10)
    b = entropic_direction([f"cat-{v}" for v in y], x, bootstrap=12, seed=10)
    assert a.margin_bits == pytest.approx(-b.margin_bits)
    assert a.bootstrap_fractions[0] == b.bootstrap_fractions[1]
    assert sum(a.bootstrap_fractions) == pytest.approx(1)
    assert a == entropic_direction(x, y, bootstrap=12, seed=10)
    assert a.margin_interval_bits[0] > 0


def test_bijection_constant_and_independent_ties():
    assert entropic_direction([0, 1] * 20, ["a", "b"] * 20).direction is None
    assert entropic_direction([0] * 10, [1] * 10).direction is None
    x, y = np.indices((3, 3)).reshape(2, -1)
    assert entropic_direction(x, y, criterion="total").direction is None


@pytest.mark.parametrize(
    "x,y,kwargs",
    [
        ([], [], {}),
        ([1], [1, 2], {}),
        ([None], [1], {}),
        ([1.0], [1], {}),
        ([np.nan], [1], {}),
        ([[1]], [1], {}),
        ([0], [0], {"criterion": "bad"}),
        ([0], [0], {"names": ("X", "X")}),
        ([0], [0], {"names": ("", "Y")}),
        ([0], [0], {"bootstrap": -1}),
        ([0], [0], {"bootstrap": 1.5}),
        ([0], [0], {"tolerance": float("nan")}),
        ([0], [0], {"tolerance": -1}),
        ([0, 1], [0, 1], {"max_table_cells": 3}),
    ],
)
def test_invalid_pair_inputs(x, y, kwargs):
    with pytest.raises(ValueError):
        entropic_direction(x, y, **kwargs)


def test_graph_pair_score_matches_total_criterion():
    x = [0, 0, 0, 1, 1, 2, 2, 2]
    y = [0, 0, 1, 1, 1, 0, 0, 0]
    pair = entropic_direction(x, y, criterion="total")
    graph = orient_graph({"X": x, "Y": y}, [("X", "Y")])
    scores = {c.edges: c.score_bits for c in graph.candidates}
    assert scores[(("X", "Y"),)] == pytest.approx(pair.forward_bits)
    assert scores[(("Y", "X"),)] == pytest.approx(pair.reverse_bits)


def test_graph_ties_and_constraints():
    observations = {name: [0, 1] * 30 for name in "XYZ"}
    skeleton = [("X", "Y"), ("Y", "Z"), ("X", "Z")]
    result = orient_graph(observations, skeleton)
    assert len(result.candidates) == 6  # Eight orientations minus the two cycles.
    assert result.edges == ()
    assert len(result.unresolved) == 3
    fixed = orient_graph(observations, skeleton, required_edges=[("X", "Y"), ("Y", "Z")])
    assert fixed.edges == (("X", "Y"), ("X", "Z"), ("Y", "Z"))
    assert fixed.unresolved == ()
    # Enumeration bound accounts for fixed directions.
    orient_graph(observations, skeleton, required_edges=list(fixed.edges), max_orientations=1)


def test_empty_skeleton_keeps_isolated_variables():
    result = orient_graph({"X": [0, 1], "Y": [0, 0]}, [])
    assert len(result.candidates) == 1
    assert result.candidates[0].score_bits == pytest.approx(1)
    assert result.edges == result.unresolved == ()


@pytest.mark.parametrize(
    "skeleton,kwargs",
    [
        ([("X", "Q")], {}),
        ([("X", "X")], {}),
        (["XY"], {}),
        ([("X", "Y"), ("Y", "X")], {}),
        ([("X", "Y")], {"required_edges": [("X", "Z")]}),
        ([("X", "Y")], {"required_edges": [("X", "Y"), ("Y", "X")]}),
        (
            [("X", "Y"), ("Y", "Z"), ("X", "Z")],
            {"required_edges": [("X", "Y"), ("Y", "Z"), ("Z", "X")]},
        ),
        ([("X", "Y")], {"max_orientations": 1}),
        ([("X", "Y")], {"max_table_cells": 1}),
    ],
)
def test_invalid_graph_constraints(skeleton, kwargs):
    with pytest.raises(ValueError):
        orient_graph({"X": [0, 1], "Y": [0, 1], "Z": [0, 1]}, skeleton, **kwargs)


def test_graph_invalid_observations():
    for observations in [{}, {"X": []}, {"X": [1], "Y": [1, 2]}, {"": [1]}]:
        with pytest.raises(ValueError):
            orient_graph(observations, [])


def test_independence_is_not_a_noise_score_test():
    x, y = np.indices((8, 2)).reshape(2, -1)
    assert entropic_direction(x, y).margin_bits == pytest.approx(2)
    assert entropic_direction(x, y, criterion="total").direction is None


def test_graph_order_invariance():
    rng = np.random.default_rng(29)
    data = rng.integers(0, 4, size=(1200, 3))
    obs = dict(zip("XYZ", data.T, strict=True))
    skeleton = [("X", "Y"), ("Y", "Z")]
    a = orient_graph(obs, skeleton)
    order = rng.permutation(len(data))
    reordered = {name: obs[name][order] for name in "ZYX"}
    b = orient_graph(reordered, [("Z", "Y"), ("Y", "X")])
    a_scores = {c.edges: c.score_bits for c in a.candidates}
    b_scores = {c.edges: c.score_bits for c in b.candidates}
    assert a_scores == pytest.approx(b_scores)
    assert a.edges == b.edges


def test_bootstrap_omitted_states_and_full_tie():
    result = entropic_direction(["rare", "common"], [True, False], bootstrap=30, seed=3)
    assert result.bootstrap_fractions == (0, 0, 1)
    assert result.margin_interval_bits == (0, 0)
