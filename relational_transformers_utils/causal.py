"""Experimental entropic causal discovery for explicitly categorical observations.

Implements the pairwise exogenous/total criteria and small-skeleton enumeration
in Compton et al., https://arxiv.org/html/2509.16463v1. All scores are in bits.
Scores use greedy approximate minimum-entropy coupling; they are neither causal
effect sizes nor probabilities that an edge is causal. See docs/causal.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from numbers import Integral

import numpy as np

from ._greedy_mec import coupling_masses

__all__ = [
    "DirectionResult",
    "GraphCandidate",
    "GraphResult",
    "entropic_direction",
    "greedy_coupling_entropy",
    "orient_graph",
]


def _entropy(p):
    p = np.asarray(p, dtype=float)
    positive = p[p > 0]
    return float(-np.sum(positive * np.log2(positive)))


def greedy_coupling_entropy(marginals: Sequence[Sequence[float]]) -> float:
    """Greedy upper bound on minimum coupling entropy, in bits.

    Each marginal must be a finite, nonnegative probability vector summing to
    one. Different support sizes are allowed. Inputs are never modified.
    """
    return _entropy(coupling_masses(marginals))


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _tolerance(value):
    if not np.isfinite(value) or value < 0:
        raise ValueError("tolerance must be finite and nonnegative")


def _encode(values):
    # Do not silently discretize floats or stringify mixed category types.
    codes, labels = [], {}
    for value in values:
        if not isinstance(value, (str, Integral, np.bool_)):
            raise ValueError(
                "categories must be strings, integers, or booleans; encode missing "
                "values and discretize continuous variables explicitly"
            )
        key = ("str" if isinstance(value, str) else "int", value)
        codes.append(labels.setdefault(key, len(labels)))
    if not codes:
        raise ValueError("observations must be nonempty")
    return np.asarray(codes, dtype=np.int64)


def _conditionals(data, node, parents, max_table_cells):
    if parents:
        _, groups = np.unique(data[:, parents], axis=0, return_inverse=True)
    else:
        groups = np.zeros(len(data), dtype=np.int64)
    # Bootstrap samples can omit categories: compress both axes each time.
    _, values = np.unique(data[:, node], return_inverse=True)
    width = int(values.max()) + 1
    height = int(groups.max()) + 1
    if width * height > max_table_cells:
        raise ValueError("conditional table exceeds max_table_cells; reduce categorical support")
    counts = np.bincount(groups * width + values, minlength=height * width).reshape(height, width)
    support = counts.sum(axis=1)
    return counts / support[:, None], int(support.min())


@dataclass(frozen=True)
class DirectionResult:
    """Pair scores and optional bootstrap stability (not causal confidence).

    ``margin_bits = reverse_bits - forward_bits``; positive favors names[0]
    causing names[1]. ``direction`` is None when within the tolerance.
    Bootstrap fractions are forward, reverse, unresolved; the interval is a
    percentile 95% interval for the score margin. ``min_context_count`` is the
    smallest observed category count across both variables.
    """

    names: tuple[str, str]
    criterion: str
    forward_bits: float
    reverse_bits: float
    forward_noise_bits: float
    reverse_noise_bits: float
    direction: tuple[str, str] | None
    n_samples: int
    min_context_count: int
    bootstrap_fractions: tuple[float, float, float] | None = None
    margin_interval_bits: tuple[float, float] | None = None

    @property
    def margin_bits(self) -> float:
        return self.reverse_bits - self.forward_bits


def _pair_scores(data, criterion, max_table_cells):
    conditional_y, nx = _conditionals(data, 1, [0], max_table_cells)
    conditional_x, ny = _conditionals(data, 0, [1], max_table_cells)
    forward_noise = greedy_coupling_entropy(conditional_y)
    reverse_noise = greedy_coupling_entropy(conditional_x)
    forward, reverse = forward_noise, reverse_noise
    if criterion == "total":
        forward += _entropy(np.bincount(data[:, 0]) / len(data))
        reverse += _entropy(np.bincount(data[:, 1]) / len(data))
    return forward, reverse, forward_noise, reverse_noise, min(nx, ny)


def entropic_direction(
    x: Sequence[str | int],
    y: Sequence[str | int],
    *,
    names: tuple[str, str] = ("X", "Y"),
    criterion: str = "exogenous",
    tolerance: float = 1e-9,
    bootstrap: int = 0,
    seed: int | None = None,
    max_table_cells: int = 1_000_000,
) -> DirectionResult:
    """Compare X→Y and Y→X from paired categorical observations.

    ``exogenous`` compares approximate H(E) in each direction. ``total`` adds
    the proposed cause's marginal entropy. Neither criterion tests independence;
    a direction is a model preference, not a discovered direct edge. Bootstrap
    resamples paired rows and assumes independent observations. No smoothing or
    imputation is performed. Tolerance is an absolute score difference in bits.
    """
    if criterion not in ("exogenous", "total"):
        raise ValueError("criterion must be 'exogenous' or 'total'")
    if (
        isinstance(names, str)
        or len(names) != 2
        or any(not isinstance(n, str) or not n for n in names)
    ):
        raise ValueError("names must contain two nonempty strings")
    if names[0] == names[1]:
        raise ValueError("variable names must be distinct")
    _tolerance(tolerance)
    _integer(bootstrap, "bootstrap", 0)
    _integer(max_table_cells, "max_table_cells", 1)
    x, y = _encode(x), _encode(y)
    if len(x) != len(y):
        raise ValueError("x and y must have equal lengths")
    data = np.column_stack((x, y))
    forward, reverse, fn, rn, minimum = _pair_scores(data, criterion, max_table_cells)
    margin = reverse - forward
    direction = None
    if abs(margin) > tolerance:
        direction = tuple(names) if margin > 0 else tuple(reversed(names))
    fractions = interval = None
    if bootstrap:
        rng = np.random.default_rng(seed)
        margins = []
        for _ in range(bootstrap):
            sample = data[rng.integers(len(data), size=len(data))]
            f, r, *_ = _pair_scores(sample, criterion, max_table_cells)
            margins.append(r - f)
        margins = np.asarray(margins)
        fractions = (
            float(np.mean(margins > tolerance)),
            float(np.mean(margins < -tolerance)),
            float(np.mean(np.abs(margins) <= tolerance)),
        )
        interval = tuple(float(v) for v in np.quantile(margins, [0.025, 0.975]))
    return DirectionResult(
        tuple(names),
        criterion,
        forward,
        reverse,
        fn,
        rn,
        direction,
        len(data),
        minimum,
        fractions,
        interval,
    )


@dataclass(frozen=True)
class GraphCandidate:
    """One acyclic orientation and its sum of approximate noise entropies."""

    edges: tuple[tuple[str, str], ...]
    score_bits: float


@dataclass(frozen=True)
class GraphResult:
    """Ranked acyclic orientations of a supplied skeleton.

    ``edges`` are common to every candidate within tolerance of the best score;
    ``unresolved`` contains the remaining undirected skeleton edges. These are
    score ties, not a Markov equivalence class or uncertainty interval. The first
    candidate is only one representative when tied. No edges are added/removed.
    """

    names: tuple[str, ...]
    candidates: tuple[GraphCandidate, ...]
    edges: tuple[tuple[str, str], ...]
    unresolved: tuple[tuple[str, str], ...]
    n_samples: int
    min_context_count: int


def _acyclic(size, edges):
    children = [[] for _ in range(size)]
    indegree = [0] * size
    for source, target in edges:
        children[source].append(target)
        indegree[target] += 1
    sources = [node for node in range(size) if indegree[node] == 0]
    visited = 0
    while sources:
        node = sources.pop()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                sources.append(child)
    return visited == size


def orient_graph(
    observations: Mapping[str, Sequence[str | int]],
    skeleton: Sequence[tuple[str, str]],
    *,
    required_edges: Sequence[tuple[str, str]] = (),
    tolerance: float = 1e-9,
    max_orientations: int = 4096,
    max_table_cells: int = 1_000_000,
) -> GraphResult:
    """Rank orientations of a small, caller-supplied undirected skeleton.

    Uses the paper's heuristic total entropy criterion, summing greedy coupling
    entropies for each node given its parents (marginal entropy for roots).
    Only observed parent configurations are scored. Required directed edges must
    belong to the skeleton and be acyclic. Enumeration is capped *before* scoring
    at 2**(number of unfixed edges), including cyclic orientations later rejected.
    This does not learn a skeleton or test conditional independence.
    """
    _tolerance(tolerance)
    _integer(max_orientations, "max_orientations", 1)
    _integer(max_table_cells, "max_table_cells", 1)
    names = tuple(observations)
    if not names or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("observations must map nonempty variable names to categories")
    columns = [_encode(observations[name]) for name in names]
    if len({len(c) for c in columns}) != 1:
        raise ValueError("all variables must have equal observation counts")
    data = np.column_stack(columns)
    indices = {name: i for i, name in enumerate(names)}

    def edge_indices(edge):
        if isinstance(edge, str) or len(edge) != 2 or any(n not in indices for n in edge):
            raise ValueError("edges must be pairs of known variable names")
        a, b = (indices[n] for n in edge)
        if a == b:
            raise ValueError("self edges are not allowed")
        return a, b

    undirected = set()
    for edge in skeleton:
        pair = tuple(sorted(edge_indices(edge)))
        if pair in undirected:
            raise ValueError("duplicate skeleton edge")
        undirected.add(pair)
    fixed = {}
    for edge in required_edges:
        directed = edge_indices(edge)
        pair = tuple(sorted(directed))
        if pair not in undirected:
            raise ValueError("required edges must belong to the skeleton")
        if pair in fixed and fixed[pair] != directed:
            raise ValueError("conflicting required directions")
        fixed[pair] = directed
    if not _acyclic(len(names), list(fixed.values())):
        raise ValueError("required edges contain a cycle")
    free = sorted(undirected - fixed.keys())
    if 2 ** len(free) > max_orientations:
        raise ValueError(
            "search exceeds max_orientations; supply more required edges or a smaller skeleton"
        )
    local_scores = {}
    minimum = len(data)
    candidates = []
    for choices in product((False, True), repeat=len(free)):
        edges = list(fixed.values()) + [
            pair[::-1] if reverse else pair for pair, reverse in zip(free, choices, strict=True)
        ]
        if not _acyclic(len(names), edges):
            continue
        score = 0.0
        for node in range(len(names)):
            parents = tuple(sorted(a for a, b in edges if b == node))
            key = node, parents
            if key not in local_scores:
                conditionals, support = _conditionals(data, node, list(parents), max_table_cells)
                local_scores[key] = greedy_coupling_entropy(conditionals)
                minimum = min(minimum, support)
            score += local_scores[key]
        named = tuple(sorted((names[a], names[b]) for a, b in edges))
        candidates.append(GraphCandidate(named, score))
    candidates.sort(key=lambda candidate: (candidate.score_bits, candidate.edges))
    best = candidates[0].score_bits
    tied = [set(c.edges) for c in candidates if c.score_bits <= best + tolerance]
    common = set.intersection(*tied)
    unresolved = tuple(
        (names[a], names[b])
        for a, b in sorted(undirected)
        if (names[a], names[b]) not in common and (names[b], names[a]) not in common
    )
    return GraphResult(
        names, tuple(candidates), tuple(sorted(common)), unresolved, len(data), minimum
    )
