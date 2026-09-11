"""Bounded validation-based feature-group selection guided by causal candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .causal import GraphResult


@dataclass(frozen=True)
class FeatureSubsetScore:
    """One evaluated group subset, its validation metric, and proposal source."""

    groups: tuple[str, ...]
    score: float
    proposal: str


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Selection and complete validation trace; scores are not causal effects."""

    selected_groups: tuple[str, ...]
    validation_score: float
    excluded_unavailable: tuple[str, ...]
    evaluations: tuple[FeatureSubsetScore, ...]
    budget_exhausted: bool


class CausalFeatureSelector:
    """Select named column/join-path groups with a caller's validation evaluator.

    ``feature_groups`` maps each selectable group to graph variable names. A
    group is atomic: all its columns/joins must be available at prediction time
    for it to appear in the explicit ``available_groups`` allowlist. The target
    cannot belong to a feature group. Overlapping groups are allowed; the caller
    must deduplicate their columns/joins when building contexts.

    Proposals are the empty baseline, all available groups, and unions of target
    parents/ancestors across graph candidates within ``graph_tolerance`` bits of
    the best score. Membership uses any variable in a group. All groups remain
    eligible in the full baseline, even if the graph does not mark them causal.
    Greedy backward elimination then explores deletions from the preferred subset.

    ``select(evaluate)`` calls ``evaluate(tuple_of_group_names) -> float`` at most
    ``max_evaluations`` times. The evaluator owns context construction, prediction,
    and a fixed validation split. It must handle the empty subset. No model is
    fitted by this class. Keep final test data separate from selection.

    Metric tolerance is absolute: prefer fewer groups among evaluated subsets
    within ``score_tolerance`` of the best observed validation score. This bound
    is always relative to the global best, so tolerances do not accumulate over
    successive deletions. Equal-size ties use alphabetical group order.
    """

    def __init__(
        self,
        graph: GraphResult,
        *,
        target: str,
        feature_groups: Mapping[str, Sequence[str]],
        available_groups: Sequence[str],
        greater_is_better: bool = True,
        score_tolerance: float = 0.0,
        graph_tolerance: float = 1e-9,
        max_evaluations: int = 64,
    ) -> None:
        if target not in graph.names or not graph.candidates:
            raise ValueError("target must be in a graph with at least one candidate")
        if not isinstance(greater_is_better, bool):
            raise ValueError("greater_is_better must be boolean")
        for name, value in (
            ("score_tolerance", score_tolerance),
            ("graph_tolerance", graph_tolerance),
        ):
            if not np.isscalar(value) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            isinstance(max_evaluations, bool)
            or not isinstance(max_evaluations, (int, np.integer))
            or max_evaluations < 1
        ):
            raise ValueError("max_evaluations must be a positive integer")
        groups = {}
        for name, variables in feature_groups.items():
            if not isinstance(name, str) or not name:
                raise ValueError("feature group names must be nonempty strings")
            if isinstance(variables, str):
                raise ValueError("group variables must be a sequence, not a string")
            variables = tuple(variables)
            if not variables or any(v not in graph.names or v == target for v in variables):
                raise ValueError("group variables must be known non-target graph variables")
            groups[name] = frozenset(variables)
        if isinstance(available_groups, str):
            raise ValueError("available_groups must be a sequence of group names")
        available = tuple(sorted(set(available_groups)))
        if any(name not in groups for name in available):
            raise ValueError("available_groups contains an unknown group")
        parents, ancestors = set(), set()
        scores = [candidate.score_bits for candidate in graph.candidates]
        if not np.isfinite(scores).all():
            raise ValueError("graph candidate scores must be finite")
        best = min(scores)
        for candidate in graph.candidates:
            if candidate.score_bits > best + graph_tolerance:
                continue
            incoming = {node: set() for node in graph.names}
            for source, destination in candidate.edges:
                if source not in incoming or destination not in incoming or source == destination:
                    raise ValueError("graph candidate contains an invalid edge")
                incoming[destination].add(source)
            parents.update(incoming[target])
            reached = set()
            frontier = list(incoming[target])
            while frontier:
                node = frontier.pop()
                if node == target:
                    raise ValueError("graph candidate contains a cycle through target")
                if node not in reached:
                    reached.add(node)
                    frontier.extend(incoming[node])
            ancestors.update(reached)
        proposals = [
            ((), "empty"),
            (available, "all_available"),
            (tuple(g for g in available if groups[g] & parents), "causal_parents"),
            (tuple(g for g in available if groups[g] & ancestors), "causal_ancestors"),
        ]
        unique = {}
        for subset, reason in proposals:
            unique.setdefault(subset, reason)
        if len(unique) > max_evaluations:
            raise ValueError("max_evaluations is too small for the distinct initial proposals")
        self._proposals = tuple(unique.items())
        self._unavailable = tuple(sorted(set(groups) - set(available)))
        self._sign = 1 if greater_is_better else -1
        self._score_tolerance = float(score_tolerance)
        self._max_evaluations = max_evaluations

    def select(self, evaluate: Callable[[tuple[str, ...]], float]) -> FeatureSelectionResult:
        """Evaluate graph-guided proposals, then bounded greedy backward deletions.

        Every unique subset is evaluated once per call. Nonfinite/non-scalar
        metrics fail explicitly. Callback errors propagate. Budget exhaustion
        returns the best visited subset and an explicit flag; there is no claim
        of finding an optimum among all possible feature combinations.
        """
        if not callable(evaluate):
            raise TypeError("evaluate must be callable")
        visited = {}

        def measure(groups, reason):
            score = evaluate(groups)
            if isinstance(score, (bool, np.bool_)) or not np.isscalar(score):
                raise ValueError("evaluate must return a finite scalar metric")
            try:
                score = float(score)
            except (ValueError, TypeError) as exc:
                raise ValueError("evaluate must return a finite scalar metric") from exc
            if not np.isfinite(score):
                raise ValueError("evaluate must return a finite scalar metric")
            visited[groups] = FeatureSubsetScore(groups, score, reason)

        def preferred():
            best = max(self._sign * item.score for item in visited.values())
            eligible = [
                item
                for item in visited.values()
                if best - self._sign * item.score <= self._score_tolerance + 1e-15
            ]
            return min(eligible, key=lambda item: (len(item.groups), item.groups))

        for subset, reason in self._proposals:
            measure(subset, reason)
        exhausted = False
        while True:
            current = preferred()
            deletions = [
                current.groups[:i] + current.groups[i + 1 :] for i in range(len(current.groups))
            ]
            pending = [subset for subset in deletions if subset not in visited]
            if not pending:
                break
            for subset in pending:
                if len(visited) == self._max_evaluations:
                    exhausted = True
                    break
                measure(subset, "backward_elimination")
            if exhausted or preferred().groups == current.groups:
                break
        chosen = preferred()
        return FeatureSelectionResult(
            chosen.groups, chosen.score, self._unavailable, tuple(visited.values()), exhausted
        )
