"""Dataset-level ablation measurement over relational contexts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from relational_transformers import RelationalExample


@dataclass
class AblationEvaluator:
    """Measure prediction deltas for caller-defined groups of cell positions.

    Each named ablation lists cell positions to remove through
    :meth:`RelationalBatch.ablate <relational_transformers.RelationalBatch.ablate>`.
    Calling the evaluator with a model returns ``{name}_mean_delta`` and
    ``{name}_mean_absolute_delta`` per ablation, computed on identity-activation
    outputs.
    """

    examples: Sequence[RelationalExample]
    ablations: Mapping[str, Sequence[int]]

    def __call__(self, model) -> dict[str, float]:
        if not self.examples:
            raise ValueError("evaluation requires at least one example")
        if not self.ablations:
            raise ValueError("at least one named ablation is required")
        inputs = [
            model._batch(example.input, target=example.target) for example in self.examples
        ]
        baseline = np.asarray(model.predict(inputs, activation="identity")).reshape(-1)
        metrics = {}
        for name, positions in self.ablations.items():
            ablated = [batch.ablate(positions) for batch in inputs]
            changed = np.asarray(model.predict(ablated, activation="identity")).reshape(-1)
            difference = changed - baseline
            metrics[f"{name}_mean_delta"] = float(difference.mean())
            metrics[f"{name}_mean_absolute_delta"] = float(np.abs(difference).mean())
        return metrics
