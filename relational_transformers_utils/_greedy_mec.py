"""Entropy-only adaptation of ssokota/mec's NumPy greedy coupling.

Source: mec/approximation/algorithms.py at
https://github.com/ssokota/mec/tree/e56ad6619498914fe289d91516febb1fa3be050f
Changes: retain only coupling masses, validate strictly, use bits, omit Rust dispatch.
This is a greedy approximation, not an exact minimum-entropy solver.

MIT License

Copyright (c) 2024 Samuel Sokota

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np


def coupling_masses(marginals):
    """Return the masses of a feasible greedy coupling without Cartesian storage."""
    arrays = [np.asarray(m, dtype=np.float64) for m in marginals]
    if not arrays:
        raise ValueError("at least one marginal is required")
    for marginal in arrays:
        if marginal.ndim != 1 or not marginal.size:
            raise ValueError("marginals must be nonempty one-dimensional arrays")
        if not np.isfinite(marginal).all() or (marginal < 0).any():
            raise ValueError("marginals must contain finite nonnegative probabilities")
        if not np.isclose(marginal.sum(), 1.0, rtol=1e-7, atol=1e-12):
            raise ValueError("each marginal must sum to one")
    max_len = max(map(len, arrays))
    remaining = np.stack([np.pad(m / m.sum(), (0, max_len - len(m))) for m in arrays])
    rows = np.arange(len(arrays))
    masses = []
    while True:
        indices = remaining.argmax(axis=1)
        mass = remaining[rows, indices].min()
        if mass == 0:
            break
        remaining[rows, indices] -= mass
        masses.append(mass)
    if not np.isclose(sum(masses), 1.0, rtol=1e-10, atol=1e-12):
        raise RuntimeError("greedy coupling lost probability mass")
    return np.asarray(masses)
