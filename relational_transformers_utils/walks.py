"""Deterministic sampling primitives for context collection.

The RT-J reference sampler's observable row ordering depends on these exact
streams: a rand-0.9.1 ``StdRng``-compatible ChaCha12 stream (including
``seed_from_u64``'s PCG expansion and Canon integer sampling), rand's
``seq::index::sample`` selection strategy, and the vectorized peer-ranking
walk. Ports must match statement for statement; one extra or missing draw
shifts every subsequent choice.
"""
from __future__ import annotations

import numpy as np

__all__ = ["StdRng", "rand_sample", "reference_walk_counts",
           "stdrng_first_u64_batch", "U32", "U64"]

U32 = 0xFFFF_FFFF
U64 = 0xFFFF_FFFF_FFFF_FFFF


class StdRng:
    """rand 0.9.1 StdRng-compatible ChaCha12 stream.

    The reference sampler's observable ordering depends on this exact stream,
    including seed_from_u64's PCG expansion and Canon integer sampling.
    """

    def __init__(self, seed: int):
        state = seed & U64
        key = []
        for _ in range(8):
            state = (state * 6364136223846793005 + 11634580027462260723) & U64
            x = (((state >> 18) ^ state) >> 27) & U32
            rot = state >> 59
            key.append(((x >> rot) | (x << ((-rot) & 31))) & U32)
        self._key = key
        self._counter = 0
        self._buf: list[int] = []
        self._at = 0

    @staticmethod
    def _rotl(x: int, n: int) -> int:
        return ((x << n) | (x >> (32 - n))) & U32

    @classmethod
    def _quarter(cls, x: list[int], a: int, b: int, c: int, d: int):
        x[a] = (x[a] + x[b]) & U32; x[d] ^= x[a]; x[d] = cls._rotl(x[d], 16)
        x[c] = (x[c] + x[d]) & U32; x[b] ^= x[c]; x[b] = cls._rotl(x[b], 12)
        x[a] = (x[a] + x[b]) & U32; x[d] ^= x[a]; x[d] = cls._rotl(x[d], 8)
        x[c] = (x[c] + x[d]) & U32; x[b] ^= x[c]; x[b] = cls._rotl(x[b], 7)

    def _refill(self):
        out = []
        constants = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574]
        for block in range(4):
            counter = (self._counter + block) & U64
            initial = constants + self._key + [counter & U32, counter >> 32, 0, 0]
            x = initial.copy()
            for _ in range(6):
                self._quarter(x, 0, 4, 8, 12); self._quarter(x, 1, 5, 9, 13)
                self._quarter(x, 2, 6, 10, 14); self._quarter(x, 3, 7, 11, 15)
                self._quarter(x, 0, 5, 10, 15); self._quarter(x, 1, 6, 11, 12)
                self._quarter(x, 2, 7, 8, 13); self._quarter(x, 3, 4, 9, 14)
            out.extend((a + b) & U32 for a, b in zip(x, initial))
        self._counter = (self._counter + 4) & U64
        self._buf, self._at = out, 0

    def u32(self) -> int:
        if self._at >= len(self._buf):
            self._refill()
        value = self._buf[self._at]
        self._at += 1
        return value

    def u64(self) -> int:
        return self.u32() | (self.u32() << 32)

    def range(self, stop: int, start: int = 0) -> int:
        if not start < stop:
            raise ValueError("empty RNG range")
        width = stop - start
        # usize delegates to u32 for all graph sizes representable here.
        product = self.u32() * width
        result, low = product >> 32, product & U32
        if low > ((-width) & U32):
            new_hi = (self.u32() * width) >> 32
            if low + new_hi > U32:
                result += 1
        return start + result

    def range_inclusive(self, stop: int) -> int:
        return self.range(stop + 1)

    def uniform_range(self, stop: int) -> int:
        """Sampling from a constructed Uniform(0, stop), as rand's rejection
        index sampler does (distinct from random_range's Canon path)."""
        threshold = ((-stop) & U32) % stop
        while True:
            product = self.u32() * stop
            high, low = product >> 32, product & U32
            if low >= threshold:
                return high


def reference_walk_counts(node_count: int, offsets, neighbors, target: int,
                          eligible, seed: int, num_walks: int,
                          walk_length: int):
    """The peer-ranking graph walk, vectorized across walks.

    CSR graph with deterministic neighbor order; ``eligible`` marks nodes
    whose visits are counted. All walks advance one step per RNG call — a
    single ``num_walks``-wide draw from a raw PCG64 stream. Sampling is
    deterministic per seed (``PCG64.random_raw`` is a fixed stream; the
    modulo bias at context-sized degrees is < 2^-50).

    The draw protocol — per step, one vector over walks still alive, in walk
    order — is shared with :meth:`ContextGraph.assemble
    <relational_transformers_utils.graph.ContextGraph.assemble>`, so row and
    columnar context paths sample identical contexts.
    """
    counts = np.zeros(node_count, dtype=np.int64)
    off = np.asarray(offsets, dtype=np.int64)
    nbr = np.asarray(neighbors, dtype=np.int64)
    elig = np.asarray(eligible, dtype=bool)
    if num_walks <= 0 or walk_length <= 0:
        return counts.astype(np.uint32)
    bg = np.random.PCG64(seed & U64)
    cur = np.full(num_walks, target, dtype=np.int64)
    alive = np.ones(num_walks, dtype=bool)
    for _ in range(walk_length):
        live = cur[alive]
        visited = live[elig[live]]
        if visited.size:
            np.add.at(counts, visited, 1)
        degree = off[live + 1] - off[live]
        stepping = degree > 0
        if not stepping.any():
            break
        raw = bg.random_raw(int(stepping.sum()))
        step_from = live[stepping]
        r = (raw % degree[stepping].astype(np.uint64)).astype(np.int64)
        moved = nbr[off[step_from] + r]
        nxt = live.copy()
        nxt[stepping] = moved
        idx = np.flatnonzero(alive)
        cur[idx] = nxt
        alive[idx[~stepping]] = False
        if not alive.any():
            break
    return counts.astype(np.uint32)


def stdrng_first_u64_batch(seeds) -> np.ndarray:
    """First u64 from independently seeded rand-0.9.1 StdRng streams."""
    return np.asarray([StdRng(int(s) & U64).u64() for s in seeds],
                      dtype=np.uint64)


def rand_sample(rng: StdRng, length: int, amount: int) -> list[int]:
    """rand::seq::index::sample for the u32-sized cases used by contexts."""
    if not 0 <= amount <= length:
        raise ValueError("invalid sample size")
    if amount < 163:
        j = int(length >= 500_000)
        use_inplace = (amount > 11 and length < ([10.0, 70.0 / 9.0][j]
                       + [1.6, 8.0 / 45.0][j] * amount) * amount)
    else:
        j = int(length >= 500_000)
        use_inplace = length < [270.0, 330.0 / 9.0][j] * amount
    if use_inplace:
        indices = list(range(length))
        for i in range(amount):
            k = rng.range(length, i)
            indices[i], indices[k] = indices[k], indices[i]
        return indices[:amount]
    # Floyd is the reference path for the small fallback/BFS samples typical
    # of the 8K-cell evaluator. Rejection is only selected for huge samples.
    if amount < 163:
        indices: list[int] = []
        for j in range(length - amount, length):
            t = rng.range_inclusive(j)
            try:
                pos = indices.index(t)
            except ValueError:
                pass
            else:
                indices[pos] = j
            indices.append(t)
        return indices
    # rand's rejection sampler preserves unique draw order.
    chosen: set[int] = set()
    values: list[int] = []
    while len(values) < amount:
        value = rng.uniform_range(length)
        if value not in chosen:
            chosen.add(value); values.append(value)
    return values
