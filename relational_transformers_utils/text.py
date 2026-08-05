"""Embedding caches for application-owned text encoders.

The model packages never encode text; applications do, and they call the
same encoder for the same strings across every context of a batch, a cohort,
and a session. :class:`CachedEncoder` puts a per-process cache and an
optional precomputed-embedding table in front of any encode callable. Cell
values and normalized class-label embeddings cache separately, since the
two spaces must never mix.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

__all__ = ["CachedEncoder", "PrecomputedEmbeddingError"]


class PrecomputedEmbeddingError(RuntimeError):
    """A strict precomputed table was asked for a string it does not carry."""


class CachedEncoder:
    """A per-process embedding cache over an application's encode function.

    ``encode_fn(texts, normalize) -> [n, d] array`` runs only for cache
    misses, deduplicated. ``install_precomputed`` loads preprocessing-time
    embeddings; with ``strict=True`` an unknown string is an error, never an
    implicit switch to a newly computed embedding distribution.
    """

    def __init__(self, encode_fn: Callable | None = None):
        self._encode_fn = encode_fn
        self._cache: dict[str, np.ndarray] = {}
        self._cache_norm: dict[str, np.ndarray] = {}
        self._strict_precomputed = False

    def install_precomputed(self, values: dict[str, np.ndarray], *,
                            strict: bool = True) -> None:
        self._cache.update({str(key): np.asarray(value, np.float32)
                            for key, value in values.items()})
        self._strict_precomputed = strict

    def _encode_missing(self, texts: Sequence[str],
                        normalize: bool) -> np.ndarray:
        if self._encode_fn is None:
            raise PrecomputedEmbeddingError(
                "no encoder configured and the precomputed table has no "
                "value for " + ", ".join(repr(t) for t in texts[:3]))
        return np.asarray(self._encode_fn(list(texts), normalize=normalize),
                          np.float32)

    def encode(self, texts: Sequence[str], *,
               normalize: bool = False) -> list[np.ndarray]:
        cache = self._cache_norm if normalize else self._cache
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        if missing:
            if self._strict_precomputed:
                raise PrecomputedEmbeddingError(
                    "precomputed embedding table has no value for "
                    + ", ".join(repr(value) for value in missing[:3]))
            encoded = self._encode_missing(missing, normalize)
            for t, e in zip(missing, encoded, strict=True):
                cache[t] = np.asarray(e, np.float32)
        return [cache[t] for t in texts]

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
