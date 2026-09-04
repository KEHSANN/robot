"""Embedding provider used by Stage 0.

``EmbeddingService`` is a deterministic hashing fallback that produces a
stable vector without network access. Swap the backend (Gemini / Groq / OpenAI /
local model) by passing an object with ``embed(text) -> list[float]`` to
``EmbeddingService``. The Stage 0 pipeline only needs the ``embed`` method.
"""

from __future__ import annotations

import hashlib
import math
from typing import Callable

_DIM = 128


def hashed_vector(text: str, *, dimensions: int = _DIM) -> list[float]:
    """Produce a stable, deterministic feature vector from ``text``.

    Each character is hashed into a bucket; the resulting vector is L2
    normalised so ``cosine_similarity`` stays in ``[0, 1]``.
    """
    vector = [0.0] * dimensions
    for token in (text or "").lower().split():
        for char in token:
            digest = hashlib.sha256(char.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


class EmbeddingService:
    """Small wrapper around an embedding backend.

    If ``backend`` is omitted, ``hashed_vector`` is used. ``backend`` can be a
    callable or an object exposing ``embed(text)``.
    """

    def __init__(
        self,
        backend: Callable[[str], list[float]] | object | None = None,
        *,
        model_name: str = "fallback-hash-v1",
    ) -> None:
        self.backend = backend
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        if self.backend is None:
            return hashed_vector(text)
        if callable(self.backend):
            return list(self.backend(text))
        embedder = getattr(self.backend, "embed", None)
        if embedder is None:
            raise TypeError("embedding backend must be callable or expose .embed(text)")
        return list(embedder(text))
