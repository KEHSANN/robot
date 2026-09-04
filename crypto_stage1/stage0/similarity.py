"""Similarity primitives for Stage 0.

``cosine_similarity`` works on dense vectors (typically from an embedding
model). ``text_similarity`` is a dependency-free token-count fallback so tests
and cold-start code have something deterministic to run on.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens from ``text``."""
    return _TOKEN_RE.findall((text or "").lower())


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Return cosine similarity in ``[0, 1]`` for two real vectors.

    Empty or zero vectors compare as ``0.0``.
    """
    if len(left) != len(right) or not left:
        return 0.0

    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for a, b in zip(left, right):
        dot += a * b
        norm_left += a * a
        norm_right += b * b

    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (math.sqrt(norm_left) * math.sqrt(norm_right))))


def text_similarity(left: str, right: str) -> float:
    """Return cosine similarity over token frequency counts."""
    tokens_left = Counter(tokenize(left))
    tokens_right = Counter(tokenize(right))
    keys = set(tokens_left) | set(tokens_right)
    if not keys:
        return 0.0
    left_vec = [float(tokens_left.get(key, 0)) for key in keys]
    right_vec = [float(tokens_right.get(key, 0)) for key in keys]
    return cosine_similarity(left_vec, right_vec)


def similarity(
    left: str,
    right: str,
    *,
    left_vector: Sequence[float] | None = None,
    right_vector: Sequence[float] | None = None,
) -> float:
    """Similarity that uses embeddings when both are provided.

    Falls back to token similarity when either vector is missing or empty.
    """
    if left_vector and right_vector:
        return cosine_similarity(left_vector, right_vector)
    return text_similarity(left, right)
