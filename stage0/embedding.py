"""Embeddings and similarity for Stage 0.

Only reached when both hashes miss, because an embedding call costs money and a
hash does not.

Similarity is cosine. When Postgres/pgvector is available the search happens in
the database; the pure-Python helpers here serve the in-memory store and the
tests, and they are also what verifies the vectors coming back from the provider
are usable at all.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence

from services.config import Stage0Settings, settings as global_settings
from services.llm import LLMClient
from services.types import NewsItem
from stage0.normalize import embedding_text

log = logging.getLogger(__name__)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, clamped to [-1, 1] against float drift."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    value = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    return max(-1.0, min(1.0, value))


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """Unit-length copy, so cosine reduces to a dot product downstream."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0.0:
        return list(vector)
    return [v / norm for v in vector]


def top_similar(
    query: Sequence[float],
    candidates: Iterable[tuple[int, Sequence[float]]],
    *,
    limit: int = 8,
    threshold: float = 0.0,
) -> list[tuple[int, float]]:
    """Rank ``(id, vector)`` pairs by similarity to ``query``."""
    scored = [
        (item_id, cosine_similarity(query, vector))
        for item_id, vector in candidates
        if vector
    ]
    scored = [pair for pair in scored if pair[1] >= threshold]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:limit]


class Embedder:
    """Embeds articles, with a small in-process cache.

    Returning ``None`` on failure is deliberate: if the embedding provider is
    down, Stage 0 should fall back to hash-and-identity comparison and let the
    article through as NEW rather than stall the pipeline. A false NEW costs one
    analysis; a stalled pipeline costs every alert.
    """

    def __init__(self, client: LLMClient, config: Stage0Settings | None = None) -> None:
        self.client = client
        self.config = config or global_settings.stage0
        self._cache: dict[str, list[float]] = {}
        self.calls = 0
        self.failures = 0

    async def embed_item(self, news: NewsItem) -> list[float] | None:
        text = embedding_text(news.title, news.body, self.config.embed_max_chars)
        if not text:
            return None
        if text in self._cache:
            return self._cache[text]

        self.calls += 1
        vector = await self.client.embed(text, dim=self.config.embed_dim)
        if vector is None:
            self.failures += 1
            log.warning("embedding unavailable for %r", news.title[:80])
            return None

        unit = normalize_vector(vector)
        # Bounded so a long `ingest` run cannot grow the cache without limit.
        if len(self._cache) < 4096:
            self._cache[text] = unit
        news.embedding = unit
        return unit


__all__ = [
    "Embedder",
    "cosine_similarity",
    "normalize_vector",
    "top_similar",
]
