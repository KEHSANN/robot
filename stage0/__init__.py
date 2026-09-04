"""Stage 0 — event detection, memory and deduplication.

The gate that turns ~1000 articles a day into ~300 events. Everything downstream
is priced per event, so this is where the system's economics are decided.

Public entry point is :class:`Stage0Pipeline`; the leaf modules are importable on
their own because the tests exercise them directly and because a few of them
(:mod:`stage0.normalize`, :mod:`stage0.identity`) are useful to the later stages.
"""

from __future__ import annotations

from stage0.embedding import Embedder, cosine_similarity, top_similar
from stage0.facts import FactExtractor, heuristic_facts
from stage0.hashing import exact_hash, fingerprints, norm_hash, title_hash, url_hash
from stage0.identity import (
    StateDiff,
    compare_state,
    identity_similarity,
    merge_state,
    same_event,
)
from stage0.normalize import (
    canonical_url,
    embedding_text,
    normalize_for_hash,
    normalize_text,
    normalize_title,
)
from stage0.pipeline import Stage0Pipeline, Stage0Result, Stage0Stats

__all__ = [
    "Embedder",
    "FactExtractor",
    "Stage0Pipeline",
    "Stage0Result",
    "Stage0Stats",
    "StateDiff",
    "canonical_url",
    "compare_state",
    "cosine_similarity",
    "embedding_text",
    "exact_hash",
    "fingerprints",
    "heuristic_facts",
    "identity_similarity",
    "merge_state",
    "norm_hash",
    "normalize_for_hash",
    "normalize_text",
    "normalize_title",
    "same_event",
    "title_hash",
    "top_similar",
    "url_hash",
]
