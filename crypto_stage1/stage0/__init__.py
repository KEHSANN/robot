"""Stage 0: normalise, deduplicate and cluster incoming events."""

from __future__ import annotations

from .dedup import DedupIndex, content_hash
from .event_assignment import assign_to_nearest_event
from .fact_engine import extract_facts
from .normalizer import normalize_text
from .pipeline import Stage0Record, Stage0Result, run_stage0
from .similarity import cosine_similarity, text_similarity

__all__ = [
    "DedupIndex",
    "assign_to_nearest_event",
    "content_hash",
    "cosine_similarity",
    "extract_facts",
    "normalize_text",
    "run_stage0",
    "text_similarity",
    "Stage0Record",
    "Stage0Result",
]
