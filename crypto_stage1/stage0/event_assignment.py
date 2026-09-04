"""Assign an incoming record to an existing or new event cluster.

Stage 0 produces *events*: cluster representatives with a stable id, a centroid
vector and the already assigned records. This module is the pure decision layer;
it does not know how embeddings are produced or where clusters are persisted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class EventCluster:
    """A lightweight event cluster used by Stage 0."""

    event_id: str
    label: str
    vector: Sequence[float] = field(default_factory=list)
    record_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_record(self, record_id: str, *,
                   update_vector: bool = False, new_vector: Sequence[float] | None = None) -> None:
        if record_id not in self.record_ids:
            self.record_ids.append(record_id)
        if update_vector and new_vector:
            self.vector = list(new_vector)


def _similarity_score(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    from .similarity import cosine_similarity

    return cosine_similarity(vector_a, vector_b)


def _text_score(text_a: str, text_b: str) -> float:
    from .similarity import text_similarity

    return text_similarity(text_a, text_b)


def assign_to_nearest_event(
    *,
    record_id: str,
    text: str,
    vector: Sequence[float] | None,
    events: list[EventCluster],
    threshold: float = 0.55,
) -> tuple[str, EventCluster | None]:
    """Return ``(event_id, cluster_or_None_for_new_event)``.

    ``vector`` is preferred when available. When there are no events, or when no
    event clears ``threshold``, the caller should create a new cluster using the
    returned ``EventCluster is None`` signal.
    """
    if not events:
        return record_id, None

    best_event: EventCluster | None = None
    best_score = -1.0
    for event in events:
        if vector:
            score = _similarity_score(vector, event.vector) if event.vector else _text_score(text, event.label)
        else:
            score = _text_score(text, event.label)
        if score > best_score:
            best_score = score
            best_event = event

    if best_event is not None and best_score >= threshold:
        return best_event.event_id, best_event
    return record_id, None
