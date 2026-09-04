"""Schemas shared between Stage 0 and Stage 1.

These are plain dataclasses so the package works with no third-party
application. When ``pydantic`` becomes a project requirement, ``Stage0Output``
can be mirrored as a Pydantic model without changing the runner API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Stage0Output:
    """What Stage 0 hands to Stage 1 for one record."""

    record_id: str
    normalised_title: str
    deduplicated: bool
    canonical_id: str
    event_id: str
    event_label: str
    facts: list[dict[str, Any]]
    vector: list[float]
    similarity_score: float
    source: str = ""


@dataclass
class Stage1Decision:
    """One model's decision about an event assignment."""

    record_id: str
    event_id: str
    confidence: float = 0.0
    reasoning: str = ""
    model: str = ""


@dataclass
class Stage1Result:
    """The agreed Stage 1 result for one record."""

    record_id: str
    event_id: str
    confidence: float
    votes: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
