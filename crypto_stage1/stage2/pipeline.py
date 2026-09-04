"""Stage 2 pipeline.

This is the placeholder implementation for the second stage. It accepts Stage 1
results and produces a basic narrative/payload. Replace the heuristic body with
the real LLM/rendering implementation while keeping this signature stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Stage2Input:
    """Input for Stage 2."""

    record_id: str
    event_id: str
    confidence: float
    votes: dict[str, float] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class Stage2Output:
    """Output produced by Stage 2."""

    record_id: str
    event_id: str
    narrative: str
    risk_label: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Stage2Pipeline:
    """Heuristic Stage 2 pipeline."""

    def run(self, record: Stage2Input) -> Stage2Output:
        if record.confidence >= 0.75:
            risk_label = "high-confidence"
        elif record.confidence >= 0.5:
            risk_label = "medium"
        else:
            risk_label = "low-confidence"

        narrative = (
            f"{record.event_id}: event {record.event_id} "
            f"(confidence {record.confidence:.2f}, rationale: {record.rationale or 'n/a'})"
        )
        return Stage2Output(
            record_id=record.record_id,
            event_id=record.event_id,
            narrative=narrative,
            risk_label=risk_label,
            confidence=record.confidence,
            metadata={"votes": dict(record.votes)},
        )


def run_stage2(records, *, pipeline: Stage2Pipeline | None = None) -> list[Stage2Output]:
    """Run Stage 2 over an iterable of ``Stage2Input`` or mapping objects."""
    active = pipeline or Stage2Pipeline()
    outputs: list[Stage2Output] = []
    for record in records:
        if isinstance(record, Stage2Input):
            outputs.append(active.run(record))
        else:
            outputs.append(active.run(Stage2Input(**record)))
    return outputs
