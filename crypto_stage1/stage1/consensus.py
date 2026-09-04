"""Consensus logic for Stage 1.

A deterministic majority vote is used by default. When multiple sources vote,
the final confidence is the average confidence of the winning event.
"""

from __future__ import annotations

from collections import defaultdict

from .schemas import Stage1Decision, Stage1Result


def majority_consensus(decisions: list[Stage1Decision]) -> Stage1Result:
    """Combine ``decisions`` into a single Stage 1 result.

    Ties are resolved by the first event id that reached the top vote count.
    """
    if not decisions:
        raise ValueError("majority_consensus requires at least one decision")

    record_id = decisions[0].record_id
    weighted: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for decision in decisions:
        weighted[decision.event_id] += max(0.0, decision.confidence)
        counts[decision.event_id] += 1

    winner = max(counts, key=lambda event_id: (counts[event_id], weighted[event_id]))

    winning = [d for d in decisions if d.event_id == winner]
    confidence = (
        sum(max(0.0, d.confidence) for d in winning) / len(winning)
        if winning
        else 0.0
    )
    rationale = "; ".join(d.reasoning for d in winning if d.reasoning) or "majority consensus"

    return Stage1Result(
        record_id=record_id,
        event_id=winner,
        confidence=confidence,
        votes={event_id: float(count) for event_id, count in counts.items()},
        rationale=rationale,
    )
