"""Stage 1 runner: transform Stage 0 output into consensus decisions.

The runner owns the orchestration boundary. It accepts ``Stage0Output``
records, asks registrations for decisions (defaulting to a deterministic
decision derived from the Stage 0 score), and applies ``majority_consensus``.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Iterable

from .consensus import majority_consensus
from .schemas import Stage0Output, Stage1Decision, Stage1Result


def _default_decision(record: Stage0Output) -> Stage1Decision:
    """A deterministic fallback based on Stage 0's own score."""
    return Stage1Decision(
        record_id=record.record_id,
        event_id=record.event_id,
        confidence=record.similarity_score or 1.0,
        reasoning="inherited stage0 assignment",
        model="rule",
    )


class Stage1Runner:
    """Configurable Stage 1 orchestrator."""

    def __init__(self, voters: Iterable[Callable[[Stage0Output], Stage1Decision]] | None = None) -> None:
        self.voters: list[Callable[[Stage0Output], Stage1Decision]] = list(voters or [])
        if not self.voters:
            self.voters.append(_default_decision)

    def decide(self, record: Stage0Output) -> Stage1Result:
        decisions = [voter(record) for voter in self.voters]
        return majority_consensus(decisions)

    def run(self, records: Iterable[Stage0Output]) -> list[Stage1Result]:
        return [self.decide(record) for record in records]


def run_stage1(records: Iterable[Any], *, voters: Iterable[Callable[[Stage0Output], Stage1Decision]] | None = None) -> list[Stage1Result]:
    """Convenience entry point for Stage 1.

    ``records`` may be ``Stage0Output`` instances, ``Stage0Result``-like
    dataclasses, or plain dictionaries with the same keys.
    """
    runner = Stage1Runner(voters=voters)
    normalized: list[Stage0Output] = []
    for record in records:
        if isinstance(record, Stage0Output):
            normalized.append(record)
        elif isinstance(record, dict):
            normalized.append(Stage0Output(**record))
        elif is_dataclass(record):
            normalized.append(Stage0Output(**asdict(record)))
        else:
            normalized.append(Stage0Output(**vars(record)))
    return runner.run(normalized)
