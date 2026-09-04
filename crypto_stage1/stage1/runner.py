"""Stage 1 runner: transform Stage 0 output into consensus decisions.

The runner owns the orchestration boundary. It accepts ``Stage0Output``
records, asks registrations for decisions (defaulting to a deterministic
decision derived from the Stage 0 score), and applies ``majority_consensus``.
An optional ``LLMClient`` adds one model-based voter to each decision round.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Iterable

from services.llm import LLMClient

from .consensus import majority_consensus
from .llm_voter import build_event_candidates, llm_event_decision
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


def _coerce_record(record: Any) -> Stage0Output:
    if isinstance(record, Stage0Output):
        return record
    if isinstance(record, dict):
        return Stage0Output(**record)
    if is_dataclass(record):
        return Stage0Output(**asdict(record))
    return Stage0Output(**vars(record))


class Stage1Runner:
    """Configurable Stage 1 orchestrator."""

    def __init__(
        self,
        voters: Iterable[Callable[[Stage0Output], Stage1Decision]] | None = None,
        *,
        model_client: LLMClient | None = None,
        use_model_voter: bool = True,
    ) -> None:
        self.voters: list[Callable[[Stage0Output], Stage1Decision]] = list(voters or [])
        if not self.voters:
            self.voters.append(_default_decision)
        self.model_client = model_client
        self.use_model_voter = use_model_voter

    def decide(self, record: Stage0Output, candidates: list[tuple[str, str]]) -> Stage1Result:
        decisions = [voter(record) for voter in self.voters]
        if (
            self.use_model_voter
            and self.model_client is not None
            and self.model_client.configured
        ):
            decisions.append(llm_event_decision(self.model_client, record, candidates))
        return majority_consensus(decisions)

    def run(self, records: Iterable[Any]) -> list[Stage1Result]:
        normalized = [_coerce_record(record) for record in records]
        candidates = build_event_candidates(normalized)
        return [self.decide(record, candidates) for record in normalized]


def run_stage1(
    records: Iterable[Any],
    *,
    voters: Iterable[Callable[[Stage0Output], Stage1Decision]] | None = None,
    model_client: LLMClient | None = None,
    use_model_voter: bool = True,
) -> list[Stage1Result]:
    """Convenience entry point for Stage 1.

    ``records`` may be ``Stage0Output`` instances, ``Stage0Result``-like
    dataclasses, or plain dictionaries with the same keys.
    """
    runner = Stage1Runner(
        voters=voters,
        model_client=model_client,
        use_model_voter=use_model_voter,
    )
    return runner.run(records)
