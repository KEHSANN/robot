"""Vote counting for the multi-model stages.

Two rules matter beyond simple majority:

* **Asymmetric thresholds.** Stage 1 drops an item only when enough models
  actively object (default 2 of 5), because the cost of dropping real news is
  higher than the cost of passing noise to a cheap next stage.

* **Quorum.** If four of five models are unreachable, one lone NO must not read
  as a unanimous rejection. Below ``min_votes`` the verdict is INCONCLUSIVE, and
  ``fail_open`` decides whether it advances. Without this, a provider outage
  looks exactly like "the news was irrelevant".
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from services.config import ConsensusSettings, settings as global_settings
from services.llm import LLMResult
from services.models import ModelSpec


class Verdict(str, Enum):
    PASS = "PASS"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class Vote:
    """One model's answer, successful or not."""

    model: ModelSpec
    ok: bool
    value: bool | None = None
    detail: dict = field(default_factory=dict)
    result: LLMResult | None = None

    @property
    def label(self) -> str:
        return self.model.label

    @property
    def reason(self) -> str:
        if not self.ok:
            return self.result.error if self.result and self.result.error else "no answer"
        reason = self.detail.get("reason") or self.detail.get("evidence") or ""
        return str(reason)[:400]


@dataclass
class Consensus:
    """Aggregated panel decision."""

    verdict: Verdict
    votes_pass: int = 0
    votes_reject: int = 0
    votes_failed: int = 0
    votes: list[Vote] = field(default_factory=list)
    score: float | None = None
    #: Share of the models that answered which agreed with the majority side.
    agreement: float = 0.0
    #: Set when the verdict was forced by the quorum rule rather than the votes.
    note: str | None = None

    @property
    def answered(self) -> int:
        return self.votes_pass + self.votes_reject

    @property
    def total(self) -> int:
        return self.answered + self.votes_failed

    @property
    def passed(self) -> bool:
        """Whether the item advances to the next stage."""
        if self.verdict is Verdict.PASS:
            return True
        if self.verdict is Verdict.INCONCLUSIVE:
            return global_settings.consensus.fail_open
        return False

    def summary(self) -> str:
        parts = [f"{self.verdict.value} {self.votes_pass}/{self.answered}"]
        if self.votes_failed:
            parts.append(f"{self.votes_failed} no-answer")
        if self.score is not None:
            parts.append(f"score={self.score:.0f}")
        if self.note:
            parts.append(self.note)
        return " · ".join(parts)

    def as_record(self) -> dict:
        """Row shape for the ``stage_consensus`` table."""
        return {
            "verdict": self.verdict.value,
            "votes_pass": self.votes_pass,
            "votes_reject": self.votes_reject,
            "votes_failed": self.votes_failed,
            "agreement": round(self.agreement, 3),
            "score": self.score,
            "note": self.note,
            "detail": {
                vote.model.id: {
                    "ok": vote.ok,
                    "value": vote.value,
                    "reason": vote.reason,
                    "latency": round(vote.result.latency, 2) if vote.result else None,
                    "key_index": vote.result.key_index if vote.result else None,
                }
                for vote in self.votes
            },
        }


def build_votes(
    results: Sequence[LLMResult],
    parse: Callable[[dict], tuple[bool | None, dict]],
) -> list[Vote]:
    """Turn raw model results into votes using a stage-specific parser.

    ``parse`` receives the model's JSON object and returns
    ``(vote_or_None, normalised_detail)``. Returning ``None`` marks the answer
    unusable — the model replied but not in a way we can score.
    """
    votes: list[Vote] = []
    for result in results:
        if not result.ok or result.data is None:
            votes.append(Vote(model=result.model, ok=False, result=result))
            continue
        try:
            value, detail = parse(result.data)
        except Exception as exc:  # a malformed but parseable object
            votes.append(
                Vote(
                    model=result.model,
                    ok=False,
                    detail={"reason": f"unreadable answer: {exc}"},
                    result=result,
                )
            )
            continue

        if value is None:
            votes.append(
                Vote(model=result.model, ok=False, detail=detail, result=result)
            )
        else:
            votes.append(
                Vote(model=result.model, ok=True, value=value, detail=detail, result=result)
            )
    return votes


def tally(
    votes: Sequence[Vote],
    *,
    reject_votes: int,
    config: ConsensusSettings | None = None,
    score_key: str | None = None,
) -> Consensus:
    """Apply the reject-threshold rule with a quorum guard."""
    config = config or global_settings.consensus

    passes = sum(1 for vote in votes if vote.ok and vote.value is True)
    rejects = sum(1 for vote in votes if vote.ok and vote.value is False)
    failed = sum(1 for vote in votes if not vote.ok)
    answered = passes + rejects

    consensus = Consensus(
        verdict=Verdict.INCONCLUSIVE,
        votes_pass=passes,
        votes_reject=rejects,
        votes_failed=failed,
        votes=list(votes),
    )

    if score_key:
        consensus.score = aggregate_numeric(
            [
                vote.detail.get(score_key)
                for vote in votes
                if vote.ok and vote.detail.get(score_key) is not None
            ]
        )

    if answered:
        consensus.agreement = max(passes, rejects) / answered

    if answered < config.min_votes:
        consensus.note = (
            f"quorum not met ({answered} of {len(votes)} models answered; "
            f"need {config.min_votes})"
        )
        return consensus

    consensus.verdict = Verdict.REJECT if rejects >= reject_votes else Verdict.PASS
    return consensus


def aggregate_numeric(values: Sequence[Any], method: str = "median") -> float | None:
    """Combine numeric model outputs, defaulting to the median.

    The median is deliberate: one model returning 0 or 100 should not drag the
    panel's score with it.
    """
    numbers = [float(v) for v in values if isinstance(v, (int, float))]
    if not numbers:
        return None
    if method == "mean":
        return statistics.fmean(numbers)
    return statistics.median(numbers)


def majority_label(labels: Sequence[str]) -> tuple[str | None, float]:
    """Most common label plus the share of votes it received."""
    cleaned = [label for label in labels if label]
    if not cleaned:
        return None, 0.0
    counts: dict[str, int] = {}
    for label in cleaned:
        counts[label] = counts.get(label, 0) + 1
    winner = max(counts, key=lambda k: (counts[k], k))
    return winner, counts[winner] / len(cleaned)


__all__ = [
    "Consensus",
    "Verdict",
    "Vote",
    "aggregate_numeric",
    "build_votes",
    "majority_label",
    "tally",
]
