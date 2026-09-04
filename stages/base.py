"""Shared machinery for the multi-model stages.

Every stage from 1 to 5 has the same skeleton: build a prompt, fan out to the
panel, parse each answer, count the votes, persist both the individual answers and
the aggregate. Only the question and the parser differ. That shape lives here so
the stage modules contain the *judgement* and nothing else.

Two invariants this module enforces for all of them:

* **Individual answers are always stored**, even the failures. Per-model accuracy
  is the whole point of the feedback loop, and a model that fails 30% of the time
  is a fact worth knowing about it — one that only shows up if the failures are
  recorded rather than dropped.

* **A stage never raises.** :class:`services.llm.LLMClient` returns failures as
  results, and the consensus quorum rule turns "nobody answered" into
  INCONCLUSIVE rather than a rejection. One dead provider must not look like
  editorial judgement.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from services.config import Settings, settings as global_settings
from services.consensus import Consensus, Verdict, Vote, build_votes, tally
from services.llm import LLMClient, LLMResult
from services.models import ModelSpec, panel_for_stage
from services.types import AssetImpact, AssetLink, FactSet, NewsItem
from database.base import Store

log = logging.getLogger(__name__)

#: ``parse`` contract shared by every stage: given the model's JSON object, return
#: its vote (or ``None`` if the answer is unusable) plus a normalised detail dict.
Parser = Callable[[dict], tuple[bool | None, dict]]


@dataclass
class StageContext:
    """Everything the stages accumulate about one article.

    Threaded through the whole chain so that Stage 5 and the final layer can see
    what the earlier stages concluded — the spec's requirement that the deep
    review gets the model disagreement, not just the winning answer.
    """

    news: NewsItem
    facts: FactSet
    news_id: int | None = None
    event_id: int | None = None
    event_context: str = ""
    source_count: int = 1

    stage1: Consensus | None = None
    stage2: Consensus | None = None
    links: list[AssetLink] = field(default_factory=list)
    market_wide: bool = False
    #: Stage 4 output, then overwritten in place by Stage 5 and the final layer,
    #: so ``impacts`` always holds the current best answer per asset.
    impacts: dict[str, AssetImpact] = field(default_factory=dict)
    #: Per-asset record of which stages ran and what they contributed.
    trail: dict[str, list[str]] = field(default_factory=dict)

    #: Stage that rejected the item, or None if it reached the end.
    dropped_at: int | None = None
    drop_reason: str = ""

    @property
    def stage2_score(self) -> float | None:
        return self.stage2.score if self.stage2 else None

    @property
    def event_type(self) -> str:
        return self.facts.event_type or "OTHER"

    def note(self, asset: str, text: str) -> None:
        self.trail.setdefault(asset, []).append(text)

    def drop(self, stage: int, reason: str) -> None:
        self.dropped_at = stage
        self.drop_reason = reason

    def headline(self) -> str:
        return self.facts.headline or self.news.title


class PanelRunner:
    """Runs one stage's question across the panel and records the result."""

    def __init__(
        self,
        client: LLMClient,
        store: Store,
        config: Settings | None = None,
    ) -> None:
        self.client = client
        self.store = store
        self.settings = config or global_settings
        self.llm_calls = 0
        self.llm_failures = 0

    async def run(
        self,
        ctx: StageContext,
        *,
        stage: int,
        system: str,
        user: str,
        parse: Parser,
        reject_votes: int,
        score_key: str | None = None,
        asset: str = "",
        specs: Sequence[ModelSpec] | None = None,
    ) -> tuple[Consensus, list[LLMResult]]:
        """Fan out, tally, persist. Never raises."""
        panel = tuple(specs) if specs is not None else panel_for_stage(stage)
        results = await self.client.fan_out(panel, system, user)

        self.llm_calls += len(results)
        self.llm_failures += sum(1 for result in results if not result.ok)

        votes = build_votes(results, parse)
        consensus = tally(
            votes,
            reject_votes=reject_votes,
            config=self.settings.consensus,
            score_key=score_key,
        )
        await self._persist(ctx, stage, asset, votes, consensus)

        if consensus.verdict is Verdict.INCONCLUSIVE:
            log.warning(
                "stage %d inconclusive for %r: %s (fail_open=%s)",
                stage, ctx.headline()[:60], consensus.note,
                self.settings.consensus.fail_open,
            )
        return consensus, results

    async def _persist(
        self,
        ctx: StageContext,
        stage: int,
        asset: str,
        votes: Sequence[Vote],
        consensus: Consensus,
    ) -> None:
        if ctx.news_id is None:
            return  # dry run: nothing to attach the rows to
        rows = []
        for vote in votes:
            if vote.result is None:
                continue
            row = vote.result.as_record()
            row["vote"] = _vote_label(vote)
            rows.append(row)
        try:
            await self.store.save_stage_results(
                ctx.news_id, ctx.event_id, stage, rows, asset=asset
            )
            await self.store.save_consensus(
                ctx.news_id, ctx.event_id, stage, consensus.as_record(), asset=asset
            )
        except Exception as exc:
            # Losing the audit trail must not lose the alert.
            log.error("could not persist stage %d results: %s", stage, exc)


def _vote_label(vote: Vote) -> str:
    if not vote.ok:
        return "NO_ANSWER"
    return "PASS" if vote.value else "REJECT"


# --------------------------------------------------------------------------- #
# aggregation helpers used by the asset-level stages
# --------------------------------------------------------------------------- #

def best_evidence(candidates: Sequence[str]) -> str:
    """Pick the most informative of several one-line explanations.

    Longest wins, capped — with a floor on how short is acceptable, because models
    that answer "positive news" are saying nothing and should lose to a model that
    named a mechanism.
    """
    cleaned = [text.strip() for text in candidates if text and text.strip()]
    if not cleaned:
        return ""
    return max(cleaned, key=len)[:600]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def majority_of(panel_size: int) -> int:
    """Votes needed for a majority, with a floor of 2.

    The floor matters: with a two-model panel a bare majority is one model, and one
    model is not a consensus. Below that the quorum rule in
    :func:`services.consensus.tally` takes over anyway.
    """
    return max(2, panel_size // 2 + 1)


__all__ = [
    "PanelRunner",
    "Parser",
    "StageContext",
    "best_evidence",
    "clamp",
    "majority_of",
]
