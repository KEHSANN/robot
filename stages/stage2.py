"""Stage 2 — market significance.

Deliberately a separate stage from relevance, and the separation is the point. "Is
this real crypto news?" and "would prices actually react?" are different questions
with different failure modes, and a model asked both at once conflates them: it
passes anything topical, or rejects anything undramatic.

A great deal of genuine, current, correctly-reported crypto news moves no price at
all. Filtering that out here is what keeps the expensive stages affordable, and the
funnel's whole economics — roughly 180 relevant items down to 60 worth analysing —
happens in this one vote.

The panel also returns a 0-100 score, aggregated by median. That score drives the
escalation decisions later, so an outlier model cannot promote a nothing event into
the NVIDIA layer.
"""

from __future__ import annotations

import logging

from services.consensus import Consensus, aggregate_numeric
from services.jsonparse import as_bool, as_float
from services.prompts import STAGE2_SYSTEM, stage2_user
from stages.base import PanelRunner, StageContext, clamp

log = logging.getLogger(__name__)

#: A model that says "not worth it" but scores it high has contradicted itself.
#: Above this score we trust the number over the boolean, because the score is what
#: the calibration bands in the prompt actually pin down.
SCORE_OVERRIDES_WORTH = 65.0

#: Likewise downward: "worth: true, score: 8" is a model hedging, not a signal.
SCORE_OVERRIDES_UNWORTH = 15.0

URGENCY_ORDER = {"low": 0, "medium": 1, "high": 2}


def parse(data: dict) -> tuple[bool | None, dict]:
    """Read one model's significance answer.

    A vote of True means the event is worth analysing.
    """
    worth = as_bool(data.get("worth"))
    score = as_float(data.get("score"))
    if score is not None:
        score = clamp(score, 0.0, 100.0)

    urgency = str(data.get("urgency") or "").strip().lower()
    if urgency not in URGENCY_ORDER:
        urgency = "medium"

    detail = {
        "worth": worth,
        "score": score,
        "urgency": urgency,
        "reason": str(data.get("reason") or "")[:400],
    }

    if worth is None and score is None:
        return None, detail

    if worth is None:
        # Only a number came back; the bands in the prompt make 41+ the "moderate
        # or better" boundary, so read the number as the answer.
        worth = score >= 41.0
        detail["worth"] = worth
        detail["note"] = "inferred from score"
    elif score is not None:
        resolved = _resolve_contradiction(worth, score)
        if resolved != worth:
            detail["note"] = f"score {score:.0f} overrides worth={worth}"
            worth = resolved
            detail["worth"] = worth

    return worth, detail


def _resolve_contradiction(worth: bool, score: float) -> bool:
    if not worth and score >= SCORE_OVERRIDES_WORTH:
        return True
    if worth and score <= SCORE_OVERRIDES_UNWORTH:
        return False
    return worth


async def run(runner: PanelRunner, ctx: StageContext) -> Consensus:
    """Decide whether the event can move a crypto price."""
    consensus, _ = await runner.run(
        ctx,
        stage=2,
        system=STAGE2_SYSTEM,
        user=stage2_user(ctx.news, ctx.facts, ctx.event_context),
        parse=parse,
        reject_votes=runner.settings.consensus.stage2_reject_votes,
        score_key="score",
    )
    ctx.stage2 = consensus

    if not consensus.passed:
        ctx.drop(2, _panel_reason(consensus))
        log.info("stage 2 NO (%s): %s", consensus.summary(), ctx.headline()[:70])
    else:
        log.info(
            "stage 2 YES (%s): %s",
            consensus.summary(), ctx.headline()[:70],
        )
    return consensus


def urgency(consensus: Consensus | None) -> str:
    """The panel's median urgency, for alert ordering."""
    if not consensus:
        return "medium"
    ranks = [
        URGENCY_ORDER.get(vote.detail.get("urgency", ""), 1)
        for vote in consensus.votes
        if vote.ok
    ]
    median = aggregate_numeric(ranks)
    if median is None:
        return "medium"
    for name, rank in URGENCY_ORDER.items():
        if rank == int(round(median)):
            return name
    return "medium"


def reason(consensus: Consensus | None) -> str:
    """The transmission mechanism the panel named, for the Stage 3 prompt."""
    if not consensus:
        return ""
    reasons = [
        vote.detail.get("reason", "")
        for vote in consensus.votes
        if vote.ok and vote.value is True and vote.detail.get("reason")
    ]
    return max(reasons, key=len)[:400] if reasons else ""


def _panel_reason(consensus: Consensus) -> str:
    reasons = [
        vote.detail.get("reason", "")
        for vote in consensus.votes
        if vote.ok and vote.value is False and vote.detail.get("reason")
    ]
    score = f", median score {consensus.score:.0f}" if consensus.score is not None else ""
    if not reasons:
        return f"{consensus.summary()}{score}"
    return f"{consensus.votes_reject}/{consensus.answered} models: {reasons[0]}{score}"


__all__ = ["parse", "reason", "run", "urgency"]
