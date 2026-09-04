"""Stage 1 — relevance.

The question is narrow on purpose: *is this a real, current, crypto-relevant
event?* Not whether it matters, not whether it is bullish. Stage 2 asks whether it
matters, and asking both at once measurably degrades both answers.

The vote is asymmetric. Two of five models objecting is enough to drop an item,
because what follows Stage 1 is still cheap — passing noise costs one more panel
call, while dropping a real event costs the alert entirely. The threshold is
``CONSENSUS_STAGE1_DROP_VOTES``.
"""

from __future__ import annotations

import logging

from services.consensus import Consensus
from services.jsonparse import as_bool
from services.prompts import STAGE1_SYSTEM, stage1_user
from stages.base import PanelRunner, StageContext

log = logging.getLogger(__name__)

#: Content types that are never a market event, whatever else the model said.
#: Kept to the unambiguous cases: "analysis" and "opinion" can carry a real event,
#: and the two booleans below already reject them when they do not.
DROP_TYPES = frozenset({"advertisement", "spam", "educational", "price_commentary"})


def parse(data: dict) -> tuple[bool | None, dict]:
    """Read one model's relevance answer.

    A vote of True means KEEP. Both booleans must be present and true, and the
    content type must not be outright noise.
    """
    specific = as_bool(data.get("specific_event"))
    current = as_bool(data.get("current_event"))
    content_type = str(data.get("type") or "").strip().lower().replace(" ", "_").replace("-", "_")

    detail = {
        "specific_event": specific,
        "current_event": current,
        "type": content_type,
        "reason": str(data.get("reason") or "")[:400],
    }

    if specific is None and current is None:
        return None, detail  # the model answered, but not the question

    keep = bool(specific) and bool(current) and content_type not in DROP_TYPES
    if not keep and not detail["reason"]:
        detail["reason"] = _implicit_reason(specific, current, content_type)
    return keep, detail


def _implicit_reason(specific: bool | None, current: bool | None, content_type: str) -> str:
    if content_type in DROP_TYPES:
        return f"content type is {content_type}"
    if specific is False:
        return "no specific event"
    if current is False:
        return "not a current event"
    return "rejected"


async def run(runner: PanelRunner, ctx: StageContext) -> Consensus:
    """Filter noise. Returns the consensus; ``ctx.stage1`` is set either way."""
    consensus, _ = await runner.run(
        ctx,
        stage=1,
        system=STAGE1_SYSTEM,
        user=stage1_user(ctx.news, ctx.facts),
        parse=parse,
        reject_votes=runner.settings.consensus.stage1_drop_votes,
    )
    ctx.stage1 = consensus

    if not consensus.passed:
        reason = _panel_reason(consensus)
        ctx.drop(1, reason)
        log.info("stage 1 REMOVE (%s): %s", consensus.summary(), ctx.headline()[:70])
    return consensus


def _panel_reason(consensus: Consensus) -> str:
    """The most common objection, so the drop is explainable later."""
    reasons = [
        vote.detail.get("reason", "")
        for vote in consensus.votes
        if vote.ok and vote.value is False and vote.detail.get("reason")
    ]
    if not reasons:
        return consensus.summary()
    return f"{consensus.votes_reject}/{consensus.answered} models: {reasons[0]}"


__all__ = ["DROP_TYPES", "parse", "run"]
