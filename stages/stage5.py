"""Stage 5 — deep cross-check.

An audit, not a second opinion. The prompt tells the panel to find what Stage 4 got
wrong rather than to agree with it, and this module is built around the asymmetry
that follows: **an audit that finds problems can never raise confidence.**

That rule is the whole point. Without it, running an extra stage on the events that
matter most would systematically inflate certainty — five models saying "yes" then
five more saying "yes, but…" would come out more confident than the first five
alone, when the second panel's actual finding was doubt.

Two more rules follow from the same logic:

**A direction flip needs its own majority.** Auditors who reject Stage 4's direction
but scatter across three alternatives have established that Stage 4 might be wrong,
not what the right answer is. That is MIXED with low confidence, not a flip.

**``priced_in`` shrinks the move.** This is the correction Stage 4 structurally
cannot make: it sees one article, while Stage 5 sees the event's whole history. If
prior updates already moved the price, what remains is the surprise, not the news.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from services.consensus import Consensus, aggregate_numeric, majority_label
from services.jsonparse import as_bool, as_float, as_probability, as_str_list
from services.models import panel_for_stage
from services.prompts import STAGE5_SYSTEM, stage5_user
from services.types import AssetImpact, Direction, Magnitude
from stages.base import PanelRunner, StageContext, best_evidence, clamp, majority_of
from stages.stage4 import (
    aggregate_magnitude,
    disagreement_summary,
    normalise_range,
    snap_horizon,
)

log = logging.getLogger(__name__)

#: How much of the expected move survives when the panel says the event is already
#: priced in. Not zero: "priced in" is a judgement about degree, and the market
#: rarely has it fully discounted.
PRICED_IN_RETENTION = 0.45

#: A flip needs this share of the auditors behind the same alternative direction.
FLIP_THRESHOLD = 0.5

#: Ceiling on confidence once the audit has rejected the preliminary call. Whatever
#: the auditors' own certainty, two panels that contradict each other are not
#: grounds for a confident alert.
CONTESTED_CONFIDENCE_CAP = 0.55


def parse(data: dict) -> tuple[bool | None, dict]:
    """Read one auditor's verdict.

    The vote is ``confirms``: True means the preliminary analysis stands.
    """
    confirms = as_bool(data.get("confirms"))
    direction_raw = data.get("direction")
    if confirms is None and direction_raw is None:
        return None, {"reason": "no verdict returned"}

    magnitude = Magnitude.parse(data.get("magnitude"))
    low, high = normalise_range(
        as_float(data.get("expected_move_pct_low")),
        as_float(data.get("expected_move_pct_high")),
        magnitude,
    )

    conflicts = as_str_list(data.get("conflicts"))[:4]
    overlooked = as_str_list(data.get("overlooked"))[:4]

    detail = {
        "confirms": confirms,
        "direction": Direction.parse(direction_raw).value if direction_raw is not None else "",
        "magnitude": magnitude.value,
        "low": low,
        "high": high,
        "midpoint": (low + high) / 2,
        "confidence": as_probability(data.get("confidence"), 0.5),
        "horizon": snap_horizon(as_float(data.get("horizon_minutes"))),
        "conflicts": conflicts,
        "overlooked": overlooked,
        "priced_in": as_bool(data.get("priced_in"), False),
        "reason": str(data.get("reason") or "")[:600],
    }

    if confirms is None:
        # Only a corrected forecast came back. Silence on `confirms` while
        # restating the same direction reads as agreement.
        confirms = not conflicts
        detail["confirms"] = confirms
        detail["note"] = "inferred from conflicts"

    return confirms, detail


async def run(
    runner: PanelRunner,
    ctx: StageContext,
    assets: list[str] | None = None,
) -> dict[str, Consensus]:
    """Audit the named assets' impacts. Revises ``ctx.impacts`` in place.

    ``assets`` comes from the router; omitting it audits everything Stage 4
    produced, which is what the dry-run and test paths want.
    """
    targets = assets if assets is not None else list(ctx.impacts)
    panel = panel_for_stage(5)
    results: dict[str, Consensus] = {}

    for asset in targets:
        impact = ctx.impacts.get(asset)
        if impact is None:
            continue

        stage4_disagreement = _stage4_disagreement(ctx, asset)
        consensus, _ = await runner.run(
            ctx,
            stage=5,
            system=STAGE5_SYSTEM,
            user=stage5_user(
                ctx.news,
                ctx.facts,
                impact,
                event_context=ctx.event_context,
                disagreement=stage4_disagreement,
                source_count=ctx.source_count,
            ),
            parse=parse,
            reject_votes=majority_of(len(panel)),
            asset=asset,
        )
        results[asset] = consensus

        revised = revise(impact, consensus)
        ctx.impacts[asset] = revised
        ctx.note(asset, f"stage5 {'confirmed' if consensus.passed else 'revised'}")
        log.info(
            "stage 5 %s: %s -> %s %s %.2f-%.2f%% conf %.2f -> %.2f (%s)",
            asset, impact.direction.value, revised.direction.value,
            revised.magnitude.value, revised.expected_low, revised.expected_high,
            impact.confidence, revised.confidence, consensus.summary(),
        )
    return results


def revise(impact: AssetImpact, consensus: Consensus) -> AssetImpact:
    """Fold the audit back into the impact.

    Returns the Stage 4 impact unchanged when the audit produced nothing usable —
    a dead panel is not evidence, and must not quietly degrade a good analysis.
    """
    answers = [vote.detail for vote in consensus.votes if vote.ok and "confirms" in vote.detail]
    if not answers:
        log.warning("stage 5 returned nothing usable for %s; keeping stage 4", impact.asset)
        return impact

    confirmed = sum(1 for answer in answers if answer["confirms"])
    contested = confirmed * 2 <= len(answers)

    direction = _revised_direction(impact, answers, contested)
    low, high, magnitude = _revised_size(impact, answers, direction)

    priced_in = sum(1 for answer in answers if answer["priced_in"]) * 2 > len(answers)
    if priced_in:
        low, high = low * PRICED_IN_RETENTION, high * PRICED_IN_RETENTION
        magnitude = aggregate_magnitude(
            [{"magnitude": magnitude.value, "midpoint": (low + high) / 2}]
        )

    confidence = _revised_confidence(impact, answers, contested, direction)

    horizon = snap_horizon(aggregate_numeric([answer["horizon"] for answer in answers]))
    notes = _audit_notes(answers)

    return replace(
        impact,
        direction=direction,
        magnitude=magnitude,
        expected_low=round(low, 3),
        expected_high=round(high, 3),
        confidence=confidence,
        horizon_minutes=horizon,
        mechanism=impact.mechanism,
        risks=best_evidence([impact.risks, notes]) if notes else impact.risks,
        agreement=confirmed / len(answers),
        model_count=len(answers),
        source="stage5",
    )


def _revised_direction(
    impact: AssetImpact, answers: list[dict], contested: bool
) -> Direction:
    """Change direction only when the auditors agree on a replacement."""
    if not contested:
        return impact.direction

    alternatives = [
        answer["direction"]
        for answer in answers
        if not answer["confirms"] and answer["direction"]
        and answer["direction"] != impact.direction.value
    ]
    if not alternatives:
        return impact.direction

    winner, share = majority_label(alternatives)
    if share >= FLIP_THRESHOLD and len(alternatives) * 2 > len(answers):
        return Direction.parse(winner)

    # Rejected, but with no agreed replacement: the honest reading is that the
    # direction is unresolved, which is what MIXED means.
    if impact.direction in (Direction.BULLISH, Direction.BEARISH):
        return Direction.MIXED
    return impact.direction


def _revised_size(
    impact: AssetImpact, answers: list[dict], direction: Direction
) -> tuple[float, float, Magnitude]:
    """Blend the auditors' sizes with Stage 4's, weighted by who was contested."""
    aligned = [
        answer for answer in answers
        if not answer["direction"] or answer["direction"] == direction.value
    ] or answers

    audit_low = aggregate_numeric([answer["low"] for answer in aligned])
    audit_high = aggregate_numeric([answer["high"] for answer in aligned])
    if audit_low is None or audit_high is None:
        return impact.expected_low, impact.expected_high, impact.magnitude

    # Auditors saw the event history and cross-source evidence, so they get the
    # larger share — but Stage 4 read the article most closely, and discarding it
    # entirely would throw away the only opinion grounded in the primary text.
    low = 0.35 * impact.expected_low + 0.65 * float(audit_low)
    high = 0.35 * impact.expected_high + 0.65 * float(audit_high)
    if high < low:
        low, high = high, low

    magnitude = aggregate_magnitude(
        [{"magnitude": answer["magnitude"], "midpoint": answer["midpoint"]} for answer in aligned]
    )
    return low, high, magnitude


def _revised_confidence(
    impact: AssetImpact, answers: list[dict], contested: bool, direction: Direction
) -> float:
    """The asymmetry: an audit can lower confidence freely, raise it only slightly.

    Confirmation is weak evidence — the auditors were shown Stage 4's answer and
    anchoring is real. Rejection is strong evidence, because the prompt gave them
    every opportunity to agree and they declined.
    """
    audit_confidence = aggregate_numeric([answer["confidence"] for answer in answers])
    audit_confidence = float(audit_confidence) if audit_confidence is not None else impact.confidence

    if contested:
        # Take the lower of the two panels, then cap it. Whichever panel is right,
        # the disagreement itself is a reason for caution.
        confidence = min(impact.confidence, audit_confidence)
        confidence = min(confidence, CONTESTED_CONFIDENCE_CAP)
    else:
        agreement = sum(1 for a in answers if a["confirms"]) / len(answers)
        confidence = min(audit_confidence, impact.confidence * 1.15)
        confidence *= clamp(agreement, 0.6, 1.0)

    if direction is Direction.MIXED and impact.direction is not Direction.MIXED:
        confidence *= 0.7  # the direction itself is now in doubt

    if any(answer["overlooked"] for answer in answers):
        confidence *= 0.9  # the analysis was incomplete, whatever its conclusion

    return round(clamp(confidence, 0.0, 1.0), 3)


def _audit_notes(answers: list[dict]) -> str:
    """The auditors' objections, for the alert's risk line and the final prompt."""
    items: list[str] = []
    for answer in answers:
        items.extend(answer.get("conflicts", []))
        items.extend(answer.get("overlooked", []))
    unique = list(dict.fromkeys(item.strip() for item in items if item and item.strip()))
    return "; ".join(unique[:3])[:400]


def _stage4_disagreement(ctx: StageContext, asset: str) -> str:
    """Where Stage 4's panel split on this asset, so the auditors see it.

    The spec is explicit that the deep review receives the model disagreement, not
    just the winning answer: the split is often the most informative thing the
    cheap panel produced.
    """
    impact = ctx.impacts.get(asset)
    if impact is None or impact.agreement >= 0.99 or impact.model_count < 2:
        return ""
    return (
        f"{impact.agreement:.0%} of {impact.model_count} models backed "
        f"{impact.direction.value}"
    )


def summary(consensus: Consensus | None) -> str:
    """One line for the final layer's prompt."""
    if consensus is None:
        return ""
    answers = [vote.detail for vote in consensus.votes if vote.ok and "confirms" in vote.detail]
    if not answers:
        return "no usable cross-check"

    confirmed = sum(1 for answer in answers if answer["confirms"])
    parts = [f"{confirmed}/{len(answers)} confirmed"]
    if sum(1 for answer in answers if answer["priced_in"]) * 2 > len(answers):
        parts.append("majority say already priced in")
    notes = _audit_notes(answers)
    if notes:
        parts.append(f"objections: {notes}")
    reasons = [answer["reason"] for answer in answers if answer.get("reason")]
    if reasons:
        parts.append(max(reasons, key=len)[:300])
    return "; ".join(parts)


__all__ = [
    "CONTESTED_CONFIDENCE_CAP",
    "PRICED_IN_RETENTION",
    "disagreement_summary",
    "parse",
    "revise",
    "run",
    "summary",
]
