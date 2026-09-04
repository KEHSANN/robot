"""Stage 4 — per-asset impact.

The stage that produces the actual forecast: direction, size, confidence, horizon,
causal channel, and the mechanism behind it. One panel run per asset, because the
answer for BNB and the answer for BTC on the same event are genuinely different
analyses and averaging them would produce a number that is true of neither.

Two aggregation choices carry most of the weight here:

**Direction is a vote, size is a median.** Direction has no meaningful average —
halfway between BULLISH and BEARISH is not NEUTRAL, it is disagreement, and saying
so is more useful than splitting the difference. Sizes are medians so that one
model answering 15% cannot drag a panel that said 1%.

**Disagreement lowers confidence.** If three models say up and two say down, the
panel does not know, and the published confidence has to say that even when every
individual model was sure. Without this the system reports high confidence on
exactly the events where it is least reliable — and those are the events big enough
to be worth escalating.
"""

from __future__ import annotations

import logging

from services.consensus import Consensus, aggregate_numeric, majority_label
from services.jsonparse import as_float, as_probability
from services.models import panel_for_stage
from services.prompts import STAGE4_SYSTEM, stage4_user
from services.types import AssetImpact, AssetLink, Causality, Direction, Magnitude
from stages.base import PanelRunner, StageContext, best_evidence, clamp, majority_of

log = logging.getLogger(__name__)

#: Assets analysed per event, strongest link first. The cap is a spending control:
#: each one costs a full panel, and links ranked below third are almost always weak
#: INDIRECT read-across that the alert threshold would reject anyway.
MAX_ASSETS_ANALYSED = 3

#: Horizons the prompt offers. A model answering 45 gets snapped to the nearest,
#: so the feedback loop always has an observation window that matches.
HORIZONS = (15, 60, 180, 360, 1440)

#: Below this expected move, an asset is not meaningfully affected whatever label
#: the model attached — it is inside normal intraday noise for a large cap.
NOISE_FLOOR_PCT = 0.15

#: A direction split this even means the panel does not know. Reported as MIXED
#: rather than as whichever side happened to have one more vote.
MIXED_THRESHOLD = 0.6


def snap_horizon(minutes: float | None) -> int:
    if minutes is None:
        return 180
    return min(HORIZONS, key=lambda h: abs(h - minutes))


def parse(data: dict) -> tuple[bool | None, dict]:
    """Read one model's impact answer for a single asset.

    The vote is whether the asset moves measurably. It is not used to drop the
    asset — a well-argued NEUTRAL is a valid finding — but it is what the agreement
    figure and the escalation rules are computed from.
    """
    direction_raw = data.get("direction")
    if direction_raw is None and "magnitude" not in data:
        return None, {"reason": "no direction or magnitude returned"}

    direction = Direction.parse(direction_raw)
    magnitude = Magnitude.parse(data.get("magnitude"))

    low = as_float(data.get("expected_move_pct_low"))
    high = as_float(data.get("expected_move_pct_high"))
    low, high = normalise_range(low, high, magnitude)

    detail = {
        "direction": direction.value,
        "magnitude": magnitude.value,
        "low": low,
        "high": high,
        "midpoint": (low + high) / 2,
        "confidence": as_probability(data.get("confidence"), 0.5),
        "horizon": snap_horizon(as_float(data.get("horizon_minutes"))),
        "causality": Causality.parse(data.get("causality")).value,
        "mechanism": str(data.get("mechanism") or "")[:800],
        "risks": str(data.get("risks") or "")[:600],
        "reason": str(data.get("mechanism") or "")[:400],
    }

    moves = direction is not Direction.NEUTRAL and detail["midpoint"] >= NOISE_FLOOR_PCT
    return moves, detail


def normalise_range(
    low: float | None, high: float | None, magnitude: Magnitude
) -> tuple[float, float]:
    """Coerce a model's percentage range into a usable absolute band.

    Models return these in every shape: signed, reversed, one side only, or the
    label with no numbers at all. The label's calibration band is the fallback,
    which is why the prompt states those bands numerically.

    Shared with Stage 5 and the final layer, which ask for the same two fields.
    """
    if low is None and high is None:
        return magnitude.default_range

    # The prompt asks for absolute sizes with direction carrying the sign, but
    # models leak the sign in anyway.
    values = [abs(v) for v in (low, high) if v is not None]
    if len(values) == 1:
        band = magnitude.default_range
        single = values[0]
        # Widen a single number into a band around it, keeping it inside the
        # label's range so the two answers do not contradict each other.
        return min(single, band[0]), max(single, band[0])

    lo, hi = sorted(values)
    return lo, hi


async def run(runner: PanelRunner, ctx: StageContext) -> dict[str, AssetImpact]:
    """Analyse each linked asset. Fills ``ctx.impacts``."""
    links = ctx.links[:MAX_ASSETS_ANALYSED]
    if not links:
        return {}

    panel = panel_for_stage(4)
    for link in links:
        consensus, _ = await runner.run(
            ctx,
            stage=4,
            system=STAGE4_SYSTEM,
            user=stage4_user(
                ctx.news,
                ctx.facts,
                link,
                stage2_score=ctx.stage2_score,
                event_context=ctx.event_context,
            ),
            parse=parse,
            reject_votes=majority_of(len(panel)),
            asset=link.asset,
        )
        impact = aggregate_impact(link, consensus)
        ctx.impacts[link.asset] = impact
        ctx.note(link.asset, f"stage4 {impact.direction.value} {impact.magnitude.value}")
        log.info(
            "stage 4 %s: %s %s %.2f-%.2f%% conf=%.2f %s (agreement %.0f%% of %d)",
            link.asset, impact.direction.value, impact.magnitude.value,
            impact.expected_low, impact.expected_high, impact.confidence,
            impact.horizon_label, impact.agreement * 100, impact.model_count,
        )
    return ctx.impacts


def aggregate_impact(link: AssetLink, consensus: Consensus) -> AssetImpact:
    """Fold the panel's answers into one forecast for one asset."""
    answers = [vote.detail for vote in consensus.votes if vote.ok and vote.detail.get("direction")]

    if not answers:
        # Nothing usable came back. A NEUTRAL with zero confidence is the honest
        # record, and it will not clear any alert threshold.
        return AssetImpact(
            asset=link.asset,
            direction=Direction.NEUTRAL,
            magnitude=Magnitude.LOW,
            confidence=0.0,
            relation=link.relation,
            mechanism="no model returned a usable answer",
            agreement=0.0,
            model_count=0,
        )

    direction, agreement = _aggregate_direction(answers)

    # Size is taken from the models that agreed on the winning direction. Averaging
    # in the dissenters' numbers would shrink the move toward zero and describe a
    # scenario no model actually forecast.
    aligned = [
        answer for answer in answers
        if answer["direction"] == direction.value
    ] or answers

    magnitude = aggregate_magnitude(aligned)
    low = aggregate_numeric([answer["low"] for answer in aligned]) or 0.0
    high = aggregate_numeric([answer["high"] for answer in aligned]) or 0.0
    if high < low:
        low, high = high, low

    confidence = aggregate_numeric([answer["confidence"] for answer in aligned]) or 0.5
    confidence = _damp_confidence(float(confidence), agreement, link)

    horizon = snap_horizon(aggregate_numeric([answer["horizon"] for answer in aligned]))
    causality, _ = majority_label([answer["causality"] for answer in aligned])

    if direction is Direction.NEUTRAL:
        # A neutral call still needs a band, but the models that said "no move"
        # gave sizes for a move they do not expect.
        low, high = min(low, NOISE_FLOOR_PCT), min(high, Magnitude.LOW.default_range[1])

    return AssetImpact(
        asset=link.asset,
        direction=direction,
        magnitude=magnitude,
        expected_low=float(low),
        expected_high=float(high),
        confidence=confidence,
        horizon_minutes=horizon,
        causality=Causality.parse(causality),
        mechanism=best_evidence([answer.get("mechanism", "") for answer in aligned]),
        risks=best_evidence([answer.get("risks", "") for answer in answers]),
        relation=link.relation,
        agreement=agreement,
        model_count=len(answers),
        source="stage4",
    )


def _aggregate_direction(answers: list[dict]) -> tuple[Direction, float]:
    """Majority direction, plus how much of the panel backed it.

    A near-even split between BULLISH and BEARISH is reported as MIXED. Reporting
    the winner of a 3-2 vote as though it were the panel's view is how a coin flip
    gets published as a forecast.
    """
    labels = [answer["direction"] for answer in answers]
    winner, share = majority_label(labels)
    direction = Direction.parse(winner)

    bulls = labels.count(Direction.BULLISH.value)
    bears = labels.count(Direction.BEARISH.value)
    directional = bulls + bears
    if directional >= 2:
        lead = max(bulls, bears) / directional
        if lead < MIXED_THRESHOLD:
            return Direction.MIXED, lead
    return direction, share


def aggregate_magnitude(answers: list[dict]) -> Magnitude:
    """Median magnitude by rank, cross-checked against the numeric forecast.

    When the labels and the numbers disagree the numbers win: they are what gets
    scored against the actual price move, so the label has to describe them.
    """
    ranks = [Magnitude.parse(answer["magnitude"]).rank for answer in answers]
    median = aggregate_numeric(ranks) or 1
    by_label = {m.rank: m for m in Magnitude}[int(clamp(round(median), 1, 4))]

    midpoint = aggregate_numeric([answer["midpoint"] for answer in answers])
    if midpoint is None:
        return by_label
    by_number = _magnitude_for_move(float(midpoint))
    # Only correct a full step or more of disagreement; adjacent bands overlap at
    # the edges and quibbling there adds noise rather than accuracy.
    return by_number if abs(by_number.rank - by_label.rank) >= 2 else by_label


def _magnitude_for_move(pct: float) -> Magnitude:
    for magnitude in (Magnitude.LOW, Magnitude.MEDIUM, Magnitude.HIGH):
        if pct <= magnitude.default_range[1]:
            return magnitude
    return Magnitude.EXTREME


def _damp_confidence(confidence: float, agreement: float, link: AssetLink) -> float:
    """Lower the published confidence when the evidence for it is thin.

    Three sources of doubt compound here: the panel disagreeing about direction,
    Stage 3 being unsure the asset is even affected, and the link being indirect.
    A model is confident about its own reasoning; none of these are visible to it.
    """
    damped = confidence * clamp(agreement, 0.4, 1.0)
    damped *= clamp(0.6 + 0.4 * link.confidence, 0.6, 1.0)
    if link.relation.value == "INDIRECT":
        damped *= 0.85
    return round(clamp(damped, 0.0, 1.0), 3)


def disagreement_summary(consensus: Consensus) -> str:
    """Where the panel split, phrased for the Stage 5 prompt."""
    tallies: dict[str, int] = {}
    for vote in consensus.votes:
        if vote.ok and vote.detail.get("direction"):
            label = vote.detail["direction"]
            tallies[label] = tallies.get(label, 0) + 1
    if len(tallies) <= 1:
        return ""
    return ", ".join(f"{count}x {label}" for label, count in sorted(tallies.items()))


__all__ = [
    "HORIZONS",
    "MAX_ASSETS_ANALYSED",
    "MIXED_THRESHOLD",
    "NOISE_FLOOR_PCT",
    "aggregate_impact",
    "aggregate_magnitude",
    "disagreement_summary",
    "normalise_range",
    "parse",
    "run",
    "snap_horizon",
]
