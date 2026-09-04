"""Final layer — the heavy NVIDIA panel.

Three large models, each doing an independent analysis of the whole chain, then
cross-checked against each other. Per the architecture this layer is **forbidden in
Stages 0-5** and runs only on events that survived every cheaper filter — the last
three or so out of a thousand daily items. The router enforces the budget; this
module assumes the decision to spend has already been made.

The design question here is what authority the layer has. It sees everything the
pipeline concluded, so it can overrule — that is what it is for, and a final layer
that could only ratify would be worth nothing. But its own agreement is what
licenses that authority:

* All three agreeing overrules the pipeline outright.
* Two of three sets the verdict with the dissent recorded and confidence trimmed.
* Three-way disagreement means the event is genuinely ambiguous. The verdict is
  MIXED and untradeable, and that is a real finding: the pipeline's neat 3-2 answer
  from the cheap panel was hiding an unresolvable question.

The published record always keeps ``disagreement_with_pipeline``. When the outcome
is scored, that field is what makes it possible to ask whether the expensive models
were actually right to overrule — the question the whole routing layer depends on.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from services.consensus import Consensus, aggregate_numeric, majority_label
from services.jsonparse import as_bool, as_float, as_probability
from services.models import panel_for_stage
from services.prompts import FINAL_SYSTEM, final_user
from services.types import AssetImpact, Causality, Direction, Magnitude
from stages.base import PanelRunner, StageContext, best_evidence, clamp
from stages.stage4 import aggregate_magnitude, normalise_range, snap_horizon

log = logging.getLogger(__name__)

#: Confidence ceiling when the three heavy models cannot agree on a direction.
#: Reaching this layer means the event is important, which makes an overconfident
#: call here the most expensive mistake the system can make.
SPLIT_CONFIDENCE_CAP = 0.4

#: Applied when two of three agree: the dissent is real and gets priced in.
MINORITY_DISSENT_FACTOR = 0.85


def parse(data: dict) -> tuple[bool | None, dict]:
    """Read one heavy model's verdict. The vote is ``tradeable``."""
    direction_raw = data.get("direction")
    if direction_raw is None and "magnitude" not in data:
        return None, {"reason": "no verdict returned"}

    magnitude = Magnitude.parse(data.get("magnitude"))
    low, high = normalise_range(
        as_float(data.get("expected_move_pct_low")),
        as_float(data.get("expected_move_pct_high")),
        magnitude,
    )
    direction = Direction.parse(direction_raw)
    confidence = as_probability(data.get("confidence"), 0.5)

    tradeable = as_bool(data.get("tradeable"))
    if tradeable is None:
        # Not answered: derive it the way the prompt defines it — a move too small
        # to clear spread and noise is not tradeable whatever the direction.
        tradeable = direction is not Direction.NEUTRAL and (low + high) / 2 >= 0.5

    detail = {
        "direction": direction.value,
        "magnitude": magnitude.value,
        "low": low,
        "high": high,
        "midpoint": (low + high) / 2,
        "confidence": confidence,
        "horizon": snap_horizon(as_float(data.get("horizon_minutes"))),
        "causality": Causality.parse(data.get("causality")).value,
        "mechanism": str(data.get("mechanism") or "")[:1000],
        "risks": str(data.get("risks") or "")[:800],
        "key_uncertainty": str(data.get("key_uncertainty") or "")[:400],
        "disagreement": str(data.get("disagreement_with_pipeline") or "")[:600],
        "tradeable": bool(tradeable),
        "reason": str(data.get("mechanism") or "")[:400],
    }
    return bool(tradeable), detail


async def run(
    runner: PanelRunner,
    ctx: StageContext,
    assets: list[str],
    *,
    stage5_summaries: dict[str, str] | None = None,
) -> dict[str, Consensus]:
    """Issue the final verdict for the named assets. Overwrites ``ctx.impacts``."""
    panel = panel_for_stage(6)
    if not panel:
        log.error("final layer requested but no NVIDIA models are configured")
        return {}

    summaries = stage5_summaries or {}
    results: dict[str, Consensus] = {}

    for asset in assets:
        impact = ctx.impacts.get(asset)
        if impact is None:
            continue

        consensus, _ = await runner.run(
            ctx,
            stage=6,
            system=FINAL_SYSTEM,
            user=final_user(
                ctx.news,
                ctx.facts,
                impact,
                stage1_summary=ctx.stage1.summary() if ctx.stage1 else "",
                stage2_summary=ctx.stage2.summary() if ctx.stage2 else "",
                stage2_score=ctx.stage2_score,
                links=ctx.links,
                stage5_summary=summaries.get(asset, ""),
                event_context=ctx.event_context,
                source_count=ctx.source_count,
            ),
            parse=parse,
            # Two of three. The floor in majority_of() would give the same answer,
            # but stating it here documents the intent for a three-model panel.
            reject_votes=2,
            asset=asset,
            specs=panel,
        )
        results[asset] = consensus

        verdict = decide(impact, consensus)
        ctx.impacts[asset] = verdict
        ctx.note(asset, f"final {verdict.direction.value} tradeable={verdict.tradeable}")
        log.info(
            "final %s: %s %s %.2f-%.2f%% conf=%.2f tradeable=%s (%d models)%s",
            asset, verdict.direction.value, verdict.magnitude.value,
            verdict.expected_low, verdict.expected_high, verdict.confidence,
            verdict.tradeable, verdict.model_count,
            f" — differs: {verdict.notes[:80]}" if verdict.notes else "",
        )
    return results


def decide(impact: AssetImpact, consensus: Consensus) -> AssetImpact:
    """Fold the heavy panel into the published verdict.

    Falls back to the incoming impact when nothing usable came back. A dead NVIDIA
    provider must not discard five models' work — the alert still goes out, marked
    with whatever the deepest stage that *did* answer concluded.
    """
    answers = [vote.detail for vote in consensus.votes if vote.ok and vote.detail.get("direction")]
    if not answers:
        log.warning("final layer returned nothing usable for %s; keeping %s",
                    impact.asset, impact.source)
        return replace(impact, notes="final layer unavailable")

    direction, agreement = _final_direction(answers)
    split = agreement < 0.6

    aligned = [a for a in answers if a["direction"] == direction.value] or answers

    low = aggregate_numeric([a["low"] for a in aligned]) or impact.expected_low
    high = aggregate_numeric([a["high"] for a in aligned]) or impact.expected_high
    if high < low:
        low, high = high, low

    magnitude = aggregate_magnitude(
        [{"magnitude": a["magnitude"], "midpoint": a["midpoint"]} for a in aligned]
    )

    confidence = float(aggregate_numeric([a["confidence"] for a in aligned]) or 0.5)
    if split:
        confidence = min(confidence, SPLIT_CONFIDENCE_CAP)
    elif len(aligned) < len(answers):
        confidence *= MINORITY_DISSENT_FACTOR

    tradeable = sum(1 for a in answers if a["tradeable"]) * 2 > len(answers)
    if split:
        # Three models that cannot agree on direction have not identified a trade.
        tradeable = False

    causality, _ = majority_label([a["causality"] for a in aligned])

    return replace(
        impact,
        direction=direction,
        magnitude=magnitude,
        expected_low=round(float(low), 3),
        expected_high=round(float(high), 3),
        confidence=round(clamp(confidence, 0.0, 1.0), 3),
        horizon_minutes=snap_horizon(aggregate_numeric([a["horizon"] for a in aligned])),
        causality=Causality.parse(causality),
        mechanism=best_evidence([a["mechanism"] for a in aligned]) or impact.mechanism,
        risks=best_evidence([a["risks"] for a in answers]) or impact.risks,
        key_uncertainty=best_evidence([a["key_uncertainty"] for a in answers]),
        notes=_divergence(impact, answers, direction, split),
        agreement=agreement,
        model_count=len(answers),
        source="final",
        tradeable=tradeable,
    )


def _final_direction(answers: list[dict]) -> tuple[Direction, float]:
    """Majority direction among the heavy models, plus its share.

    Unlike Stage 4 this does not synthesise MIXED from a bull/bear split — with
    three models a 2-1 split is a decision, and the dissent is recorded in the
    notes rather than erased by averaging. A genuine three-way split has share
    1/3, which trips the ``split`` branch in :func:`decide`.
    """
    labels = [answer["direction"] for answer in answers]
    winner, share = majority_label(labels)
    if len(set(labels)) == len(labels) and len(labels) >= 3:
        return Direction.MIXED, share
    return Direction.parse(winner), share


def _divergence(
    impact: AssetImpact, answers: list[dict], direction: Direction, split: bool
) -> str:
    """What the heavy panel says it disagrees with, plus what we can see ourselves.

    The models' own ``disagreement_with_pipeline`` is unreliable — they often leave
    it empty while returning different numbers — so the structural comparison is
    made here rather than trusted to the prompt.
    """
    parts: list[str] = []

    if split:
        seen = sorted({answer["direction"] for answer in answers})
        parts.append(f"final panel split three ways ({', '.join(seen)})")
    elif direction is not impact.direction:
        parts.append(f"overrules {impact.source} {impact.direction.value} -> {direction.value}")

    stated = best_evidence([answer["disagreement"] for answer in answers])
    if stated:
        parts.append(stated)

    return "; ".join(parts)[:600]


def summary(consensus: Consensus | None) -> str:
    """One line describing the final layer's own agreement, for the alert footer."""
    if consensus is None:
        return ""
    answers = [vote.detail for vote in consensus.votes if vote.ok and vote.detail.get("direction")]
    if not answers:
        return "final layer unavailable"
    labels = [answer["direction"] for answer in answers]
    winner, share = majority_label(labels)
    tradeable = sum(1 for answer in answers if answer["tradeable"])
    return (
        f"{labels.count(winner)}/{len(answers)} heavy models {winner} "
        f"({share:.0%}), {tradeable}/{len(answers)} tradeable"
    )


__all__ = [
    "MINORITY_DISSENT_FACTOR",
    "SPLIT_CONFIDENCE_CAP",
    "decide",
    "parse",
    "run",
    "summary",
]
