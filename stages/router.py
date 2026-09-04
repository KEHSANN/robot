"""Escalation routing.

The funnel's economics live here. Roughly a thousand items a day reach Stage 0; ten
should reach Stage 5 and three the NVIDIA panel. Those ratios are not a target to
hit for its own sake — they are what makes running five models on everything
affordable in the first place, and every rule below exists to protect them.

Three ideas shape the rules:

**Importance and uncertainty both justify spending.** A high-scoring event is worth
a closer look because being wrong about it is expensive. A *low-agreement* event is
worth a closer look because the pipeline has admitted it does not know — and the
cheap panel cannot resolve its own disagreement by being asked again. Both routes
escalate; ``final_max_confidence`` is the second one.

**Some event types escalate on identity, not score.** Hacks, enforcement actions and
ETF decisions are where a missed call costs the most and where the cheap models are
weakest, because the market impact turns on legal and structural detail rather than
on sentiment. Those go deep regardless of what the score said.

**The budget is per cycle and it is hard.** A breaking story arrives as fifteen
articles in ten minutes; without a ceiling, one event could spend the day's entire
heavy-model budget in a single pass. When more candidates qualify than the budget
allows, the router spends on the highest-scoring and records the rest as skipped —
visible in the logs rather than silently dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from services.config import Settings, settings as global_settings
from services.types import AssetImpact, Direction, Magnitude
from stages.base import StageContext

log = logging.getLogger(__name__)

#: Assets below this magnitude are not audited even when the event qualifies. The
#: event being important does not make a 0.3% read-across worth a second panel.
STAGE5_MIN_MAGNITUDE = Magnitude.MEDIUM

#: Confidence at/above which a NEUTRAL call is accepted without auditing. A model
#: confidently saying "this changes nothing" is a useful answer, and re-asking it is
#: how a system talks itself into finding an impact that is not there.
CONFIDENT_NEUTRAL = 0.6


@dataclass
class Escalation:
    """One asset selected for a deeper stage, with the reason it qualified."""

    asset: str
    reason: str
    priority: float = 0.0


@dataclass
class Router:
    """Decides which assets go deeper. Stateful: holds the per-cycle budget."""

    config: Settings = field(default_factory=lambda: global_settings)
    final_spent: int = 0
    skipped_for_budget: list[str] = field(default_factory=list)

    @property
    def routing(self):
        return self.config.routing

    def reset_cycle(self) -> None:
        self.final_spent = 0
        self.skipped_for_budget.clear()

    @property
    def final_budget_left(self) -> int:
        return max(0, self.routing.final_max_per_cycle - self.final_spent)

    # ----------------------------------------------------------------- stage 5

    def stage5_assets(self, ctx: StageContext) -> list[Escalation]:
        """Which of this event's assets deserve the deep cross-check."""
        event_gate, event_reason = self._event_qualifies_for_stage5(ctx)

        picks: list[Escalation] = []
        for asset, impact in ctx.impacts.items():
            asset_gate, asset_reason = self._asset_qualifies_for_stage5(impact)
            if not (event_gate or asset_gate):
                continue
            if not self._worth_auditing(impact):
                continue
            reason = asset_reason or event_reason
            picks.append(Escalation(asset, reason, self._priority(ctx, impact)))

        picks.sort(key=lambda pick: pick.priority, reverse=True)
        if picks:
            log.info(
                "stage 5 escalation: %s",
                ", ".join(f"{p.asset} ({p.reason})" for p in picks),
            )
        return picks

    def _event_qualifies_for_stage5(self, ctx: StageContext) -> tuple[bool, str]:
        event_type = ctx.event_type.upper()
        if event_type in self.routing.stage5_always_types:
            return True, f"event type {event_type} always escalates"

        score = ctx.stage2_score
        if score is not None and score >= self.routing.stage5_min_score:
            return True, f"significance {score:.0f} >= {self.routing.stage5_min_score}"

        # Several outlets independently reporting the same thing is corroboration,
        # and corroborated events are the ones where a real move is likeliest.
        if ctx.source_count >= 4 and (score or 0) >= self.routing.stage5_min_score - 15:
            return True, f"{ctx.source_count} sources reporting"

        return False, ""

    def _asset_qualifies_for_stage5(self, impact: AssetImpact) -> tuple[bool, str]:
        """Per-asset routes that fire regardless of the event's score."""
        if impact.model_count >= 2 and impact.agreement < self.routing.stage5_max_agreement:
            return True, f"panel agreement {impact.agreement:.0%} is low"
        if impact.magnitude in (Magnitude.HIGH, Magnitude.EXTREME):
            return True, f"{impact.magnitude.value} forecast needs a second look"
        if impact.direction is Direction.MIXED:
            return True, "direction unresolved"
        return False, ""

    def _worth_auditing(self, impact: AssetImpact) -> bool:
        """Filter out the assets where a deeper look cannot change anything."""
        if impact.model_count == 0:
            return False  # nothing to audit; Stage 4 never got an answer
        if impact.magnitude.rank < STAGE5_MIN_MAGNITUDE.rank:
            # Small forecasts are only worth auditing when the panel was unsure.
            return impact.agreement < self.routing.stage5_max_agreement
        if impact.direction is Direction.NEUTRAL and impact.confidence >= CONFIDENT_NEUTRAL:
            return False
        return True

    # ------------------------------------------------------------ final layer

    def final_assets(self, ctx: StageContext) -> list[Escalation]:
        """Which assets get the NVIDIA panel, inside the per-cycle budget."""
        if self.final_budget_left <= 0:
            for asset in ctx.impacts:
                self.skipped_for_budget.append(f"{asset} ({ctx.headline()[:40]})")
            log.warning(
                "heavy-panel budget exhausted (%d/%d this cycle); skipping %s",
                self.final_spent, self.routing.final_max_per_cycle, ctx.headline()[:60],
            )
            return []

        candidates: list[Escalation] = []
        for asset, impact in ctx.impacts.items():
            qualifies, reason = self._qualifies_for_final(ctx, impact)
            if qualifies:
                candidates.append(Escalation(asset, reason, self._priority(ctx, impact)))

        candidates.sort(key=lambda pick: pick.priority, reverse=True)
        picks = candidates[: self.final_budget_left]

        for skipped in candidates[len(picks):]:
            self.skipped_for_budget.append(f"{skipped.asset} ({skipped.reason})")

        self.final_spent += len(picks)
        if picks:
            log.info(
                "final layer: %s [budget %d/%d]",
                ", ".join(f"{p.asset} ({p.reason})" for p in picks),
                self.final_spent, self.routing.final_max_per_cycle,
            )
        return picks

    def _qualifies_for_final(
        self, ctx: StageContext, impact: AssetImpact
    ) -> tuple[bool, str]:
        if impact.model_count == 0:
            return False, ""

        score = ctx.stage2_score or 0.0
        if score >= self.routing.final_min_score:
            return True, f"significance {score:.0f} >= {self.routing.final_min_score}"

        # The uncertainty route. Only for events already important enough that
        # being wrong matters — otherwise every low-confidence trickle would buy
        # itself three large models.
        if (
            impact.confidence < self.routing.final_max_confidence
            and impact.magnitude in (Magnitude.HIGH, Magnitude.EXTREME)
        ):
            return True, (
                f"{impact.magnitude.value} forecast at only "
                f"{impact.confidence:.0%} confidence"
            )

        if impact.magnitude is Magnitude.EXTREME:
            return True, "EXTREME forecast always gets the heavy panel"

        # A cross-check that overturned the preliminary analysis is a live
        # disagreement between two panels; the heavy models exist to settle it.
        if impact.source == "stage5" and impact.direction is Direction.MIXED:
            return True, "stage 5 could not resolve the direction"

        return False, ""

    # ------------------------------------------------------------------ alerts

    def _priority(self, ctx: StageContext, impact: AssetImpact) -> float:
        """Ordering key for spending a limited budget.

        Expected size times confidence, nudged by the event's significance score.
        Deliberately not confidence alone: a 90%-confident 0.2% move is not worth
        a heavy panel, and a 50%-confident 6% move is.
        """
        size = (abs(impact.expected_low) + abs(impact.expected_high)) / 2
        return size * max(impact.confidence, 0.1) + (ctx.stage2_score or 0.0) / 100.0

    def alert_score(self, ctx: StageContext, impact: AssetImpact) -> float:
        """0-100 publication score for one asset's verdict.

        Blends how much the market cares (Stage 2), how big the move is, and how
        sure the system is. Depth counts too: a verdict that survived the deep
        stages has been tested in ways a Stage 4-only answer has not.
        """
        significance = (ctx.stage2_score or 40.0) / 100.0
        size = min((abs(impact.expected_low) + abs(impact.expected_high)) / 2, 8.0) / 8.0
        depth = {"stage4": 0.0, "stage5": 0.06, "final": 0.12}.get(impact.source, 0.0)

        raw = 0.40 * significance + 0.35 * size + 0.25 * impact.confidence + depth
        if impact.direction is Direction.NEUTRAL:
            raw *= 0.35  # recorded, effectively never alerted
        elif impact.direction is Direction.MIXED:
            raw *= 0.75
        if impact.tradeable is False:
            raw *= 0.6
        if impact.model_count == 0:
            raw *= 0.4  # no model actually answered for this asset
        return round(min(100.0, raw * 100.0), 1)

    def should_alert(self, ctx: StageContext, impact: AssetImpact) -> tuple[bool, float]:
        """Whether this verdict is worth someone's attention, and its score."""
        score = self.alert_score(ctx, impact)
        threshold = self.config.telegram.min_alert_score
        if impact.direction is Direction.NEUTRAL:
            return False, score
        return score >= threshold, score


__all__ = [
    "CONFIDENT_NEUTRAL",
    "STAGE5_MIN_MAGNITUDE",
    "Escalation",
    "Router",
]
