"""The pipeline: Stage 0 through the final layer, for one article.

Ordering here is the architecture's core economic claim — cheap filters before
expensive analysis — so the sequence is worth stating as costs:

    stage 0   0-2 calls    dedup; most items stop here for free
    stage 1   5 calls      relevance
    stage 2   5 calls      market significance
    stage 3   5 calls      which assets
    stage 4   5 x assets   the forecast
    stage 5   5 x assets   audit, escalated only
    final     3 x assets   NVIDIA, critical only

An item that fails Stage 1 costs five small calls. One that reaches the final layer
costs upwards of forty, three of them large. The router decides which, and the
funnel only works because the early stages are ruthless.

Two behaviours are deliberate and easy to mistake for oversights:

**Dropped items are still recorded.** Every drop is written with the stage and
reason. Without that the filters are unauditable — there is no way to discover that
Stage 1 has been quietly discarding a whole category of real events.

**Predictions are stored even when no alert is sent.** The unpublished forecasts are
what make the calibration data honest; keeping only the published ones would measure
the system solely on the events it already felt confident about.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from database.base import Store
from services.config import Settings, settings as global_settings
from services.llm import LLMClient
from services.types import (
    AssetImpact,
    Direction,
    FactSet,
    NewsItem,
    Prediction,
    Stage0Decision,
)
from stage0.pipeline import Stage0Pipeline, Stage0Result
from stages import final as final_stage
from stages import stage1, stage2, stage3, stage4, stage5
from stages.base import PanelRunner, StageContext
from stages.router import Router

log = logging.getLogger(__name__)

#: Maps the deepest stage that contributed to a verdict onto the integer the
#: ``analyses`` table records. 6 is the NVIDIA layer.
_DEPTH = {"stage4": 4, "stage5": 5, "final": 6}


@dataclass
class Alert:
    """A verdict judged worth someone's attention."""

    news: NewsItem
    facts: FactSet
    impact: AssetImpact
    score: float
    urgency: str = "medium"
    event_id: int | None = None
    news_id: int | None = None
    analysis_id: int | None = None
    source_count: int = 1
    decision: Stage0Decision = Stage0Decision.NEW
    #: Stages that touched this asset, for the alert footer.
    trail: list[str] = field(default_factory=list)

    @property
    def is_update(self) -> bool:
        return self.decision is Stage0Decision.UPDATE

    @property
    def deepest_stage(self) -> int:
        return _DEPTH.get(self.impact.source, 4)


@dataclass
class PipelineStats:
    ingested: int = 0
    duplicates: int = 0
    new_events: int = 0
    updates: int = 0
    dropped_stage1: int = 0
    dropped_stage2: int = 0
    dropped_stage3: int = 0
    analysed: int = 0
    escalated_stage5: int = 0
    escalated_final: int = 0
    alerts: int = 0
    errors: int = 0
    llm_calls: int = 0
    llm_failures: int = 0

    def as_dict(self) -> dict:
        return {
            "ingested": self.ingested,
            "duplicates": self.duplicates,
            "new_events": self.new_events,
            "updates": self.updates,
            "dropped_stage1": self.dropped_stage1,
            "dropped_stage2": self.dropped_stage2,
            "dropped_stage3": self.dropped_stage3,
            "analysed": self.analysed,
            "escalated_stage5": self.escalated_stage5,
            "escalated_final": self.escalated_final,
            "alerts": self.alerts,
            "errors": self.errors,
            "llm_calls": self.llm_calls,
            "llm_failures": self.llm_failures,
        }


@dataclass
class Result:
    """What the pipeline concluded about one article."""

    stage0: Stage0Result
    ctx: StageContext | None = None
    alerts: list[Alert] = field(default_factory=list)

    @property
    def dropped_at(self) -> int | None:
        if self.ctx is None:
            return 0 if not self.stage0.advances else None
        return self.ctx.dropped_at

    @property
    def drop_reason(self) -> str:
        if self.ctx is None:
            return self.stage0.outcome.reason
        return self.ctx.drop_reason


class Pipeline:
    """Runs the whole chain. One instance per process; safe to reuse."""

    def __init__(
        self,
        client: LLMClient,
        store: Store,
        config: Settings | None = None,
    ) -> None:
        self.settings = config or global_settings
        self.client = client
        self.store = store
        self.stage0 = Stage0Pipeline(client, store, self.settings)
        self.runner = PanelRunner(client, store, self.settings)
        self.router = Router(self.settings)
        self.stats = PipelineStats()

    # ------------------------------------------------------------------ batch

    async def run_batch(self, items: list[NewsItem]) -> list[Alert]:
        """Process a batch, returning every alert worth sending.

        The heavy-model budget is per batch, so it resets here. One article failing
        must not take the batch with it — a malformed feed entry is routine.
        """
        self.router.reset_cycle()
        alerts: list[Alert] = []

        for item in items:
            try:
                result = await self.process(item)
            except Exception:
                self.stats.errors += 1
                log.exception("pipeline failed on %r", item.title[:80])
                continue
            alerts.extend(result.alerts)

        self.stats.llm_calls = self.runner.llm_calls
        self.stats.llm_failures = self.runner.llm_failures
        # Highest-scoring first: if a burst of news arrives, the reader sees what
        # matters before the tail.
        alerts.sort(key=lambda alert: alert.score, reverse=True)
        return alerts

    # ------------------------------------------------------------ one article

    async def process(self, news: NewsItem) -> Result:
        """Take one article as far down the chain as it earns."""
        self.stats.ingested += 1

        stage0_result = await self.stage0.process(news)
        outcome = stage0_result.outcome
        decision = outcome.decision

        if decision is Stage0Decision.NEW:
            self.stats.new_events += 1
        elif decision is Stage0Decision.UPDATE:
            self.stats.updates += 1
        else:
            self.stats.duplicates += 1

        if not stage0_result.advances:
            await self._mark(stage0_result.news_id, stage=0, dropped_at=0,
                             reason=outcome.reason)
            return Result(stage0=stage0_result)

        ctx = StageContext(
            news=stage0_result.news,
            facts=outcome.facts,
            news_id=stage0_result.news_id,
            event_id=stage0_result.event_id,
            event_context=stage0_result.event_context,
            source_count=stage0_result.source_count,
        )
        result = Result(stage0=stage0_result, ctx=ctx)

        # --- stage 1: relevance ------------------------------------------- #
        await stage1.run(self.runner, ctx)
        if ctx.dropped_at:
            self.stats.dropped_stage1 += 1
            await self._mark(ctx.news_id, stage=1, dropped_at=1, reason=ctx.drop_reason)
            return result

        # --- stage 2: market significance --------------------------------- #
        await stage2.run(self.runner, ctx)
        if ctx.dropped_at:
            self.stats.dropped_stage2 += 1
            await self._mark(ctx.news_id, stage=2, dropped_at=2, reason=ctx.drop_reason)
            return result

        # --- stage 3: affected assets ------------------------------------- #
        await stage3.run(self.runner, ctx, stage2_reason=stage2.reason(ctx.stage2))
        if ctx.dropped_at:
            self.stats.dropped_stage3 += 1
            await self._mark(ctx.news_id, stage=3, dropped_at=3, reason=ctx.drop_reason)
            return result

        # --- stage 4: the forecast ---------------------------------------- #
        await stage4.run(self.runner, ctx)
        if not ctx.impacts:
            await self._mark(ctx.news_id, stage=4, dropped_at=4,
                             reason="no impact could be computed")
            return result
        self.stats.analysed += 1

        # --- stage 5: audit, escalated only ------------------------------- #
        stage5_summaries: dict[str, str] = {}
        escalations = self.router.stage5_assets(ctx)
        if escalations:
            consensuses = await stage5.run(
                self.runner, ctx, [pick.asset for pick in escalations]
            )
            stage5_summaries = {
                asset: stage5.summary(consensus)
                for asset, consensus in consensuses.items()
            }
            self.stats.escalated_stage5 += len(consensuses)

        # --- final layer: NVIDIA, critical only --------------------------- #
        final_picks = self.router.final_assets(ctx)
        if final_picks:
            consensuses = await final_stage.run(
                self.runner,
                ctx,
                [pick.asset for pick in final_picks],
                stage5_summaries=stage5_summaries,
            )
            self.stats.escalated_final += len(consensuses)

        # --- persist and decide what to publish --------------------------- #
        result.alerts = await self._publish(ctx, decision, escalated=bool(escalations))
        deepest = max((_DEPTH.get(i.source, 4) for i in ctx.impacts.values()), default=4)
        await self._mark(ctx.news_id, stage=deepest)
        return result

    # -------------------------------------------------------------- persisting

    async def _publish(
        self,
        ctx: StageContext,
        decision: Stage0Decision,
        *,
        escalated: bool,
    ) -> list[Alert]:
        """Store every verdict and its prediction; return the ones worth sending."""
        alerts: list[Alert] = []
        urgency = stage2.urgency(ctx.stage2)

        for asset, impact in ctx.impacts.items():
            analysis_id = await self._save_analysis(ctx, impact, escalated=escalated)
            await self._save_prediction(ctx, impact, analysis_id)

            worth, score = self.router.should_alert(ctx, impact)
            if not worth:
                log.debug("not alerting %s (score %.1f, %s)", asset, score, impact.source)
                continue
            if not await self._alert_is_new(ctx, asset, decision):
                continue

            alerts.append(
                Alert(
                    news=ctx.news,
                    facts=ctx.facts,
                    impact=impact,
                    score=score,
                    urgency=urgency,
                    event_id=ctx.event_id,
                    news_id=ctx.news_id,
                    analysis_id=analysis_id,
                    source_count=ctx.source_count,
                    decision=decision,
                    trail=list(ctx.trail.get(asset, [])),
                )
            )
            self.stats.alerts += 1
        return alerts

    async def _alert_is_new(
        self, ctx: StageContext, asset: str, decision: Stage0Decision
    ) -> bool:
        """Suppress a repeat alert for an event/asset already published.

        Only for NEW. An UPDATE reached this point precisely because Stage 0 found
        the event's *state* had changed — that is the thing worth telling someone
        about a story they have already seen, and suppressing it would make the
        event-tracking machinery pointless.
        """
        if decision is not Stage0Decision.NEW or ctx.event_id is None:
            return True
        try:
            if await self.store.already_alerted(ctx.event_id, asset):
                log.info("suppressing repeat alert for event %s / %s", ctx.event_id, asset)
                return False
        except Exception as exc:
            log.warning("could not check alert history: %s", exc)
        return True

    async def _save_analysis(
        self, ctx: StageContext, impact: AssetImpact, *, escalated: bool
    ) -> int | None:
        if ctx.news_id is None:
            return None
        try:
            return await self.store.save_analysis(
                ctx.news_id,
                ctx.event_id,
                impact,
                deepest_stage=_DEPTH.get(impact.source, 4),
                stage2_score=ctx.stage2_score,
                escalated=escalated,
                final_reviewed=impact.source == "final",
                tradeable=impact.tradeable,
                detail={
                    "trail": ctx.trail.get(impact.asset, []),
                    "market_wide": ctx.market_wide,
                    "links": [link.to_json() for link in ctx.links],
                    "key_uncertainty": impact.key_uncertainty,
                    "notes": impact.notes,
                },
            )
        except Exception as exc:
            log.error("could not save analysis for %s: %s", impact.asset, exc)
            return None

    async def _save_prediction(
        self, ctx: StageContext, impact: AssetImpact, analysis_id: int | None
    ) -> None:
        """Record the falsifiable claim, published or not.

        Skipped only when no model answered — there is no claim to score, and a
        placeholder row would pollute the calibration data with a forecast the
        system never actually made.
        """
        if ctx.news_id is None or impact.model_count == 0:
            return
        if impact.direction is Direction.MIXED:
            return  # no directional claim to score
        try:
            await self.store.save_prediction(
                Prediction.from_impact(impact, event_id=ctx.event_id),
                analysis_id=analysis_id,
                news_id=ctx.news_id,
                model_ids=[],
                deepest_stage=_DEPTH.get(impact.source, 4),
            )
        except Exception as exc:
            log.error("could not save prediction for %s: %s", impact.asset, exc)

    async def _mark(
        self,
        news_id: int | None,
        *,
        stage: int,
        dropped_at: int | None = None,
        reason: str = "",
    ) -> None:
        if news_id is None:
            return
        try:
            await self.store.mark_processed(
                news_id, stage=stage, dropped_at=dropped_at,
                drop_reason=reason or None,
            )
        except Exception as exc:
            log.error("could not mark news %s processed: %s", news_id, exc)

    # ------------------------------------------------------------------ stats

    def stats_snapshot(self) -> dict:
        """Combined Stage 0 and analysis counters, for ``save_run`` and /status."""
        merged = {**self.stage0.stats.as_dict(), **self.stats.as_dict()}
        merged["heavy_budget_used"] = self.router.final_spent
        merged["heavy_budget_skipped"] = len(self.router.skipped_for_budget)
        return merged


__all__ = ["Alert", "Pipeline", "PipelineStats", "Result"]
