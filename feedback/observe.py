"""The observation loop: sampling prices and closing out predictions.

This is the half of the system that makes it accountable. Everything before it is
opinion; this is where an opinion becomes a record.

How a prediction moves through it:

1. **Baseline.** Read once, from the minute the prediction was made — not from
   whenever the observer first sees it. A prediction with no baseline cannot be
   scored at all, so this is the one step retried on later passes.
2. **Observation.** At each configured horizon, once that horizon has elapsed, the
   price is sampled and the move from baseline recorded. Horizons already observed
   are skipped, so a pass that runs every five minutes does not re-record the
   15-minute sample nineteen times.
3. **Resolution.** When the prediction's own horizon has been observed, it is
   scored, an outcome is written, and each model that contributed gets the result
   folded into its running averages.

Three decisions worth stating.

**A prediction is resolved at its own horizon, not at the last one.** A 15-minute
call should be scored at 15 minutes; keeping it open until the 24-hour sample
arrives would mean the fast calls — the ones the system is most useful for — are
the last to produce any feedback.

**An unpriceable asset closes unscored rather than staying open.** A prediction
about ``CRYPTO`` as an asset has no price series to check, and leaving it open
means the observer re-reads it on every pass forever. It is closed as unscoreable
and excluded from model averages, because counting it as a miss would blame the
models for a limitation of the price feed.

**Prices are fetched once per (asset, minute), not once per prediction.** One news
item produces a prediction per asset, and a high-impact item produces stage 4,
stage 5 and final predictions for the *same* asset at the same timestamp. Without
the per-pass cache those all fetch the same number separately, and the current
prices they need are collected into a single batch request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from database.base import OpenPrediction, Store
from feedback.prices import (
    PriceSource,
    PriceUnavailable,
    is_priceable,
    normalise_asset,
    pct_change,
)
from feedback.score import Scored, score_prediction
from services.config import Settings, settings as global_settings
from services.types import Observation, Outcome

log = logging.getLogger(__name__)

#: Grace period before a horizon is considered due. Sampling exactly on the
#: minute would race the exchange's own candle close.
HORIZON_GRACE = timedelta(seconds=30)

#: How long a prediction may go without a usable baseline before it is abandoned.
#: Past this the price history is still available, but the move it was supposed to
#: measure is long over, so scoring it would be fiction.
BASELINE_DEADLINE = timedelta(hours=6)

#: A moment this close to now is answered by the spot ticker rather than a candle.
#: Must match :meth:`feedback.prices.PriceSource.price_at`, which is what decides
#: it — this constant only tells the batch warm-up which samples spot can serve.
SPOT_WINDOW = timedelta(minutes=2)


@dataclass
class ObserveStats:
    considered: int = 0
    baselines_set: int = 0
    observations: int = 0
    resolved: int = 0
    unpriceable: int = 0
    pending: int = 0
    abandoned: int = 0
    requests: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "considered": self.considered,
            "baselines_set": self.baselines_set,
            "observations": self.observations,
            "resolved": self.resolved,
            "unpriceable": self.unpriceable,
            "pending": self.pending,
            "abandoned": self.abandoned,
            "requests": self.requests,
            "errors": len(self.errors),
        }

    def summary(self) -> str:
        return (
            f"{self.observations} samples, {self.resolved} resolved, "
            f"{self.pending} open, {self.baselines_set} baselines, "
            f"{self.unpriceable} unpriceable, {self.requests} requests, "
            f"{len(self.errors)} errors"
        )


class Observer:
    """Samples prices for open predictions and scores the ones that are due."""

    def __init__(
        self,
        store: Store,
        prices: PriceSource | None = None,
        config: Settings | None = None,
    ) -> None:
        self.settings = config or global_settings
        self.store = store
        self._prices = prices
        self._owned: PriceSource | None = None
        #: (asset, minute) -> price, cleared at the start of every pass.
        self._cache: dict[tuple[str, int], float] = {}

    async def __aenter__(self) -> "Observer":
        if self._prices is None:
            self._owned = PriceSource(self.settings.feedback)
            self._prices = await self._owned.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._owned is not None:
            await self._owned.__aexit__(*exc)
            self._owned = None
            self._prices = None

    @property
    def prices(self) -> PriceSource:
        if self._prices is None:
            raise RuntimeError("Observer must be used as an async context manager")
        return self._prices

    @property
    def horizons(self) -> list[int]:
        return sorted({int(h) for h in self.settings.feedback.horizons_minutes if int(h) > 0})

    # -------------------------------------------------------------------- pass

    async def run_once(self, limit: int = 200) -> ObserveStats:
        """One observation pass over every open prediction."""
        stats = ObserveStats()
        now = datetime.now(timezone.utc)
        self._cache.clear()

        before = self.prices.requests
        predictions = await self.store.open_predictions(now=now, limit=limit)
        stats.considered = len(predictions)
        if not predictions:
            return stats

        await self._warm_spot_cache(predictions, now)

        for prediction in predictions:
            try:
                await self._advance(prediction, now, stats)
            except PriceUnavailable as exc:
                # The exchange's problem, not the prediction's. Leave it open.
                stats.pending += 1
                stats.errors.append(f"{prediction.asset}: {exc}"[:200])
                log.debug("price unavailable for prediction %s: %s", prediction.id, exc)
            except Exception as exc:  # noqa: BLE001 - one bad row must not end the pass
                stats.errors.append(f"{prediction.asset}: {type(exc).__name__}: {exc}"[:200])
                log.exception("observing prediction %s failed", prediction.id)

        stats.requests = self.prices.requests - before
        log.info("observe: %s", stats.summary())
        return stats

    async def _advance(
        self, prediction: OpenPrediction, now: datetime, stats: ObserveStats
    ) -> None:
        if not is_priceable(prediction.asset):
            await self._close_unscored(prediction, "asset has no tradeable price series")
            stats.unpriceable += 1
            return

        baseline = prediction.baseline_price
        if not baseline or baseline <= 0:
            baseline = await self._establish_baseline(prediction, now, stats)
            if baseline is None:
                return

        observations = dict(prediction.observations)
        for offset in self._due_horizons(prediction, now, observations):
            price = await self._price_at(
                prediction.asset, prediction.created_at + timedelta(minutes=offset)
            )
            change = pct_change(baseline, price)
            observations[offset] = change

            await self.store.save_observation(
                Observation(
                    prediction_id=prediction.id,
                    asset=prediction.asset,
                    offset_minutes=offset,
                    price=price,
                    pct_change=change,
                )
            )
            stats.observations += 1

        if self._is_resolvable(prediction, now, observations):
            await self._resolve(prediction, observations, stats)
        else:
            stats.pending += 1

    # ------------------------------------------------------------------ prices

    async def _price_at(self, asset: str, when: datetime) -> float:
        """Price at a moment, fetched at most once per pass per minute."""
        key = self._cache_key(asset, when)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        price = await self.prices.price_at(asset, when)
        if price > 0:
            self._cache[key] = price
        return price

    @staticmethod
    def _cache_key(asset: str, when: datetime) -> tuple[str, int]:
        moment = when.astimezone(timezone.utc).replace(second=0, microsecond=0)
        return (normalise_asset(asset), int(moment.timestamp()))

    async def _warm_spot_cache(
        self, predictions: list[OpenPrediction], now: datetime
    ) -> None:
        """Fill the cache for every sample the spot ticker can answer, in one call.

        Only samples inside :data:`SPOT_WINDOW` qualify — anything older needs its
        own historical candle, which the batch endpoint cannot serve. Failure here
        is not an error: the per-prediction path will fetch what is missing.
        """
        wanted: dict[str, tuple[str, int]] = {}
        for prediction in predictions:
            if not is_priceable(prediction.asset):
                continue
            for moment in self._samples_for(prediction, now):
                if abs(now - moment) >= SPOT_WINDOW:
                    continue
                key = self._cache_key(prediction.asset, moment)
                wanted[key[0]] = key

        if len(wanted) < 2:
            # One asset is not a batch; let the normal path fetch it.
            return

        try:
            prices = await self.prices.spot_many(sorted(wanted))
        except PriceUnavailable as exc:
            log.debug("batch price warm-up failed, falling back per asset: %s", exc)
            return

        for asset, price in prices.items():
            key = wanted.get(asset)
            if key and price > 0:
                self._cache[key] = price

    def _samples_for(
        self, prediction: OpenPrediction, now: datetime
    ) -> list[datetime]:
        """The moments this prediction will ask about on this pass."""
        moments: list[datetime] = []
        if not prediction.baseline_price or prediction.baseline_price <= 0:
            moments.append(prediction.created_at)
        for offset in self._due_horizons(prediction, now, prediction.observations):
            moments.append(prediction.created_at + timedelta(minutes=offset))
        return moments

    # ---------------------------------------------------------------- baseline

    async def _establish_baseline(
        self, prediction: OpenPrediction, now: datetime, stats: ObserveStats
    ) -> float | None:
        """Read and store the price at the moment of the prediction.

        Taken from the candle containing ``created_at``, so a delayed first pass
        does not silently redefine the starting point to be after the move.
        """
        if now - prediction.created_at > BASELINE_DEADLINE:
            hours = BASELINE_DEADLINE.total_seconds() / 3600
            await self._close_unscored(prediction, f"no baseline within {hours:.0f}h")
            stats.abandoned += 1
            return None

        price = await self._price_at(prediction.asset, prediction.created_at)
        if price <= 0:
            stats.pending += 1
            return None

        await self.store.set_baseline_price(prediction.id, price)
        stats.baselines_set += 1
        log.debug(
            "baseline for %s at %s: %s",
            prediction.asset, prediction.created_at.isoformat(), price,
        )
        return price

    # -------------------------------------------------------------- scheduling

    def _due_horizons(
        self, prediction: OpenPrediction, now: datetime, observations: dict[int, float]
    ) -> list[int]:
        """Horizons that have elapsed and have not been sampled yet.

        The prediction's own horizon is always included, even when it is not one of
        the configured offsets, because that is the sample it gets scored on.
        """
        wanted = set(self.horizons)
        wanted.add(int(prediction.horizon_minutes))

        elapsed = now - prediction.created_at
        return sorted(
            offset
            for offset in wanted
            if offset > 0
            and offset not in observations
            and elapsed >= timedelta(minutes=offset) + HORIZON_GRACE
        )

    def _is_resolvable(
        self, prediction: OpenPrediction, now: datetime, observations: dict[int, float]
    ) -> bool:
        """Whether enough is known to score it.

        Either the prediction's own horizon has been sampled, or every configured
        horizon has passed — the second case closes out a prediction whose horizon
        is longer than anything the observer samples.
        """
        if not observations:
            return False
        if int(prediction.horizon_minutes) in observations:
            return True
        longest = max(self.horizons) if self.horizons else int(prediction.horizon_minutes)
        return now - prediction.created_at >= timedelta(minutes=longest) + HORIZON_GRACE

    # -------------------------------------------------------------- resolution

    async def _resolve(
        self, prediction: OpenPrediction, observations: dict[int, float], stats: ObserveStats
    ) -> None:
        scored = score_prediction(
            direction=prediction.direction,
            expected_low=prediction.expected_low,
            expected_high=prediction.expected_high,
            horizon_minutes=int(prediction.horizon_minutes),
            observations=observations,
            config=self.settings.feedback,
        )
        if scored is None:
            stats.pending += 1
            return

        await self.store.save_outcome(
            scored.as_outcome(
                prediction.id,
                asset=prediction.asset,
                called=str(prediction.direction),
                confidence=round(prediction.confidence, 3),
                horizon_called=int(prediction.horizon_minutes),
                event_type=prediction.event_type,
                stage=prediction.deepest_stage,
            )
        )
        stats.resolved += 1

        log.info(
            "prediction %s (%s %s) scored %.2f — direction %s, actual %+.2f%% at %dm, "
            "best horizon %dm",
            prediction.id, prediction.asset, prediction.direction, scored.score,
            "correct" if scored.direction_correct else "wrong",
            scored.actual_pct, prediction.horizon_minutes, scored.best_horizon_minutes,
        )

        await self._credit_models(prediction, scored)

    async def _credit_models(self, prediction: OpenPrediction, scored: Scored) -> None:
        """Fold the result into each contributing model's record.

        Attribution is per ``model × stage × event_type × asset``, as the plan
        specifies. Every model that voted gets the same result, because the verdict
        was a consensus — a model that dissented from a wrong call is not credited
        separately, which is a known limitation: crediting dissent would need the
        per-model votes carried this far.

        A failure to credit one model does not abandon the rest, and never undoes
        the outcome that was already written.
        """
        for model_id in prediction.model_ids or []:
            try:
                await self.store.bump_model_performance(
                    model_id=model_id,
                    stage=int(prediction.deepest_stage),
                    event_type=prediction.event_type or "ALL",
                    asset=normalise_asset(prediction.asset),
                    direction_correct=scored.direction_correct,
                    magnitude_error=scored.magnitude_error,
                    score=scored.score,
                )
            except Exception:  # noqa: BLE001 - a stats write must not lose an outcome
                log.exception("could not credit model %s", model_id)

    async def _close_unscored(self, prediction: OpenPrediction, reason: str) -> None:
        """Resolve a prediction that cannot be checked, without scoring it.

        Written with ``score=0`` and a reason, and credited to no model: the models
        did not fail here, the price feed cannot answer the question. The
        ``unscored`` flag in the detail is what keeps these out of the accuracy
        numbers later.
        """
        log.info(
            "closing prediction %s (%s) unscored: %s", prediction.id, prediction.asset, reason
        )
        await self.store.save_outcome(
            Outcome(
                prediction_id=prediction.id,
                direction_correct=False,
                actual_pct=0.0,
                expected_pct=0.0,
                magnitude_error=0.0,
                score=0.0,
                best_horizon_minutes=0,
                detail={
                    "unscored": True,
                    "reason": reason,
                    "asset": prediction.asset,
                    "called": str(prediction.direction),
                },
            )
        )


async def observe_once(
    store: Store, config: Settings | None = None, limit: int = 200
) -> ObserveStats:
    """One pass, building and closing its own price source."""
    async with Observer(store, config=config) as observer:
        return await observer.run_once(limit)


__all__ = [
    "BASELINE_DEADLINE",
    "HORIZON_GRACE",
    "SPOT_WINDOW",
    "ObserveStats",
    "Observer",
    "observe_once",
]
