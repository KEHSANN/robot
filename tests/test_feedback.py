"""Tests for the feedback loop.

The failure modes here are quiet ones. A scoring bug does not crash anything — it
produces plausible numbers that are wrong, and the system then optimises toward
them. So these tests are mostly about the definitions:

- a baseline read at the wrong time (the single most damaging bug in this package,
  because it makes fast-moving events score as misses and slow ones as hits)
- a horizon sampled twice, or never
- a wrong direction that scores near a right one
- an unpriceable asset kept open forever
- a stats write failing and taking a resolved outcome with it
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from database.base import OpenPrediction
from feedback.observe import BASELINE_DEADLINE, Observer
from feedback.prices import (
    Candle,
    PriceSource,
    PriceUnavailable,
    is_priceable,
    normalise_asset,
    pct_change,
)
from feedback.score import (
    combined_score,
    direction_of,
    expected_range,
    magnitude_error,
    performance_summary,
    score_prediction,
)
from services.config import FeedbackSettings, Settings
from services.types import Direction

#: The observer reads the wall clock, so the fixtures are anchored to the real
#: now and every prediction is expressed as an offset from it. A fixed date here
#: would make every horizon permanently overdue and the scheduling tests vacuous.
#: Truncated to the minute because a candle is identified by its minute.
NOW = datetime.now(timezone.utc).replace(second=0, microsecond=0)


def feedback_settings(**kwargs) -> FeedbackSettings:
    base = {
        "horizons_minutes": [15, 60, 180, 360, 1440],
        "binance_base_url": "https://price.test",
        "quote_asset": "USDT",
        "direction_deadband_pct": 0.15,
    }
    base.update(kwargs)
    return FeedbackSettings(**base)


def app_settings(**kwargs) -> Settings:
    return Settings(feedback=feedback_settings(**kwargs))


def prediction(**kwargs) -> OpenPrediction:
    base = dict(
        id=1,
        asset="BTC",
        direction="BULLISH",
        expected_low=3.0,
        expected_high=7.5,
        confidence=0.82,
        horizon_minutes=180,
        created_at=NOW - timedelta(hours=4),
        baseline_price=100.0,
        model_ids=["a/model-1", "b/model-2"],
        event_type="REGULATION",
        deepest_stage=4,
        observations={},
    )
    base.update(kwargs)
    return OpenPrediction(**base)


# --------------------------------------------------------------------------- #
# price plumbing
# --------------------------------------------------------------------------- #


class FakePrices:
    """A price API with a fixed series, recording exactly what was asked for."""

    def __init__(self, series: dict[str, dict[int, float]] | None = None) -> None:
        #: asset -> {epoch minute -> price}. A missing minute falls back to `flat`.
        self.series = series or {}
        self.flat: dict[str, float] = {"BTC": 100.0, "ETH": 50.0, "SOL": 20.0}
        self.klines: list[tuple[str, int]] = []
        self.tickers: list[list[str]] = []
        self.unknown: set[str] = set()
        self.fail_with: Exception | None = None

    def price(self, symbol: str, minute: int) -> float:
        asset = symbol.removesuffix("USDT")
        return self.series.get(asset, {}).get(minute, self.flat.get(asset, 0.0))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.fail_with is not None:
            raise self.fail_with

        params = dict(request.url.params)
        if request.url.path == "/api/v3/klines":
            symbol = params["symbol"]
            start = int(params["startTime"])
            self.klines.append((symbol, start))
            if symbol in self.unknown:
                return httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."})
            minute = start // 60_000
            price = self.price(symbol, minute)
            return httpx.Response(
                200,
                json=[[start, str(price), str(price * 1.01), str(price * 0.99), str(price * 1.005),
                       "1.0", start + 59_999, "0", 1, "0", "0", "0"]],
            )

        if request.url.path == "/api/v3/ticker/price":
            minute = int(NOW.timestamp()) // 60
            if "symbols" in params:
                symbols = json.loads(params["symbols"])
                self.tickers.append(symbols)
                return httpx.Response(
                    200,
                    json=[
                        {"symbol": symbol, "price": str(self.price(symbol, minute))}
                        for symbol in symbols
                        if symbol not in self.unknown
                    ],
                )
            symbol = params["symbol"]
            self.tickers.append([symbol])
            if symbol in self.unknown:
                return httpx.Response(400, json={"code": -1121, "msg": "Invalid symbol."})
            return httpx.Response(200, json={"symbol": symbol, "price": str(self.price(symbol, minute))})

        return httpx.Response(404, json={"msg": "not found"})


def source_for(fake: FakePrices, config: FeedbackSettings | None = None) -> PriceSource:
    http = httpx.AsyncClient(transport=httpx.MockTransport(fake), base_url="https://price.test")
    return PriceSource(config or feedback_settings(), http=http)


class FakeStore:
    def __init__(self, predictions: list[OpenPrediction] | None = None) -> None:
        self.predictions = predictions or []
        self.baselines: dict[int, float] = {}
        self.observations: list = []
        self.outcomes: list = []
        self.bumps: list[dict] = []
        self.bump_fails = False

    async def open_predictions(self, *, now, limit=200):
        return list(self.predictions)[:limit]

    async def set_baseline_price(self, prediction_id: int, price: float) -> None:
        self.baselines[prediction_id] = price

    async def save_observation(self, observation) -> int:
        self.observations.append(observation)
        return len(self.observations)

    async def save_outcome(self, outcome) -> int:
        self.outcomes.append(outcome)
        return len(self.outcomes)

    async def bump_model_performance(self, **kwargs) -> None:
        if self.bump_fails:
            raise RuntimeError("stats table is locked")
        self.bumps.append(kwargs)


def observer_for(store: FakeStore, fake: FakePrices, **settings_kwargs) -> Observer:
    config = app_settings(**settings_kwargs)
    return Observer(store, prices=source_for(fake, config.feedback), config=config)


# --------------------------------------------------------------------------- #
# asset normalisation
# --------------------------------------------------------------------------- #


def test_model_names_map_to_tickers():
    assert normalise_asset("bitcoin") == "BTC"
    assert normalise_asset("$ETH") == "ETH"
    assert normalise_asset("SOL/USD") == "SOL"
    assert normalise_asset(" MATIC ") == "POL"


def test_market_wide_assets_are_not_priceable():
    # Stage 3 returns these deliberately; they must not be treated as bad output.
    for asset in ("CRYPTO", "MARKET", "ALTCOINS", "DEFI", "SPX"):
        assert not is_priceable(asset), asset


def test_stablecoins_are_not_priceable():
    # USDT/USDT is 1.0 by construction, so a direction call on it is meaningless.
    assert not is_priceable("USDT")
    assert not is_priceable("usdc")


def test_real_assets_are_priceable():
    for asset in ("BTC", "eth", "bitcoin", "PEPE"):
        assert is_priceable(asset), asset


def test_pct_change_is_signed():
    assert pct_change(100.0, 105.0) == pytest.approx(5.0)
    assert pct_change(100.0, 95.0) == pytest.approx(-5.0)
    # No baseline means no measurable move, not an exception in the hot path.
    assert pct_change(0.0, 95.0) == 0.0


# --------------------------------------------------------------------------- #
# the baseline
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_baseline_uses_the_candle_open_at_the_prediction_time():
    """The single most damaging bug this package can have.

    If the baseline is read as "the price now" instead of "the price then", a move
    that already happened becomes invisible and every fast event scores as a miss.
    """
    created = NOW - timedelta(hours=2)
    minute = int(created.timestamp()) // 60
    fake = FakePrices({"BTC": {minute: 100.0, int(NOW.timestamp()) // 60: 180.0}})

    async with source_for(fake) as prices:
        price = await prices.price_at("BTC", created)

    assert price == 100.0, "baseline must be the price when the news broke"
    assert fake.klines, "a historical price must come from a candle, not the ticker"


@pytest.mark.asyncio
async def test_a_fresh_prediction_reads_spot_not_a_candle():
    # A prediction made seconds ago has no completed candle yet, so requiring one
    # would fail on exactly the predictions that matter most.
    fake = FakePrices()
    async with source_for(fake) as prices:
        price = await prices.price_at("BTC", datetime.now(timezone.utc))

    assert price == 100.0
    assert fake.tickers and not fake.klines


@pytest.mark.asyncio
async def test_candle_open_is_used_not_close():
    when = NOW - timedelta(hours=1)
    minute = int(when.timestamp()) // 60
    fake = FakePrices({"BTC": {minute: 200.0}})

    async with source_for(fake) as prices:
        candle = await prices.candle_at("BTC", when)

    assert isinstance(candle, Candle)
    assert candle.open == 200.0
    assert candle.close != candle.open, "the fake must distinguish them for this to prove anything"


@pytest.mark.asyncio
async def test_unknown_symbol_is_asked_for_once():
    fake = FakePrices()
    fake.unknown.add("WHATEVERUSDT")

    async with source_for(fake) as prices:
        with pytest.raises(PriceUnavailable):
            await prices.spot("WHATEVER")
        first = len(fake.tickers)
        with pytest.raises(PriceUnavailable):
            await prices.spot("WHATEVER")

    assert len(fake.tickers) == first, "a delisted token must not be re-requested every pass"


@pytest.mark.asyncio
async def test_network_failure_raises_price_unavailable_not_httpx():
    fake = FakePrices()
    fake.fail_with = httpx.ConnectError("no route")

    async with source_for(fake) as prices:
        with pytest.raises(PriceUnavailable):
            await prices.spot("BTC")


@pytest.mark.asyncio
async def test_spot_many_is_one_request():
    fake = FakePrices()
    async with source_for(fake) as prices:
        prices_out = await prices.spot_many(["BTC", "ETH", "SOL"])

    assert prices_out == {"BTC": 100.0, "ETH": 50.0, "SOL": 20.0}
    assert len(fake.tickers) == 1, "forty requests per pass is how a public API starts refusing"


@pytest.mark.asyncio
async def test_spot_many_skips_unpriceable_assets_without_losing_the_batch():
    fake = FakePrices()
    async with source_for(fake) as prices:
        out = await prices.spot_many(["BTC", "CRYPTO", "USDT", "ETH"])

    assert set(out) == {"BTC", "ETH"}, "one unlisted asset must not cost the batch its prices"


# --------------------------------------------------------------------------- #
# scoring: direction
# --------------------------------------------------------------------------- #


def test_noise_is_not_a_correct_direction_call():
    # +0.02% is not a bull call coming true; without the deadband every model
    # scores ~50% on direction by chance and the metric says nothing.
    scored = score_prediction(
        direction="BULLISH", expected_low=3.0, expected_high=7.5,
        horizon_minutes=60, observations={60: 0.02}, config=feedback_settings(),
    )
    assert scored is not None
    assert not scored.direction_correct
    assert not scored.moved


def test_a_real_move_in_the_called_direction_is_correct():
    scored = score_prediction(
        direction="BULLISH", expected_low=3.0, expected_high=7.5,
        horizon_minutes=60, observations={60: 4.0}, config=feedback_settings(),
    )
    assert scored.direction_correct
    assert scored.magnitude_error == 0.0


def test_bearish_call_is_scored_against_a_negative_range():
    # The stages report magnitude unsigned with the sign carried by direction, so
    # a bearish call that fell 5% must land inside its range, not outside it.
    scored = score_prediction(
        direction="BEARISH", expected_low=3.0, expected_high=7.5,
        horizon_minutes=60, observations={60: -5.0}, config=feedback_settings(),
    )
    assert scored.direction_correct
    assert scored.magnitude_error == 0.0


def test_signed_magnitudes_from_a_model_are_accepted():
    # Some models return the range already signed; rejecting that form would
    # silently drop every forecast from that model.
    assert expected_range(-7.5, -3.0, Direction.BEARISH) == (-7.5, -3.0)
    assert expected_range(3.0, 7.5, Direction.BEARISH) == (-7.5, -3.0)


def test_neutral_is_a_real_claim_that_can_be_right():
    scored = score_prediction(
        direction="NEUTRAL", expected_low=0.0, expected_high=0.5,
        horizon_minutes=60, observations={60: 0.05}, config=feedback_settings(),
    )
    assert scored.direction_correct, "a NEUTRAL call is correct when the price stayed put"


def test_neutral_is_wrong_when_the_market_moved():
    scored = score_prediction(
        direction="NEUTRAL", expected_low=0.0, expected_high=0.5,
        horizon_minutes=60, observations={60: 4.0}, config=feedback_settings(),
    )
    assert not scored.direction_correct, "NEUTRAL must not be a free pass"


def test_direction_of_respects_the_deadband():
    assert direction_of(0.5, 0.15) is Direction.BULLISH
    assert direction_of(-0.5, 0.15) is Direction.BEARISH
    assert direction_of(0.1, 0.15) is Direction.NEUTRAL
    assert direction_of(-0.1, 0.15) is Direction.NEUTRAL


# --------------------------------------------------------------------------- #
# scoring: magnitude
# --------------------------------------------------------------------------- #


def test_anywhere_inside_the_range_is_zero_error():
    # The models were asked for a range, so 5% is not a worse answer than 4%.
    assert magnitude_error(3.0, 3.0, 7.5) == 0.0
    assert magnitude_error(5.0, 3.0, 7.5) == 0.0
    assert magnitude_error(7.5, 3.0, 7.5) == 0.0


def test_error_is_distance_to_the_nearest_edge():
    assert magnitude_error(9.0, 3.0, 7.5) == pytest.approx(1.5)
    assert magnitude_error(1.0, 3.0, 7.5) == pytest.approx(2.0)


def test_a_narrow_range_is_not_rewarded_over_an_honest_one():
    # Both got the direction right and the move was 5%. The wide range said 3-7.5
    # and was right; the narrow one said 4.9-5.1 and was also right. Neither
    # should be penalised, so the tight range must not out-score the honest one
    # merely for being tight.
    wide = score_prediction(
        direction="BULLISH", expected_low=3.0, expected_high=7.5,
        horizon_minutes=60, observations={60: 5.0}, config=feedback_settings(),
    )
    narrow = score_prediction(
        direction="BULLISH", expected_low=4.9, expected_high=5.1,
        horizon_minutes=60, observations={60: 5.0}, config=feedback_settings(),
    )
    assert wide.score == narrow.score == 1.0


# --------------------------------------------------------------------------- #
# scoring: the combined number
# --------------------------------------------------------------------------- #


def test_a_wrong_direction_never_outscores_a_right_one():
    """A wrong direction reads as a reason to take the losing side.

    So the best possible wrong call must score below the worst possible right one,
    whatever the magnitudes were.
    """
    best_wrong = combined_score(False, 0.0, 3.0, 7.5)
    worst_right = combined_score(True, 500.0, 3.0, 7.5)
    assert best_wrong < worst_right


def test_a_perfect_call_scores_one():
    assert combined_score(True, 0.0, 3.0, 7.5) == 1.0


def test_a_right_direction_with_a_bad_size_still_pays_for_direction():
    score = combined_score(True, 50.0, 3.0, 7.5)
    assert score == pytest.approx(0.65, abs=0.01), "direction is 65% of the score"


def test_size_error_is_scaled_by_the_width_of_the_forecast():
    # Being 2% off a 10% call is a better forecast than being 2% off a 0.5% call.
    big = combined_score(True, 2.0, 8.0, 12.0)
    small = combined_score(True, 2.0, 0.2, 0.5)
    assert big > small


# --------------------------------------------------------------------------- #
# horizon diagnostic
# --------------------------------------------------------------------------- #


def test_best_horizon_prefers_the_called_horizon_on_a_tie():
    # 60m and 180m both land inside 3-7.5. Reporting 60m would say a well-timed
    # prediction was mistimed, biasing the diagnostic in one direction.
    scored = score_prediction(
        direction="BULLISH", expected_low=3.0, expected_high=7.5,
        horizon_minutes=180,
        observations={15: 1.2, 60: 3.4, 180: 5.1, 360: 8.0},
        config=feedback_settings(),
    )
    assert scored.best_horizon_minutes == 180


def test_best_horizon_reveals_a_slow_market():
    # The call was for 15 minutes; the move only arrived at 6 hours. That is a
    # different correction from being wrong outright, so it must be visible.
    scored = score_prediction(
        direction="BULLISH", expected_low=3.0, expected_high=7.5,
        horizon_minutes=15,
        observations={15: 0.1, 60: 0.4, 180: 1.2, 360: 5.0},
        config=feedback_settings(),
    )
    assert not scored.direction_correct, "at 15m the call had not come true"
    assert scored.best_horizon_minutes == 360


def test_best_horizon_never_picks_a_wrong_side_sample():
    # -4% is 7% from the bottom of a +3..+7.5 range; +1% is 2% away. But -4% is
    # on the wrong side, so it can never be the "best" match.
    scored = score_prediction(
        direction="BULLISH", expected_low=3.0, expected_high=7.5,
        horizon_minutes=60, observations={60: 1.0, 180: -4.0},
        config=feedback_settings(),
    )
    assert scored.best_horizon_minutes == 60


def test_scoring_needs_at_least_one_observation():
    assert score_prediction(
        direction="BULLISH", expected_low=3.0, expected_high=7.5,
        horizon_minutes=60, observations={}, config=feedback_settings(),
    ) is None


def test_the_nearest_observation_is_used_when_the_horizon_is_missing():
    # A prediction must not go unscored because one sample was missed.
    scored = score_prediction(
        direction="BULLISH", expected_low=3.0, expected_high=7.5,
        horizon_minutes=180, observations={60: 4.0, 360: -9.0},
        config=feedback_settings(),
    )
    assert scored.actual_pct == 4.0


# --------------------------------------------------------------------------- #
# model ranking
# --------------------------------------------------------------------------- #


def test_three_for_three_does_not_top_the_ranking():
    """Ranking without a sample-size floor promotes whichever model answered least.

    That is precisely the wrong signal to route traffic on.
    """
    ranked = performance_summary([
        {"model_id": "lucky", "predictions": 3, "direction_correct": 3},
        {"model_id": "proven", "predictions": 200, "direction_correct": 150},
    ])
    assert ranked[0]["model_id"] == "proven"
    assert ranked[0]["hit_rate"] == pytest.approx(0.75)


def test_models_with_no_predictions_are_omitted():
    ranked = performance_summary([
        {"model_id": "silent", "predictions": 0, "direction_correct": 0},
        {"model_id": "active", "predictions": 10, "direction_correct": 6},
    ])
    assert [row["model_id"] for row in ranked] == ["active"]


# --------------------------------------------------------------------------- #
# the observer
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_missing_baseline_is_backfilled_from_the_prediction_time():
    created = NOW - timedelta(minutes=30)
    minute = int(created.timestamp()) // 60
    fake = FakePrices({"BTC": {minute: 100.0}})
    store = FakeStore([prediction(baseline_price=None, created_at=created, horizon_minutes=15)])

    async with observer_for(store, fake) as observer:
        stats = await observer.run_once()

    assert store.baselines == {1: 100.0}
    assert stats.baselines_set == 1


@pytest.mark.asyncio
async def test_due_horizons_are_sampled_and_earlier_ones_are_not_redone():
    created = NOW - timedelta(hours=4)
    fake = FakePrices()
    store = FakeStore([
        prediction(created_at=created, observations={15: 1.0, 60: 2.0}),
    ])

    async with observer_for(store, fake) as observer:
        await observer.run_once()

    sampled = sorted(obs.offset_minutes for obs in store.observations)
    assert sampled == [180], "15 and 60 were already recorded; 360 and 1440 are not due"


@pytest.mark.asyncio
async def test_an_undue_horizon_is_not_sampled_early():
    fake = FakePrices()
    store = FakeStore([prediction(created_at=NOW - timedelta(minutes=5), horizon_minutes=15)])

    async with observer_for(store, fake) as observer:
        stats = await observer.run_once()

    assert store.observations == []
    assert stats.pending == 1


@pytest.mark.asyncio
async def test_the_move_is_measured_from_the_baseline_not_from_the_previous_sample():
    created = NOW - timedelta(hours=4)
    at_180 = int((created + timedelta(minutes=180)).timestamp()) // 60
    fake = FakePrices({"BTC": {at_180: 105.0}})
    store = FakeStore([prediction(created_at=created, baseline_price=100.0)])

    async with observer_for(store, fake) as observer:
        await observer.run_once()

    observation = next(o for o in store.observations if o.offset_minutes == 180)
    assert observation.pct_change == pytest.approx(5.0)
    assert observation.price == 105.0


@pytest.mark.asyncio
async def test_a_prediction_resolves_at_its_own_horizon_not_the_last_one():
    # A 15-minute call kept open until the 24-hour sample means the fast calls are
    # the last to produce feedback, which is backwards.
    created = NOW - timedelta(minutes=20)
    at_15 = int((created + timedelta(minutes=15)).timestamp()) // 60
    fake = FakePrices({"BTC": {at_15: 104.0}})
    store = FakeStore([prediction(created_at=created, horizon_minutes=15)])

    async with observer_for(store, fake) as observer:
        stats = await observer.run_once()

    assert stats.resolved == 1
    assert store.outcomes[0].direction_correct
    assert store.outcomes[0].actual_pct == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_resolving_credits_every_contributing_model():
    created = NOW - timedelta(minutes=20)
    at_15 = int((created + timedelta(minutes=15)).timestamp()) // 60
    fake = FakePrices({"BTC": {at_15: 104.0}})
    store = FakeStore([
        prediction(created_at=created, horizon_minutes=15, deepest_stage=5,
                   model_ids=["a/one", "b/two", "c/three"]),
    ])

    async with observer_for(store, fake) as observer:
        await observer.run_once()

    assert [bump["model_id"] for bump in store.bumps] == ["a/one", "b/two", "c/three"]
    assert store.bumps[0]["stage"] == 5
    assert store.bumps[0]["event_type"] == "REGULATION"
    assert store.bumps[0]["asset"] == "BTC"
    assert store.bumps[0]["direction_correct"] is True


@pytest.mark.asyncio
async def test_a_failing_stats_write_does_not_lose_the_outcome():
    created = NOW - timedelta(minutes=20)
    fake = FakePrices()
    store = FakeStore([prediction(created_at=created, horizon_minutes=15)])
    store.bump_fails = True

    async with observer_for(store, fake) as observer:
        stats = await observer.run_once()

    assert len(store.outcomes) == 1, "the scored result is the thing that matters"
    assert stats.resolved == 1
    assert store.bumps == []


@pytest.mark.asyncio
async def test_an_unpriceable_asset_is_closed_not_left_open():
    fake = FakePrices()
    store = FakeStore([prediction(asset="CRYPTO", baseline_price=None)])

    async with observer_for(store, fake) as observer:
        stats = await observer.run_once()

    assert stats.unpriceable == 1
    assert len(store.outcomes) == 1
    assert store.outcomes[0].detail["unscored"] is True
    assert store.bumps == [], "the models did not fail; the price feed cannot answer"
    assert fake.tickers == [] and fake.klines == [], "and it must cost no requests"


@pytest.mark.asyncio
async def test_a_prediction_too_old_to_baseline_is_abandoned():
    fake = FakePrices()
    store = FakeStore([
        prediction(baseline_price=None, created_at=NOW - BASELINE_DEADLINE - timedelta(hours=1)),
    ])

    async with observer_for(store, fake) as observer:
        stats = await observer.run_once()

    assert stats.abandoned == 1
    assert store.outcomes[0].detail["unscored"] is True
    assert "baseline" in store.outcomes[0].detail["reason"]


@pytest.mark.asyncio
async def test_an_exchange_outage_leaves_the_prediction_open():
    fake = FakePrices()
    fake.fail_with = httpx.ConnectError("no route")
    store = FakeStore([prediction(created_at=NOW - timedelta(minutes=20), horizon_minutes=15)])

    async with observer_for(store, fake) as observer:
        stats = await observer.run_once()

    assert store.outcomes == [], "a network problem must not resolve a prediction"
    assert stats.pending == 1
    assert stats.errors


@pytest.mark.asyncio
async def test_one_broken_prediction_does_not_end_the_pass():
    class Exploding(FakeStore):
        async def save_observation(self, observation):
            if observation.prediction_id == 1:
                raise RuntimeError("constraint violation")
            return await FakeStore.save_observation(self, observation)

    created = NOW - timedelta(minutes=20)
    fake = FakePrices()
    store = Exploding([
        prediction(id=1, created_at=created, horizon_minutes=15),
        prediction(id=2, asset="ETH", created_at=created, horizon_minutes=15, baseline_price=50.0),
    ])

    async with observer_for(store, fake) as observer:
        stats = await observer.run_once()

    assert stats.errors, "the failure must be reported"
    assert any(obs.prediction_id == 2 for obs in store.observations), "and the rest still run"


@pytest.mark.asyncio
async def test_predictions_sharing_an_asset_and_minute_fetch_one_price():
    """Stage 4, stage 5 and final produce three predictions for the same asset.

    They share a timestamp, so they need the same prices. Fetching each number
    three times is how a public endpoint starts refusing.
    """
    created = NOW - timedelta(hours=4)
    fake = FakePrices()
    store = FakeStore([
        prediction(id=1, created_at=created, deepest_stage=4),
        prediction(id=2, created_at=created, deepest_stage=5),
        prediction(id=3, created_at=created, deepest_stage=6),
    ])

    async with observer_for(store, fake) as observer:
        stats = await observer.run_once()

    # Three predictions, three due horizons each (15, 60, 180 at four hours old).
    assert len(store.observations) == 9, "every prediction is still observed in full"
    assert stats.resolved == 3
    # But only three distinct (asset, minute) pairs behind them.
    assert len(fake.klines) == 3, "nine identical fetches collapse to three"
    assert len(set(fake.klines)) == 3


@pytest.mark.asyncio
async def test_current_prices_for_many_assets_come_back_in_one_request():
    created = NOW - timedelta(minutes=15, seconds=45)
    fake = FakePrices()
    store = FakeStore([
        prediction(id=1, asset="BTC", created_at=created, horizon_minutes=15, baseline_price=100.0),
        prediction(id=2, asset="ETH", created_at=created, horizon_minutes=15, baseline_price=50.0),
        prediction(id=3, asset="SOL", created_at=created, horizon_minutes=15, baseline_price=20.0),
    ])

    async with observer_for(store, fake) as observer:
        await observer.run_once()

    assert len(store.observations) == 3
    assert len(fake.tickers) == 1, "one batch call, not one per asset"
    assert sorted(fake.tickers[0]) == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


@pytest.mark.asyncio
async def test_a_batch_failure_falls_back_to_individual_prices():
    class BatchBreaks(FakePrices):
        def __call__(self, request):
            if "symbols" in dict(request.url.params):
                return httpx.Response(500, json={"msg": "nope"})
            return FakePrices.__call__(self, request)

    created = NOW - timedelta(minutes=15, seconds=45)
    fake = BatchBreaks()
    store = FakeStore([
        prediction(id=1, asset="BTC", created_at=created, horizon_minutes=15, baseline_price=100.0),
        prediction(id=2, asset="ETH", created_at=created, horizon_minutes=15, baseline_price=50.0),
    ])

    async with observer_for(store, fake) as observer:
        await observer.run_once()

    assert len(store.observations) == 2, "the warm-up is an optimisation, not a dependency"


@pytest.mark.asyncio
async def test_an_empty_queue_costs_nothing():
    fake = FakePrices()
    store = FakeStore([])

    async with observer_for(store, fake) as observer:
        stats = await observer.run_once()

    assert stats.considered == 0
    assert fake.tickers == [] and fake.klines == []


@pytest.mark.asyncio
async def test_a_horizon_longer_than_any_configured_sample_still_resolves():
    # A 48-hour call would otherwise stay open forever, re-read on every pass.
    created = NOW - timedelta(days=3)
    fake = FakePrices()
    store = FakeStore([
        prediction(created_at=created, horizon_minutes=2880, observations={1440: 3.0}),
    ])

    async with observer_for(store, fake) as observer:
        stats = await observer.run_once()

    assert stats.resolved == 1
    assert store.outcomes[0].detail["horizon_called"] == 2880


@pytest.mark.asyncio
async def test_the_outcome_detail_carries_every_horizon_seen():
    created = NOW - timedelta(hours=4)
    fake = FakePrices()
    store = FakeStore([prediction(created_at=created, observations={15: 1.0, 60: 2.0})])

    async with observer_for(store, fake) as observer:
        await observer.run_once()

    by_horizon = store.outcomes[0].detail["by_horizon"]
    assert set(by_horizon) == {"15", "60", "180"}
