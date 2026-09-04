"""Aggregation tests for stages 1-5 and the final layer.

These run without an API. Each test feeds the stage's own ``parse`` the JSON a model
would return, tallies it the way :class:`stages.base.PanelRunner` does, and checks
what the aggregation concluded — so a change to either the parser or the fold is
caught, which is where the judgement actually lives.

The invariants worth protecting, in rough order of how badly a regression would
hurt:

1. An audit that finds problems cannot raise confidence. Otherwise running more
   stages on the events that matter most inflates certainty exactly where being
   wrong is most expensive.
2. A split panel reports MIXED, not the winner of a coin flip.
3. A dead panel changes nothing. An outage must never look like a finding.
4. Size comes from the models that agreed on the direction — averaging in the
   dissenters describes a scenario nobody forecast.
"""

from __future__ import annotations

import pytest

from services.consensus import Consensus, Verdict, Vote, tally
from services.models import ModelSpec
from services.types import (
    AssetImpact,
    AssetLink,
    Causality,
    Direction,
    FactSet,
    Magnitude,
    NewsItem,
    Relation,
)
from stages import final as final_stage
from stages import stage1, stage2, stage3, stage4, stage5
from stages.base import StageContext
from stages.router import Router

# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #


def _spec(index: int) -> ModelSpec:
    return ModelSpec(id=f"test/model-{index}", provider="test", label=f"M{index}")


def votes_for(payloads, parse) -> list[Vote]:
    """Build votes exactly as ``build_votes`` would. ``None`` means no answer."""
    out: list[Vote] = []
    for index, payload in enumerate(payloads):
        spec = _spec(index)
        if payload is None:
            out.append(Vote(model=spec, ok=False, detail={"reason": "no answer"}))
            continue
        value, detail = parse(payload)
        if value is None:
            out.append(Vote(model=spec, ok=False, detail=detail))
        else:
            out.append(Vote(model=spec, ok=True, value=value, detail=detail))
    return out


def consensus_for(payloads, parse, *, reject_votes: int = 3, score_key=None) -> Consensus:
    return tally(
        votes_for(payloads, parse),
        reject_votes=reject_votes,
        score_key=score_key,
    )


def link(asset="BTC", relation=Relation.DIRECT, confidence=0.9) -> AssetLink:
    return AssetLink(asset=asset, relation=relation, confidence=confidence, votes=3)


def impact_of(**kwargs) -> AssetImpact:
    base = dict(
        asset="BTC",
        direction=Direction.BULLISH,
        magnitude=Magnitude.MEDIUM,
        expected_low=1.0,
        expected_high=2.0,
        confidence=0.7,
        horizon_minutes=180,
        causality=Causality.FUNDAMENTAL,
        mechanism="spot ETF inflows create mechanical buy pressure",
        risks="approval could be delayed",
        agreement=0.8,
        model_count=5,
        source="stage4",
    )
    base.update(kwargs)
    return AssetImpact(**base)


def context(score: float | None = None, **kwargs) -> StageContext:
    ctx = StageContext(
        news=NewsItem(title="SEC approves spot bitcoin ETF", source="test"),
        facts=FactSet(event_type="ETF_APPROVAL", entity="SEC", assets=["BTC"]),
        news_id=None,
        **kwargs,
    )
    if score is not None:
        ctx.stage2 = Consensus(verdict=Verdict.PASS, votes_pass=5, score=score)
    return ctx


# --------------------------------------------------------------------------- #
# stage 1 — relevance
# --------------------------------------------------------------------------- #


def test_stage1_keeps_a_real_event():
    vote, _ = stage1.parse(
        {"specific_event": True, "current_event": True, "type": "news", "reason": "filing"}
    )
    assert vote is True


@pytest.mark.parametrize(
    "payload",
    [
        {"specific_event": False, "current_event": True, "type": "news"},
        {"specific_event": True, "current_event": False, "type": "news"},
        {"specific_event": True, "current_event": True, "type": "advertisement"},
        {"specific_event": True, "current_event": True, "type": "price commentary"},
    ],
    ids=["not-specific", "not-current", "advertisement", "price-commentary"],
)
def test_stage1_drops_noise(payload):
    vote, detail = stage1.parse(payload)
    assert vote is False
    assert detail["reason"], "a drop must always carry a reason"


def test_stage1_unanswered_is_not_a_reject():
    """A model that ignored the question must not be counted as objecting."""
    vote, _ = stage1.parse({"comment": "interesting article"})
    assert vote is None

    consensus = consensus_for(
        [
            {"specific_event": True, "current_event": True, "type": "news"},
            {"specific_event": True, "current_event": True, "type": "news"},
            {"comment": "no idea"},
            None,
            None,
        ],
        stage1.parse,
        reject_votes=2,
    )
    assert consensus.verdict is Verdict.PASS
    assert consensus.votes_failed == 3


# --------------------------------------------------------------------------- #
# stage 2 — significance
# --------------------------------------------------------------------------- #


def test_stage2_infers_worth_from_score_alone():
    high, detail = stage2.parse({"score": 72})
    low, _ = stage2.parse({"score": 22})
    assert high is True and low is False
    assert detail["note"] == "inferred from score"


def test_stage2_score_overrides_a_contradictory_boolean():
    up, detail_up = stage2.parse({"worth": False, "score": 88, "reason": "big"})
    down, detail_down = stage2.parse({"worth": True, "score": 6, "reason": "meh"})
    assert up is True and "overrides" in detail_up["note"]
    assert down is False and "overrides" in detail_down["note"]


def test_stage2_median_score_resists_an_outlier():
    consensus = consensus_for(
        [
            {"worth": True, "score": 55},
            {"worth": True, "score": 60},
            {"worth": True, "score": 58},
            {"worth": True, "score": 57},
            {"worth": True, "score": 99},
        ],
        stage2.parse,
        score_key="score",
    )
    assert consensus.score == pytest.approx(58.0)


def test_stage2_urgency_is_the_median():
    consensus = consensus_for(
        [
            {"worth": True, "score": 60, "urgency": "high"},
            {"worth": True, "score": 60, "urgency": "high"},
            {"worth": True, "score": 60, "urgency": "medium"},
            {"worth": True, "score": 60, "urgency": "low"},
            {"worth": True, "score": 60, "urgency": "medium"},
        ],
        stage2.parse,
    )
    assert stage2.urgency(consensus) == "medium"


# --------------------------------------------------------------------------- #
# stage 3 — assets
# --------------------------------------------------------------------------- #


def test_stage3_requires_two_mentions():
    consensus = consensus_for(
        [
            {"assets": [{"asset": "BTC", "relation": "DIRECT", "confidence": 0.9}]},
            {"assets": [{"asset": "BTC", "relation": "DIRECT", "confidence": 0.8}]},
            {"assets": [{"asset": "BTC", "relation": "INDIRECT", "confidence": 0.7}]},
            {"assets": [{"asset": "DOGE", "relation": "INDIRECT", "confidence": 0.3}]},
            {"assets": [{"asset": "BTC", "relation": "DIRECT", "confidence": 0.85}]},
        ],
        stage3.parse,
    )
    links = stage3.aggregate_links(consensus)
    assert [item.asset for item in links] == ["BTC"]
    assert links[0].votes == 4
    assert links[0].relation is Relation.DIRECT  # 3 of 4 said DIRECT


def test_stage3_one_model_cannot_vote_twice():
    consensus = consensus_for(
        [
            {"assets": [{"asset": "SOL"}, {"asset": "SOL"}, {"asset": "SOL"}]},
            {"assets": [{"asset": "BTC"}]},
            {"assets": [{"asset": "BTC"}]},
        ],
        stage3.parse,
    )
    links = {item.asset: item.votes for item in stage3.aggregate_links(consensus)}
    assert links == {"BTC": 2}, "SOL was named three times by one model"


@pytest.mark.parametrize(
    "value", ["the whole market", "altcoins", "N/A", "USD", "", "a very long asset name"]
)
def test_stage3_rejects_prose_in_the_asset_field(value):
    assert stage3.normalize_ticker(value) == ""


def test_stage3_accepts_pair_notation():
    assert stage3.normalize_ticker("BTC/USDT") == "BTC"
    assert stage3.normalize_ticker("eth-usd") == "ETH"
    assert stage3.normalize_ticker("$SOL") == "SOL"


def test_stage3_lowers_the_bar_for_a_decimated_panel():
    """With most of the panel down, insisting on two mentions discards everything."""
    consensus = consensus_for(
        [{"assets": [{"asset": "BTC", "confidence": 0.9}]}, None, None, None, None],
        stage3.parse,
    )
    assert [item.asset for item in stage3.aggregate_links(consensus)] == ["BTC"]


def test_stage3_market_wide_needs_a_majority():
    def build(flags):
        return consensus_for(
            [{"assets": [{"asset": "BTC"}], "market_wide": flag} for flag in flags],
            stage3.parse,
        )

    assert stage3._market_wide(build([True, True, True, False, False])) is True
    assert stage3._market_wide(build([True, True, False, False, False])) is False


# --------------------------------------------------------------------------- #
# stage 4 — impact
# --------------------------------------------------------------------------- #


def _impact_payload(direction="BULLISH", magnitude="MEDIUM", low=1.0, high=2.0, **extra):
    payload = {
        "direction": direction,
        "magnitude": magnitude,
        "expected_move_pct_low": low,
        "expected_move_pct_high": high,
        "confidence": 0.8,
        "horizon_minutes": 180,
        "causality": "FUNDAMENTAL",
        "mechanism": "authorised participants must buy spot to create shares",
        "risks": "outflows could offset",
    }
    payload.update(extra)
    return payload


def test_stage4_size_ignores_the_dissenters():
    """A lone bear must not shrink the bulls' forecast toward zero."""
    consensus = consensus_for(
        [
            _impact_payload(low=2.0, high=4.0),
            _impact_payload(low=2.0, high=4.0),
            _impact_payload(low=2.0, high=4.0),
            _impact_payload(low=2.0, high=4.0),
            _impact_payload(direction="BEARISH", low=0.1, high=0.2),
        ],
        stage4.parse,
    )
    result = stage4.aggregate_impact(link(), consensus)
    assert result.direction is Direction.BULLISH
    assert (result.expected_low, result.expected_high) == (2.0, 4.0)


def test_stage4_reports_mixed_on_an_even_split():
    consensus = consensus_for(
        [
            _impact_payload(direction="BULLISH"),
            _impact_payload(direction="BULLISH"),
            _impact_payload(direction="BEARISH"),
            _impact_payload(direction="BEARISH"),
        ],
        stage4.parse,
    )
    result = stage4.aggregate_impact(link(), consensus)
    assert result.direction is Direction.MIXED
    assert result.agreement == pytest.approx(0.5)


def test_stage4_a_three_two_split_still_has_a_direction():
    """The boundary: 60% is a lead, not a coin flip."""
    consensus = consensus_for(
        [
            _impact_payload(direction="BULLISH"),
            _impact_payload(direction="BULLISH"),
            _impact_payload(direction="BULLISH"),
            _impact_payload(direction="BEARISH"),
            _impact_payload(direction="BEARISH"),
        ],
        stage4.parse,
    )
    assert stage4.aggregate_impact(link(), consensus).direction is Direction.BULLISH


def test_stage4_disagreement_lowers_confidence():
    def confidence_for(payloads):
        return stage4.aggregate_impact(link(), consensus_for(payloads, stage4.parse)).confidence

    unanimous = confidence_for([_impact_payload() for _ in range(5)])
    divided = confidence_for(
        [_impact_payload()] * 3 + [_impact_payload(direction="BEARISH")] * 2
    )
    assert divided < unanimous
    assert unanimous <= 0.8, "confidence must never exceed what the models claimed"


def test_stage4_indirect_links_are_discounted():
    payloads = [_impact_payload() for _ in range(5)]
    direct = stage4.aggregate_impact(link(relation=Relation.DIRECT), consensus_for(payloads, stage4.parse))
    indirect = stage4.aggregate_impact(
        link(relation=Relation.INDIRECT), consensus_for(payloads, stage4.parse)
    )
    assert indirect.confidence < direct.confidence


def test_stage4_a_shaky_link_is_discounted():
    payloads = [_impact_payload() for _ in range(5)]
    strong = stage4.aggregate_impact(link(confidence=1.0), consensus_for(payloads, stage4.parse))
    weak = stage4.aggregate_impact(link(confidence=0.2), consensus_for(payloads, stage4.parse))
    assert weak.confidence < strong.confidence


def test_stage4_dead_panel_produces_an_honest_null():
    consensus = consensus_for([None] * 5, stage4.parse)
    result = stage4.aggregate_impact(link(), consensus)
    assert result.direction is Direction.NEUTRAL
    assert result.confidence == 0.0
    assert result.model_count == 0


def test_stage4_numbers_win_over_a_contradictory_label():
    """Six percent is not LOW, whatever the model called it."""
    consensus = consensus_for(
        [_impact_payload(magnitude="LOW", low=6.0, high=8.0) for _ in range(3)],
        stage4.parse,
    )
    assert stage4.aggregate_impact(link(), consensus).magnitude is Magnitude.EXTREME


def test_stage4_adjacent_label_disagreement_is_left_alone():
    consensus = consensus_for(
        [_impact_payload(magnitude="MEDIUM", low=2.1, high=2.4) for _ in range(3)],
        stage4.parse,
    )
    assert stage4.aggregate_impact(link(), consensus).magnitude is Magnitude.MEDIUM


@pytest.mark.parametrize(
    "low,high,expected",
    [
        (-3.0, -1.0, (1.0, 3.0)),   # sign leaked into the size
        (4.0, 2.0, (2.0, 4.0)),     # reversed
        (None, None, (0.7, 2.0)),   # label only; MEDIUM's calibration band
    ],
    ids=["signed", "reversed", "label-only"],
)
def test_stage4_normalises_awkward_ranges(low, high, expected):
    assert stage4.normalise_range(low, high, Magnitude.MEDIUM) == expected


def test_stage4_widens_a_single_number_into_a_band():
    assert stage4.normalise_range(3.0, None, Magnitude.HIGH) == (2.0, 3.0)


def test_stage4_snaps_horizons_to_observable_windows():
    assert stage4.snap_horizon(45) == 60
    assert stage4.snap_horizon(1000) == 1440
    assert stage4.snap_horizon(None) == 180


def test_stage4_neutral_is_not_given_a_large_band():
    consensus = consensus_for(
        [_impact_payload(direction="NEUTRAL", magnitude="HIGH", low=3.0, high=5.0)] * 3,
        stage4.parse,
    )
    result = stage4.aggregate_impact(link(), consensus)
    assert result.direction is Direction.NEUTRAL
    assert result.expected_high <= Magnitude.LOW.default_range[1]


# --------------------------------------------------------------------------- #
# stage 5 — cross-check
# --------------------------------------------------------------------------- #


def _audit_payload(confirms=True, direction="BULLISH", low=1.0, high=2.0, **extra):
    payload = {
        "confirms": confirms,
        "direction": direction,
        "magnitude": "MEDIUM",
        "expected_move_pct_low": low,
        "expected_move_pct_high": high,
        "confidence": 0.8,
        "horizon_minutes": 180,
        "conflicts": [],
        "overlooked": [],
        "priced_in": False,
        "reason": "the mechanism holds",
    }
    payload.update(extra)
    return payload


def test_stage5_confirmation_keeps_the_direction():
    consensus = consensus_for([_audit_payload() for _ in range(5)], stage5.parse)
    revised = stage5.revise(impact_of(), consensus)
    assert revised.direction is Direction.BULLISH
    assert revised.source == "stage5"


def test_stage5_an_audit_that_objects_never_raises_confidence():
    """The invariant. Without it, escalation manufactures certainty."""
    original = impact_of(confidence=0.7)
    consensus = consensus_for(
        [
            _audit_payload(confirms=False, direction="BEARISH", confidence=0.95,
                           conflicts=["direction is backwards"]),
            _audit_payload(confirms=False, direction="BEARISH", confidence=0.95,
                           conflicts=["direction is backwards"]),
            _audit_payload(confirms=False, direction="BEARISH", confidence=0.9,
                           conflicts=["direction is backwards"]),
            _audit_payload(confidence=0.9),
            _audit_payload(confidence=0.9),
        ],
        stage5.parse,
    )
    revised = stage5.revise(original, consensus)
    assert revised.direction is Direction.BEARISH, "three auditors agreed on the flip"
    assert revised.confidence < original.confidence
    assert revised.confidence <= stage5.CONTESTED_CONFIDENCE_CAP


def test_stage5_rejection_without_agreement_yields_mixed():
    """Auditors who agree it is wrong but not on what have found doubt, not an answer."""
    consensus = consensus_for(
        [
            _audit_payload(confirms=False, direction="BEARISH", conflicts=["wrong"]),
            _audit_payload(confirms=False, direction="NEUTRAL", conflicts=["wrong"]),
            _audit_payload(confirms=False, direction="MIXED", conflicts=["wrong"]),
            _audit_payload(),
            _audit_payload(),
        ],
        stage5.parse,
    )
    revised = stage5.revise(impact_of(), consensus)
    assert revised.direction is Direction.MIXED
    assert revised.confidence <= stage5.CONTESTED_CONFIDENCE_CAP


def test_stage5_priced_in_shrinks_the_move():
    original = impact_of(expected_low=2.0, expected_high=4.0)
    consensus = consensus_for(
        [
            _audit_payload(low=2.0, high=4.0, priced_in=True),
            _audit_payload(low=2.0, high=4.0, priced_in=True),
            _audit_payload(low=2.0, high=4.0, priced_in=True),
            _audit_payload(low=2.0, high=4.0),
            _audit_payload(low=2.0, high=4.0),
        ],
        stage5.parse,
    )
    revised = stage5.revise(original, consensus)
    assert revised.expected_high < original.expected_high
    assert revised.expected_high == pytest.approx(4.0 * stage5.PRICED_IN_RETENTION, abs=0.01)


def test_stage5_a_dead_panel_changes_nothing():
    original = impact_of()
    consensus = consensus_for([None] * 5, stage5.parse)
    assert stage5.revise(original, consensus) is original


def test_stage5_confirmation_cannot_inflate_confidence_much():
    original = impact_of(confidence=0.5)
    consensus = consensus_for(
        [_audit_payload(confidence=0.99) for _ in range(5)], stage5.parse
    )
    revised = stage5.revise(original, consensus)
    assert revised.confidence <= original.confidence * 1.15


def test_stage5_overlooked_factors_reduce_confidence():
    payloads = [_audit_payload(overlooked=["ignored the ETF outflow data"])] * 5
    with_gaps = stage5.revise(impact_of(), consensus_for(payloads, stage5.parse))
    clean = stage5.revise(impact_of(), consensus_for([_audit_payload()] * 5, stage5.parse))
    assert with_gaps.confidence < clean.confidence
    assert "outflow" in with_gaps.risks


def test_stage5_infers_a_verdict_from_conflicts_alone():
    silent_but_objecting, _ = stage5.parse(
        {"direction": "BEARISH", "conflicts": ["the direction is wrong"]}
    )
    silent_and_agreeing, _ = stage5.parse({"direction": "BULLISH", "conflicts": []})
    assert silent_but_objecting is False
    assert silent_and_agreeing is True


# --------------------------------------------------------------------------- #
# final layer
# --------------------------------------------------------------------------- #


def _final_payload(direction="BULLISH", tradeable=True, **extra):
    payload = {
        "asset": "BTC",
        "direction": direction,
        "magnitude": "HIGH",
        "expected_move_pct_low": 2.0,
        "expected_move_pct_high": 4.0,
        "confidence": 0.8,
        "horizon_minutes": 360,
        "causality": "FUNDAMENTAL",
        "mechanism": "creation baskets force spot purchases within T+1",
        "risks": "a stay pending appeal",
        "key_uncertainty": "the size of day-one inflows",
        "disagreement_with_pipeline": "",
        "tradeable": tradeable,
    }
    payload.update(extra)
    return payload


def test_final_unanimity_overrules_the_pipeline():
    consensus = consensus_for(
        [_final_payload(direction="BEARISH") for _ in range(3)],
        final_stage.parse,
        reject_votes=2,
    )
    verdict = final_stage.decide(impact_of(direction=Direction.BULLISH), consensus)
    assert verdict.direction is Direction.BEARISH
    assert verdict.source == "final"
    assert "overrules" in verdict.notes


def test_final_three_way_split_is_not_tradeable():
    """Three heavy models disagreeing is a finding, not a forecast."""
    consensus = consensus_for(
        [
            _final_payload(direction="BULLISH"),
            _final_payload(direction="BEARISH"),
            _final_payload(direction="NEUTRAL"),
        ],
        final_stage.parse,
        reject_votes=2,
    )
    verdict = final_stage.decide(impact_of(), consensus)
    assert verdict.direction is Direction.MIXED
    assert verdict.tradeable is False
    assert verdict.confidence <= final_stage.SPLIT_CONFIDENCE_CAP
    assert "split three ways" in verdict.notes


def test_final_two_of_three_decides_but_prices_the_dissent():
    agreeing = consensus_for([_final_payload() for _ in range(3)], final_stage.parse, reject_votes=2)
    divided = consensus_for(
        [_final_payload(), _final_payload(), _final_payload(direction="BEARISH")],
        final_stage.parse,
        reject_votes=2,
    )
    unanimous_verdict = final_stage.decide(impact_of(), agreeing)
    split_verdict = final_stage.decide(impact_of(), divided)
    assert split_verdict.direction is Direction.BULLISH
    assert split_verdict.confidence < unanimous_verdict.confidence


def test_final_unavailable_keeps_the_earlier_verdict():
    original = impact_of(source="stage5", confidence=0.66)
    consensus = consensus_for([None] * 3, final_stage.parse, reject_votes=2)
    verdict = final_stage.decide(original, consensus)
    assert verdict.direction is original.direction
    assert verdict.confidence == original.confidence
    assert verdict.source == "stage5", "must not claim a review that never happened"
    assert verdict.notes == "final layer unavailable"


def test_final_derives_tradeable_when_the_model_omits_it():
    tiny, detail_tiny = final_stage.parse(
        _final_payload(expected_move_pct_low=0.1, expected_move_pct_high=0.2, tradeable=None)
    )
    big, _ = final_stage.parse(_final_payload(tradeable=None))
    assert tiny is False and detail_tiny["tradeable"] is False
    assert big is True


# --------------------------------------------------------------------------- #
# router
# --------------------------------------------------------------------------- #


def test_router_escalates_a_low_agreement_asset_regardless_of_score():
    router = Router()
    ctx = context(score=10.0)
    ctx.impacts["BTC"] = impact_of(agreement=0.5, magnitude=Magnitude.MEDIUM)
    picks = router.stage5_assets(ctx)
    assert [pick.asset for pick in picks] == ["BTC"]
    assert "agreement" in picks[0].reason


def test_router_leaves_a_confident_neutral_alone():
    router = Router()
    ctx = context(score=95.0)
    ctx.impacts["BTC"] = impact_of(
        direction=Direction.NEUTRAL, magnitude=Magnitude.LOW, confidence=0.8, agreement=1.0
    )
    assert router.stage5_assets(ctx) == []


def test_router_skips_assets_with_no_answer():
    router = Router()
    ctx = context(score=95.0)
    ctx.impacts["BTC"] = impact_of(model_count=0, confidence=0.0)
    assert router.stage5_assets(ctx) == []
    assert router.final_assets(ctx) == []


def test_router_always_escalates_the_listed_event_types():
    router = Router()
    always = router.routing.stage5_always_types[0]
    ctx = context(score=5.0)
    ctx.facts.event_type = always
    ctx.impacts["BTC"] = impact_of(agreement=1.0, magnitude=Magnitude.MEDIUM)
    picks = router.stage5_assets(ctx)
    assert picks and always in picks[0].reason


def test_router_enforces_the_heavy_budget_across_events():
    router = Router()
    budget = router.routing.final_max_per_cycle
    spent = 0
    for round_index in range(budget + 2):
        ctx = context(score=router.routing.final_min_score + 5.0)
        ctx.impacts[f"A{round_index}"] = impact_of(asset=f"A{round_index}")
        spent += len(router.final_assets(ctx))
    assert spent == budget
    assert router.final_budget_left == 0
    assert router.skipped_for_budget


def test_router_spends_the_budget_on_the_biggest_moves():
    router = Router()
    ctx = context(score=router.routing.final_min_score + 5.0)
    ctx.impacts["TINY"] = impact_of(
        asset="TINY", expected_low=0.1, expected_high=0.2, confidence=0.95
    )
    ctx.impacts["BIG"] = impact_of(
        asset="BIG", expected_low=4.0, expected_high=8.0, confidence=0.5,
        magnitude=Magnitude.HIGH,
    )
    picks = router.final_assets(ctx)
    assert picks[0].asset == "BIG"


def test_router_never_alerts_on_neutral():
    router = Router()
    ctx = context(score=99.0)
    neutral = impact_of(direction=Direction.NEUTRAL, confidence=0.9)
    worth, _ = router.should_alert(ctx, neutral)
    assert worth is False


def test_router_alert_score_rewards_size_confidence_and_depth():
    router = Router()
    ctx = context(score=80.0)
    weak = impact_of(expected_low=0.2, expected_high=0.4, confidence=0.3)
    strong = impact_of(
        expected_low=3.0, expected_high=6.0, confidence=0.85,
        magnitude=Magnitude.HIGH, source="final", tradeable=True,
    )
    assert router.alert_score(ctx, strong) > router.alert_score(ctx, weak)

    shallow = impact_of(source="stage4")
    deep = impact_of(source="final")
    assert router.alert_score(ctx, deep) > router.alert_score(ctx, shallow)


def test_router_discounts_an_untradeable_verdict():
    router = Router()
    ctx = context(score=80.0)
    tradeable = impact_of(tradeable=True)
    untradeable = impact_of(tradeable=False)
    assert router.alert_score(ctx, untradeable) < router.alert_score(ctx, tradeable)
