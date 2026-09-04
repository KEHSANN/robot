"""Event identity: which pairs must merge, and which must stay apart.

This is the highest-leverage test in the project. A false merge silently buries a
real event inside an unrelated one — no alert is ever sent, and nothing in the logs
says so. A false split costs one duplicate analysis. So the cases below are written
as two lists, and the asymmetry in consequences is why the thresholds lean toward
splitting when they are unsure.

Every pair here is a shape seen in real crypto news, not a synthetic edge case.
"""

from __future__ import annotations

import pytest

from services.types import FactSet
from stage0.identity import (
    compare_state,
    identity_conflict,
    identity_similarity,
    merge_state,
    same_event,
)


def facts(**kwargs) -> FactSet:
    return FactSet(**kwargs)


# --------------------------------------------------------------------------- #
# pairs that describe ONE event and must merge
# --------------------------------------------------------------------------- #

MUST_MERGE = [
    pytest.param(
        facts(event_type="ETF_APPROVAL", entity="SEC", assets=["BTC"],
              action="approve", target="BlackRock spot bitcoin ETF"),
        facts(event_type="ETF_APPROVAL", entity="US Securities and Exchange Commission",
              assets=["BTC"], action="approves", target="BlackRock's spot BTC fund"),
        id="regulator-spelled-out",
    ),
    pytest.param(
        facts(event_type="ENFORCEMENT", entity="CFTC", assets=["BTC"],
              action="charge", target="Bitfinex"),
        facts(event_type="ENFORCEMENT", entity="Commodity Futures Trading Commission",
              assets=["BTC"], action="charges", target="Bitfinex"),
        id="acronym-expansion",
    ),
    pytest.param(
        facts(event_type="MACRO", entity="Fed", assets=["BTC"],
              action="cut rates", target="interest rates"),
        facts(event_type="MACRO", entity="Federal Reserve", assets=["BTC"],
              action="cuts rates", target="interest rates"),
        id="short-form-institution",
    ),
    pytest.param(
        facts(event_type="ETF_APPROVAL", entity="SEC", assets=["BTC"],
              action="approve", target="BlackRock spot bitcoin ETF"),
        facts(event_type="ETF_APPROVAL", entity="SEC", assets=["BTC"],
              action="greenlight", target="BlackRock spot bitcoin ETF"),
        id="synonym-verb",
    ),
    pytest.param(
        facts(event_type="ETF_FLOWS", entity="Bitcoin ETFs", assets=["BTC"],
              action="record inflow", target="spot bitcoin ETFs"),
        facts(event_type="ETF_FLOWS", entity="Bitcoin ETF", assets=["BTC"],
              action="see inflows", target="spot BTC funds"),
        id="reworded-headline",
    ),
    pytest.param(
        facts(event_type="TREASURY_PURCHASE", entity="Strategy", assets=["BTC"],
              action="buy", target="bitcoin"),
        facts(event_type="TREASURY_PURCHASE", entity="MicroStrategy", assets=["BTC"],
              action="buys", target="bitcoin"),
        id="company-renamed",
    ),
    pytest.param(
        facts(event_type="OTHER", entity="Tether", assets=["USDT"]),
        facts(event_type="OTHER", entity="Tether Limited", assets=["USDT"]),
        id="corporate-suffix",
    ),
]


# --------------------------------------------------------------------------- #
# pairs that describe TWO events and must stay apart
# --------------------------------------------------------------------------- #

MUST_SPLIT = [
    pytest.param(
        facts(event_type="ETF_APPROVAL", entity="SEC", assets=["BTC"],
              action="approve", target="BlackRock spot bitcoin ETF"),
        facts(event_type="ETF_APPROVAL", entity="SEC", assets=["BTC"],
              action="approve", target="Fidelity spot bitcoin ETF"),
        id="different-etf-sponsor",
    ),
    pytest.param(
        facts(event_type="ENFORCEMENT", entity="SEC", assets=["BTC"],
              action="charge", target="Bitfinex"),
        facts(event_type="ENFORCEMENT", entity="CFTC", assets=["BTC"],
              action="charge", target="Bitfinex"),
        id="different-regulator",
    ),
    pytest.param(
        facts(event_type="EXCHANGE_HACK", entity="Binance", assets=["BNB"],
              action="hacked", target="hot wallet"),
        facts(event_type="EXCHANGE_HACK", entity="Coinbase", assets=["BNB"],
              action="hacked", target="hot wallet"),
        id="different-exchange",
    ),
    pytest.param(
        facts(event_type="MACRO", entity="Federal Reserve", assets=["BTC"],
              action="cut rates", target="interest rates"),
        facts(event_type="MACRO", entity="European Central Bank", assets=["BTC"],
              action="cut rates", target="interest rates"),
        id="different-central-bank",
    ),
    pytest.param(
        facts(event_type="LISTING", entity="Coinbase", assets=["SOL"],
              action="list", target="SOL"),
        facts(event_type="LISTING", entity="Coinbase", assets=["SUI"],
              action="list", target="SUI"),
        id="different-asset",
    ),
    pytest.param(
        facts(event_type="ETF_APPROVAL", entity="SEC", assets=["BTC"],
              action="approve", target="spot ETF"),
        facts(event_type="ENFORCEMENT", entity="SEC", assets=["BTC"],
              action="charge", target="Ripple"),
        id="different-event-type",
    ),
]


@pytest.mark.parametrize("left,right", MUST_MERGE)
def test_same_event_merges(left: FactSet, right: FactSet) -> None:
    matched, score = same_event(left, right)
    assert matched, (
        f"should be one event but split (score {score:.3f}, "
        f"conflict {identity_conflict(left, right) or 'none'})"
    )


@pytest.mark.parametrize("left,right", MUST_SPLIT)
def test_different_events_split(left: FactSet, right: FactSet) -> None:
    matched, score = same_event(left, right)
    assert not matched, f"should be two events but merged (score {score:.3f})"


@pytest.mark.parametrize("left,right", MUST_SPLIT)
def test_split_survives_high_text_similarity(left: FactSet, right: FactSet) -> None:
    """A conflict must score 0.0, not merely below threshold.

    The pipeline lets a near-verbatim reprint override a sub-threshold identity
    score. That override reads the returned score, so a conflicting pair has to
    come back at zero — otherwise two different ETF approvals written from the same
    press release would merge on text similarity alone.
    """
    if not identity_conflict(left, right):
        pytest.skip("this pair splits on score, not on a field conflict")
    _, score = same_event(left, right)
    assert score == 0.0


def test_identity_similarity_ignores_fields_neither_side_asserts() -> None:
    left = facts(event_type="OTHER", entity="Tether")
    right = facts(event_type="OTHER", entity="Tether")
    assert identity_similarity(left, right) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# state comparison: UPDATE vs DUPLICATE
# --------------------------------------------------------------------------- #

def test_status_moving_forward_is_an_update() -> None:
    before = facts(status="reported").state_fields()
    after = facts(status="confirmed").state_fields()
    assert compare_state(before, after).is_material


def test_status_synonym_is_not_an_update() -> None:
    before = facts(status="announced").state_fields()
    after = facts(status="confirmed").state_fields()
    assert not compare_state(before, after).has_changes


def test_rounded_number_is_not_an_update() -> None:
    before = facts(amount=742_000_000).state_fields()
    after = facts(amount=740_000_000).state_fields()
    assert not compare_state(before, after).has_changes


def test_revised_number_is_an_update() -> None:
    before = facts(amount=742_000_000).state_fields()
    after = facts(amount=1_200_000_000).state_fields()
    assert compare_state(before, after).is_material


def test_reversal_is_an_update() -> None:
    before = facts(status="approved", decision="approve").state_fields()
    after = facts(status="rejected", decision="deny").state_fields()
    diff = compare_state(before, after)
    assert diff.is_material
    assert "decision" in diff.changed


def test_omitted_field_is_not_a_change() -> None:
    """A thinner follow-up article is not a development."""
    before = facts(status="confirmed", location="New York").state_fields()
    after = facts(status="confirmed").state_fields()
    assert not compare_state(before, after).has_changes


def test_single_extra_claim_is_cosmetic() -> None:
    before = facts(status="confirmed", key_claims=["etf approved"]).state_fields()
    after = facts(status="confirmed", key_claims=["etf approved", "trading starts monday"]).state_fields()
    diff = compare_state(before, after)
    assert diff.has_changes
    assert not diff.is_material


def test_merge_never_walks_status_backwards() -> None:
    """A stale wire story must not undo a confirmation."""
    before = facts(status="confirmed", amount=742_000_000).state_fields()
    stale = facts(status="reported").state_fields()
    merged = merge_state(before, stale)
    assert merged["status"] == "confirmed"
    assert merged["amount"] == 742_000_000


def test_merge_accumulates_detail() -> None:
    before = facts(status="reported", amount=742_000_000).state_fields()
    after = facts(status="confirmed", location="New York").state_fields()
    merged = merge_state(before, after)
    assert merged["status"] == "confirmed"
    assert merged["amount"] == 742_000_000          # kept, not lost to the thinner item
    assert merged["location"] == "new york"
