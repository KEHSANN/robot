"""Event identity and state comparison — the NEW / UPDATE / DUPLICATE decision.

The spec's rule, restated precisely:

* same identity + same state  -> DUPLICATE (another outlet, nothing new)
* same identity + new state    -> UPDATE    (the story developed)
* different identity           -> NEW       (even if the text reads similarly)

The third case is why text similarity alone cannot drive this. "SEC approves
BlackRock's spot ETF" and "SEC approves Fidelity's spot ETF" embed almost
identically and are two different events. Conversely "Bitcoin ETFs see $742M
inflow" and "Spot BTC funds pull in three-quarters of a billion" embed further
apart and are one event.

So similarity only *nominates* candidates. Identity decides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from services.types import EventRecord, FactSet
from stage0.facts import KNOWN_ASSETS

#: Status values ordered by how settled the event is. Movement along this axis is
#: a real development even when every number stays the same: "SEC *proposes*" and
#: "SEC *confirms*" are the same event at two very different stages.
STATUS_ORDER = {
    "": 0,
    "rumored": 1, "rumoured": 1, "speculated": 1,
    "proposed": 2, "filed": 2, "planned": 2, "expected": 2,
    "reported": 3, "alleged": 3,
    "ongoing": 4, "developing": 4, "in progress": 4,
    "confirmed": 5, "announced": 5, "approved": 5,
    "completed": 6, "settled": 6, "closed": 6, "executed": 6,
    "denied": 7, "rejected": 7, "cancelled": 7, "canceled": 7, "withdrawn": 7,
}

#: Fields whose change makes the item worth re-analysing.
NUMERIC_STATE_FIELDS = ("amount", "price", "percentage", "count")
TEXT_STATE_FIELDS = ("status", "decision", "location", "time_reference", "event_date")

#: A number must move by more than this fraction to count as a development.
#: Outlets round the same figure differently ("$742M" vs "$740M"); that is not
#: news. A 12% revision to a headline number is.
NUMERIC_TOLERANCE = 0.05

#: Identity similarity at/above which two fact sets describe the same happening
#: even though their identity hashes differ.
IDENTITY_MATCH_THRESHOLD = 0.82

#: Weights are deliberately flat-ish. An earlier version gave `target` almost no
#: weight, which broke the case this module exists for: two ETF approvals agree on
#: type, regulator, asset and action, and differ only in *whose* ETF it is. If the
#: distinguishing field is worth 6% of the score, two different events score 0.98.
_IDENTITY_WEIGHTS = {
    "event_type": 0.22,
    "entity": 0.24,
    "asset": 0.14,
    "action": 0.18,
    "target": 0.22,
}

#: Tokens shorter than this are too generic to distinguish two events ("us", "etf",
#: "btc"), so they never trigger the conflict veto.
_SPECIFIC_MIN_LEN = 4

#: Two tokens at/above this similarity are the same word wearing different clothes
#: ("blackrocks" / "blackrock"), not competing specifics.
_TOKEN_MATCH = 0.8

#: Generic vocabulary that shows up in almost every identity field in this domain.
#: A token here is never treated as a distinguishing specific. Kept short on
#: purpose: the length floor above does most of the work, and every word added
#: here is a chance to merge two events that should stay apart.
_GENERIC_TOKENS = frozenset({
    "spot", "etf", "etfs", "fund", "funds", "coin", "token", "crypto",
    "cryptocurrency", "digital", "asset", "assets", "market", "markets",
    "price", "prices", "trading", "exchange", "network", "protocol",
    "bitcoin", "ethereum", "futures", "options", "product", "products",
    "filing", "application", "approval", "decision", "rule", "rules",
})


#: Corporate suffixes and honorifics that differ between outlets describing the
#: same actor.
_ENTITY_NOISE = re.compile(
    r"\b(inc|incorporated|corp|corporation|ltd|limited|llc|lp|llp|plc|gmbh|ag|sa|"
    r"nv|bv|co|company|group|holdings?|labs?|foundation|the|us|u s|united states)\b"
)


def _fold_entity(value: str) -> str:
    text = _ENTITY_NOISE.sub(" ", (value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


#: Words skipped when reading initials off a name, so "department of justice"
#: yields both "doj" and "dj" and either spelling of the acronym matches.
_ACRONYM_SKIP = frozenset({"of", "and", "for", "the", "in", "on", "de", "du"})


def _initialisms(value: str) -> set[str]:
    """The acronyms a multi-word name could be abbreviated to."""
    words = [word for word in (value or "").split() if word]
    if len(words) < 2:
        return set()
    full = "".join(word[0] for word in words)
    trimmed = "".join(word[0] for word in words if word not in _ACRONYM_SKIP)
    return {form for form in (full, trimmed) if len(form) >= 2}


def _acronym_match(a: str, b: str) -> bool:
    """Whether one side is the other's acronym.

    Every institution that moves this market has two names — "SEC" and
    "Securities and Exchange Commission", "ECB" and "European Central Bank" — and
    which one an extraction returns is arbitrary. Character similarity scores those
    pairs near zero, so without this the same event splits in two whenever two
    outlets pick different spellings of the regulator.
    """
    short, long = (a, b) if len(a.split()) == 1 else (b, a)
    if len(short.split()) != 1 or len(long.split()) < 2:
        return False
    return short.replace(".", "") in _initialisms(long)


def _ratio(a: str, b: str) -> float:
    """Token-aware string similarity in [0, 1]."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if _acronym_match(a, b):
        return 1.0
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if len(tokens_a) == 1 and len(tokens_b) == 1:
        # "strategy" inside "microstrategy", "blackrock" inside "blackrocks".
        # The floor keeps short tickers and abbreviations out of it, where a
        # substring hit would be coincidence rather than the same name.
        short, long = sorted((a, b), key=len)
        if len(short) >= 5 and short in long:
            return 0.9
    if tokens_a and tokens_b:
        overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        # One name fully contained in the other ("sec" in "sec enforcement
        # division") is a strong match that Jaccard undercounts.
        if tokens_a <= tokens_b or tokens_b <= tokens_a:
            overlap = max(overlap, 0.85)
    else:
        overlap = 0.0
    return max(overlap, SequenceMatcher(None, a, b).ratio())


def _specifics(value: str) -> set[str]:
    """Tokens in a field that actually identify *which* thing is meant."""
    return {
        token for token in (value or "").split()
        if len(token) >= _SPECIFIC_MIN_LEN and token not in _GENERIC_TOKENS
    }


def _competing_specifics(left: str, right: str) -> bool:
    """Whether two fields each name something the other does not.

    This is the test that separates "SEC approves BlackRock's ETF" from "SEC
    approves Fidelity's ETF": both name a specific fund sponsor, and neither name
    appears on the other side. A weighted average cannot see that, because the two
    fact sets agree on everything else.

    The test is deliberately *mutual*. One-sided extra detail is verbosity — "SEC"
    against "Securities and Exchange Commission" must still match — so a conflict
    requires each side to assert a specific the other lacks.
    """
    if not left or not right:
        return False
    if _acronym_match(left, right):
        return False  # "CFTC" and "Commodity Futures Trading Commission"
    left_tokens, right_tokens = _specifics(left), _specifics(right)
    if not left_tokens or not right_tokens:
        return False

    def unmatched(source: set[str], other: set[str]) -> bool:
        return any(
            not any(_ratio(token, peer) >= _TOKEN_MATCH for peer in other)
            for token in source
        )

    return unmatched(left_tokens, right_tokens) and unmatched(right_tokens, left_tokens)


def _fold_asset(value: str) -> str:
    """Resolve an asset name to its ticker so BTC and "bitcoin" compare equal."""
    text = (value or "").strip().lower()
    if not text:
        return ""
    if text.upper() in KNOWN_ASSETS:
        return text.upper()
    for ticker, aliases in KNOWN_ASSETS.items():
        if text in aliases:
            return ticker
    return text.upper()


def identity_conflict(left: FactSet, right: FactSet | EventRecord) -> str:
    """Name the identity field that proves these are different events, if any.

    Returns an empty string when nothing conflicts. A conflict is decisive: it
    overrules the weighted score and any text-similarity evidence, because text
    similarity is precisely what cannot tell these cases apart.

    Only the fields that name *specific parties and objects* are checked. Verbs are
    not, because "approve" and "greenlight" are the same action, and a synonym must
    not split one event into two. Nor is ``event_type``, which is a controlled
    vocabulary already compared exactly by the identity hash.
    """
    a = left.identity_fields()
    b = _identity_of(right)

    # Tickers are a controlled vocabulary, so unequal means unequal — there is no
    # sense in which BTC is 80% of the way to ETH.
    left_asset, right_asset = _fold_asset(a.get("asset", "")), _fold_asset(b.get("asset", ""))
    if left_asset and right_asset and left_asset != right_asset:
        return "asset"

    for field_name in ("entity", "target"):
        left_value, right_value = a.get(field_name, ""), b.get(field_name, "")
        if field_name == "entity":
            left_value, right_value = _fold_entity(left_value), _fold_entity(right_value)
            # A one-word entity is a name, and the specifics floor is exactly wrong
            # for it: "SEC" and "CFTC" are the shortest and most decisive
            # difference two enforcement stories can have. Acronym expansion is
            # already resolved by _ratio, so anything still unlike here is a
            # different actor.
            if (
                left_value and right_value
                and len(left_value.split()) == 1 and len(right_value.split()) == 1
                and _ratio(left_value, right_value) < _TOKEN_MATCH
            ):
                return "entity"
        if _competing_specifics(left_value, right_value):
            return field_name
    return ""


def _identity_of(record: FactSet | EventRecord) -> dict[str, str]:
    if isinstance(record, EventRecord):
        return {
            "event_type": (record.event_type or "").lower(),
            "entity": (record.entity or "").lower(),
            "asset": (record.primary_asset or "").lower(),
            "action": (record.action or "").lower(),
            "target": (record.target or "").lower(),
        }
    return record.identity_fields()


def identity_similarity(left: FactSet, right: FactSet | EventRecord) -> float:
    """Weighted agreement between two event identities, in [0, 1]."""
    a = left.identity_fields()
    b = _identity_of(right)

    score = 0.0
    weight_used = 0.0
    for field_name, weight in _IDENTITY_WEIGHTS.items():
        left_value = a.get(field_name, "")
        right_value = b.get(field_name, "")
        if field_name == "entity":
            left_value, right_value = _fold_entity(left_value), _fold_entity(right_value)
        if not left_value and not right_value:
            continue  # neither side asserts it; it should not drag the score
        weight_used += weight
        score += weight * _ratio(left_value, right_value)

    if weight_used <= 0.0:
        return 0.0
    return score / weight_used


def same_event(
    left: FactSet,
    right: FactSet | EventRecord,
    *,
    threshold: float = IDENTITY_MATCH_THRESHOLD,
) -> tuple[bool, float]:
    """Whether two fact sets describe the same happening, and by how much.

    A conflicting identity field returns ``(False, 0.0)``. The zero is not a
    guess at the similarity — it is a signal, so that callers holding stronger
    text-similarity evidence (a near-verbatim reprint, say) cannot talk themselves
    past a conflict they have no way to see.
    """
    right_key = right.identity_key if isinstance(right, EventRecord) else right.identity_key()
    if left.identity_key() == right_key:
        return True, 1.0
    if identity_conflict(left, right):
        return False, 0.0
    score = identity_similarity(left, right)
    return score >= threshold, score


# --------------------------------------------------------------------------- #
# state comparison
# --------------------------------------------------------------------------- #

@dataclass
class StateDiff:
    """What changed between a stored event's state and a new article's."""

    changed: dict[str, dict[str, Any]] = field(default_factory=dict)
    added_claims: list[str] = field(default_factory=list)
    #: True when the change is only cosmetic (rounding, rewording).
    cosmetic_only: bool = False

    @property
    def has_changes(self) -> bool:
        return bool(self.changed or self.added_claims)

    @property
    def is_material(self) -> bool:
        return self.has_changes and not self.cosmetic_only

    def summary(self) -> str:
        parts = [
            f"{name}: {entry['from']!r} -> {entry['to']!r}"
            for name, entry in self.changed.items()
        ]
        if self.added_claims:
            parts.append(f"{len(self.added_claims)} new claim(s)")
        return "; ".join(parts) or "no change"

    def as_json(self) -> dict:
        return {"changed": self.changed, "added_claims": self.added_claims}


def _numeric_changed(old: Any, new: Any) -> bool:
    """Whether a number moved enough to matter."""
    if old is None and new is None:
        return False
    if old is None or new is None:
        return True  # a figure appearing or disappearing is information
    try:
        old_f, new_f = float(old), float(new)
    except (TypeError, ValueError):
        return old != new
    if old_f == new_f:
        return False
    scale = max(abs(old_f), abs(new_f))
    if scale == 0:
        return True
    return abs(new_f - old_f) / scale > NUMERIC_TOLERANCE


def _status_rank(value: Any) -> int:
    return STATUS_ORDER.get(str(value or "").strip().lower(), 0)


def compare_state(previous: dict[str, Any], current: dict[str, Any]) -> StateDiff:
    """Diff two ``FactSet.state_fields()`` dicts.

    A rewording of the same claim is not an update; a changed ruling, a revised
    figure, or a status that moved along :data:`STATUS_ORDER` is.
    """
    diff = StateDiff()

    for name in NUMERIC_STATE_FIELDS:
        old, new = previous.get(name), current.get(name)
        if _numeric_changed(old, new):
            diff.changed[name] = {"from": old, "to": new}

    for name in TEXT_STATE_FIELDS:
        old = str(previous.get(name) or "")
        new = str(current.get(name) or "")
        if old == new:
            continue
        if not new:
            # The new article simply does not mention it. Absence is not a change.
            continue
        if name == "status":
            if _status_rank(old) == _status_rank(new):
                continue  # synonyms: "announced" vs "confirmed"
            diff.changed[name] = {"from": old, "to": new}
            continue
        if old and _ratio(old, new) >= 0.9:
            continue  # same content, different wording
        diff.changed[name] = {"from": old, "to": new}

    old_claims = {str(c) for c in (previous.get("key_claims") or [])}
    for claim in current.get("key_claims") or []:
        text = str(claim)
        if not text or text in old_claims:
            continue
        if any(_ratio(text, existing) >= 0.85 for existing in old_claims):
            continue  # the same claim restated
        diff.added_claims.append(text)

    if diff.changed:
        diff.cosmetic_only = False
    elif diff.added_claims:
        # Extra detail with no changed field is a fuller retelling, not a
        # development — unless there is a lot of it.
        diff.cosmetic_only = len(diff.added_claims) < 2

    return diff


def merge_state(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Fold a new article's state into the stored event state.

    Later reporting wins on any field it actually asserts; fields it omits keep
    their previous value, so the event record accumulates detail instead of losing
    it to a thinner follow-up article. Status only moves forward, so a stale wire
    story cannot walk a CONFIRMED event back to REPORTED.
    """
    merged = dict(previous)
    for name, value in current.items():
        if name == "key_claims":
            continue
        if value in (None, ""):
            continue
        if name == "status" and _status_rank(value) < _status_rank(previous.get("status")):
            continue
        merged[name] = value

    claims = list(previous.get("key_claims") or [])
    for claim in current.get("key_claims") or []:
        if claim and not any(_ratio(str(claim), str(existing)) >= 0.85 for existing in claims):
            claims.append(claim)
    merged["key_claims"] = claims[:20]
    return merged


__all__ = [
    "IDENTITY_MATCH_THRESHOLD",
    "NUMERIC_TOLERANCE",
    "STATUS_ORDER",
    "StateDiff",
    "compare_state",
    "identity_conflict",
    "identity_similarity",
    "merge_state",
    "same_event",
]
