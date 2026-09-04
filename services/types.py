"""Domain types shared across the pipeline stages.

These are the objects that flow from ingestion through Stage 0's event memory to
the analysis stages, the Telegram alert, and finally the prediction/outcome
feedback loop.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# enums
# --------------------------------------------------------------------------- #

class Stage0Decision(str, Enum):
    NEW = "NEW"
    UPDATE = "UPDATE"
    DUPLICATE = "DUPLICATE"


class Direction(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"

    @classmethod
    def parse(cls, value: Any) -> "Direction":
        text = str(value or "").strip().upper()
        aliases = {
            "UP": cls.BULLISH, "POSITIVE": cls.BULLISH, "BULL": cls.BULLISH, "LONG": cls.BULLISH,
            "DOWN": cls.BEARISH, "NEGATIVE": cls.BEARISH, "BEAR": cls.BEARISH, "SHORT": cls.BEARISH,
            "FLAT": cls.NEUTRAL, "NONE": cls.NEUTRAL, "NO IMPACT": cls.NEUTRAL,
            "UNCERTAIN": cls.MIXED, "BOTH": cls.MIXED, "UNCLEAR": cls.MIXED,
        }
        if text in cls.__members__:
            return cls[text]
        return aliases.get(text, cls.NEUTRAL)

    @property
    def emoji(self) -> str:
        return {
            Direction.BULLISH: "🟢",
            Direction.BEARISH: "🔴",
            Direction.NEUTRAL: "⚪",
            Direction.MIXED: "🟡",
        }[self]

    @property
    def sign(self) -> int:
        return {Direction.BULLISH: 1, Direction.BEARISH: -1}.get(self, 0)


class Magnitude(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

    @classmethod
    def parse(cls, value: Any) -> "Magnitude":
        text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
        if text in cls.__members__:
            return cls[text]
        # Models like to answer "MEDIUM-HIGH"; take the stronger half.
        for name in ("EXTREME", "HIGH", "MEDIUM", "LOW"):
            if name in text:
                return cls[name]
        return cls.LOW

    #: Default expected absolute move (percent) when a model gives a label but
    #: no numeric range.
    @property
    def default_range(self) -> tuple[float, float]:
        return {
            Magnitude.LOW: (0.1, 0.7),
            Magnitude.MEDIUM: (0.7, 2.0),
            Magnitude.HIGH: (2.0, 5.0),
            Magnitude.EXTREME: (5.0, 15.0),
        }[self]

    @property
    def rank(self) -> int:
        return {Magnitude.LOW: 1, Magnitude.MEDIUM: 2, Magnitude.HIGH: 3, Magnitude.EXTREME: 4}[self]


class Relation(str, Enum):
    DIRECT = "DIRECT"
    INDIRECT = "INDIRECT"

    @classmethod
    def parse(cls, value: Any) -> "Relation":
        text = str(value or "").strip().upper()
        return cls.DIRECT if text.startswith("DIR") else cls.INDIRECT


class Causality(str, Enum):
    DIRECT = "DIRECT"
    INDIRECT = "INDIRECT"
    MACRO = "MACRO"
    LIQUIDITY = "LIQUIDITY"
    SENTIMENT = "SENTIMENT"
    REGULATORY = "REGULATORY"
    FUNDAMENTAL = "FUNDAMENTAL"

    @classmethod
    def parse(cls, value: Any) -> "Causality":
        text = str(value or "").strip().upper()
        return cls[text] if text in cls.__members__ else cls.SENTIMENT


# --------------------------------------------------------------------------- #
# news + events
# --------------------------------------------------------------------------- #

@dataclass
class NewsItem:
    """One ingested article."""

    title: str
    body: str = ""
    url: str = ""
    source: str = ""
    source_type: str = "rss"  # rss | telegram | x | web | manual
    published_at: datetime | None = None
    fetched_at: datetime = field(default_factory=utcnow)

    id: int | None = None
    exact_hash: str = ""
    norm_hash: str = ""
    normalized: str = ""
    embedding: list[float] | None = None
    event_id: int | None = None
    decision: Stage0Decision | None = None

    @property
    def text(self) -> str:
        """Title + body, which is what every stage prompt reads."""
        body = self.body.strip()
        title = self.title.strip()
        if body and title and not body.startswith(title):
            return f"{title}\n\n{body}"
        return body or title

    def snippet(self, limit: int = 2200) -> str:
        text = self.text
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


@dataclass
class FactSet:
    """Structured facts extracted from an article by Stage 0.

    Split into *identity* (does this describe the same happening?) and *state*
    (has anything about it changed?) — the distinction that lets the system tell
    an UPDATE from a DUPLICATE.
    """

    event_type: str = "OTHER"
    entity: str = ""
    assets: list[str] = field(default_factory=list)
    action: str = ""
    target: str = ""

    status: str = ""
    decision: str = ""
    amount: float | None = None
    price: float | None = None
    percentage: float | None = None
    count: float | None = None
    location: str = ""
    time_reference: str = ""
    event_date: str = ""
    key_claims: list[str] = field(default_factory=list)
    headline: str = ""

    @property
    def primary_asset(self) -> str:
        return self.assets[0] if self.assets else ""

    def identity_fields(self) -> dict[str, str]:
        return {
            "event_type": _norm_token(self.event_type),
            "entity": _norm_token(self.entity),
            "asset": _norm_token(self.primary_asset),
            "action": _norm_token(self.action),
            "target": _norm_token(self.target),
        }

    def identity_key(self) -> str:
        fields = self.identity_fields()
        raw = "|".join(fields[name] for name in ("event_type", "entity", "asset", "action", "target"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def state_fields(self) -> dict[str, Any]:
        return {
            "status": _norm_token(self.status),
            "decision": _norm_token(self.decision),
            "amount": self.amount,
            "price": self.price,
            "percentage": self.percentage,
            "count": self.count,
            "location": _norm_token(self.location),
            "time_reference": _norm_token(self.time_reference),
            "event_date": _norm_token(self.event_date),
            "key_claims": sorted({_norm_token(c) for c in self.key_claims if c}),
        }

    def to_json(self) -> dict:
        return {
            "event_type": self.event_type,
            "entity": self.entity,
            "assets": self.assets,
            "action": self.action,
            "target": self.target,
            "status": self.status,
            "decision": self.decision,
            "amount": self.amount,
            "price": self.price,
            "percentage": self.percentage,
            "count": self.count,
            "location": self.location,
            "time_reference": self.time_reference,
            "event_date": self.event_date,
            "key_claims": self.key_claims,
            "headline": self.headline,
        }

    @classmethod
    def from_json(cls, data: dict) -> "FactSet":
        from services.jsonparse import as_float, as_str_list

        return cls(
            event_type=str(data.get("event_type") or "OTHER").upper().replace(" ", "_"),
            entity=str(data.get("entity") or "").strip(),
            assets=[a.upper() for a in as_str_list(data.get("assets") or data.get("asset"))],
            action=str(data.get("action") or "").strip(),
            target=str(data.get("target") or "").strip(),
            status=str(data.get("status") or "").strip(),
            decision=str(data.get("decision") or "").strip(),
            amount=as_float(data.get("amount")),
            price=as_float(data.get("price")),
            percentage=as_float(data.get("percentage")),
            count=as_float(data.get("count")),
            location=str(data.get("location") or "").strip(),
            time_reference=str(data.get("time_reference") or "").strip(),
            event_date=str(data.get("event_date") or data.get("date") or "").strip(),
            key_claims=as_str_list(data.get("key_claims")),
            headline=str(data.get("headline") or "").strip(),
        )


def _norm_token(value: Any) -> str:
    """Fold a field to a comparable token so trivial wording differences in the
    same fact do not register as a state change."""
    if value is None:
        return ""
    text = re.sub(r"[^a-z0-9$%.+\- ]+", " ", str(value).lower())
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class EventRecord:
    """A happening in the world, tracked across the articles that report it."""

    identity_key: str
    event_type: str = "OTHER"
    entity: str = ""
    primary_asset: str = ""
    action: str = ""
    target: str = ""
    headline: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    status: str = ""

    id: int | None = None
    first_seen: datetime = field(default_factory=utcnow)
    last_seen: datetime = field(default_factory=utcnow)
    article_count: int = 1
    update_count: int = 0
    importance: float = 0.0

    @classmethod
    def from_facts(cls, facts: FactSet) -> "EventRecord":
        identity = facts.identity_fields()
        return cls(
            identity_key=facts.identity_key(),
            event_type=facts.event_type,
            entity=facts.entity,
            primary_asset=facts.primary_asset,
            action=facts.action,
            target=facts.target,
            headline=facts.headline,
            state=facts.state_fields(),
            status=identity.get("status", "") or facts.status,
        )


@dataclass
class Stage0Outcome:
    decision: Stage0Decision
    facts: FactSet
    event: EventRecord | None = None
    reason: str = ""
    similarity: float | None = None
    matched_news_id: int | None = None
    changed_fields: dict[str, Any] = field(default_factory=dict)
    previous_state: dict[str, Any] = field(default_factory=dict)
    #: True when the decision needed no embedding or LLM call at all.
    cheap_path: bool = False

    @property
    def advances(self) -> bool:
        return self.decision in (Stage0Decision.NEW, Stage0Decision.UPDATE)


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #

@dataclass
class AssetLink:
    """Stage 3 output: an asset the event touches."""

    asset: str
    relation: Relation = Relation.DIRECT
    confidence: float = 0.5
    evidence: str = ""
    #: How many panel models named this asset.
    votes: int = 0

    def to_json(self) -> dict:
        return {
            "asset": self.asset,
            "relation": self.relation.value,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "votes": self.votes,
        }


@dataclass
class AssetImpact:
    """Stage 4 output: the expected market effect on one asset."""

    asset: str
    direction: Direction = Direction.NEUTRAL
    magnitude: Magnitude = Magnitude.LOW
    expected_low: float = 0.0
    expected_high: float = 0.0
    confidence: float = 0.5
    horizon_minutes: int = 180
    causality: Causality = Causality.SENTIMENT
    mechanism: str = ""
    risks: str = ""
    relation: Relation = Relation.DIRECT
    #: Share of answering models that agreed on the direction.
    agreement: float = 0.0
    model_count: int = 0
    source: str = "stage4"
    #: Set only by the final layer: whether the expected move clears spread and
    #: normal volatility. ``None`` means no model was asked.
    tradeable: bool | None = None
    key_uncertainty: str = ""
    #: Free-form provenance — where a later stage differed from an earlier one.
    notes: str = ""

    def to_json(self) -> dict:
        return {
            "asset": self.asset,
            "direction": self.direction.value,
            "magnitude": self.magnitude.value,
            "expected_low": round(self.expected_low, 3),
            "expected_high": round(self.expected_high, 3),
            "confidence": round(self.confidence, 3),
            "horizon_minutes": self.horizon_minutes,
            "causality": self.causality.value,
            "mechanism": self.mechanism,
            "risks": self.risks,
            "relation": self.relation.value,
            "agreement": round(self.agreement, 3),
            "model_count": self.model_count,
            "source": self.source,
            "tradeable": self.tradeable,
            "key_uncertainty": self.key_uncertainty,
            "notes": self.notes,
        }

    @property
    def signed_midpoint(self) -> float:
        """Expected move as a signed percentage, for scoring."""
        midpoint = (abs(self.expected_low) + abs(self.expected_high)) / 2
        return midpoint * self.direction.sign

    @property
    def horizon_label(self) -> str:
        minutes = self.horizon_minutes
        if minutes < 60:
            return f"{minutes}m"
        if minutes % 1440 == 0:
            return f"{minutes // 1440}d"
        if minutes % 60 == 0:
            return f"{minutes // 60}h"
        return f"{minutes // 60}h{minutes % 60:02d}m"


@dataclass
class Prediction:
    """A falsifiable claim, stored so the outcome can be scored later."""

    asset: str
    direction: Direction
    expected_low: float
    expected_high: float
    confidence: float
    horizon_minutes: int
    baseline_price: float | None = None
    id: int | None = None
    event_id: int | None = None
    analysis_id: int | None = None
    created_at: datetime = field(default_factory=utcnow)
    resolved: bool = False

    @classmethod
    def from_impact(cls, impact: AssetImpact, *, event_id: int | None = None) -> "Prediction":
        return cls(
            asset=impact.asset,
            direction=impact.direction,
            expected_low=impact.expected_low,
            expected_high=impact.expected_high,
            confidence=impact.confidence,
            horizon_minutes=impact.horizon_minutes,
            event_id=event_id,
        )


@dataclass
class Observation:
    prediction_id: int
    asset: str
    offset_minutes: int
    price: float
    pct_change: float
    observed_at: datetime = field(default_factory=utcnow)


@dataclass
class Outcome:
    prediction_id: int
    direction_correct: bool
    actual_pct: float
    expected_pct: float
    magnitude_error: float
    score: float
    best_horizon_minutes: int
    detail: dict[str, Any] = field(default_factory=dict)
    resolved_at: datetime = field(default_factory=utcnow)


__all__ = [
    "AssetImpact",
    "AssetLink",
    "Causality",
    "Direction",
    "EventRecord",
    "FactSet",
    "Magnitude",
    "NewsItem",
    "Observation",
    "Outcome",
    "Prediction",
    "Relation",
    "Stage0Decision",
    "Stage0Outcome",
    "utcnow",
]
