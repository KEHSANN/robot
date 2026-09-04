"""Storage interface.

Two implementations sit behind this: :class:`database.postgres.PostgresStore` for
real deployments (pgvector does the similarity search) and
:class:`database.memory.MemoryStore` for tests and for running the pipeline before
Postgres exists. The interface is written so Stage 0 does not know or care which
one it has — the only difference the pipeline can observe is that the memory store
forgets everything on exit.

Every method that writes returns the row id, because the pipeline threads ids
through from news to event to analysis to prediction, and the feedback loop needs
that chain intact months later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from services.types import (
    AssetImpact,
    EventRecord,
    FactSet,
    NewsItem,
    Observation,
    Outcome,
    Prediction,
)


@dataclass
class StoredNews:
    """A news row as the store hands it back — enough to make a Stage 0 decision
    against it without loading the full article text."""

    id: int
    title: str = ""
    url: str = ""
    source: str = ""
    norm_hash: str = ""
    identity_key: str = ""
    event_id: int | None = None
    decision: str = ""
    facts: dict | None = None
    fetched_at: datetime | None = None
    published_at: datetime | None = None

    def fact_set(self) -> FactSet | None:
        return FactSet.from_json(self.facts) if self.facts else None


@dataclass
class SimilarNews:
    """A nearest-neighbour hit from the embedding search."""

    news: StoredNews
    similarity: float


@dataclass
class EventUpdate:
    """One recorded development of an event, newest first when listed."""

    id: int
    event_id: int
    news_id: int | None
    changed_fields: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    created_at: datetime | None = None


@dataclass
class OpenPrediction:
    """A prediction awaiting observation, with what the feedback loop needs."""

    id: int
    asset: str
    direction: str
    expected_low: float
    expected_high: float
    confidence: float
    horizon_minutes: int
    created_at: datetime
    baseline_price: float | None = None
    model_ids: list[str] = field(default_factory=list)
    event_type: str = "ALL"
    deepest_stage: int = 4
    observations: dict[int, float] = field(default_factory=dict)


class Store(ABC):
    """Persistence for the whole pipeline."""

    # -- lifecycle --------------------------------------------------------- #

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    async def __aenter__(self) -> "Store":
        await self.connect()
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    @abstractmethod
    async def init_schema(self) -> None:
        """Create tables and indexes. Idempotent."""

    @property
    def supports_vector_search(self) -> bool:
        """Whether :meth:`find_similar` uses a real ANN index."""
        return False

    # -- Stage 0: dedup lookups -------------------------------------------- #

    @abstractmethod
    async def find_by_exact_hash(self, exact_hash: str) -> StoredNews | None: ...

    @abstractmethod
    async def find_by_norm_hash(self, norm_hash: str) -> StoredNews | None: ...

    @abstractmethod
    async def find_by_url_hash(self, url_hash: str) -> StoredNews | None: ...

    @abstractmethod
    async def find_similar(
        self,
        embedding: Sequence[float],
        *,
        since: datetime,
        limit: int = 8,
        min_similarity: float = 0.0,
    ) -> list[SimilarNews]:
        """Nearest neighbours inside the lookback window, best first."""

    # -- Stage 0: writes ---------------------------------------------------- #

    @abstractmethod
    async def insert_news(
        self,
        news: NewsItem,
        *,
        fingerprints: dict[str, str],
        facts: FactSet | None = None,
        identity_key: str = "",
        event_id: int | None = None,
        decision: str = "",
        reason: str = "",
        similarity: float | None = None,
        matched_news_id: int | None = None,
        cheap_path: bool = False,
    ) -> int: ...

    @abstractmethod
    async def get_event(self, event_id: int) -> EventRecord | None: ...

    @abstractmethod
    async def get_event_by_identity(self, identity_key: str) -> EventRecord | None: ...

    @abstractmethod
    async def create_event(self, event: EventRecord) -> int: ...

    @abstractmethod
    async def touch_event(
        self,
        event_id: int,
        *,
        last_seen: datetime,
        state: dict[str, Any] | None = None,
        headline: str | None = None,
        status: str | None = None,
        article_delta: int = 1,
        update_delta: int = 0,
        importance: float | None = None,
    ) -> None:
        """Fold a new sighting into an existing event."""

    @abstractmethod
    async def record_event_update(
        self,
        event_id: int,
        news_id: int | None,
        *,
        changed_fields: dict[str, Any],
        previous_state: dict[str, Any],
        new_state: dict[str, Any],
        summary: str,
    ) -> int: ...

    @abstractmethod
    async def event_updates(self, event_id: int, limit: int = 5) -> list[EventUpdate]: ...

    @abstractmethod
    async def event_article_count(self, event_id: int) -> int:
        """How many distinct articles reported this event — the source count the
        deep stages weigh as corroboration."""

    @abstractmethod
    async def related_events(
        self,
        *,
        event_type: str,
        asset: str,
        since: datetime,
        limit: int = 5,
        exclude_event_id: int | None = None,
    ) -> list[EventRecord]:
        """Historical events of the same type and asset, for Stage 5 context."""

    # -- Stages 1-6 --------------------------------------------------------- #

    @abstractmethod
    async def save_stage_results(
        self,
        news_id: int,
        event_id: int | None,
        stage: int,
        rows: Sequence[dict],
        *,
        asset: str = "",
    ) -> None:
        """Persist one row per model. ``rows`` come from ``LLMResult.as_record()``
        plus a ``vote`` key."""

    @abstractmethod
    async def save_consensus(
        self,
        news_id: int,
        event_id: int | None,
        stage: int,
        record: dict,
        *,
        asset: str = "",
    ) -> int: ...

    @abstractmethod
    async def save_analysis(
        self,
        news_id: int,
        event_id: int | None,
        impact: AssetImpact,
        *,
        deepest_stage: int = 4,
        stage2_score: float | None = None,
        escalated: bool = False,
        final_reviewed: bool = False,
        tradeable: bool | None = None,
        detail: dict | None = None,
    ) -> int: ...

    @abstractmethod
    async def save_prediction(
        self,
        prediction: Prediction,
        *,
        analysis_id: int | None = None,
        news_id: int | None = None,
        model_ids: Sequence[str] = (),
        deepest_stage: int = 4,
    ) -> int: ...

    @abstractmethod
    async def mark_processed(
        self,
        news_id: int,
        *,
        stage: int,
        dropped_at: int | None = None,
        drop_reason: str | None = None,
    ) -> None: ...

    @abstractmethod
    async def pending_news(self, limit: int = 25) -> list[NewsItem]:
        """Articles ingested but not yet run through the stages."""

    # -- feedback loop ------------------------------------------------------ #

    @abstractmethod
    async def open_predictions(self, *, now: datetime, limit: int = 200) -> list[OpenPrediction]: ...

    @abstractmethod
    async def set_baseline_price(self, prediction_id: int, price: float) -> None: ...

    @abstractmethod
    async def save_observation(self, observation: Observation) -> int: ...

    @abstractmethod
    async def save_outcome(self, outcome: Outcome) -> int: ...

    @abstractmethod
    async def bump_model_performance(
        self,
        *,
        model_id: str,
        stage: int,
        event_type: str,
        asset: str,
        direction_correct: bool,
        magnitude_error: float,
        score: float,
        latency_ms: int = 0,
        failed: bool = False,
    ) -> None:
        """Fold one scored prediction into the running per-model averages."""

    @abstractmethod
    async def model_performance(
        self,
        *,
        stage: int | None = None,
        event_type: str | None = None,
        asset: str | None = None,
        limit: int = 50,
    ) -> list[dict]: ...

    # -- operations --------------------------------------------------------- #

    @abstractmethod
    async def save_key_health(self, rows: Sequence[dict]) -> None: ...

    @abstractmethod
    async def load_key_health(self) -> list[dict]: ...

    @abstractmethod
    async def record_alert(
        self,
        *,
        chat_id: str,
        text: str,
        news_id: int | None = None,
        event_id: int | None = None,
        analysis_id: int | None = None,
        asset: str = "",
        kind: str = "alert",
        message_id: int | None = None,
        sent: bool = False,
        error: str | None = None,
    ) -> int: ...

    @abstractmethod
    async def already_alerted(self, event_id: int, asset: str) -> bool:
        """Whether this event/asset pair has already gone out.

        Guards against re-alerting on an UPDATE that says nothing new.
        """

    @abstractmethod
    async def save_run(self, stats: dict) -> int: ...

    @abstractmethod
    async def recent_analyses(self, limit: int = 10) -> list[dict]:
        """Newest published verdicts, for the bot's ``/last`` command."""

    @abstractmethod
    async def counts(self) -> dict[str, int]:
        """Row counts for ``/status`` and ``doctor``."""


__all__ = [
    "EventUpdate",
    "OpenPrediction",
    "SimilarNews",
    "Store",
    "StoredNews",
]
