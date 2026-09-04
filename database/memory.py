"""In-memory store with optional JSON persistence.

This exists so the pipeline is runnable before Postgres is set up — `run.py` falls
back to it automatically when ``DATABASE_URL`` is unset. It is the reference
implementation of :class:`database.base.Store`: the Postgres version must behave
the same way, and the tests assert against this one.

Persistence is deliberately modest. State is written to a JSON file on close and
read back on connect, so `run` followed by `observe` in a separate process still
finds its open predictions. Embeddings are the one thing that is capped
(``MEMORY_EMBED_PERSIST``, default 1500 newest) because 1536 float32s per article
is 6 KB, and a week of a busy feed would produce a file nobody wants. Hash-based
dedup is unaffected by that cap; only similarity search over the very oldest rows
degrades.

Similarity search here is a linear scan. At the spec's ~7000-row working set that
is a few milliseconds, which is fine — but it is O(n), so a real deployment wants
Postgres with pgvector.
"""

from __future__ import annotations

import array
import base64
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from services.config import ROOT, settings as global_settings
from services.types import (
    AssetImpact,
    EventRecord,
    FactSet,
    NewsItem,
    Observation,
    Outcome,
    Prediction,
    utcnow,
)
from database.base import (
    EventUpdate,
    OpenPrediction,
    SimilarNews,
    Store,
    StoredNews,
)
from stage0.embedding import cosine_similarity

log = logging.getLogger(__name__)

DEFAULT_PATH = ROOT / "data" / "store.json"


def _encode_vector(vector: Sequence[float]) -> str:
    return base64.b64encode(array.array("f", vector).tobytes()).decode("ascii")


def _decode_vector(blob: str) -> list[float]:
    buffer = array.array("f")
    buffer.frombytes(base64.b64decode(blob))
    return list(buffer)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class MemoryStore(Store):
    def __init__(self, path: Path | str | None = None, *, persist: bool = True) -> None:
        self.path = Path(path) if path else DEFAULT_PATH
        self.persist = persist
        self.embed_persist_limit = int(os.getenv("MEMORY_EMBED_PERSIST", "1500"))

        self._news: dict[int, dict] = {}
        self._events: dict[int, dict] = {}
        self._event_updates: list[dict] = []
        self._stage_results: list[dict] = []
        self._consensus: list[dict] = []
        self._analyses: dict[int, dict] = {}
        self._predictions: dict[int, dict] = {}
        self._observations: list[dict] = []
        self._outcomes: dict[int, dict] = {}
        self._performance: dict[tuple, dict] = {}
        self._key_health: list[dict] = []
        self._alerts: list[dict] = []
        self._runs: list[dict] = []
        self._counters: dict[str, int] = {}

        # Secondary indexes, rebuilt on load.
        self._by_exact: dict[str, int] = {}
        self._by_norm: dict[str, int] = {}
        self._by_url: dict[str, int] = {}
        self._by_identity: dict[str, int] = {}

    # -- lifecycle --------------------------------------------------------- #

    def _next_id(self, table: str) -> int:
        self._counters[table] = self._counters.get(table, 0) + 1
        return self._counters[table]

    async def connect(self) -> None:
        if self.persist and self.path.is_file():
            try:
                self._load()
            except Exception as exc:  # a corrupt file must not block a run
                log.warning("could not read %s (%s); starting empty", self.path, exc)

    async def close(self) -> None:
        if self.persist:
            try:
                self._save()
            except Exception as exc:
                log.warning("could not write %s: %s", self.path, exc)

    async def init_schema(self) -> None:
        """Nothing to create — but make sure the directory is writable now rather
        than discovering it is not at the end of a long run."""
        if self.persist:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- persistence ------------------------------------------------------- #

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._news = {int(k): v for k, v in data.get("news", {}).items()}
        self._events = {int(k): v for k, v in data.get("events", {}).items()}
        self._event_updates = data.get("event_updates", [])
        self._stage_results = data.get("stage_results", [])
        self._consensus = data.get("consensus", [])
        self._analyses = {int(k): v for k, v in data.get("analyses", {}).items()}
        self._predictions = {int(k): v for k, v in data.get("predictions", {}).items()}
        self._observations = data.get("observations", [])
        self._outcomes = {int(k): v for k, v in data.get("outcomes", {}).items()}
        self._performance = {
            tuple(row["key"]): row["value"] for row in data.get("performance", [])
        }
        self._key_health = data.get("key_health", [])
        self._alerts = data.get("alerts", [])
        self._runs = data.get("runs", [])
        self._counters = data.get("counters", {})

        for row in self._news.values():
            blob = row.pop("embedding_b64", None)
            if blob:
                row["embedding"] = _decode_vector(blob)

        self._reindex()

    def _reindex(self) -> None:
        self._by_exact.clear()
        self._by_norm.clear()
        self._by_url.clear()
        for news_id, row in self._news.items():
            if row.get("exact_hash"):
                self._by_exact[row["exact_hash"]] = news_id
            # First writer wins: the earliest article is the canonical one.
            self._by_norm.setdefault(row.get("norm_hash", ""), news_id)
            if row.get("url_hash"):
                self._by_url.setdefault(row["url_hash"], news_id)
        self._by_norm.pop("", None)

        self._by_identity = {
            row["identity_key"]: event_id
            for event_id, row in self._events.items()
            if row.get("identity_key")
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        keep_embeddings = {
            news_id
            for news_id in sorted(self._news, reverse=True)[: self.embed_persist_limit]
        }
        news_out: dict[str, dict] = {}
        for news_id, row in self._news.items():
            copy = {k: v for k, v in row.items() if k != "embedding"}
            vector = row.get("embedding")
            if vector and news_id in keep_embeddings:
                copy["embedding_b64"] = _encode_vector(vector)
            news_out[str(news_id)] = copy

        payload = {
            "version": 1,
            "saved_at": _iso(utcnow()),
            "counters": self._counters,
            "news": news_out,
            "events": {str(k): v for k, v in self._events.items()},
            "event_updates": self._event_updates,
            # Raw model output is the bulkiest thing here and the least useful
            # after the fact, so only the recent tail is kept.
            "stage_results": self._stage_results[-4000:],
            "consensus": self._consensus[-4000:],
            "analyses": {str(k): v for k, v in self._analyses.items()},
            "predictions": {str(k): v for k, v in self._predictions.items()},
            "observations": self._observations[-8000:],
            "outcomes": {str(k): v for k, v in self._outcomes.items()},
            "performance": [
                {"key": list(key), "value": value} for key, value in self._performance.items()
            ],
            "key_health": self._key_health,
            "alerts": self._alerts[-2000:],
            "runs": self._runs[-200:],
        }

        # Write to a sibling temp file then replace, so a crash mid-write cannot
        # leave a truncated store behind.
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    # -- Stage 0 lookups --------------------------------------------------- #

    def _stored(self, news_id: int) -> StoredNews | None:
        row = self._news.get(news_id)
        if not row:
            return None
        return StoredNews(
            id=news_id,
            title=row.get("title", ""),
            url=row.get("url", ""),
            source=row.get("source", ""),
            norm_hash=row.get("norm_hash", ""),
            identity_key=row.get("identity_key", "") or "",
            event_id=row.get("event_id"),
            decision=row.get("decision", "") or "",
            facts=row.get("facts"),
            fetched_at=_parse_dt(row.get("fetched_at")),
            published_at=_parse_dt(row.get("published_at")),
        )

    async def find_by_exact_hash(self, exact_hash: str) -> StoredNews | None:
        news_id = self._by_exact.get(exact_hash)
        return self._stored(news_id) if news_id else None

    async def find_by_norm_hash(self, norm_hash: str) -> StoredNews | None:
        news_id = self._by_norm.get(norm_hash)
        return self._stored(news_id) if news_id else None

    async def find_by_url_hash(self, url_hash: str) -> StoredNews | None:
        news_id = self._by_url.get(url_hash)
        return self._stored(news_id) if news_id else None

    async def find_similar(
        self,
        embedding: Sequence[float],
        *,
        since: datetime,
        limit: int = 8,
        min_similarity: float = 0.0,
    ) -> list[SimilarNews]:
        hits: list[SimilarNews] = []
        for news_id, row in self._news.items():
            vector = row.get("embedding")
            if not vector:
                continue
            fetched = _parse_dt(row.get("fetched_at"))
            if fetched and fetched < since:
                continue
            score = cosine_similarity(embedding, vector)
            if score >= min_similarity:
                stored = self._stored(news_id)
                if stored:
                    hits.append(SimilarNews(news=stored, similarity=score))
        hits.sort(key=lambda hit: hit.similarity, reverse=True)
        return hits[:limit]

    # -- Stage 0 writes ----------------------------------------------------- #

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
    ) -> int:
        existing = self._by_exact.get(fingerprints.get("exact_hash", ""))
        if existing:
            return existing  # unique constraint, same as Postgres

        news_id = self._next_id("news")
        self._news[news_id] = {
            "title": news.title,
            "body": news.body,
            "url": news.url,
            "source": news.source,
            "source_type": news.source_type,
            "published_at": _iso(news.published_at),
            "fetched_at": _iso(news.fetched_at or utcnow()),
            "exact_hash": fingerprints.get("exact_hash", ""),
            "norm_hash": fingerprints.get("norm_hash", ""),
            "title_hash": fingerprints.get("title_hash", ""),
            "url_hash": fingerprints.get("url_hash", ""),
            "normalized": news.normalized,
            "embedding": list(news.embedding) if news.embedding else None,
            "facts": facts.to_json() if facts else None,
            "identity_key": identity_key,
            "event_id": event_id,
            "decision": decision,
            "decision_reason": reason,
            "similarity": similarity,
            "matched_news_id": matched_news_id,
            "cheap_path": cheap_path,
            "stage": 0,
            "dropped_at": None,
            "drop_reason": None,
            "processed_at": None,
        }
        news.id = news_id

        if fingerprints.get("exact_hash"):
            self._by_exact[fingerprints["exact_hash"]] = news_id
        if fingerprints.get("norm_hash"):
            self._by_norm.setdefault(fingerprints["norm_hash"], news_id)
        if fingerprints.get("url_hash"):
            self._by_url.setdefault(fingerprints["url_hash"], news_id)
        return news_id

    def _event_record(self, event_id: int) -> EventRecord | None:
        row = self._events.get(event_id)
        if not row:
            return None
        return EventRecord(
            id=event_id,
            identity_key=row["identity_key"],
            event_type=row.get("event_type", "OTHER"),
            entity=row.get("entity", ""),
            primary_asset=row.get("primary_asset", ""),
            action=row.get("action", ""),
            target=row.get("target", ""),
            headline=row.get("headline", ""),
            state=row.get("state", {}) or {},
            status=row.get("status", ""),
            first_seen=_parse_dt(row.get("first_seen")) or utcnow(),
            last_seen=_parse_dt(row.get("last_seen")) or utcnow(),
            article_count=row.get("article_count", 1),
            update_count=row.get("update_count", 0),
            importance=row.get("importance", 0.0),
        )

    async def get_event(self, event_id: int) -> EventRecord | None:
        return self._event_record(event_id)

    async def get_event_by_identity(self, identity_key: str) -> EventRecord | None:
        event_id = self._by_identity.get(identity_key)
        return self._event_record(event_id) if event_id else None

    async def create_event(self, event: EventRecord) -> int:
        existing = self._by_identity.get(event.identity_key)
        if existing:
            return existing

        event_id = self._next_id("events")
        self._events[event_id] = {
            "identity_key": event.identity_key,
            "event_type": event.event_type,
            "entity": event.entity,
            "primary_asset": event.primary_asset,
            "action": event.action,
            "target": event.target,
            "headline": event.headline,
            "status": event.status,
            "state": event.state,
            "first_seen": _iso(event.first_seen),
            "last_seen": _iso(event.last_seen),
            "article_count": event.article_count,
            "update_count": event.update_count,
            "importance": event.importance,
        }
        self._by_identity[event.identity_key] = event_id
        event.id = event_id
        return event_id

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
        row = self._events.get(event_id)
        if not row:
            return
        row["last_seen"] = _iso(last_seen)
        row["article_count"] = row.get("article_count", 0) + article_delta
        row["update_count"] = row.get("update_count", 0) + update_delta
        if state is not None:
            row["state"] = state
        if headline:
            row["headline"] = headline
        if status:
            row["status"] = status
        if importance is not None:
            row["importance"] = max(row.get("importance", 0.0), importance)

    async def record_event_update(
        self,
        event_id: int,
        news_id: int | None,
        *,
        changed_fields: dict[str, Any],
        previous_state: dict[str, Any],
        new_state: dict[str, Any],
        summary: str,
    ) -> int:
        update_id = self._next_id("event_updates")
        self._event_updates.append(
            {
                "id": update_id,
                "event_id": event_id,
                "news_id": news_id,
                "changed_fields": changed_fields,
                "previous_state": previous_state,
                "new_state": new_state,
                "summary": summary,
                "created_at": _iso(utcnow()),
            }
        )
        return update_id

    async def event_updates(self, event_id: int, limit: int = 5) -> list[EventUpdate]:
        rows = [row for row in self._event_updates if row["event_id"] == event_id]
        rows.sort(key=lambda row: row["id"], reverse=True)
        return [
            EventUpdate(
                id=row["id"],
                event_id=row["event_id"],
                news_id=row.get("news_id"),
                changed_fields=row.get("changed_fields", {}),
                summary=row.get("summary", ""),
                created_at=_parse_dt(row.get("created_at")),
            )
            for row in rows[:limit]
        ]

    async def event_article_count(self, event_id: int) -> int:
        row = self._events.get(event_id)
        return int(row.get("article_count", 0)) if row else 0

    async def related_events(
        self,
        *,
        event_type: str,
        asset: str,
        since: datetime,
        limit: int = 5,
        exclude_event_id: int | None = None,
    ) -> list[EventRecord]:
        matches: list[EventRecord] = []
        for event_id, row in self._events.items():
            if event_id == exclude_event_id:
                continue
            if event_type and row.get("event_type") != event_type:
                continue
            if asset and row.get("primary_asset") and row["primary_asset"] != asset:
                continue
            last_seen = _parse_dt(row.get("last_seen"))
            if last_seen and last_seen < since:
                continue
            record = self._event_record(event_id)
            if record:
                matches.append(record)
        matches.sort(key=lambda event: event.last_seen, reverse=True)
        return matches[:limit]

    # -- stages ------------------------------------------------------------- #

    async def save_stage_results(
        self,
        news_id: int,
        event_id: int | None,
        stage: int,
        rows: Sequence[dict],
        *,
        asset: str = "",
    ) -> None:
        for row in rows:
            self._stage_results.append(
                {
                    "id": self._next_id("stage_results"),
                    "news_id": news_id,
                    "event_id": event_id,
                    "stage": stage,
                    "asset": asset,
                    "created_at": _iso(utcnow()),
                    **row,
                }
            )

    async def save_consensus(
        self,
        news_id: int,
        event_id: int | None,
        stage: int,
        record: dict,
        *,
        asset: str = "",
    ) -> int:
        consensus_id = self._next_id("consensus")
        self._consensus = [
            row
            for row in self._consensus
            if not (row["news_id"] == news_id and row["stage"] == stage and row["asset"] == asset)
        ]
        self._consensus.append(
            {
                "id": consensus_id,
                "news_id": news_id,
                "event_id": event_id,
                "stage": stage,
                "asset": asset,
                "created_at": _iso(utcnow()),
                **record,
            }
        )
        return consensus_id

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
    ) -> int:
        for analysis_id, row in self._analyses.items():
            if row["news_id"] == news_id and row["asset"] == impact.asset:
                row.update(impact.to_json())
                row["deepest_stage"] = deepest_stage
                row["final_reviewed"] = final_reviewed
                row["tradeable"] = tradeable
                row["detail"] = detail or {}
                return analysis_id

        analysis_id = self._next_id("analyses")
        self._analyses[analysis_id] = {
            "news_id": news_id,
            "event_id": event_id,
            "deepest_stage": deepest_stage,
            "stage2_score": stage2_score,
            "escalated": escalated,
            "final_reviewed": final_reviewed,
            "tradeable": tradeable,
            "detail": detail or {},
            "created_at": _iso(utcnow()),
            **impact.to_json(),
        }
        return analysis_id

    async def save_prediction(
        self,
        prediction: Prediction,
        *,
        analysis_id: int | None = None,
        news_id: int | None = None,
        model_ids: Sequence[str] = (),
        deepest_stage: int = 4,
    ) -> int:
        prediction_id = self._next_id("predictions")
        event_type = "ALL"
        if prediction.event_id:
            row = self._events.get(prediction.event_id)
            if row:
                event_type = row.get("event_type", "ALL")

        self._predictions[prediction_id] = {
            "analysis_id": analysis_id,
            "news_id": news_id,
            "event_id": prediction.event_id,
            "event_type": event_type,
            "asset": prediction.asset,
            "direction": prediction.direction.value,
            "expected_low": prediction.expected_low,
            "expected_high": prediction.expected_high,
            "confidence": prediction.confidence,
            "horizon_minutes": prediction.horizon_minutes,
            "baseline_price": prediction.baseline_price,
            "model_ids": list(model_ids),
            "deepest_stage": deepest_stage,
            "created_at": _iso(prediction.created_at),
            "resolved": False,
            "resolved_at": None,
        }
        prediction.id = prediction_id
        return prediction_id

    async def mark_processed(
        self,
        news_id: int,
        *,
        stage: int,
        dropped_at: int | None = None,
        drop_reason: str | None = None,
    ) -> None:
        row = self._news.get(news_id)
        if not row:
            return
        row["stage"] = stage
        row["dropped_at"] = dropped_at
        row["drop_reason"] = drop_reason
        row["processed_at"] = _iso(utcnow())

    async def pending_news(self, limit: int = 25) -> list[NewsItem]:
        items: list[NewsItem] = []
        for news_id in sorted(self._news):
            row = self._news[news_id]
            if row.get("processed_at"):
                continue
            items.append(
                NewsItem(
                    id=news_id,
                    title=row.get("title", ""),
                    body=row.get("body", ""),
                    url=row.get("url", ""),
                    source=row.get("source", ""),
                    source_type=row.get("source_type", "rss"),
                    published_at=_parse_dt(row.get("published_at")),
                    fetched_at=_parse_dt(row.get("fetched_at")) or utcnow(),
                    exact_hash=row.get("exact_hash", ""),
                    norm_hash=row.get("norm_hash", ""),
                    event_id=row.get("event_id"),
                )
            )
            if len(items) >= limit:
                break
        return items

    # -- feedback ----------------------------------------------------------- #

    async def open_predictions(self, *, now: datetime, limit: int = 200) -> list[OpenPrediction]:
        out: list[OpenPrediction] = []
        for prediction_id, row in sorted(self._predictions.items()):
            if row.get("resolved"):
                continue
            created = _parse_dt(row.get("created_at")) or now
            observed = {
                obs["offset_minutes"]: obs["pct_change"]
                for obs in self._observations
                if obs["prediction_id"] == prediction_id
            }
            out.append(
                OpenPrediction(
                    id=prediction_id,
                    asset=row["asset"],
                    direction=row["direction"],
                    expected_low=row.get("expected_low", 0.0),
                    expected_high=row.get("expected_high", 0.0),
                    confidence=row.get("confidence", 0.0),
                    horizon_minutes=row.get("horizon_minutes", 180),
                    created_at=created,
                    baseline_price=row.get("baseline_price"),
                    model_ids=list(row.get("model_ids") or []),
                    event_type=row.get("event_type", "ALL"),
                    deepest_stage=row.get("deepest_stage", 4),
                    observations=observed,
                )
            )
            if len(out) >= limit:
                break
        return out

    async def set_baseline_price(self, prediction_id: int, price: float) -> None:
        row = self._predictions.get(prediction_id)
        if row:
            row["baseline_price"] = price

    async def save_observation(self, observation: Observation) -> int:
        for row in self._observations:
            if (
                row["prediction_id"] == observation.prediction_id
                and row["offset_minutes"] == observation.offset_minutes
            ):
                return row["id"]
        observation_id = self._next_id("observations")
        self._observations.append(
            {
                "id": observation_id,
                "prediction_id": observation.prediction_id,
                "asset": observation.asset,
                "offset_minutes": observation.offset_minutes,
                "price": observation.price,
                "pct_change": observation.pct_change,
                "observed_at": _iso(observation.observed_at),
            }
        )
        return observation_id

    async def save_outcome(self, outcome: Outcome) -> int:
        outcome_id = self._next_id("outcomes")
        self._outcomes[outcome.prediction_id] = {
            "id": outcome_id,
            "direction_correct": outcome.direction_correct,
            "actual_pct": outcome.actual_pct,
            "expected_pct": outcome.expected_pct,
            "magnitude_error": outcome.magnitude_error,
            "score": outcome.score,
            "best_horizon_minutes": outcome.best_horizon_minutes,
            "detail": outcome.detail,
            "resolved_at": _iso(outcome.resolved_at),
        }
        row = self._predictions.get(outcome.prediction_id)
        if row:
            row["resolved"] = True
            row["resolved_at"] = _iso(outcome.resolved_at)
        return outcome_id

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
        key = (model_id, stage, event_type or "ALL", asset or "ALL")
        row = self._performance.setdefault(
            key,
            {
                "model_id": model_id,
                "stage": stage,
                "event_type": event_type or "ALL",
                "asset": asset or "ALL",
                "predictions": 0,
                "direction_correct": 0,
                "avg_magnitude_error": 0.0,
                "avg_score": 0.0,
                "avg_latency_ms": 0,
                "failures": 0,
            },
        )
        previous = row["predictions"]
        row["predictions"] = previous + 1
        row["direction_correct"] += 1 if direction_correct else 0
        row["failures"] += 1 if failed else 0
        # Running mean, so history is not re-read on every update.
        row["avg_magnitude_error"] = (
            row["avg_magnitude_error"] * previous + magnitude_error
        ) / row["predictions"]
        row["avg_score"] = (row["avg_score"] * previous + score) / row["predictions"]
        if latency_ms:
            row["avg_latency_ms"] = int(
                (row["avg_latency_ms"] * previous + latency_ms) / row["predictions"]
            )
        row["updated_at"] = _iso(utcnow())

    async def model_performance(
        self,
        *,
        stage: int | None = None,
        event_type: str | None = None,
        asset: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        rows = list(self._performance.values())
        if stage is not None:
            rows = [row for row in rows if row["stage"] == stage]
        if event_type:
            rows = [row for row in rows if row["event_type"] == event_type]
        if asset:
            rows = [row for row in rows if row["asset"] == asset]
        rows.sort(key=lambda row: (row["predictions"], row["avg_score"]), reverse=True)
        return rows[:limit]

    # -- operations --------------------------------------------------------- #

    async def save_key_health(self, rows: Sequence[dict]) -> None:
        self._key_health = [dict(row) for row in rows]

    async def load_key_health(self) -> list[dict]:
        return [dict(row) for row in self._key_health]

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
    ) -> int:
        alert_id = self._next_id("alerts")
        self._alerts.append(
            {
                "id": alert_id,
                "news_id": news_id,
                "event_id": event_id,
                "analysis_id": analysis_id,
                "asset": asset,
                "chat_id": chat_id,
                "message_id": message_id,
                "kind": kind,
                "text": text,
                "sent": sent,
                "error": error,
                "created_at": _iso(utcnow()),
            }
        )
        return alert_id

    async def already_alerted(self, event_id: int, asset: str) -> bool:
        return any(
            row.get("event_id") == event_id
            and row.get("asset") == asset
            and row.get("sent")
            for row in self._alerts
        )

    async def save_run(self, stats: dict) -> int:
        run_id = self._next_id("runs")
        self._runs.append({"id": run_id, **stats})
        return run_id

    async def recent_analyses(self, limit: int = 10) -> list[dict]:
        rows = [
            {"id": analysis_id, **row}
            for analysis_id, row in sorted(self._analyses.items(), reverse=True)
        ]
        for row in rows:
            event = self._events.get(row.get("event_id") or -1)
            row["headline"] = event.get("headline", "") if event else ""
            row["event_type"] = event.get("event_type", "") if event else ""
        return rows[:limit]

    async def counts(self) -> dict[str, int]:
        return {
            "news": len(self._news),
            "events": len(self._events),
            "event_updates": len(self._event_updates),
            "analyses": len(self._analyses),
            "predictions": len(self._predictions),
            "observations": len(self._observations),
            "outcomes": len(self._outcomes),
            "alerts": len(self._alerts),
        }

    # -- helpers used by tests --------------------------------------------- #

    def purge_older_than(self, days: int | None = None) -> int:
        """Drop news outside the lookback window, so a long-lived process does
        not grow without bound."""
        days = days or global_settings.stage0.lookback_days
        cutoff = utcnow() - timedelta(days=days)
        stale = [
            news_id
            for news_id, row in self._news.items()
            if (_parse_dt(row.get("fetched_at")) or utcnow()) < cutoff
        ]
        for news_id in stale:
            self._news.pop(news_id, None)
        if stale:
            self._reindex()
        return len(stale)


__all__ = ["MemoryStore"]
