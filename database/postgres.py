"""Postgres + pgvector store.

The real deployment target. Two things it does that the in-memory store cannot:
the Stage 0 similarity search runs as an indexed ANN query instead of a linear
scan, and state survives a restart in full — including key cooldowns, so a key
that hit its daily quota is not probed again the moment the process comes back.

Every query is written out longhand rather than through an ORM. The pipeline has
perhaps twenty queries, several of them vector operations that an ORM would fight
about, and being able to read exactly what hits the database is worth more here
than the abstraction.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Sequence

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

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

log = logging.getLogger(__name__)

SCHEMA_PATH = ROOT / "database" / "schema.sql"

_NEWS_COLUMNS = """
    id, title, url, source, norm_hash, identity_key, event_id,
    decision, facts, fetched_at, published_at
"""


def _vector_literal(values: Sequence[float]) -> str:
    """pgvector's text input form.

    Passed as a string rather than via the pgvector adapter so the store works
    whether or not ``pgvector.psycopg.register_vector`` has run — one less thing
    that can silently break a deployment.
    """
    return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"


def _stored_news(row: dict) -> StoredNews:
    return StoredNews(
        id=row["id"],
        title=row.get("title") or "",
        url=row.get("url") or "",
        source=row.get("source") or "",
        norm_hash=row.get("norm_hash") or "",
        identity_key=row.get("identity_key") or "",
        event_id=row.get("event_id"),
        decision=row.get("decision") or "",
        facts=row.get("facts"),
        fetched_at=row.get("fetched_at"),
        published_at=row.get("published_at"),
    )


def _event_record(row: dict) -> EventRecord:
    return EventRecord(
        id=row["id"],
        identity_key=row["identity_key"],
        event_type=row.get("event_type") or "OTHER",
        entity=row.get("entity") or "",
        primary_asset=row.get("primary_asset") or "",
        action=row.get("action") or "",
        target=row.get("target") or "",
        headline=row.get("headline") or "",
        state=row.get("state") or {},
        status=row.get("status") or "",
        first_seen=row.get("first_seen") or utcnow(),
        last_seen=row.get("last_seen") or utcnow(),
        article_count=row.get("article_count") or 0,
        update_count=row.get("update_count") or 0,
        importance=row.get("importance") or 0.0,
    )


class PostgresStore(Store):
    def __init__(self, dsn: str | None = None, *, min_size: int = 1, max_size: int = 8) -> None:
        self.dsn = dsn or global_settings.database_url
        if not self.dsn:
            raise ValueError("DATABASE_URL is not set")
        self._pool: AsyncConnectionPool | None = None
        self._min_size = min_size
        self._max_size = max_size

    # -- lifecycle --------------------------------------------------------- #

    async def connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = AsyncConnectionPool(
            self.dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            open=False,
            kwargs={"row_factory": dict_row, "autocommit": True},
        )
        await self._pool.open(wait=True, timeout=20)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            raise RuntimeError("PostgresStore.connect() has not been called")
        return self._pool

    @property
    def supports_vector_search(self) -> bool:
        return True

    async def init_schema(self) -> None:
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        sql = sql.replace("__EMBED_DIM__", str(global_settings.stage0.embed_dim))
        async with self.pool.connection() as conn:
            await conn.execute(sql)
        log.info("schema applied (embedding dimension %d)", global_settings.stage0.embed_dim)

    # -- small helpers ------------------------------------------------------ #

    async def _fetchone(self, sql: str, params: Sequence[Any] = ()) -> dict | None:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(sql, params)
            return await cursor.fetchone()

    async def _fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[dict]:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(sql, params)
            return await cursor.fetchall()

    async def _execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(sql, params)

    async def _scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = await self._fetchone(sql, params)
        if not row:
            return None
        return next(iter(row.values()))

    # -- Stage 0 lookups --------------------------------------------------- #

    async def find_by_exact_hash(self, exact_hash: str) -> StoredNews | None:
        row = await self._fetchone(
            f"SELECT {_NEWS_COLUMNS} FROM news WHERE exact_hash = %s", (exact_hash,)
        )
        return _stored_news(row) if row else None

    async def find_by_norm_hash(self, norm_hash: str) -> StoredNews | None:
        # Oldest match wins: the first article to carry this text is canonical.
        row = await self._fetchone(
            f"SELECT {_NEWS_COLUMNS} FROM news WHERE norm_hash = %s ORDER BY id LIMIT 1",
            (norm_hash,),
        )
        return _stored_news(row) if row else None

    async def find_by_url_hash(self, url_hash: str) -> StoredNews | None:
        if not url_hash:
            return None
        row = await self._fetchone(
            f"SELECT {_NEWS_COLUMNS} FROM news WHERE url_hash = %s ORDER BY id LIMIT 1",
            (url_hash,),
        )
        return _stored_news(row) if row else None

    async def find_similar(
        self,
        embedding: Sequence[float],
        *,
        since: datetime,
        limit: int = 8,
        min_similarity: float = 0.0,
    ) -> list[SimilarNews]:
        if not embedding:
            return []
        # `<=>` is cosine *distance*; similarity is 1 - distance. The window on
        # fetched_at is what keeps this query cheap as the table grows.
        rows = await self._fetchall(
            f"""
            SELECT {_NEWS_COLUMNS}, 1 - (embedding <=> %s::vector) AS similarity
              FROM news
             WHERE embedding IS NOT NULL
               AND fetched_at >= %s
             ORDER BY embedding <=> %s::vector
             LIMIT %s
            """,
            (_vector_literal(embedding), since, _vector_literal(embedding), limit),
        )
        return [
            SimilarNews(news=_stored_news(row), similarity=float(row["similarity"]))
            for row in rows
            if row["similarity"] is not None and float(row["similarity"]) >= min_similarity
        ]

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
        from stage0.normalize import canonical_url

        embedding = _vector_literal(news.embedding) if news.embedding else None
        row = await self._fetchone(
            """
            INSERT INTO news (
                title, body, url, canonical_url, source, source_type,
                published_at, fetched_at,
                exact_hash, norm_hash, title_hash, url_hash, normalized, embedding,
                facts, identity_key, event_id, decision, decision_reason,
                similarity, matched_news_id, cheap_path
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s, %s, %s::vector,
                %s, %s, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (exact_hash) DO UPDATE
                -- A re-fetch of the same article should not create a second row,
                -- but it should still refresh what we learned about it.
                SET event_id     = COALESCE(EXCLUDED.event_id, news.event_id),
                    decision     = COALESCE(NULLIF(EXCLUDED.decision, ''), news.decision),
                    facts        = COALESCE(EXCLUDED.facts, news.facts),
                    identity_key = COALESCE(NULLIF(EXCLUDED.identity_key, ''), news.identity_key)
            RETURNING id
            """,
            (
                news.title,
                news.body,
                news.url,
                canonical_url(news.url),
                news.source,
                news.source_type,
                news.published_at,
                news.fetched_at or utcnow(),
                fingerprints.get("exact_hash", ""),
                fingerprints.get("norm_hash", ""),
                fingerprints.get("title_hash", ""),
                fingerprints.get("url_hash", ""),
                news.normalized,
                embedding,
                Jsonb(facts.to_json()) if facts else None,
                identity_key or None,
                event_id,
                decision,
                reason,
                similarity,
                matched_news_id,
                cheap_path,
            ),
        )
        news.id = int(row["id"])
        return news.id

    async def get_event(self, event_id: int) -> EventRecord | None:
        row = await self._fetchone("SELECT * FROM events WHERE id = %s", (event_id,))
        return _event_record(row) if row else None

    async def get_event_by_identity(self, identity_key: str) -> EventRecord | None:
        row = await self._fetchone(
            "SELECT * FROM events WHERE identity_key = %s", (identity_key,)
        )
        return _event_record(row) if row else None

    async def create_event(self, event: EventRecord) -> int:
        row = await self._fetchone(
            """
            INSERT INTO events (
                identity_key, event_type, entity, primary_asset, action, target,
                headline, status, state, first_seen, last_seen,
                article_count, update_count, importance
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (identity_key) DO UPDATE
                SET last_seen     = GREATEST(events.last_seen, EXCLUDED.last_seen),
                    article_count = events.article_count + 1
            RETURNING id
            """,
            (
                event.identity_key,
                event.event_type,
                event.entity,
                event.primary_asset,
                event.action,
                event.target,
                event.headline,
                event.status,
                Jsonb(event.state),
                event.first_seen,
                event.last_seen,
                event.article_count,
                event.update_count,
                event.importance,
            ),
        )
        event.id = int(row["id"])
        return event.id

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
        await self._execute(
            """
            UPDATE events
               SET last_seen     = GREATEST(last_seen, %s),
                   state         = COALESCE(%s, state),
                   headline      = COALESCE(NULLIF(%s, ''), headline),
                   status        = COALESCE(NULLIF(%s, ''), status),
                   article_count = article_count + %s,
                   update_count  = update_count + %s,
                   importance    = GREATEST(importance, COALESCE(%s, 0))
             WHERE id = %s
            """,
            (
                last_seen,
                Jsonb(state) if state is not None else None,
                headline or "",
                status or "",
                article_delta,
                update_delta,
                importance,
                event_id,
            ),
        )

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
        row = await self._fetchone(
            """
            INSERT INTO event_updates
                (event_id, news_id, changed_fields, previous_state, new_state, summary)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                event_id,
                news_id,
                Jsonb(changed_fields),
                Jsonb(previous_state),
                Jsonb(new_state),
                summary,
            ),
        )
        return int(row["id"])

    async def event_updates(self, event_id: int, limit: int = 5) -> list[EventUpdate]:
        rows = await self._fetchall(
            """
            SELECT id, event_id, news_id, changed_fields, summary, created_at
              FROM event_updates
             WHERE event_id = %s
             ORDER BY created_at DESC, id DESC
             LIMIT %s
            """,
            (event_id, limit),
        )
        return [
            EventUpdate(
                id=row["id"],
                event_id=row["event_id"],
                news_id=row.get("news_id"),
                changed_fields=row.get("changed_fields") or {},
                summary=row.get("summary") or "",
                created_at=row.get("created_at"),
            )
            for row in rows
        ]

    async def event_article_count(self, event_id: int) -> int:
        value = await self._scalar(
            "SELECT article_count FROM events WHERE id = %s", (event_id,)
        )
        return int(value or 0)

    async def related_events(
        self,
        *,
        event_type: str,
        asset: str,
        since: datetime,
        limit: int = 5,
        exclude_event_id: int | None = None,
    ) -> list[EventRecord]:
        rows = await self._fetchall(
            """
            SELECT * FROM events
             WHERE last_seen >= %s
               AND (%s = '' OR event_type = %s)
               AND (%s = '' OR primary_asset = %s OR primary_asset = '')
               AND (%s::bigint IS NULL OR id <> %s::bigint)
             ORDER BY last_seen DESC
             LIMIT %s
            """,
            (
                since,
                event_type, event_type,
                asset, asset,
                exclude_event_id, exclude_event_id,
                limit,
            ),
        )
        return [_event_record(row) for row in rows]

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
        if not rows:
            return
        params = [
            (
                news_id,
                event_id,
                stage,
                asset,
                row.get("model_id", ""),
                row.get("provider", ""),
                row.get("key_index"),
                bool(row.get("ok")),
                row.get("vote"),
                row.get("latency_ms", 0),
                row.get("attempts", 1),
                row.get("error"),
                row.get("finish_reason"),
                row.get("prompt_tokens"),
                row.get("completion_tokens"),
                Jsonb(row.get("raw") or {}),
            )
            for row in rows
        ]
        async with self.pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.executemany(
                    """
                    INSERT INTO stage_results (
                        news_id, event_id, stage, asset, model_id, provider,
                        key_index, ok, vote, latency_ms, attempts, error,
                        finish_reason, prompt_tokens, completion_tokens, raw
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    params,
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
        row = await self._fetchone(
            """
            INSERT INTO stage_consensus (
                news_id, event_id, stage, asset, verdict,
                votes_pass, votes_reject, votes_failed, agreement, score, note, detail
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (news_id, stage, asset) DO UPDATE
                SET verdict = EXCLUDED.verdict,
                    votes_pass = EXCLUDED.votes_pass,
                    votes_reject = EXCLUDED.votes_reject,
                    votes_failed = EXCLUDED.votes_failed,
                    agreement = EXCLUDED.agreement,
                    score = EXCLUDED.score,
                    note = EXCLUDED.note,
                    detail = EXCLUDED.detail
            RETURNING id
            """,
            (
                news_id,
                event_id,
                stage,
                asset,
                record.get("verdict", "INCONCLUSIVE"),
                record.get("votes_pass", 0),
                record.get("votes_reject", 0),
                record.get("votes_failed", 0),
                record.get("agreement", 0.0),
                record.get("score"),
                record.get("note"),
                Jsonb(record.get("detail") or {}),
            ),
        )
        return int(row["id"])

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
        row = await self._fetchone(
            """
            INSERT INTO analyses (
                news_id, event_id, asset, direction, magnitude,
                expected_low, expected_high, confidence, horizon_minutes,
                causality, relation, mechanism, risks, agreement, model_count,
                deepest_stage, stage2_score, escalated, final_reviewed, tradeable, detail
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (news_id, asset) DO UPDATE
                -- A deeper stage always supersedes a shallower one.
                SET direction = EXCLUDED.direction,
                    magnitude = EXCLUDED.magnitude,
                    expected_low = EXCLUDED.expected_low,
                    expected_high = EXCLUDED.expected_high,
                    confidence = EXCLUDED.confidence,
                    horizon_minutes = EXCLUDED.horizon_minutes,
                    causality = EXCLUDED.causality,
                    mechanism = EXCLUDED.mechanism,
                    risks = EXCLUDED.risks,
                    agreement = EXCLUDED.agreement,
                    model_count = EXCLUDED.model_count,
                    deepest_stage = GREATEST(analyses.deepest_stage, EXCLUDED.deepest_stage),
                    escalated = analyses.escalated OR EXCLUDED.escalated,
                    final_reviewed = analyses.final_reviewed OR EXCLUDED.final_reviewed,
                    tradeable = EXCLUDED.tradeable,
                    detail = EXCLUDED.detail
            RETURNING id
            """,
            (
                news_id,
                event_id,
                impact.asset,
                impact.direction.value,
                impact.magnitude.value,
                impact.expected_low,
                impact.expected_high,
                impact.confidence,
                impact.horizon_minutes,
                impact.causality.value,
                impact.relation.value,
                impact.mechanism,
                impact.risks,
                impact.agreement,
                impact.model_count,
                deepest_stage,
                stage2_score,
                escalated,
                final_reviewed,
                tradeable,
                Jsonb(detail or {}),
            ),
        )
        return int(row["id"])

    async def save_prediction(
        self,
        prediction: Prediction,
        *,
        analysis_id: int | None = None,
        news_id: int | None = None,
        model_ids: Sequence[str] = (),
        deepest_stage: int = 4,
    ) -> int:
        row = await self._fetchone(
            """
            INSERT INTO predictions (
                analysis_id, news_id, event_id, asset, direction,
                expected_low, expected_high, confidence, horizon_minutes,
                baseline_price, model_ids, deepest_stage, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                analysis_id,
                news_id,
                prediction.event_id,
                prediction.asset,
                prediction.direction.value,
                prediction.expected_low,
                prediction.expected_high,
                prediction.confidence,
                prediction.horizon_minutes,
                prediction.baseline_price,
                list(model_ids),
                deepest_stage,
                prediction.created_at,
            ),
        )
        prediction.id = int(row["id"])
        return prediction.id

    async def mark_processed(
        self,
        news_id: int,
        *,
        stage: int,
        dropped_at: int | None = None,
        drop_reason: str | None = None,
    ) -> None:
        await self._execute(
            """
            UPDATE news
               SET stage = %s, dropped_at = %s, drop_reason = %s, processed_at = now()
             WHERE id = %s
            """,
            (stage, dropped_at, drop_reason, news_id),
        )

    async def pending_news(self, limit: int = 25) -> list[NewsItem]:
        rows = await self._fetchall(
            """
            SELECT id, title, body, url, source, source_type, published_at,
                   fetched_at, exact_hash, norm_hash, event_id
              FROM news
             WHERE processed_at IS NULL
             ORDER BY id
             LIMIT %s
            """,
            (limit,),
        )
        return [
            NewsItem(
                id=row["id"],
                title=row["title"],
                body=row.get("body") or "",
                url=row.get("url") or "",
                source=row.get("source") or "",
                source_type=row.get("source_type") or "rss",
                published_at=row.get("published_at"),
                fetched_at=row.get("fetched_at") or utcnow(),
                exact_hash=row.get("exact_hash") or "",
                norm_hash=row.get("norm_hash") or "",
                event_id=row.get("event_id"),
            )
            for row in rows
        ]

    # -- feedback ----------------------------------------------------------- #

    async def open_predictions(self, *, now: datetime, limit: int = 200) -> list[OpenPrediction]:
        rows = await self._fetchall(
            """
            SELECT p.id, p.asset, p.direction, p.expected_low, p.expected_high,
                   p.confidence, p.horizon_minutes, p.created_at, p.baseline_price,
                   p.model_ids, p.deepest_stage,
                   COALESCE(e.event_type, 'ALL') AS event_type,
                   COALESCE(
                       jsonb_object_agg(o.offset_minutes::text, o.pct_change)
                           FILTER (WHERE o.id IS NOT NULL),
                       '{}'::jsonb
                   ) AS observations
              FROM predictions p
              LEFT JOIN events e ON e.id = p.event_id
              LEFT JOIN observations o ON o.prediction_id = p.id
             WHERE NOT p.resolved
             GROUP BY p.id, e.event_type
             ORDER BY p.created_at
             LIMIT %s
            """,
            (limit,),
        )
        return [
            OpenPrediction(
                id=row["id"],
                asset=row["asset"],
                direction=row["direction"],
                expected_low=float(row.get("expected_low") or 0.0),
                expected_high=float(row.get("expected_high") or 0.0),
                confidence=float(row.get("confidence") or 0.0),
                horizon_minutes=int(row.get("horizon_minutes") or 180),
                created_at=row["created_at"],
                baseline_price=row.get("baseline_price"),
                model_ids=list(row.get("model_ids") or []),
                event_type=row.get("event_type") or "ALL",
                deepest_stage=int(row.get("deepest_stage") or 4),
                observations={
                    int(offset): float(pct)
                    for offset, pct in (row.get("observations") or {}).items()
                },
            )
            for row in rows
        ]

    async def set_baseline_price(self, prediction_id: int, price: float) -> None:
        await self._execute(
            "UPDATE predictions SET baseline_price = %s WHERE id = %s AND baseline_price IS NULL",
            (price, prediction_id),
        )

    async def save_observation(self, observation: Observation) -> int:
        row = await self._fetchone(
            """
            INSERT INTO observations
                (prediction_id, asset, offset_minutes, price, pct_change, observed_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (prediction_id, offset_minutes) DO UPDATE
                SET price = EXCLUDED.price, pct_change = EXCLUDED.pct_change
            RETURNING id
            """,
            (
                observation.prediction_id,
                observation.asset,
                observation.offset_minutes,
                observation.price,
                observation.pct_change,
                observation.observed_at,
            ),
        )
        return int(row["id"])

    async def save_outcome(self, outcome: Outcome) -> int:
        row = await self._fetchone(
            """
            INSERT INTO outcomes (
                prediction_id, direction_correct, actual_pct, expected_pct,
                magnitude_error, score, best_horizon_minutes, detail, resolved_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (prediction_id) DO UPDATE
                SET direction_correct = EXCLUDED.direction_correct,
                    actual_pct = EXCLUDED.actual_pct,
                    magnitude_error = EXCLUDED.magnitude_error,
                    score = EXCLUDED.score,
                    best_horizon_minutes = EXCLUDED.best_horizon_minutes,
                    detail = EXCLUDED.detail,
                    resolved_at = EXCLUDED.resolved_at
            RETURNING id
            """,
            (
                outcome.prediction_id,
                outcome.direction_correct,
                outcome.actual_pct,
                outcome.expected_pct,
                outcome.magnitude_error,
                outcome.score,
                outcome.best_horizon_minutes,
                Jsonb(outcome.detail),
                outcome.resolved_at,
            ),
        )
        await self._execute(
            "UPDATE predictions SET resolved = TRUE, resolved_at = %s WHERE id = %s",
            (outcome.resolved_at, outcome.prediction_id),
        )
        return int(row["id"])

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
        # The averages are maintained incrementally in SQL so this stays one
        # round trip and never re-reads the history it is summarising.
        await self._execute(
            """
            INSERT INTO model_performance (
                model_id, stage, event_type, asset, predictions, direction_correct,
                avg_magnitude_error, avg_score, avg_latency_ms, failures, updated_at
            ) VALUES (%s, %s, %s, %s, 1, %s, %s, %s, %s, %s, now())
            ON CONFLICT (model_id, stage, event_type, asset) DO UPDATE
                SET avg_magnitude_error =
                        (model_performance.avg_magnitude_error * model_performance.predictions
                         + EXCLUDED.avg_magnitude_error) / (model_performance.predictions + 1),
                    avg_score =
                        (model_performance.avg_score * model_performance.predictions
                         + EXCLUDED.avg_score) / (model_performance.predictions + 1),
                    avg_latency_ms =
                        CASE WHEN EXCLUDED.avg_latency_ms = 0 THEN model_performance.avg_latency_ms
                             ELSE ((model_performance.avg_latency_ms * model_performance.predictions
                                    + EXCLUDED.avg_latency_ms) / (model_performance.predictions + 1))::int
                        END,
                    predictions = model_performance.predictions + 1,
                    direction_correct = model_performance.direction_correct + EXCLUDED.direction_correct,
                    failures = model_performance.failures + EXCLUDED.failures,
                    updated_at = now()
            """,
            (
                model_id,
                stage,
                event_type or "ALL",
                asset or "ALL",
                1 if direction_correct else 0,
                magnitude_error,
                score,
                latency_ms,
                1 if failed else 0,
            ),
        )

    async def model_performance(
        self,
        *,
        stage: int | None = None,
        event_type: str | None = None,
        asset: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        return await self._fetchall(
            """
            SELECT * FROM model_performance
             WHERE (%s::smallint IS NULL OR stage = %s::smallint)
               AND (%s::text IS NULL OR event_type = %s::text)
               AND (%s::text IS NULL OR asset = %s::text)
             ORDER BY predictions DESC, avg_score DESC
             LIMIT %s
            """,
            (stage, stage, event_type, event_type, asset, asset, limit),
        )

    # -- operations --------------------------------------------------------- #

    async def save_key_health(self, rows: Sequence[dict]) -> None:
        if not rows:
            return
        params = [
            (
                row.get("provider", ""),
                row.get("key_index", 0),
                row.get("fingerprint", ""),
                row.get("status", "HEALTHY"),
                row.get("consecutive_failures", 0),
                row.get("total_requests", 0),
                row.get("total_failures", 0),
                row.get("cooldown_until"),
                row.get("last_error"),
                row.get("last_success_at"),
                row.get("last_failure_at"),
            )
            for row in rows
        ]
        async with self.pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.executemany(
                    """
                    INSERT INTO api_key_health (
                        provider, key_index, fingerprint, status,
                        consecutive_failures, total_requests, total_failures,
                        cooldown_until, last_error, last_success_at, last_failure_at,
                        updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (provider, key_index) DO UPDATE
                        SET fingerprint = EXCLUDED.fingerprint,
                            status = EXCLUDED.status,
                            consecutive_failures = EXCLUDED.consecutive_failures,
                            total_requests = EXCLUDED.total_requests,
                            total_failures = EXCLUDED.total_failures,
                            cooldown_until = EXCLUDED.cooldown_until,
                            last_error = EXCLUDED.last_error,
                            last_success_at = EXCLUDED.last_success_at,
                            last_failure_at = EXCLUDED.last_failure_at,
                            updated_at = now()
                    """,
                    params,
                )

    async def load_key_health(self) -> list[dict]:
        return await self._fetchall(
            "SELECT * FROM api_key_health ORDER BY provider, key_index"
        )

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
        row = await self._fetchone(
            """
            INSERT INTO telegram_alerts
                (news_id, event_id, analysis_id, asset, chat_id, message_id,
                 kind, text, sent, error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (news_id, event_id, analysis_id, asset, chat_id, message_id, kind, text, sent, error),
        )
        return int(row["id"])

    async def already_alerted(self, event_id: int, asset: str) -> bool:
        value = await self._scalar(
            """
            SELECT 1 FROM telegram_alerts
             WHERE event_id = %s AND asset = %s AND sent
             LIMIT 1
            """,
            (event_id, asset),
        )
        return bool(value)

    async def save_run(self, stats: dict) -> int:
        row = await self._fetchone(
            """
            INSERT INTO pipeline_runs (
                started_at, finished_at, ingested,
                stage0_new, stage0_updates, stage0_duplicates,
                stage1_kept, stage2_kept, stage3_assets, stage4_impacts,
                stage5_runs, final_runs, alerts_sent, llm_calls, llm_failures, detail
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                stats.get("started_at") or utcnow(),
                stats.get("finished_at"),
                stats.get("ingested", 0),
                stats.get("stage0_new", 0),
                stats.get("stage0_updates", 0),
                stats.get("stage0_duplicates", 0),
                stats.get("stage1_kept", 0),
                stats.get("stage2_kept", 0),
                stats.get("stage3_assets", 0),
                stats.get("stage4_impacts", 0),
                stats.get("stage5_runs", 0),
                stats.get("final_runs", 0),
                stats.get("alerts_sent", 0),
                stats.get("llm_calls", 0),
                stats.get("llm_failures", 0),
                Jsonb(stats.get("detail") or {}),
            ),
        )
        return int(row["id"])

    async def recent_analyses(self, limit: int = 10) -> list[dict]:
        return await self._fetchall(
            """
            SELECT a.*, e.headline, e.event_type, n.url, n.source
              FROM analyses a
              LEFT JOIN events e ON e.id = a.event_id
              LEFT JOIN news n ON n.id = a.news_id
             ORDER BY a.created_at DESC
             LIMIT %s
            """,
            (limit,),
        )

    async def counts(self) -> dict[str, int]:
        row = await self._fetchone(
            """
            SELECT (SELECT count(*) FROM news)            AS news,
                   (SELECT count(*) FROM events)          AS events,
                   (SELECT count(*) FROM event_updates)   AS event_updates,
                   (SELECT count(*) FROM analyses)        AS analyses,
                   (SELECT count(*) FROM predictions)     AS predictions,
                   (SELECT count(*) FROM observations)    AS observations,
                   (SELECT count(*) FROM outcomes)        AS outcomes,
                   (SELECT count(*) FROM telegram_alerts) AS alerts
            """
        )
        return {key: int(value or 0) for key, value in (row or {}).items()}

    # -- maintenance -------------------------------------------------------- #

    async def purge_older_than(self, days: int | None = None) -> int:
        """Delete news outside the similarity window.

        Events, analyses and the feedback tables are kept: they are the history
        the routing layer learns from. Only the bulky article text and embeddings
        age out.
        """
        days = days or global_settings.stage0.lookback_days
        value = await self._scalar(
            """
            WITH deleted AS (
                DELETE FROM news
                 WHERE fetched_at < now() - make_interval(days => %s)
                   AND processed_at IS NOT NULL
                RETURNING 1
            )
            SELECT count(*) FROM deleted
            """,
            (days,),
        )
        return int(value or 0)

    async def raw_connection(self) -> AsyncConnection:
        """Escape hatch for ad-hoc queries in the CLI."""
        return await AsyncConnection.connect(self.dsn, row_factory=dict_row)


__all__ = ["PostgresStore"]
