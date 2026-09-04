-- Crypto Event Intelligence System — Postgres schema.
--
-- `__EMBED_DIM__` is replaced by services.config.Stage0Settings.embed_dim when
-- `python run.py initdb` runs, so changing EMBED_DIM does not mean hand-editing
-- this file. If you change it after data exists, the vector column has to be
-- rebuilt: the old and new embeddings are not comparable.
--
-- Safe to re-run: every statement is IF NOT EXISTS.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;


-- ===========================================================================
-- Stage 0 — ingestion and event memory
-- ===========================================================================

CREATE TABLE IF NOT EXISTS news (
    id              BIGSERIAL PRIMARY KEY,
    title           TEXT        NOT NULL,
    body            TEXT        NOT NULL DEFAULT '',
    url             TEXT        NOT NULL DEFAULT '',
    canonical_url   TEXT        NOT NULL DEFAULT '',
    source          TEXT        NOT NULL DEFAULT '',
    source_type     TEXT        NOT NULL DEFAULT 'rss',
    published_at    TIMESTAMPTZ,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Stage 0 fingerprints. exact_hash catches a literal re-fetch; norm_hash
    -- catches syndication of the same wire copy.
    exact_hash      CHAR(64)    NOT NULL,
    norm_hash       CHAR(64)    NOT NULL,
    title_hash      CHAR(64)    NOT NULL DEFAULT '',
    url_hash        CHAR(64)    NOT NULL DEFAULT '',
    normalized      TEXT        NOT NULL DEFAULT '',
    embedding       vector(__EMBED_DIM__),

    facts           JSONB,
    identity_key    CHAR(32),
    event_id        BIGINT,
    decision        TEXT,                       -- NEW | UPDATE | DUPLICATE
    decision_reason TEXT        NOT NULL DEFAULT '',
    similarity      REAL,
    matched_news_id BIGINT,
    -- True when the decision needed no embedding and no model call.
    cheap_path      BOOLEAN     NOT NULL DEFAULT FALSE,

    -- Pipeline progress, so a crashed run can be resumed without re-paying.
    stage           SMALLINT    NOT NULL DEFAULT 0,
    dropped_at      SMALLINT,                   -- stage that rejected it
    drop_reason     TEXT,
    processed_at    TIMESTAMPTZ,

    CONSTRAINT news_exact_hash_uniq UNIQUE (exact_hash)
);

CREATE INDEX IF NOT EXISTS news_norm_hash_idx    ON news (norm_hash);
CREATE INDEX IF NOT EXISTS news_url_hash_idx     ON news (url_hash) WHERE url_hash <> '';
CREATE INDEX IF NOT EXISTS news_fetched_at_idx   ON news (fetched_at DESC);
CREATE INDEX IF NOT EXISTS news_event_id_idx     ON news (event_id);
CREATE INDEX IF NOT EXISTS news_identity_key_idx ON news (identity_key);
CREATE INDEX IF NOT EXISTS news_pending_idx      ON news (id) WHERE processed_at IS NULL;

-- Similarity search is always windowed to the last STAGE0_LOOKBACK_DAYS, so the
-- ANN index only needs to serve recent rows well. Lists are sized for the
-- ~7k-row working set implied by 1000 articles/day over a 7-day window.
CREATE INDEX IF NOT EXISTS news_embedding_idx
    ON news USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);


CREATE TABLE IF NOT EXISTS events (
    id             BIGSERIAL PRIMARY KEY,
    identity_key   CHAR(32)    NOT NULL UNIQUE,
    event_type     TEXT        NOT NULL DEFAULT 'OTHER',
    entity         TEXT        NOT NULL DEFAULT '',
    primary_asset  TEXT        NOT NULL DEFAULT '',
    action         TEXT        NOT NULL DEFAULT '',
    target         TEXT        NOT NULL DEFAULT '',
    headline       TEXT        NOT NULL DEFAULT '',
    status         TEXT        NOT NULL DEFAULT '',

    -- Current merged state: status, decision, amount, price, percentage, count,
    -- location, time_reference, event_date, key_claims.
    state          JSONB       NOT NULL DEFAULT '{}'::jsonb,

    first_seen     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen      TIMESTAMPTZ NOT NULL DEFAULT now(),
    article_count  INTEGER     NOT NULL DEFAULT 1,
    update_count   INTEGER     NOT NULL DEFAULT 0,
    importance     REAL        NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS events_last_seen_idx  ON events (last_seen DESC);
CREATE INDEX IF NOT EXISTS events_type_idx       ON events (event_type);
CREATE INDEX IF NOT EXISTS events_asset_idx      ON events (primary_asset);
CREATE INDEX IF NOT EXISTS events_entity_trgm_idx ON events USING gin (entity gin_trgm_ops);


-- One row per UPDATE decision: the audit trail of how a story developed.
CREATE TABLE IF NOT EXISTS event_updates (
    id             BIGSERIAL PRIMARY KEY,
    event_id       BIGINT      NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    news_id        BIGINT      REFERENCES news (id) ON DELETE SET NULL,
    changed_fields JSONB       NOT NULL DEFAULT '{}'::jsonb,
    previous_state JSONB       NOT NULL DEFAULT '{}'::jsonb,
    new_state      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    summary        TEXT        NOT NULL DEFAULT '',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS event_updates_event_idx ON event_updates (event_id, created_at DESC);


-- ===========================================================================
-- Stages 1-5 and the final layer
-- ===========================================================================

-- One row per model per stage. This is what makes per-model accuracy
-- measurable later, so it stores the raw answer, not just the vote.
CREATE TABLE IF NOT EXISTS stage_results (
    id                BIGSERIAL PRIMARY KEY,
    news_id           BIGINT      NOT NULL REFERENCES news (id) ON DELETE CASCADE,
    event_id          BIGINT      REFERENCES events (id) ON DELETE SET NULL,
    stage             SMALLINT    NOT NULL,
    asset             TEXT        NOT NULL DEFAULT '',
    model_id          TEXT        NOT NULL,
    provider          TEXT        NOT NULL,
    key_index         SMALLINT,
    ok                BOOLEAN     NOT NULL,
    vote              TEXT,
    latency_ms        INTEGER     NOT NULL DEFAULT 0,
    attempts          SMALLINT    NOT NULL DEFAULT 1,
    error             TEXT,
    finish_reason     TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    raw               JSONB,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS stage_results_news_idx  ON stage_results (news_id, stage);
CREATE INDEX IF NOT EXISTS stage_results_model_idx ON stage_results (model_id, stage, created_at DESC);


CREATE TABLE IF NOT EXISTS stage_consensus (
    id           BIGSERIAL PRIMARY KEY,
    news_id      BIGINT      NOT NULL REFERENCES news (id) ON DELETE CASCADE,
    event_id     BIGINT      REFERENCES events (id) ON DELETE SET NULL,
    stage        SMALLINT    NOT NULL,
    asset        TEXT        NOT NULL DEFAULT '',
    verdict      TEXT        NOT NULL,          -- PASS | REJECT | INCONCLUSIVE
    votes_pass   SMALLINT    NOT NULL DEFAULT 0,
    votes_reject SMALLINT    NOT NULL DEFAULT 0,
    votes_failed SMALLINT    NOT NULL DEFAULT 0,
    agreement    REAL        NOT NULL DEFAULT 0,
    score        REAL,
    note         TEXT,
    detail       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT stage_consensus_uniq UNIQUE (news_id, stage, asset)
);

CREATE INDEX IF NOT EXISTS stage_consensus_news_idx ON stage_consensus (news_id, stage);


-- The published verdict for one asset: whatever the deepest stage that ran said.
CREATE TABLE IF NOT EXISTS analyses (
    id             BIGSERIAL PRIMARY KEY,
    news_id        BIGINT      NOT NULL REFERENCES news (id) ON DELETE CASCADE,
    event_id       BIGINT      REFERENCES events (id) ON DELETE SET NULL,
    asset          TEXT        NOT NULL,
    direction      TEXT        NOT NULL,
    magnitude      TEXT        NOT NULL,
    expected_low   REAL        NOT NULL DEFAULT 0,
    expected_high  REAL        NOT NULL DEFAULT 0,
    confidence     REAL        NOT NULL DEFAULT 0,
    horizon_minutes INTEGER    NOT NULL DEFAULT 180,
    causality      TEXT        NOT NULL DEFAULT 'SENTIMENT',
    relation       TEXT        NOT NULL DEFAULT 'DIRECT',
    mechanism      TEXT        NOT NULL DEFAULT '',
    risks          TEXT        NOT NULL DEFAULT '',
    agreement      REAL        NOT NULL DEFAULT 0,
    model_count    SMALLINT    NOT NULL DEFAULT 0,
    -- Deepest stage that contributed: 4, 5, or 6 for the NVIDIA final layer.
    deepest_stage  SMALLINT    NOT NULL DEFAULT 4,
    stage2_score   REAL,
    escalated      BOOLEAN     NOT NULL DEFAULT FALSE,
    final_reviewed BOOLEAN     NOT NULL DEFAULT FALSE,
    tradeable      BOOLEAN,
    detail         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT analyses_uniq UNIQUE (news_id, asset)
);

CREATE INDEX IF NOT EXISTS analyses_event_idx ON analyses (event_id);
CREATE INDEX IF NOT EXISTS analyses_asset_idx ON analyses (asset, created_at DESC);


-- ===========================================================================
-- Feedback loop: prediction -> observation -> outcome -> model performance
-- ===========================================================================

CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    analysis_id     BIGINT      REFERENCES analyses (id) ON DELETE CASCADE,
    news_id         BIGINT      REFERENCES news (id) ON DELETE SET NULL,
    event_id        BIGINT      REFERENCES events (id) ON DELETE SET NULL,
    asset           TEXT        NOT NULL,
    direction       TEXT        NOT NULL,
    expected_low    REAL        NOT NULL DEFAULT 0,
    expected_high   REAL        NOT NULL DEFAULT 0,
    confidence      REAL        NOT NULL DEFAULT 0,
    horizon_minutes INTEGER     NOT NULL DEFAULT 180,
    baseline_price  DOUBLE PRECISION,
    -- Which models produced this call, so credit and blame can be assigned.
    model_ids       TEXT[]      NOT NULL DEFAULT '{}',
    deepest_stage   SMALLINT    NOT NULL DEFAULT 4,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved        BOOLEAN     NOT NULL DEFAULT FALSE,
    resolved_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS predictions_open_idx  ON predictions (created_at) WHERE NOT resolved;
CREATE INDEX IF NOT EXISTS predictions_asset_idx ON predictions (asset, created_at DESC);


CREATE TABLE IF NOT EXISTS observations (
    id             BIGSERIAL PRIMARY KEY,
    prediction_id  BIGINT      NOT NULL REFERENCES predictions (id) ON DELETE CASCADE,
    asset          TEXT        NOT NULL,
    offset_minutes INTEGER     NOT NULL,       -- 15 | 60 | 180 | 360 | 1440
    price          DOUBLE PRECISION NOT NULL,
    pct_change     REAL        NOT NULL,
    observed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT observations_uniq UNIQUE (prediction_id, offset_minutes)
);


CREATE TABLE IF NOT EXISTS outcomes (
    id                  BIGSERIAL PRIMARY KEY,
    prediction_id       BIGINT      NOT NULL UNIQUE REFERENCES predictions (id) ON DELETE CASCADE,
    direction_correct   BOOLEAN     NOT NULL,
    actual_pct          REAL        NOT NULL,
    expected_pct        REAL        NOT NULL,
    magnitude_error     REAL        NOT NULL,
    score               REAL        NOT NULL,
    best_horizon_minutes INTEGER    NOT NULL,
    detail              JSONB       NOT NULL DEFAULT '{}'::jsonb,
    resolved_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- Rolled up from outcomes, keyed the way the router wants to query it:
-- model x stage x event_type x asset.
CREATE TABLE IF NOT EXISTS model_performance (
    id                BIGSERIAL PRIMARY KEY,
    model_id          TEXT        NOT NULL,
    stage             SMALLINT    NOT NULL,
    event_type        TEXT        NOT NULL DEFAULT 'ALL',
    asset             TEXT        NOT NULL DEFAULT 'ALL',
    predictions       INTEGER     NOT NULL DEFAULT 0,
    direction_correct INTEGER     NOT NULL DEFAULT 0,
    avg_magnitude_error REAL      NOT NULL DEFAULT 0,
    avg_score         REAL        NOT NULL DEFAULT 0,
    avg_latency_ms    INTEGER     NOT NULL DEFAULT 0,
    failures          INTEGER     NOT NULL DEFAULT 0,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT model_performance_uniq UNIQUE (model_id, stage, event_type, asset)
);


-- ===========================================================================
-- Operations
-- ===========================================================================

-- Key health survives restarts, so a key that hit its daily quota is not
-- retried immediately on every process start.
CREATE TABLE IF NOT EXISTS api_key_health (
    id                BIGSERIAL PRIMARY KEY,
    provider          TEXT        NOT NULL,
    key_index         SMALLINT    NOT NULL,
    fingerprint       TEXT        NOT NULL,     -- masked, never the secret
    status            TEXT        NOT NULL DEFAULT 'HEALTHY',
    consecutive_failures SMALLINT NOT NULL DEFAULT 0,
    total_requests    BIGINT      NOT NULL DEFAULT 0,
    total_failures    BIGINT      NOT NULL DEFAULT 0,
    cooldown_until    TIMESTAMPTZ,
    last_error        TEXT,
    last_success_at   TIMESTAMPTZ,
    last_failure_at   TIMESTAMPTZ,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT api_key_health_uniq UNIQUE (provider, key_index)
);


CREATE TABLE IF NOT EXISTS telegram_alerts (
    id          BIGSERIAL PRIMARY KEY,
    news_id     BIGINT      REFERENCES news (id) ON DELETE SET NULL,
    event_id    BIGINT      REFERENCES events (id) ON DELETE SET NULL,
    analysis_id BIGINT      REFERENCES analyses (id) ON DELETE SET NULL,
    asset       TEXT        NOT NULL DEFAULT '',
    chat_id     TEXT        NOT NULL,
    message_id  BIGINT,
    kind        TEXT        NOT NULL DEFAULT 'alert',  -- alert | update | digest
    text        TEXT        NOT NULL DEFAULT '',
    sent        BOOLEAN     NOT NULL DEFAULT FALSE,
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS telegram_alerts_event_idx ON telegram_alerts (event_id, created_at DESC);
-- Answers "have we already alerted on this event/asset pair?" in one index hit.
CREATE INDEX IF NOT EXISTS telegram_alerts_sent_idx
    ON telegram_alerts (event_id, asset) WHERE sent;


-- Cheap funnel accounting: how many items each stage saw and dropped.
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              BIGSERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    ingested        INTEGER     NOT NULL DEFAULT 0,
    stage0_new      INTEGER     NOT NULL DEFAULT 0,
    stage0_updates  INTEGER     NOT NULL DEFAULT 0,
    stage0_duplicates INTEGER   NOT NULL DEFAULT 0,
    stage1_kept     INTEGER     NOT NULL DEFAULT 0,
    stage2_kept     INTEGER     NOT NULL DEFAULT 0,
    stage3_assets   INTEGER     NOT NULL DEFAULT 0,
    stage4_impacts  INTEGER     NOT NULL DEFAULT 0,
    stage5_runs     INTEGER     NOT NULL DEFAULT 0,
    final_runs      INTEGER     NOT NULL DEFAULT 0,
    alerts_sent     INTEGER     NOT NULL DEFAULT 0,
    llm_calls       INTEGER     NOT NULL DEFAULT 0,
    llm_failures    INTEGER     NOT NULL DEFAULT 0,
    detail          JSONB       NOT NULL DEFAULT '{}'::jsonb
);
