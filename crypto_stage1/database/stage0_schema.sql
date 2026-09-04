-- Stage 0 durable schema.
--
-- Keep this file in sync with database storage before any Stage 0 pipeline is
-- pointed at Postgres. pgvector is used for the embedding column when the
-- extension is available; the in-memory store falls back gracefully.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS stage0_events (
    event_id        TEXT PRIMARY KEY,
    label           TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'stage0',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS stage0_embeddings (
    record_id       TEXT PRIMARY KEY,
    event_id        TEXT NOT NULL REFERENCES stage0_events(event_id),
    canonical_id    TEXT,
    text            TEXT NOT NULL,
    vector          VECTOR,
    duplicate       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS stage0_facts (
    fact_id         BIGSERIAL PRIMARY KEY,
    record_id       TEXT NOT NULL,
    key             TEXT NOT NULL,
    value           JSONB NOT NULL DEFAULT '{}'::jsonb,
    source          TEXT NOT NULL DEFAULT 'rule',
    confidence      DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_stage0_embeddings_event_id
    ON stage0_embeddings (event_id);
