"""Storage layer.

:func:`open_store` picks the implementation: Postgres when ``DATABASE_URL`` is
set, otherwise the in-memory store. The fallback is not a stub — the pipeline runs
end to end on it — but similarity search is a linear scan and only the most recent
embeddings persist, so production wants Postgres.
"""

from __future__ import annotations

import logging

from services.config import Settings, settings as global_settings
from database.base import (
    EventUpdate,
    OpenPrediction,
    SimilarNews,
    Store,
    StoredNews,
)
from database.memory import MemoryStore

log = logging.getLogger(__name__)


def build_store(config: Settings | None = None) -> Store:
    """Construct the right store without connecting to it."""
    config = config or global_settings
    if not config.use_postgres:
        log.info("DATABASE_URL not set — using the in-memory store")
        return MemoryStore()

    try:
        from database.postgres import PostgresStore
    except ImportError as exc:
        # psycopg missing is a setup problem, not a reason to lose the run.
        log.error("DATABASE_URL is set but psycopg is unavailable (%s); "
                  "falling back to the in-memory store", exc)
        return MemoryStore()

    return PostgresStore(config.database_url)


async def open_store(config: Settings | None = None) -> Store:
    """Construct and connect, falling back to memory if Postgres refuses.

    A database that is down should degrade the system, not stop it: alerts from an
    in-memory run are still useful, and the operator finds out from the log line
    rather than from silence.
    """
    store = build_store(config)
    try:
        await store.connect()
    except Exception as exc:
        if isinstance(store, MemoryStore):
            raise
        log.error("could not connect to Postgres (%s); falling back to memory", exc)
        store = MemoryStore()
        await store.connect()
    return store


__all__ = [
    "EventUpdate",
    "MemoryStore",
    "OpenPrediction",
    "SimilarNews",
    "Store",
    "StoredNews",
    "build_store",
    "open_store",
]
