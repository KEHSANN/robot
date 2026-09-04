"""Collecting news from every configured source into one ordered batch.

Two decisions here shape what the rest of the pipeline sees.

**Chronological order.** Items are handed over oldest first. Stage 0 treats the
first article about something as the NEW event and later ones as UPDATEs, so
processing a batch newest-first would make the follow-up the canonical event and
the original story its own update — the event history would read backwards, and
Stage 5's "what happened before this" context would be wrong.

**Literal repeats are dropped, changed articles are not.** Feeds re-serve the same
items every pass, so a five-minute poll would re-ingest the same forty articles
288 times a day. But a URL whose *content* changed is a live-updated story, which
is exactly what the UPDATE path exists for. So the in-memory guard keys on URL
*and* content hash: same both, skip it; same URL with new text, let it through.

Sources are pluggable behind :class:`NewsSource`. RSS is the only one implemented;
the Telegram and X collectors in the plan fit the same two-method shape.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from ingest.rss import FeedResult, RSSSource
from services.config import Settings, settings as global_settings
from services.types import NewsItem
from stage0.hashing import exact_hash
from stage0.normalize import canonical_url, normalize_title

log = logging.getLogger(__name__)

#: How many (url, hash) pairs to remember between passes. At ~1000 items a day
#: this covers several days of polling in a few hundred kilobytes.
SEEN_CAPACITY = 8000

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@runtime_checkable
class NewsSource(Protocol):
    """Anything that can produce articles. RSS today; Telegram and X next."""

    source_type: str

    async def fetch_all(self) -> list[FeedResult]: ...


@dataclass
class IngestStats:
    sources: int = 0
    feeds_ok: int = 0
    feeds_failed: int = 0
    not_modified: int = 0
    #: Parsed out of the feeds before any filtering.
    fetched: int = 0
    #: Dropped by the feed parser as too old or too thin to be an article.
    skipped: int = 0
    #: Same article present in two feeds, or unchanged since the last pass.
    deduped: int = 0
    delivered: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "sources": self.sources,
            "feeds_ok": self.feeds_ok,
            "feeds_failed": self.feeds_failed,
            "not_modified": self.not_modified,
            "fetched": self.fetched,
            "skipped": self.skipped,
            "deduped": self.deduped,
            "delivered": self.delivered,
        }

    def summary(self) -> str:
        return (
            f"{self.delivered} new from {self.feeds_ok}/{self.feeds_ok + self.feeds_failed} "
            f"feeds ({self.fetched} fetched, {self.deduped} repeats, "
            f"{self.skipped} stale, {self.not_modified} unchanged)"
        )


class Ingestor:
    """Fans out over sources and returns a deduplicated, ordered batch."""

    def __init__(
        self,
        sources: list[NewsSource] | None = None,
        config: Settings | None = None,
    ) -> None:
        self.settings = config or global_settings
        self.sources: list[NewsSource] = list(sources) if sources else []
        self._owned: list[NewsSource] = []
        #: canonical url -> exact content hash of the last version we accepted
        self._seen: OrderedDict[str, str] = OrderedDict()
        self.stats = IngestStats()

    async def __aenter__(self) -> "Ingestor":
        if not self.sources:
            source = await RSSSource(
                self.settings.rss_feeds, self.settings.ingest
            ).__aenter__()
            self.sources.append(source)
            self._owned.append(source)
        return self

    async def __aexit__(self, *exc) -> None:
        for source in self._owned:
            await source.__aexit__(*exc)  # type: ignore[attr-defined]
        self._owned.clear()

    # ------------------------------------------------------------------ collect

    async def collect(self, limit: int | None = None) -> list[NewsItem]:
        """One pass over every source. Oldest first, capped at ``limit``."""
        self.stats = IngestStats(sources=len(self.sources))
        results: list[FeedResult] = []

        for source in self.sources:
            try:
                results.extend(await source.fetch_all())
            except Exception as exc:
                self.stats.errors.append(f"{type(source).__name__}: {exc}"[:200])
                log.exception("source %s failed entirely", type(source).__name__)

        collected: list[NewsItem] = []
        for result in results:
            if result.not_modified:
                self.stats.not_modified += 1
                continue
            if not result.ok:
                self.stats.feeds_failed += 1
                self.stats.errors.append(f"{result.host}: {result.error}")
                log.warning("feed %s failed: %s", result.host, result.error)
                continue

            self.stats.feeds_ok += 1
            self.stats.fetched += len(result.items)
            self.stats.skipped += result.skipped
            collected.extend(result.items)

        fresh = self._deduplicate(collected)
        fresh.sort(key=_ordering_key)

        cap = limit if limit is not None else self.settings.batch_size
        if cap and len(fresh) > cap:
            # Keeping the oldest respects event lineage; the remainder is still in
            # the feed next pass, because nothing was marked seen for it.
            log.info("holding back %d items over the batch cap", len(fresh) - cap)
            fresh = self._release_over_cap(fresh, cap)

        for item in fresh:
            self._remember(item)

        self.stats.delivered = len(fresh)
        log.info("ingest: %s", self.stats.summary())
        return fresh

    # -------------------------------------------------------------- dedup logic

    def _deduplicate(self, items: list[NewsItem]) -> list[NewsItem]:
        """Drop repeats within the batch and across passes."""
        kept: list[NewsItem] = []
        batch_urls: set[str] = set()
        batch_titles: set[str] = set()

        for item in items:
            digest = exact_hash(item.title, item.body)
            key = canonical_url(item.url) or f"title:{normalize_title(item.title)}"

            if key in batch_urls:
                self.stats.deduped += 1
                continue

            title_key = normalize_title(item.title)
            if title_key and title_key in batch_titles:
                # Two feeds carrying one wire story. Stage 0 would catch it for
                # free, but not paying for the second row is still better.
                self.stats.deduped += 1
                continue

            if self._seen.get(key) == digest:
                self.stats.deduped += 1
                continue

            batch_urls.add(key)
            if title_key:
                batch_titles.add(title_key)
            kept.append(item)

        return kept

    def _remember(self, item: NewsItem) -> None:
        key = canonical_url(item.url) or f"title:{normalize_title(item.title)}"
        self._seen[key] = exact_hash(item.title, item.body)
        self._seen.move_to_end(key)
        while len(self._seen) > SEEN_CAPACITY:
            self._seen.popitem(last=False)

    def _release_over_cap(self, items: list[NewsItem], cap: int) -> list[NewsItem]:
        """Take the oldest ``cap`` items and leave the rest for the next pass."""
        return items[:cap]

    # -------------------------------------------------------------------- state

    @property
    def seen_count(self) -> int:
        return len(self._seen)

    def forget(self) -> None:
        """Drop the cross-pass memory, so the next pass re-offers everything."""
        self._seen.clear()


def _ordering_key(item: NewsItem):
    """Oldest first; undated items sort as if they arrived now.

    An undated item cannot be placed in the timeline, and assuming it is old
    would let it claim to be the origin of an event it is actually following.
    """
    return item.published_at or item.fetched_at or _EPOCH


async def collect_once(config: Settings | None = None, limit: int | None = None) -> list[NewsItem]:
    """Convenience for one-shot use: build the default sources, collect, close."""
    async with Ingestor(config=config) as ingestor:
        return await ingestor.collect(limit)


__all__ = [
    "SEEN_CAPACITY",
    "IngestStats",
    "Ingestor",
    "NewsSource",
    "collect_once",
]
