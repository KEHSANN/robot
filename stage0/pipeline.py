"""Stage 0 — event detection, memory and deduplication.

The funnel's first and cheapest gate. Roughly 1000 articles a day arrive; about
300 distinct events are in them. Everything downstream is priced per event, so
getting this stage right is what makes the rest affordable.

The order of checks is the whole design — each one is cheaper than the next, and
each one that hits saves everything after it:

    1. exact hash        free        literal re-fetch
    2. canonical URL     free        same article, different feed
    3. normalised hash   free        syndicated wire copy
    4. embedding         1 call      nominates candidate events
    5. fact extraction   1 call      identity + state
    6. identity compare  free        NEW vs UPDATE vs DUPLICATE

Steps 1-3 resolve a large share of a real feed without spending anything. Note
that fact extraction runs *after* the similarity search, not before: there is no
point extracting facts from an article we can already prove we have seen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta

from services.config import Settings, settings as global_settings
from services.llm import LLMClient
from services.types import (
    EventRecord,
    FactSet,
    NewsItem,
    Stage0Decision,
    Stage0Outcome,
    utcnow,
)
from database.base import SimilarNews, Store
from stage0.embedding import Embedder
from stage0.facts import FactExtractor
from stage0.hashing import fingerprints
from stage0.identity import compare_state, merge_state, same_event
from stage0.normalize import normalize_for_hash

log = logging.getLogger(__name__)


@dataclass
class Stage0Stats:
    seen: int = 0
    new: int = 0
    updates: int = 0
    duplicates: int = 0
    hash_hits: int = 0
    url_hits: int = 0
    embed_calls: int = 0
    extract_calls: int = 0
    #: Decisions reached without any paid call at all.
    free_decisions: int = 0

    def as_dict(self) -> dict:
        return {
            "seen": self.seen,
            "new": self.new,
            "updates": self.updates,
            "duplicates": self.duplicates,
            "hash_hits": self.hash_hits,
            "url_hits": self.url_hits,
            "embed_calls": self.embed_calls,
            "extract_calls": self.extract_calls,
            "free_decisions": self.free_decisions,
        }


@dataclass
class Stage0Result:
    """What Stage 0 hands to Stage 1."""

    news: NewsItem
    outcome: Stage0Outcome
    news_id: int | None = None
    event_id: int | None = None
    #: Prior developments of this event, formatted for the later prompts.
    event_context: str = ""
    source_count: int = 1
    candidates: list[SimilarNews] = field(default_factory=list)

    @property
    def advances(self) -> bool:
        return self.outcome.advances


class Stage0Pipeline:
    def __init__(
        self,
        client: LLMClient,
        store: Store,
        config: Settings | None = None,
    ) -> None:
        self.settings = config or global_settings
        self.cfg = self.settings.stage0
        self.store = store
        self.embedder = Embedder(client, self.cfg)
        self.extractor = FactExtractor(client)
        self.stats = Stage0Stats()

    async def process(self, news: NewsItem) -> Stage0Result:
        """Decide NEW / UPDATE / DUPLICATE for one article and persist it."""
        self.stats.seen += 1

        prints = fingerprints(news.title, news.body, news.url)
        news.exact_hash = prints["exact_hash"]
        news.norm_hash = prints["norm_hash"]
        news.normalized = normalize_for_hash(news.text)

        # --- 1-3: free checks --------------------------------------------- #
        free_hit = await self._free_duplicate_check(news, prints)
        if free_hit is not None:
            return free_hit

        # --- 4: embedding -------------------------------------------------- #
        embedding = await self.embedder.embed_item(news)
        if embedding:
            self.stats.embed_calls += 1

        candidates: list[SimilarNews] = []
        if embedding:
            since = utcnow() - timedelta(days=self.cfg.lookback_days)
            candidates = await self.store.find_similar(
                embedding,
                since=since,
                limit=self.cfg.candidate_limit,
                min_similarity=self.cfg.candidate_similarity,
            )

        # --- 5: facts ------------------------------------------------------ #
        facts, extract_result = await self.extractor.extract(news)
        if extract_result is not None:
            self.stats.extract_calls += 1
        identity_key = facts.identity_key()

        # --- 6: identity and state ---------------------------------------- #
        outcome = await self._decide(news, facts, identity_key, candidates)
        return await self._persist(news, prints, facts, identity_key, outcome, candidates)

    # -- step 1-3 ----------------------------------------------------------- #

    async def _free_duplicate_check(
        self, news: NewsItem, prints: dict[str, str]
    ) -> Stage0Result | None:
        """Hash and URL checks. Returns a finished result on a hit."""
        exact = await self.store.find_by_exact_hash(prints["exact_hash"])
        if exact:
            self.stats.hash_hits += 1
            return self._free_duplicate(
                news, prints, exact.event_id, exact.id,
                "identical text already stored",
            )

        if prints.get("url_hash"):
            by_url = await self.store.find_by_url_hash(prints["url_hash"])
            if by_url:
                self.stats.url_hits += 1
                return self._free_duplicate(
                    news, prints, by_url.event_id, by_url.id,
                    "same article URL already stored",
                )

        by_norm = await self.store.find_by_norm_hash(prints["norm_hash"])
        if by_norm:
            self.stats.hash_hits += 1
            return self._free_duplicate(
                news, prints, by_norm.event_id, by_norm.id,
                "syndicated copy of an article already stored",
            )
        return None

    def _free_duplicate(
        self,
        news: NewsItem,
        prints: dict[str, str],
        event_id: int | None,
        matched_news_id: int,
        reason: str,
    ) -> Stage0Result:
        """A duplicate proven without spending anything.

        The row is *not* inserted: it carries no information the stored copy does
        not already have, and writing it would only bloat the similarity index.
        The matched event still gets its article count bumped, because "five
        outlets ran this" is corroboration the deep stages weigh.
        """
        self.stats.duplicates += 1
        self.stats.free_decisions += 1
        news.decision = Stage0Decision.DUPLICATE

        outcome = Stage0Outcome(
            decision=Stage0Decision.DUPLICATE,
            facts=FactSet(headline=news.title[:200]),
            reason=reason,
            matched_news_id=matched_news_id,
            cheap_path=True,
        )
        log.debug("DUPLICATE (free): %s — %s", news.title[:70], reason)
        return Stage0Result(
            news=news, outcome=outcome, event_id=event_id, news_id=None
        )

    # -- step 6 ------------------------------------------------------------- #

    async def _decide(
        self,
        news: NewsItem,
        facts: FactSet,
        identity_key: str,
        candidates: list[SimilarNews],
    ) -> Stage0Outcome:
        """Compare against the stored event, if there is one."""
        event, similarity, matched_news_id = await self._match_event(
            facts, identity_key, candidates
        )

        if event is None:
            log.debug("NEW: %s", facts.headline or news.title[:70])
            return Stage0Outcome(
                decision=Stage0Decision.NEW,
                facts=facts,
                reason="no matching event in the lookback window",
                similarity=similarity,
            )

        diff = compare_state(event.state or {}, facts.state_fields())

        if diff.is_material:
            log.info(
                "UPDATE (event #%s): %s — %s",
                event.id, facts.headline or news.title[:60], diff.summary(),
            )
            return Stage0Outcome(
                decision=Stage0Decision.UPDATE,
                facts=facts,
                event=event,
                reason=f"event state changed: {diff.summary()}",
                similarity=similarity,
                matched_news_id=matched_news_id,
                changed_fields=diff.as_json()["changed"],
                previous_state=dict(event.state or {}),
            )

        log.debug(
            "DUPLICATE (event #%s): %s", event.id, facts.headline or news.title[:60]
        )
        return Stage0Outcome(
            decision=Stage0Decision.DUPLICATE,
            facts=facts,
            event=event,
            reason="same event, no state change" if not diff.has_changes
                   else f"same event, cosmetic change only ({diff.summary()})",
            similarity=similarity,
            matched_news_id=matched_news_id,
            previous_state=dict(event.state or {}),
        )

    async def _match_event(
        self,
        facts: FactSet,
        identity_key: str,
        candidates: list[SimilarNews],
    ) -> tuple[EventRecord | None, float | None, int | None]:
        """Find the stored event this article belongs to.

        Two routes. The identity hash is exact and does not need an embedding, so
        it works even when the embedding provider is down. The similarity
        candidates catch the case the hash misses — two outlets describing the
        same happening with slightly different wording of the entity or action.
        """
        exact = await self.store.get_event_by_identity(identity_key)
        if exact:
            return exact, 1.0, None

        best: tuple[EventRecord, float, int] | None = None
        for candidate in candidates:
            if not candidate.news.event_id:
                continue
            candidate_facts = candidate.news.fact_set()
            event = await self.store.get_event(candidate.news.event_id)
            if event is None:
                continue

            # Compare against the article's own extracted facts when we have
            # them, since the stored event's fields are a merged summary.
            reference = candidate_facts or event
            matches, score = same_event(facts, reference)

            # A near-verbatim reprint with matching identity is the same event
            # even if the identity score sits just under the threshold.
            if not matches and candidate.similarity >= self.cfg.reprint_similarity:
                matches = score >= 0.6

            if matches and (best is None or candidate.similarity > best[1]):
                best = (event, candidate.similarity, candidate.news.id)

        if best:
            return best[0], best[1], best[2]

        top = candidates[0].similarity if candidates else None
        return None, top, None

    # -- persistence -------------------------------------------------------- #

    async def _persist(
        self,
        news: NewsItem,
        prints: dict[str, str],
        facts: FactSet,
        identity_key: str,
        outcome: Stage0Outcome,
        candidates: list[SimilarNews],
    ) -> Stage0Result:
        decision = outcome.decision
        event = outcome.event
        event_id = event.id if event else None
        now = utcnow()

        if decision is Stage0Decision.NEW:
            self.stats.new += 1
            record = EventRecord.from_facts(facts)
            record.first_seen = now
            record.last_seen = now
            event_id = await self.store.create_event(record)
            outcome.event = record

        elif decision is Stage0Decision.UPDATE:
            self.stats.updates += 1
        else:
            self.stats.duplicates += 1

        news.decision = decision
        news.event_id = event_id
        news_id = await self.store.insert_news(
            news,
            fingerprints=prints,
            facts=facts,
            identity_key=identity_key,
            event_id=event_id,
            decision=decision.value,
            reason=outcome.reason,
            similarity=outcome.similarity,
            matched_news_id=outcome.matched_news_id,
            cheap_path=False,
        )

        if decision is not Stage0Decision.NEW and event_id and event is not None:
            merged = merge_state(event.state or {}, facts.state_fields())
            await self.store.touch_event(
                event_id,
                last_seen=now,
                state=merged,
                # A later article's headline is usually the better-informed one,
                # but only promote it when the story actually developed.
                headline=facts.headline if decision is Stage0Decision.UPDATE else None,
                status=str(merged.get("status") or "") or None,
                article_delta=1,
                update_delta=1 if decision is Stage0Decision.UPDATE else 0,
            )
            if decision is Stage0Decision.UPDATE:
                await self.store.record_event_update(
                    event_id,
                    news_id,
                    changed_fields=outcome.changed_fields,
                    previous_state=outcome.previous_state,
                    new_state=merged,
                    summary=outcome.reason,
                )

        source_count = 1
        context = ""
        if event_id:
            source_count = max(1, await self.store.event_article_count(event_id))
            context = await self.event_context(event_id)

        return Stage0Result(
            news=news,
            outcome=outcome,
            news_id=news_id,
            event_id=event_id,
            event_context=context,
            source_count=source_count,
            candidates=candidates,
        )

    async def event_context(self, event_id: int, limit: int = 4) -> str:
        """Prior developments of an event, rendered for the analysis prompts.

        This is what lets Stage 5 catch double counting: if the market already
        reacted to "SEC delays decision", the follow-up saying the same thing has
        very little move left in it.
        """
        updates = await self.store.event_updates(event_id, limit=limit)
        if not updates:
            return ""
        lines = []
        for update in reversed(updates):
            when = update.created_at.strftime("%Y-%m-%d %H:%M") if update.created_at else "?"
            lines.append(f"- {when} UTC: {update.summary}")
        return "\n".join(lines)


def _days(count: int):
    from datetime import timedelta

    return timedelta(days=count)


__all__ = ["Stage0Pipeline", "Stage0Result", "Stage0Stats"]
