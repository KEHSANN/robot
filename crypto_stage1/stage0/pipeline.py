"""Stage 0 pipeline: normalise -> deduplicate -> embed -> cluster.

The pipeline does its best with what it is given:
- ``normalize_text`` always runs.
- ``content_hash`` / ``DedupIndex`` always runs.
- ``EmbeddingService.embed`` is optional; when no vectors are available the
  pipeline still clusters using ``text_similarity``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dedup import DedupIndex, content_hash
from .embedding_store import EmbeddingStore, StoredEmbedding
from .event_assignment import EventCluster, assign_to_nearest_event
from .fact_engine import Fact, extract_facts
from .normalizer import normalize_text
from .similarity import text_similarity


@dataclass
class Stage0Record:
    """Input record accepted by ``run_stage0``."""

    record_id: str
    title: str = ""
    body: str = ""
    source: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}".strip()


@dataclass
class Stage0Result:
    """Output produced by ``run_stage0``."""

    record_id: str
    normalised_title: str
    deduplicated: bool
    canonical_id: str
    event_id: str
    event_label: str
    facts: list[dict[str, Any]]
    vector: list[float]
    similarity_score: float


class Stage0Pipeline:
    """Small orchestrator around the Stage 0 helpers."""

    def __init__(
        self,
        *,
        embedding_service: Any | None = None,
        dedup_index: DedupIndex | None = None,
        store: EmbeddingStore | None = None,
        threshold: float = 0.55,
    ) -> None:
        if embedding_service is not None and not hasattr(embedding_service, "embed"):
            raise TypeError("embedding_service must expose .embed(text)")
        self.embedding_service = embedding_service
        self.dedup = dedup_index or DedupIndex()
        self.store = store or EmbeddingStore()
        self.threshold = threshold
        self.events: list[EventCluster] = []

    def _vector_for(self, text: str) -> list[float]:
        if self.embedding_service is None:
            return []
        try:
            return list(self.embedding_service.embed(text))
        except Exception:
            return []

    def run(self, record: Stage0Record) -> Stage0Result:
        normalised_title = normalize_text(record.title)
        normalised_body = normalize_text(record.body)
        text = f"{normalised_title}\n{normalised_body}".strip()
        if not text:
            text = normalize_text(record.raw.get("text", ""))

        is_duplicate, canonical_id = self.dedup.add(record.record_id, text)
        canonical = self.dedup.canonical_for(record.record_id)

        vector = self._vector_for(text)
        event_id, event = assign_to_nearest_event(
            record_id=record.record_id,
            text=text,
            vector=vector or None,
            events=self.events,
            threshold=self.threshold,
        )

        score = 0.0
        if event is not None:
            if vector and event.vector:
                from .similarity import cosine_similarity

                score = cosine_similarity(vector, event.vector)
            else:
                score = text_similarity(text, event.label)
            event.add_record(
                record.record_id,
                update_vector=bool(vector),
                new_vector=vector,
            )
            event_id = event.event_id
        else:
            label = text[:120] or text[:120]
            new_event = EventCluster(
                event_id=record.record_id,
                label=label,
                vector=vector,
                record_ids=[record.record_id],
            )
            self.events.append(new_event)
            event_id = new_event.event_id

        facts = [fact.as_dict() for fact in extract_facts(text)]
        self.store.upsert(
            StoredEmbedding(
                record_id=record.record_id,
                vector=vector,
                text=text,
                metadata={
                    "canonical_id": canonical,
                    "event_id": event_id,
                    "duplicate": is_duplicate,
                },
            )
        )

        return Stage0Result(
            record_id=record.record_id,
            normalised_title=normalised_title,
            deduplicated=is_duplicate,
            canonical_id=canonical or record.record_id,
            event_id=event_id,
            event_label=label if "label" in locals() else text[:120],
            facts=facts,
            vector=vector,
            similarity_score=score,
        )


def run_stage0(
    records,
    *,
    embedding_service: Any | None = None,
    threshold: float = 0.55,
) -> list[Stage0Result]:
    """Convenience function that runs Stage 0 over an iterable of records."""
    pipeline = Stage0Pipeline(
        embedding_service=embedding_service,
        threshold=threshold,
    )
    return [pipeline.run(record) if isinstance(record, Stage0Record) else pipeline.run(Stage0Record(**record)) for record in records]
