"""Embedding persistence for Stage 0.

``EmbeddingStore`` is deliberately thin so it can sit in front of an in-memory
list now and a pgvector-backed table later. The database schema in
``database/stage0_schema.sql`` mirrors the shape of these records.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Sequence


@dataclass
class StoredEmbedding:
    """One stored embedding paired with the record it belongs to."""

    record_id: str
    vector: Sequence[float]
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["vector"] = list(self.vector)
        return data


class EmbeddingStore:
    """In-memory embedding store.

    Replace with a database-backed implementation when a ``DATABASE_URL`` is
    configured. The public interface is deliberately small on purpose.
    """

    def __init__(self) -> None:
        self._rows: list[StoredEmbedding] = []

    def upsert(self, record: StoredEmbedding) -> StoredEmbedding:
        self._rows = [
            existing
            for existing in self._rows
            if existing.record_id != record.record_id
        ]
        self._rows.append(record)
        return record

    def get(self, record_id: str) -> StoredEmbedding | None:
        for row in self._rows:
            if row.record_id == record_id:
                return row
        return None

    def all(self) -> list[StoredEmbedding]:
        return list(self._rows)

    def count(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows.clear()
