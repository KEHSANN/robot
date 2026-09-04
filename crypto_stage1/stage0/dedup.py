"""Content-hash based duplicate detection for Stage 0.

The index keeps a mapping from a stable content hash to the original record id.
It is intentionally simple and in-memory; a durable version should live behind
``embedding_store`` / the database schema.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def content_hash(text: str) -> str:
    """Return a deterministic sha256 hex digest of ``text``."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


@dataclass
class DedupIndex:
    """In-memory deduplication index.

    ``seen`` maps content hashes to ids. ``aliases`` maps later ids to the
    canonical id that already existed when the duplicate first arrived.
    """

    seen: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)

    def check(self, text: str) -> tuple[bool, str | None]:
        """Return ``(is_duplicate, canonical_id)`` for ``text``."""
        digest = content_hash(text)
        canonical = self.seen.get(digest)
        if canonical is None:
            return False, None
        return True, canonical

    def add(self, record_id: str, text: str) -> tuple[bool, str | None]:
        """Register ``text`` and return ``(is_duplicate, canonical_id)``.

        The first id wins; later identical texts become aliases of it.
        """
        is_duplicate, canonical = self.check(text)
        if is_duplicate:
            self.aliases[record_id] = canonical  # type: ignore[assignment]
            return True, canonical

        digest = content_hash(text)
        self.seen[digest] = record_id
        return False, None

    def canonical_for(self, record_id: str) -> str:
        """Return the canonical id for ``record_id`` (itself if not aliased)."""
        seen = set()
        current = record_id
        while current in self.aliases and current not in seen:
            seen.add(current)
            current = self.aliases[current]
        return current
