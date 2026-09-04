"""Content hashes — the cheapest possible dedup.

Two hashes per article, computed before any money is spent:

* ``exact_hash`` over the raw text catches a literal re-fetch of the same URL.
* ``norm_hash`` over the normalised text catches syndication: the same wire copy
  republished with different quote glyphs, tracking links and whitespace.

A ``norm_hash`` hit means DUPLICATE with no embedding call and no LLM call. At the
spec's ~1000 articles a day, syndication is a large share of the feed, so this is
where most of the savings live.

There is also a *title* hash, used only as a weak signal: the same title with a
different body is usually a live-updated story, which is an UPDATE candidate
rather than a duplicate.
"""

from __future__ import annotations

import hashlib

from stage0.normalize import canonical_url, normalize_for_hash, normalize_title


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_hash(title: str, body: str) -> str:
    """Hash of the raw text, whitespace-collapsed only."""
    combined = f"{(title or '').strip()}\n{(body or '').strip()}"
    return _sha256(" ".join(combined.split()))


def norm_hash(title: str, body: str) -> str:
    """Hash of the fully normalised text."""
    title_part = normalize_title(title)
    body_part = normalize_for_hash(body)
    return _sha256(f"{title_part}\n{body_part}")


def title_hash(title: str) -> str:
    return _sha256(normalize_title(title))


def url_hash(url: str) -> str:
    """Hash of the tracking-free URL, so one article fetched from two feeds with
    different campaign parameters resolves to one row."""
    return _sha256(canonical_url(url))


def fingerprints(title: str, body: str, url: str = "") -> dict[str, str]:
    """All hashes for one article, in one pass."""
    return {
        "exact_hash": exact_hash(title, body),
        "norm_hash": norm_hash(title, body),
        "title_hash": title_hash(title),
        "url_hash": url_hash(url) if url else "",
    }


__all__ = [
    "exact_hash",
    "fingerprints",
    "norm_hash",
    "title_hash",
    "url_hash",
]
