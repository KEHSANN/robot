"""Text normalisation helpers used by Stage 0.

This module is intentionally dependency-free so it can be imported during a
cold boot and tested without Model/embedding providers being configured.
"""

from __future__ import annotations

import re
import unicodedata

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_html: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Return a stable, normalised version of ``text``.

    Normalisation order matters for deterministic deduplication: Unicode
    canonicalisation first, then optional HTML stripping, lowercasing and
    whitespace collapsing.
    """
    if text is None:
        return ""

    value = unicodedata.normalize("NFKC", str(text)).strip()
    if strip_html:
        value = _HTML_TAG_RE.sub(" ", value)
    if lowercase:
        value = value.lower()
    if collapse_whitespace:
        value = _WHITESPACE_RE.sub(" ", value).strip()
    return value


def normalize_list(items: list[str] | tuple[str, ...]) -> list[str]:
    """Normalise a list of strings, dropping empty entries."""
    result: list[str] = []
    for item in items or []:
        value = normalize_text(item)
        if value:
            result.append(value)
    return result
