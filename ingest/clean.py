"""Turning feed markup into the plain text the stages read.

Feed summaries are HTML, and the transformation has one hard constraint: Stage 0's
normalizer deliberately keeps ``$ % + - ? ! .`` because they carry the financial
meaning ("$1.2B", "+5%", "-30%"), and the models read the same text. So the
cleaner may remove markup and boilerplate but must never touch those characters —
a regex that strips punctuation would quietly turn a 5% move into a 5.

The other job is boilerplate. Every publisher appends its own promo line to the
RSS description ("The post … appeared first on …"), and left in place it becomes
part of the text that gets hashed, embedded and compared — which makes two
unrelated articles from one outlet look more similar than they are.
"""

from __future__ import annotations

import html
import re

#: Blocks whose *contents* are markup or code, not prose.
_DROP_BLOCKS = re.compile(
    r"<(script|style|noscript|iframe|figure|figcaption|form)\b.*?</\1\s*>",
    re.I | re.S,
)

#: Tags that imply a line break rather than a space.
_BLOCK_BREAK = re.compile(
    r"</?(p|div|br|li|tr|h[1-6]|blockquote|section|article|ul|ol|table)\b[^>]*>",
    re.I,
)

_ANY_TAG = re.compile(r"<[^>]+>")

#: Publisher promo lines. Anchored to the end because that is where feeds put
#: them, so a genuine mention inside the article body survives.
_BOILERPLATE = (
    re.compile(r"\bThe post\b.{0,120}?\bappeared first on\b.{0,80}$", re.I | re.S),
    re.compile(r"\bThis (?:post|article|story) (?:first )?appeared\b.{0,120}$", re.I | re.S),
    re.compile(r"\bRead more on\b.{0,80}$", re.I),
    re.compile(r"\bContinue reading\b.{0,80}$", re.I),
    re.compile(r"\bSource:?\s*(?:https?://\S+|\w[\w .]{0,40})$", re.I),
    re.compile(r"\b(?:Disclaimer|Disclosure)\s*:.{0,400}$", re.I | re.S),
    re.compile(r"\bClick here to\b.{0,80}$", re.I),
    re.compile(r"\bSubscribe to\b.{0,80}$", re.I),
    re.compile(r"\bFollow us on\b.{0,80}$", re.I),
    re.compile(r"\bPhoto(?:\s+by|:).{0,80}$", re.I),
    re.compile(r"\b(?:Image|Credit|Photograph)s?:?\s*\S{0,60}$", re.I),
)

#: Whole lines that are navigation or sharing furniture.
_JUNK_LINE = re.compile(
    r"^\s*(share (?:this|on)|tweet|advertisement|sponsored|related:|tags?:|"
    r"read also|see also|also read|newsletter|sign up)\b.*$",
    re.I,
)

_WS_RUNS = re.compile(r"[ \t   ]+")
_BLANK_RUNS = re.compile(r"\n{3,}")
_TRAILING_URL = re.compile(r"\s*https?://\S+\s*$")


def strip_html(raw: str) -> str:
    """Markup to text, preserving paragraph breaks and every symbol.

    Entities are unescaped twice: feeds routinely double-escape, so a headline
    about ``&amp;amp;`` arrives needing two passes, and one pass leaves visible
    ``&amp;`` in the text the models read.
    """
    if not raw:
        return ""

    text = _DROP_BLOCKS.sub(" ", raw)
    text = _BLOCK_BREAK.sub("\n", text)
    text = _ANY_TAG.sub(" ", text)

    text = html.unescape(text)
    if "&" in text:
        text = html.unescape(text)

    # Zero-width and bidi marks arrive from CMS copy-paste and would otherwise
    # split words for the hasher.
    text = text.translate({0x200b: None, 0x200c: None, 0x200d: None, 0xfeff: None})

    lines = []
    for line in text.split("\n"):
        line = _WS_RUNS.sub(" ", line).strip()
        if line and not _JUNK_LINE.match(line):
            lines.append(line)

    return _BLANK_RUNS.sub("\n\n", "\n".join(lines)).strip()


def strip_boilerplate(text: str) -> str:
    """Remove publisher furniture from the end of a summary."""
    cleaned = text.strip()
    for _ in range(3):  # feeds stack two or three of these
        before = cleaned
        for pattern in _BOILERPLATE:
            cleaned = pattern.sub("", cleaned).strip()
        if cleaned == before:
            break
    return _TRAILING_URL.sub("", cleaned).strip()


def clean_text(raw: str, limit: int = 6000) -> str:
    """Full cleaning pass for a feed summary or article body."""
    text = strip_boilerplate(strip_html(raw))
    if len(text) <= limit:
        return text
    # Cut on a sentence boundary when one is near the limit, so the tail is not a
    # half-sentence the models have to guess at.
    window = text[:limit]
    for mark in (". ", "! ", "? ", "\n"):
        cut = window.rfind(mark)
        if cut > limit * 0.6:
            return window[: cut + 1].strip()
    return window.rstrip() + "…"


def clean_title(raw: str) -> str:
    """Titles are single-line and carry the source suffix publishers append."""
    title = strip_html(raw).replace("\n", " ").strip()
    title = _WS_RUNS.sub(" ", title)
    # "Bitcoin ETF approved - CoinDesk" / "… | The Block"
    title = re.sub(r"\s+[|–—]\s+[\w .'&]{2,30}$", "", title).strip()
    return title


def looks_like_prose(text: str) -> bool:
    """Whether a cleaned body is worth keeping at all.

    A feed that returns only a paywall notice, an image credit or a "read more"
    stub should not have that text hashed and embedded as if it were the article.
    The bar is deliberately low: once the boilerplate stripper has run, junk is
    what is left over as a fragment, and a genuinely short summary ("Tether mints
    $1B of USDT.") still carries information the title may not.
    """
    if len(text) < 12:
        return False
    letters = sum(1 for char in text if char.isalpha())
    return letters >= len(text) * 0.5 and " " in text.strip()


__all__ = [
    "clean_text",
    "clean_title",
    "looks_like_prose",
    "strip_boilerplate",
    "strip_html",
]
