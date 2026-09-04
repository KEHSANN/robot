"""Text normalisation for Stage 0.

The goal is narrow: make two renderings of *the same text* compare equal, while
changing nothing that carries meaning. Outlets syndicate the same wire copy with
different quote characters, different whitespace, different tracking parameters
on the links, and occasionally different Unicode forms of the same letter. All of
that is noise.

What is emphatically **not** noise: ``$ % + - ? ! .`` — a headline about a
``$742M`` inflow and one about a ``742%`` gain are different events, and "SEC
approves ETF" versus "SEC approves ETF?" is the difference between a fact and a
rumour. Those characters survive normalisation untouched.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

#: Characters that carry financial or grammatical meaning and must survive.
PRESERVED = "$%+-?!."

# Typographic variants that mean the same thing as their ASCII counterpart.
_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "«": '"', "»": '"', "‹": "'", "›": "'",
    "′": "'", "″": '"',
    # Dashes fold to ASCII hyphen — which is a PRESERVED character, so the
    # information survives; only the glyph is unified.
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", "　": " ",
    "•": " ", "·": " ", "⁃": " ",
    "＄": "$", "％": "%", "＋": "+",
    "！": "!", "？": "?", "．": ".",
}
_PUNCT_RE = re.compile("|".join(map(re.escape, _PUNCT_MAP)))

#: Zero-width and directional marks: invisible, and they break equality.
_INVISIBLE_RE = re.compile(r"[​-‏‪-‮⁠-⁤﻿­]")

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANKLINE_RE = re.compile(r"\n{3,}")

_HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&apos;": "'",
    "&nbsp;": " ", "&#39;": "'", "&#039;": "'", "&#x27;": "'", "&rsquo;": "'",
    "&lsquo;": "'", "&ldquo;": '"', "&rdquo;": '"', "&mdash;": "-",
    "&ndash;": "-", "&hellip;": "...", "&bull;": " ", "&#8217;": "'",
    "&#8216;": "'", "&#8220;": '"', "&#8221;": '"', "&#8212;": "-",
}

#: Query parameters that identify the *referrer*, not the article.
TRACKING_PARAMS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "utm_id", "utm_name", "utm_reader", "utm_brand", "utm_social",
        "utm_social-type", "utm_place", "utm_pubreferrer", "utm_swu",
        "gclid", "dclid", "gbraid", "wbraid", "fbclid", "msclkid", "twclid",
        "igshid", "mc_cid", "mc_eid", "yclid", "_hsenc", "_hsmi", "vero_id",
        "vero_conv", "ref", "ref_src", "ref_url", "source", "referrer",
        "spm", "scid", "cmpid", "campaign_id", "amp", "at_medium",
        "at_campaign", "s_kwcid", "ito", "ncid", "cid", "sr_share",
        "__twitter_impression", "guccounter", "guce_referrer",
        "guce_referrer_sig", "feature", "share",
    }
)

# Boilerplate that syndication adds around otherwise identical copy.
_BOILERPLATE_RE = re.compile(
    r"(?im)^\s*(?:"
    r"disclaimer\s*:|advertisement\s*$|sponsored(?:\s+content)?\s*$"
    r"|read\s+more\s*:|related\s+(?:reading|articles?|stories)\s*:"
    r"|(?:this\s+)?article\s+(?:was\s+)?originally\s+(?:published|appeared)"
    r"|follow\s+us\s+on\s+(?:twitter|x|telegram|google\s+news)"
    r"|subscribe\s+to\s+our\s+newsletter"
    r"|share\s+this\s+article"
    r"|the\s+views?\s+(?:and\s+opinions?\s+)?expressed\s+(?:here|in\s+this)"
    r").*$"
)


def strip_html(text: str) -> str:
    """Remove markup, keeping the readable text. RSS bodies are full of it."""
    if "<" not in text:
        return _unescape(text)
    text = _SCRIPT_RE.sub(" ", text)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>", "\n", text)
    text = _TAG_RE.sub(" ", text)
    return _unescape(text)


def _unescape(text: str) -> str:
    if "&" not in text:
        return text
    for entity, char in _HTML_ENTITIES.items():
        text = text.replace(entity, char)
    return re.sub(
        r"&#(\d{2,5});",
        lambda m: chr(int(m.group(1))) if int(m.group(1)) < 0x110000 else m.group(0),
        text,
    )


def unify_punctuation(text: str) -> str:
    """Fold typographic variants to ASCII and drop invisible characters."""
    text = _INVISIBLE_RE.sub("", text)
    return _PUNCT_RE.sub(lambda m: _PUNCT_MAP[m.group(0)], text)


def canonical_url(url: str) -> str:
    """Strip tracking so two links to the same article compare equal."""
    url = (url or "").strip()
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url

    scheme = (parts.scheme or "https").lower()
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host.endswith(":80") and scheme == "http":
        host = host[:-3]
    if host.endswith(":443") and scheme == "https":
        host = host[:-4]

    kept = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=False)
        if key.lower() not in TRACKING_PARAMS
    ]
    path = parts.path.rstrip("/") or "/"
    # `?amp` and `/amp` variants are the same article.
    if path.endswith("/amp"):
        path = path[:-4] or "/"

    return urlunsplit((scheme, host, path, urlencode(kept), ""))


def strip_urls(text: str) -> str:
    """Replace URLs with a placeholder.

    Two copies of the same article often differ only in their links, and a link's
    text contributes nothing to what the article *says*. Keeping a marker means a
    story with three links still differs from one with none. The underscores keep
    the token intact through the punctuation filter and out of collision range of
    ordinary prose.
    """
    return re.sub(r"https?://\S+|www\.[^\s<>\"]+", " __url__ ", text)


def normalize_text(text: str, *, drop_urls: bool = True) -> str:
    """Full normalisation used for the dedup hash and the embedding input.

    Lowercased, HTML-free, punctuation-unified, boilerplate-trimmed — but every
    character in :data:`PRESERVED` is left exactly as it was.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = strip_html(text)
    text = unify_punctuation(text)
    if drop_urls:
        text = strip_urls(text)
    text = _BOILERPLATE_RE.sub(" ", text)
    text = text.lower()

    # Drop residual punctuation that carries no meaning, keeping PRESERVED and
    # the characters needed to read numbers (digits, comma, colon in times).
    text = re.sub(r"[^\w\s" + re.escape(PRESERVED) + r",:/]+", " ", text)

    # A thousands separator is formatting: "$1,200" and "$1200" are one number.
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    # Anything else clinging to a word boundary is noise.
    text = re.sub(r"(?<!\d),|,(?!\d)", " ", text)

    text = _WS_RE.sub(" ", text)
    text = _BLANKLINE_RE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def normalize_for_hash(text: str) -> str:
    """Single-line normalisation: line breaks are formatting, not content."""
    return re.sub(r"\s+", " ", normalize_text(text)).strip()


def normalize_title(title: str) -> str:
    """Titles get the same treatment minus the boilerplate rules."""
    if not title:
        return ""
    text = unicodedata.normalize("NFKC", strip_html(title))
    text = unify_punctuation(text).lower()
    text = re.sub(r"[^\w\s" + re.escape(PRESERVED) + r"]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def embedding_text(title: str, body: str, limit: int = 4000) -> str:
    """What actually gets embedded.

    The title is repeated once at the front deliberately: it is the densest
    summary of the event, and weighting it slightly pulls same-event articles
    closer together than a raw body-only embedding does.
    """
    title = normalize_title(title)
    body = normalize_for_hash(body)
    combined = f"{title}. {body}".strip() if body else title
    return combined[:limit]


__all__ = [
    "PRESERVED",
    "TRACKING_PARAMS",
    "canonical_url",
    "embedding_text",
    "normalize_for_hash",
    "normalize_text",
    "normalize_title",
    "strip_html",
    "strip_urls",
    "unify_punctuation",
]
