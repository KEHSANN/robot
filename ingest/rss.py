"""RSS / Atom ingestion.

No feed library: seven URLs of well-formed-ish XML do not justify a dependency,
and the parsing that matters is the awkward part any library also has to do by
hand. What the stdlib parser needs help with, and this module supplies:

* **Three feed dialects.** RSS 2.0 (``channel/item``), Atom (``feed/entry``) and
  RSS 1.0/RDF (top-level ``item``) all appear across the default feed list.
  Namespaces are matched on the local tag name, because publishers get the
  namespace URI wrong often enough that exact matching drops whole feeds.

* **Malformed bytes.** A single control character illegal in XML — routine in
  CMS-pasted copy — makes ``ElementTree`` reject the entire document, so one
  publisher's editing habits would silently cost a whole source. They are
  stripped before parsing.

* **Two date formats and a broken clock.** RSS uses RFC 822, Atom ISO 8601, and
  a fair number of feeds stamp items in the future. A future timestamp that is
  taken at face value would keep an item permanently inside the freshness
  window.

The age cutoff is the other thing worth noticing. Feeds regularly carry weeks of
history, and this system forecasts 15-minute to 24-hour moves, so an article from
last month is not a cheap item to analyse — it is a wrong one. Items older than
``INGEST_MAX_AGE_HOURS`` never enter the pipeline, which is also what stops the
first run against an empty database from ingesting a month of archives.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit
from xml.etree import ElementTree

import httpx

from ingest.clean import clean_text, clean_title, looks_like_prose
from services.config import IngestSettings, settings as global_settings
from services.types import NewsItem, utcnow

log = logging.getLogger(__name__)

#: Bytes that are illegal in XML 1.0 and appear in real feeds anyway.
_ILLEGAL_XML = re.compile(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]")

#: Item-level tags, by local name, in preference order.
_TITLE_TAGS = ("title",)
_BODY_TAGS = ("encoded", "content", "description", "summary", "subtitle")
_DATE_TAGS = ("pubdate", "published", "updated", "date", "modified", "created")
_ID_TAGS = ("guid", "id")


def _localname(tag: str) -> str:
    """``{http://www.w3.org/2005/Atom}entry`` -> ``entry``."""
    return tag.rsplit("}", 1)[-1].lower() if "}" in tag else tag.lower()


def _text_of(element: ElementTree.Element) -> str:
    """All text under an element, including tails of nested tags.

    Feeds put markup inside ``<description>`` either escaped or as real child
    elements, and ``element.text`` alone silently returns the first fragment.
    """
    return "".join(element.itertext())


def parse_date(raw: str | None) -> datetime | None:
    """RFC 822 or ISO 8601 to an aware UTC datetime."""
    value = (raw or "").strip()
    if not value:
        return None

    parsed: datetime | None = None
    if "," in value[:5] or value[:3].isalpha():
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            parsed = None

    if parsed is None:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None

    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass
class FeedResult:
    """One feed's outcome for one pass."""

    url: str
    items: list[NewsItem] = field(default_factory=list)
    status: int | None = None
    error: str = ""
    not_modified: bool = False
    #: Items parsed but dropped as too old or too thin, for the ingest log line.
    skipped: int = 0

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def host(self) -> str:
        return urlsplit(self.url).netloc or self.url


class RSSSource:
    """Fetches and parses one list of feeds.

    Long-lived: it remembers each feed's ``ETag``/``Last-Modified`` so repeated
    passes in ``loop`` mode mostly get 304s. That is what keeps a five-minute
    poll from looking like abuse to a publisher.
    """

    source_type = "rss"

    def __init__(
        self,
        feeds: list[str] | None = None,
        config: IngestSettings | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config or global_settings.ingest
        self.feeds = list(feeds if feeds is not None else global_settings.rss_feeds)
        self._http = http
        self._owns_http = http is None
        self._validators: dict[str, dict[str, str]] = {}
        self._failures: dict[str, int] = {}

    async def __aenter__(self) -> "RSSSource":
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
                headers={
                    "user-agent": self.config.user_agent,
                    "accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
                    "accept-encoding": "gzip, deflate",
                },
                follow_redirects=True,
            )
        return self

    async def __aexit__(self, *_exc) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            raise RuntimeError("RSSSource must be used as an async context manager")
        return self._http

    # ---------------------------------------------------------------- fetching

    async def fetch_all(self) -> list[FeedResult]:
        """Fetch every feed concurrently. One bad feed cannot fail the pass."""
        semaphore = asyncio.Semaphore(max(1, self.config.concurrency))

        async def one(url: str) -> FeedResult:
            async with semaphore:
                return await self.fetch(url)

        gathered = await asyncio.gather(
            *(one(url) for url in self.feeds), return_exceptions=True
        )

        results: list[FeedResult] = []
        for url, outcome in zip(self.feeds, gathered):
            if isinstance(outcome, BaseException):
                # gather() should not get here — fetch() catches — but an
                # unexpected error must still not take down the whole pass.
                log.error("feed %s raised %s", url, outcome)
                results.append(FeedResult(url=url, error=repr(outcome)[:200]))
            else:
                results.append(outcome)
        return results

    async def fetch(self, url: str) -> FeedResult:
        headers: dict[str, str] = {}
        if self.config.conditional_requests:
            headers.update(self._validators.get(url, {}))

        try:
            response = await self.http.get(url, headers=headers)
        except Exception as exc:  # network, DNS, TLS, timeout
            self._failures[url] = self._failures.get(url, 0) + 1
            log.warning("feed %s unreachable (%d in a row): %s",
                        url, self._failures[url], type(exc).__name__)
            return FeedResult(url=url, error=f"{type(exc).__name__}: {exc}"[:200])

        if response.status_code == 304:
            return FeedResult(url=url, status=304, not_modified=True)

        if response.status_code >= 400:
            self._failures[url] = self._failures.get(url, 0) + 1
            return FeedResult(
                url=url, status=response.status_code,
                error=f"HTTP {response.status_code}",
            )

        self._failures.pop(url, None)
        if self.config.conditional_requests:
            self._remember_validators(url, response.headers)

        try:
            items, skipped = self.parse(response.content, url)
        except ElementTree.ParseError as exc:
            return FeedResult(url=url, status=response.status_code,
                              error=f"unparseable XML: {exc}"[:200])

        return FeedResult(url=url, items=items, status=response.status_code,
                          skipped=skipped)

    def _remember_validators(self, url: str, headers) -> None:
        validators: dict[str, str] = {}
        if etag := headers.get("etag"):
            validators["if-none-match"] = etag
        if modified := headers.get("last-modified"):
            validators["if-modified-since"] = modified
        if validators:
            self._validators[url] = validators

    # ----------------------------------------------------------------- parsing

    def parse(self, payload: bytes, feed_url: str) -> tuple[list[NewsItem], int]:
        """Feed bytes to :class:`NewsItem` objects, newest first."""
        root = ElementTree.fromstring(_sanitise(payload))
        source_name = _channel_title(root) or urlsplit(feed_url).netloc

        cutoff = utcnow() - timedelta(hours=max(1, self.config.max_age_hours))
        items: list[NewsItem] = []
        skipped = 0

        for element in _iter_entries(root):
            news = self._to_news(element, source_name, feed_url)
            if news is None:
                skipped += 1
                continue
            if news.published_at is not None and news.published_at < cutoff:
                skipped += 1
                continue
            items.append(news)
            if len(items) >= self.config.max_items_per_feed:
                break

        return items, skipped

    def _to_news(
        self, element: ElementTree.Element, source_name: str, feed_url: str
    ) -> NewsItem | None:
        fields: dict[str, list[tuple[str, ElementTree.Element]]] = {}
        for child in element:
            fields.setdefault(_localname(child.tag), []).append((child.tag, child))

        title = clean_title(_first_text(fields, _TITLE_TAGS))
        if not title:
            return None

        body = ""
        for tag in _BODY_TAGS:
            candidate = clean_text(_join_text(fields, tag), self.config.body_max_chars)
            # Feeds carry both a teaser and the full text; take the longer one.
            if len(candidate) > len(body):
                body = candidate
        if body and not looks_like_prose(body):
            body = ""

        url = _entry_link(fields, element)
        if not url:
            url = _first_text(fields, _ID_TAGS) if _looks_like_url(
                _first_text(fields, _ID_TAGS)
            ) else ""

        published = parse_date(_first_text(fields, _DATE_TAGS))
        if published is not None and published > utcnow() + timedelta(hours=1):
            # A feed stamping items in the future would keep them permanently
            # fresh; trust the fetch time instead.
            log.debug("feed %s dated an item in the future (%s)", feed_url, published)
            published = utcnow()

        return NewsItem(
            title=title,
            body=body,
            url=url,
            source=source_name[:120],
            source_type=self.source_type,
            published_at=published,
        )


# --------------------------------------------------------------------------- #
# element helpers
# --------------------------------------------------------------------------- #

def _sanitise(payload: bytes) -> bytes:
    """Make bytes parseable: drop a BOM, leading junk and illegal characters."""
    cleaned = _ILLEGAL_XML.sub(b"", payload)
    cleaned = cleaned.lstrip(b"\xef\xbb\xbf \t\r\n")
    # Some CDNs prepend a comment or stray text before the declaration.
    if not cleaned.startswith(b"<"):
        start = cleaned.find(b"<")
        if start > 0:
            cleaned = cleaned[start:]
    return cleaned


def _iter_entries(root: ElementTree.Element):
    """Yield item/entry elements for RSS 2.0, Atom and RDF alike."""
    seen: set[int] = set()
    for element in root.iter():
        if _localname(element.tag) in ("item", "entry") and id(element) not in seen:
            seen.add(id(element))
            yield element


def _channel_title(root: ElementTree.Element) -> str:
    """The feed's own name, used as the article's source."""
    for element in root.iter():
        name = _localname(element.tag)
        if name in ("channel", "feed"):
            for child in element:
                if _localname(child.tag) == "title":
                    return clean_title(_text_of(child))
        if name in ("item", "entry"):
            break  # past the header; a channel title would have appeared by now
    return ""


def _first_text(
    fields: dict[str, list[tuple[str, ElementTree.Element]]], tags: tuple[str, ...]
) -> str:
    for tag in tags:
        for _, element in fields.get(tag, []):
            text = _text_of(element).strip()
            if text:
                return text
    return ""


def _join_text(
    fields: dict[str, list[tuple[str, ElementTree.Element]]], tag: str
) -> str:
    """All values for one tag. Atom can repeat ``content`` with type variants."""
    return "\n".join(
        _text_of(element).strip() for _, element in fields.get(tag, [])
    ).strip()


def _entry_link(
    fields: dict[str, list[tuple[str, ElementTree.Element]]],
    element: ElementTree.Element,
) -> str:
    """The article URL. Atom puts it in an attribute, RSS in the text.

    Atom entries carry several ``link`` elements; ``rel="alternate"`` (or no rel
    at all) is the article, while ``rel="replies"``/``"enclosure"`` are not.
    """
    candidates = fields.get("link", [])

    for _, link in candidates:
        rel = (link.get("rel") or "alternate").lower()
        href = (link.get("href") or "").strip()
        if href and rel == "alternate":
            return href

    for _, link in candidates:
        text = _text_of(link).strip()
        if _looks_like_url(text):
            return text

    for _, link in candidates:
        href = (link.get("href") or "").strip()
        if href:
            return href
    return ""


def _looks_like_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


__all__ = ["FeedResult", "RSSSource", "parse_date"]
