"""News collection.

``Ingestor`` fans out over sources and hands the pipeline a deduplicated batch in
chronological order; ``RSSSource`` is the only source implemented so far. The
cleaning helpers are exported because Stage 0's hashes are computed over their
output, so anything that produces a :class:`~services.types.NewsItem` outside this
package should clean text the same way.
"""

from ingest.clean import clean_text, clean_title, looks_like_prose, strip_html
from ingest.collector import IngestStats, Ingestor, NewsSource, collect_once
from ingest.rss import FeedResult, RSSSource, parse_date

__all__ = [
    "FeedResult",
    "IngestStats",
    "Ingestor",
    "NewsSource",
    "RSSSource",
    "clean_text",
    "clean_title",
    "collect_once",
    "looks_like_prose",
    "parse_date",
    "strip_html",
]
