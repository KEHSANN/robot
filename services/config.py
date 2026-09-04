"""Central configuration for the Crypto Event Intelligence System.

Everything tunable lives here and is driven by environment variables so the
pipeline can be reconfigured (models, thresholds, key pools) without touching
code. Values are read once at import of :data:`settings`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is a hard requirement, but keep import failure survivable
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_a, **_kw):  # type: ignore[misc]
        return False


ROOT = Path(__file__).resolve().parent.parent

# This project historically kept its keys in a file named `env` (no dot), so we
# accept that alongside the conventional names. Loaded lowest-precedence last:
# a real process environment variable always wins over any file.
ENV_FILENAMES = (".env.local", ".env", "env")


def load_env_files() -> list[Path]:
    """Load every env file we can find. Returns the files that were read."""
    loaded: list[Path] = []
    for name in ENV_FILENAMES:
        path = ROOT / name
        if path.is_file():
            load_dotenv(path, override=False)
            loaded.append(path)
    return loaded


load_env_files()


# --------------------------------------------------------------------------- #
# env helpers
# --------------------------------------------------------------------------- #

def _str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name) or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name) or default)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = _str(name).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "y", "on")


def _list(name: str, default: tuple[str, ...] = ()) -> list[str]:
    raw = _str(name)
    if not raw:
        return list(default)
    return [part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()]


# --------------------------------------------------------------------------- #
# API key pools
# --------------------------------------------------------------------------- #

def read_key_pool(prefix: str) -> list[str]:
    """Collect every API key configured for a provider, in priority order.

    Three spellings are accepted so existing setups keep working while new keys
    can be added in whichever style is convenient:

    1. ``<PREFIX>_API_KEYS``      — one variable, comma/newline separated
    2. ``<PREFIX>_API_KEY_1..N``  — one variable per key, 1-based priority
    3. ``<PREFIX>_API_KEY``       — a single key

    All three are merged (in that order) and de-duplicated while preserving
    order, so key #1 stays key #1. Index gaps are tolerated: the scan continues
    past a missing number up to ``MAX_KEYS_PER_PROVIDER``.
    """
    keys: list[str] = []

    def add(candidate: str | None) -> None:
        if not candidate:
            return
        cleaned = candidate.strip().strip('"').strip("'")
        if cleaned and cleaned not in keys:
            keys.append(cleaned)

    for item in _list(f"{prefix}_API_KEYS"):
        add(item)

    for index in range(1, MAX_KEYS_PER_PROVIDER + 1):
        add(os.getenv(f"{prefix}_API_KEY_{index}"))

    add(os.getenv(f"{prefix}_API_KEY"))
    return keys


MAX_KEYS_PER_PROVIDER = 20

#: Provider name -> environment variable prefix.
PROVIDER_ENV_PREFIX = {
    "gemini": "GEMINI",
    "groq": "GROQ",
    "nvidia": "NVIDIA",
}

#: Provider name -> API base URL. Overridable for proxies/mirrors, which matters
#: in regions where these hosts are blocked.
PROVIDER_BASE_URL = {
    "gemini": _str("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"),
    "groq": _str("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    "nvidia": _str("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
}


@dataclass(frozen=True)
class FailoverSettings:
    """Tuning for the multi-key failover behaviour in :mod:`services.keypool`."""

    #: Attempts per request across the whole pool. Each attempt uses the highest
    #: priority key that is currently available.
    max_attempts: int = _int("FAILOVER_MAX_ATTEMPTS", 6)
    #: First cooldown applied to a key after a soft failure (seconds). Doubles
    #: on each consecutive failure of that key.
    cooldown_base_seconds: float = _float("FAILOVER_COOLDOWN_BASE", 20.0)
    #: Ceiling for the exponential cooldown (seconds).
    cooldown_max_seconds: float = _float("FAILOVER_COOLDOWN_MAX", 900.0)
    #: Cooldown for an exhausted quota / billing failure (seconds). Longer,
    #: because these usually reset on a daily boundary.
    quota_cooldown_seconds: float = _float("FAILOVER_QUOTA_COOLDOWN", 1800.0)
    #: How long a key rejected as invalid (401/403) is benched before we allow a
    #: single probe again. Guards against a permanent loss on a transient 401.
    dead_retry_seconds: float = _float("FAILOVER_DEAD_RETRY", 1800.0)
    #: When every key is cooling, wait at most this long for one to free up
    #: before giving up on the request (seconds).
    max_wait_for_key_seconds: float = _float("FAILOVER_MAX_WAIT", 25.0)
    #: Immediate retries against the same key for a provider-side 5xx/timeout.
    same_key_retries: int = _int("FAILOVER_SAME_KEY_RETRIES", 1)
    #: Persist key health to the database so cooldowns survive a restart.
    persist_health: bool = _bool("FAILOVER_PERSIST_HEALTH", True)


@dataclass(frozen=True)
class ConsensusSettings:
    """Vote-counting rules for the multi-model stages."""

    #: Stage 1 drops the item when at least this many models vote REMOVE.
    stage1_drop_votes: int = _int("STAGE1_DROP_VOTES", 2)
    #: Stage 2 rejects the item when at least this many models vote NO.
    stage2_reject_votes: int = _int("STAGE2_REJECT_VOTES", 3)
    #: Minimum models that must answer for a verdict to count. Below this the
    #: result is INCONCLUSIVE — an API outage must not look like a unanimous NO.
    min_votes: int = _int("CONSENSUS_MIN_VOTES", 2)
    #: What to do with an INCONCLUSIVE verdict: pass it to the next stage
    #: (fail-open, default) or drop it (fail-closed).
    fail_open: bool = _bool("CONSENSUS_FAIL_OPEN", True)


@dataclass(frozen=True)
class Stage0Settings:
    """Event detection / dedup thresholds."""

    #: Only compare against events seen inside this window.
    lookback_days: int = _int("STAGE0_LOOKBACK_DAYS", 7)
    #: Cosine similarity at/above which a stored article becomes a candidate for
    #: the same event.
    candidate_similarity: float = _float("STAGE0_CANDIDATE_SIMILARITY", 0.86)
    #: Similarity at/above which text is treated as a near-verbatim reprint, so
    #: identical extracted state means DUPLICATE without further checks.
    reprint_similarity: float = _float("STAGE0_REPRINT_SIMILARITY", 0.97)
    #: How many nearest neighbours to pull for identity comparison.
    candidate_limit: int = _int("STAGE0_CANDIDATE_LIMIT", 8)
    #: Embedding model + dimensionality. Must match the DB vector column.
    embed_model: str = _str("EMBED_MODEL", "gemini-embedding-2")
    embed_dim: int = _int("EMBED_DIM", 1536)
    #: Characters of article body fed to the embedding model.
    embed_max_chars: int = _int("EMBED_MAX_CHARS", 4000)


@dataclass(frozen=True)
class RoutingSettings:
    """Which events are expensive enough to escalate."""

    #: Stage 2 score (0-100) at/above which Stage 5 deep analysis runs.
    stage5_min_score: int = _int("STAGE5_MIN_SCORE", 60)
    #: Run Stage 5 when model agreement falls below this ratio, regardless of
    #: score — disagreement is itself a reason to look harder.
    stage5_max_agreement: float = _float("STAGE5_MAX_AGREEMENT", 0.7)
    #: Event types that always escalate to Stage 5.
    stage5_always_types: list[str] = field(
        default_factory=lambda: [
            t.upper()
            for t in _list(
                "STAGE5_ALWAYS_TYPES",
                (
                    "REGULATORY",
                    "ETF_APPROVAL",
                    "EXCHANGE_HACK",
                    "SECURITY_INCIDENT",
                    "LIQUIDATION",
                    "BANKRUPTCY",
                    "DELISTING",
                ),
            )
        ]
    )
    #: Stage 2 score at/above which the heavy NVIDIA panel runs.
    final_min_score: int = _int("FINAL_MIN_SCORE", 75)
    #: Stage 4 confidence below which we escalate even a mid-score event.
    final_max_confidence: float = _float("FINAL_MAX_CONFIDENCE", 0.6)
    #: Hard ceiling on heavy-panel runs per pipeline pass, so a burst of news
    #: cannot run away with the expensive budget.
    final_max_per_cycle: int = _int("FINAL_MAX_PER_CYCLE", 5)


@dataclass(frozen=True)
class IngestSettings:
    """News collection."""

    #: Items older than this are not worth analysing: the horizons this system
    #: forecasts over are 15 minutes to 24 hours, so a three-day-old article has
    #: already been priced. Also what stops the first run of an empty database
    #: from ingesting a month of feed history.
    max_age_hours: int = _int("INGEST_MAX_AGE_HOURS", 24)
    #: Per feed, per pass. Feeds occasionally republish their whole archive.
    max_items_per_feed: int = _int("INGEST_MAX_ITEMS_PER_FEED", 40)
    #: Feeds fetched at once.
    concurrency: int = _int("INGEST_CONCURRENCY", 6)
    timeout: float = _float("INGEST_TIMEOUT", 20.0)
    #: RSS summaries are short; anything longer is boilerplate or a full reprint.
    body_max_chars: int = _int("INGEST_BODY_MAX_CHARS", 6000)
    #: Sent on every fetch. Some publishers 403 an empty or default agent.
    user_agent: str = _str(
        "INGEST_USER_AGENT",
        "Mozilla/5.0 (compatible; CryptoEventIntel/1.0; +https://example.invalid/bot)",
    )
    #: Honour ETag / Last-Modified between passes. Polling seven feeds every five
    #: minutes gets throttled without it.
    conditional_requests: bool = _bool("INGEST_CONDITIONAL", True)


@dataclass(frozen=True)
class TelegramSettings:
    bot_token: str = _str("TELEGRAM_BOT_TOKEN")
    #: Chats that receive alerts.
    chat_ids: list[str] = field(default_factory=lambda: _list("TELEGRAM_CHAT_IDS"))
    #: Users allowed to issue bot commands. Empty means everyone.
    admin_ids: list[str] = field(default_factory=lambda: _list("TELEGRAM_ADMIN_IDS"))
    #: Only alert on events whose final score reaches this.
    min_alert_score: int = _int("TELEGRAM_MIN_ALERT_SCORE", 55)
    parse_mode: str = _str("TELEGRAM_PARSE_MODE", "HTML")
    enabled: bool = _bool("TELEGRAM_ENABLED", True)

    @property
    def configured(self) -> bool:
        return bool(self.bot_token and self.chat_ids)


@dataclass(frozen=True)
class FeedbackSettings:
    """Prediction -> observation -> outcome loop."""

    #: Minutes after a prediction at which price is sampled.
    horizons_minutes: list[int] = field(
        default_factory=lambda: [
            int(x) for x in _list("FEEDBACK_HORIZONS", ("15", "60", "180", "360", "1440"))
        ]
    )
    #: Public price API (no key required).
    binance_base_url: str = _str("BINANCE_BASE_URL", "https://api.binance.com")
    quote_asset: str = _str("PRICE_QUOTE_ASSET", "USDT")
    #: A direction call counts as correct when the move exceeds this, so noise
    #: around 0% is not scored as a hit.
    direction_deadband_pct: float = _float("FEEDBACK_DEADBAND_PCT", 0.15)


@dataclass(frozen=True)
class Settings:
    database_url: str = _str("DATABASE_URL")
    log_level: str = _str("LOG_LEVEL", "INFO")
    #: HTTP timeout for a single provider call (seconds).
    http_timeout: float = _float("HTTP_TIMEOUT", 90.0)
    #: Timeout for reasoning models and long generations. Separate because a
    #: 550B reasoning model answers in minutes, and a timeout is charged to the
    #: key as a transient fault — so one number for every model would cool the
    #: NVIDIA key on every critical event.
    heavy_http_timeout: float = _float("HTTP_TIMEOUT_HEAVY", 300.0)
    #: Ceiling on concurrent provider calls across the process.
    max_concurrent_llm_calls: int = _int("MAX_CONCURRENT_LLM_CALLS", 12)
    #: Articles pulled per pipeline pass.
    batch_size: int = _int("BATCH_SIZE", 25)
    #: Seconds between passes in `loop` mode.
    loop_interval_seconds: int = _int("LOOP_INTERVAL_SECONDS", 300)
    rss_feeds: list[str] = field(
        default_factory=lambda: _list("RSS_FEEDS", DEFAULT_RSS_FEEDS)
    )

    failover: FailoverSettings = field(default_factory=FailoverSettings)
    consensus: ConsensusSettings = field(default_factory=ConsensusSettings)
    stage0: Stage0Settings = field(default_factory=Stage0Settings)
    routing: RoutingSettings = field(default_factory=RoutingSettings)
    ingest: IngestSettings = field(default_factory=IngestSettings)
    telegram: TelegramSettings = field(default_factory=TelegramSettings)
    feedback: FeedbackSettings = field(default_factory=FeedbackSettings)

    @property
    def use_postgres(self) -> bool:
        return bool(self.database_url)


DEFAULT_RSS_FEEDS: tuple[str, ...] = (
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://www.theblock.co/rss.xml",
    "https://cryptoslate.com/feed/",
    "https://bitcoinmagazine.com/feed",
    "https://blockworks.co/feed",
)


settings = Settings()
