"""Entry point for the crypto event intelligence system.

    python run.py doctor        check config, keys, database and model slugs
    python run.py initdb        create the schema (idempotent)
    python run.py keys          key pool health, no secrets
    python run.py ingest        collect articles only, no analysis
    python run.py run           one full pass: ingest -> stages -> publish
    python run.py observe       one feedback pass: sample prices, score outcomes
    python run.py loop          pipeline + feedback, forever
    python run.py bot           Telegram command bot only
    python run.py serve         everything: pipeline, feedback and bot together

Three things this file is responsible for that nothing else is.

**Nothing here prints a secret.** ``doctor`` and ``keys`` report on credentials by
fingerprint, and every error string goes through the same redaction the logs use.
A diagnostic command is exactly where a key would otherwise get pasted into a chat
window or a screenshot.

**Shutdown is orderly.** Long polling, an HTTP client and a database pool all need
closing, and a half-closed Postgres pool on a machine that restarts every deploy
turns into connection exhaustion. Ctrl-C sets one event that every loop watches.

**One failing subsystem does not take the others down.** In ``serve`` the bot, the
pipeline and the observer run as independent tasks: if the price API is unreachable
the analysis keeps running, and if a model provider is down the bot still answers.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
from datetime import datetime, timezone

from bot import (
    BotService,
    CommandHandler,
    Publisher,
    TelegramClient,
    format_keys,
    format_status,
)
from database import open_store
from feedback import Observer
from ingest import Ingestor
from services.config import Settings, settings as global_settings
from services.errors import ProviderError
from services.keypool import KeyPoolRegistry
from services.llm import LLMClient
from services.logging_setup import redact, setup_logging
from services.models import EMBED_MODEL, EXTRACTORS, FINAL_PANEL, PANEL, all_models
from stages.pipeline import Pipeline

log = logging.getLogger("run")

#: How often the feedback loop samples prices. Independent of the pipeline
#: interval: a 15-minute horizon needs checking on a tighter cadence than news
#: needs ingesting, and tying them together would make one starve the other.
OBSERVE_INTERVAL = 120.0


# --------------------------------------------------------------------------- #
# shared lifecycle
# --------------------------------------------------------------------------- #


class Runtime:
    """Everything the commands share, opened once and closed once.

    Built as a context manager rather than module globals so a command that only
    needs the database does not open an HTTP client to three providers, and so the
    tests can construct a runtime without a network.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self.settings = config or global_settings
        self.store = None
        self.registry: KeyPoolRegistry | None = None
        self.client: LLMClient | None = None
        self.telegram: TelegramClient | None = None
        self._stack: contextlib.AsyncExitStack | None = None
        self.stop = asyncio.Event()

    async def __aenter__(self) -> "Runtime":
        self._stack = contextlib.AsyncExitStack()
        await self._stack.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(*exc)
            self._stack = None

    @property
    def stack(self) -> contextlib.AsyncExitStack:
        if self._stack is None:
            raise RuntimeError("Runtime must be used as an async context manager")
        return self._stack

    async def open_db(self):
        if self.store is None:
            self.store = await open_store(self.settings)
            self.stack.push_async_callback(self.store.close)
        return self.store

    async def open_llm(self) -> LLMClient:
        """The model client, with key health persisted as it changes.

        The observer writes through to the database so a restart does not forget
        that key 2 is dead — otherwise every deploy re-discovers the same broken
        credential by spending a request on it.
        """
        if self.client is None:
            store = await self.open_db()
            pending: dict[tuple[str, int], dict] = {}

            def remember(key) -> None:
                pending[(key.provider, key.index)] = key.snapshot()

            self.registry = KeyPoolRegistry(self.settings.failover, observer=remember)

            # Start from what the last process learned, so a revoked key is not
            # rediscovered by spending a live request on it every restart.
            with contextlib.suppress(Exception):
                restored = self.registry.restore(await store.load_key_health())
                if restored:
                    log.info("restored health for %d key(s)", restored)

            self.client = LLMClient(self.registry, self.settings)
            await self.stack.enter_async_context(self.client)

            async def flush() -> None:
                if not pending:
                    return
                with contextlib.suppress(Exception):
                    await store.save_key_health(list(pending.values()))

            self.stack.push_async_callback(flush)
            self._flush_keys = flush
        return self.client

    async def open_telegram(self) -> TelegramClient:
        if self.telegram is None:
            self.telegram = TelegramClient(self.settings.telegram)
            await self.stack.enter_async_context(self.telegram)
        return self.telegram

    async def flush_key_health(self) -> None:
        flush = getattr(self, "_flush_keys", None)
        if flush is not None:
            await flush()

    def install_signals(self) -> None:
        """Ctrl-C and SIGTERM ask the loops to finish, rather than killing them.

        ``add_signal_handler`` is not implemented on Windows, so the fallback is
        the KeyboardInterrupt that :func:`main` already catches.
        """
        loop = asyncio.get_running_loop()
        for name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, self.stop.set)
            except (NotImplementedError, RuntimeError, ValueError):
                pass

    async def sleep(self, seconds: float) -> bool:
        """Wait, returning ``False`` if we were asked to stop meanwhile."""
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.stop.wait(), timeout=seconds)
        return not self.stop.is_set()


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


async def cmd_doctor(args) -> int:
    """Check every external dependency and report what is actually wrong.

    Exit code is non-zero when something would stop the system from working, so
    this can gate a deploy. A missing Telegram token is a warning, not a failure:
    the pipeline is useful without it, it just cannot announce anything.
    """
    settings = global_settings
    problems: list[str] = []
    warnings: list[str] = []

    print("── configuration ──────────────────────────────────────────")
    print(f"  database    {_dsn_summary(settings.database_url)}")
    print(f"  log level   {settings.log_level}")
    print(f"  feeds       {len(settings.rss_feeds)} configured")
    print(f"  batch       {settings.batch_size} articles per pass")
    print(f"  loop        every {settings.loop_interval_seconds}s")
    if not settings.rss_feeds:
        problems.append("no RSS feeds configured (RSS_FEEDS)")

    # -- keys ---------------------------------------------------------------- #
    print("\n── api keys ───────────────────────────────────────────────")
    registry = KeyPoolRegistry(settings.failover)
    for provider in ("groq", "gemini", "nvidia"):
        pool = registry.get(provider)
        if not pool.keys:
            problems.append(f"{provider}: no keys configured ({provider.upper()}_API_KEYS)")
            print(f"  {provider:<8} none configured")
            continue
        marks = " ".join(key.fingerprint for key in pool.keys)
        print(f"  {provider:<8} {len(pool.keys)} key(s)  {marks}")

        suspicious = [key for key in pool.keys if _looks_like_oauth(key.secret)]
        if suspicious:
            warnings.append(
                f"{provider}: {len(suspicious)} of {len(pool.keys)} keys start with 'AQ.', "
                "which is an OAuth token shape rather than an API key — these usually "
                "fail authentication"
            )

    # -- database ------------------------------------------------------------ #
    print("\n── database ───────────────────────────────────────────────")
    store = None
    try:
        store = await open_store(settings)
        counts = await store.counts()
        kind = type(store).__name__
        print(f"  connected   {kind}")
        for name, value in sorted(counts.items()):
            print(f"  {name:<11} {value}")
        if kind == "MemoryStore" and settings.database_url:
            problems.append("DATABASE_URL is set but Postgres refused; running in memory")
    except Exception as exc:
        problems.append(f"database unreachable: {type(exc).__name__}: {redact(str(exc))}")
        print(f"  failed      {type(exc).__name__}")

    # -- telegram ------------------------------------------------------------ #
    print("\n── telegram ───────────────────────────────────────────────")
    if not settings.telegram.configured:
        warnings.append("Telegram is not configured; alerts will not be delivered")
        print("  not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_IDS)")
    else:
        try:
            async with TelegramClient(settings.telegram) as client:
                me = await client.get_me()
            print(f"  bot         @{me.get('username', '?')} ({me.get('id', '?')})")
            print(f"  chats       {len(settings.telegram.chat_ids)}")
            print(f"  admins      {len(settings.telegram.admin_ids) or 'none — /keys disabled'}")
        except Exception as exc:
            problems.append(f"telegram: {type(exc).__name__}: {redact(str(exc))}")
            print(f"  failed      {type(exc).__name__}")

    # -- models -------------------------------------------------------------- #
    print("\n── models ─────────────────────────────────────────────────")
    print(f"  panel       {len(PANEL)} models (stages 1-5)")
    print(f"  final       {len(FINAL_PANEL)} models (NVIDIA, critical events only)")
    print(f"  extractors  {len(EXTRACTORS)}")
    print(f"  embedding   {EMBED_MODEL.id}")

    if args.offline:
        print("  probe       skipped (--offline)")
    elif store is not None:
        failed = await _probe_models(settings, registry)
        for model_id, detail in failed:
            problems.append(f"model {model_id}: {detail}")

    # -- price feed ---------------------------------------------------------- #
    print("\n── prices ─────────────────────────────────────────────────")
    if args.offline:
        print("  probe       skipped (--offline)")
    else:
        from feedback import PriceSource

        try:
            async with PriceSource(settings.feedback) as prices:
                price = await prices.spot("BTC")
            print(f"  BTC{settings.feedback.quote_asset:<8} {price}")
        except Exception as exc:
            warnings.append(f"price feed: {type(exc).__name__} (outcomes cannot be scored)")
            print(f"  failed      {type(exc).__name__}")

    if store is not None:
        await store.close()

    # -- verdict ------------------------------------------------------------- #
    print()
    for text in warnings:
        print(f"  ! {text}")
    for text in problems:
        print(f"  ✗ {text}")
    if not problems:
        print("  ✓ ready" + (" (with warnings)" if warnings else ""))
    return 1 if problems else 0


async def _probe_models(settings: Settings, registry: KeyPoolRegistry) -> list[tuple[str, str]]:
    """Ask every configured model for one token, to find out if the slug is real.

    Worth the handful of tokens it costs: a mistyped or retired model name is
    invisible until an event escalates, and the first time it shows up is the one
    moment the answer mattered. Reported per model rather than aggregated, because
    "one of five panel models does not exist" and "the provider is down" need
    different fixes.
    """
    failures: list[tuple[str, str]] = []
    async with LLMClient(registry, settings) as client:
        specs = [spec for spec in all_models() if spec.provider in registry.configured_providers()]
        results = await asyncio.gather(
            *(_probe_one(client, spec) for spec in specs), return_exceptions=False
        )
        for spec, detail in zip(specs, results):
            mark = "ok" if detail is None else "FAILED"
            print(f"  {mark:<11} {spec.id} ({spec.provider})"
                  + (f" — {detail}" if detail else ""))
            if detail is not None:
                failures.append((spec.id, detail))
    return failures


async def _probe_one(client: LLMClient, spec) -> str | None:
    try:
        result = await client.complete_text(spec, "Reply with OK.", "OK", max_output_tokens=8)
    except ProviderError as exc:
        return redact(f"{exc.status or ''} {exc.message}".strip())[:120]
    except Exception as exc:
        return f"{type(exc).__name__}: {redact(str(exc))}"[:120]
    return None if result.ok else redact(result.error or "empty response")[:120]


def _dsn_summary(url: str) -> str:
    """Describe a database URL without printing its password."""
    if not url:
        return "not set (memory store)"
    try:
        head, _, tail = url.rpartition("@")
        scheme = head.split("://", 1)[0] if "://" in head else "postgres"
        return f"{scheme}://…@{tail}" if tail and head else url
    except Exception:
        return "set"


def _looks_like_oauth(secret: str) -> bool:
    """Google OAuth access tokens start ``AQ.`` and are not API keys.

    Worth calling out explicitly: the pool accepts them, the request fails with a
    401 that reads like a quota problem, and the key gets marked dead for a reason
    that has nothing to do with quota.
    """
    return secret.startswith("AQ.")


# --------------------------------------------------------------------------- #
# small commands
# --------------------------------------------------------------------------- #


async def cmd_initdb(args) -> int:
    async with Runtime() as runtime:
        store = await runtime.open_db()
        await store.init_schema()
        counts = await store.counts()
    print(f"schema ready ({type(store).__name__})")
    for name, value in sorted(counts.items()):
        print(f"  {name:<14} {value}")
    return 0


async def cmd_keys(args) -> int:
    async with Runtime() as runtime:
        store = await runtime.open_db()
        registry = KeyPoolRegistry(runtime.settings.failover)

        # Health survives restarts, so show what was learned last run rather than
        # a clean slate that hides a key already known to be dead.
        with contextlib.suppress(Exception):
            registry.restore(await store.load_key_health())

        print(_plain(format_keys(registry.snapshot())))
    return 0


async def cmd_ingest(args) -> int:
    """Show what the feeds are currently returning, without storing anything.

    Deliberately read-only. Stage 0 is what inserts an article, and it decides
    NEW/UPDATE/DUPLICATE by looking for the article's hash in the database — so
    storing rows here would make the next ``run`` classify every one of them as a
    duplicate of itself and analyse none of them. This is a diagnostic for "are the
    feeds alive and is the cleaning working", which is what it is usually needed
    for at 3am.
    """
    ingestor = Ingestor(config=global_settings)
    items = await ingestor.collect(limit=args.limit)

    print(f"{len(items)} articles (not stored — use `run` to analyse them)")
    for item in items:
        age = ""
        if item.published_at:
            minutes = (datetime.now(timezone.utc) - item.published_at).total_seconds() / 60
            age = f"{int(minutes)}m ago"
        print(f"  [{item.source:<16}] {age:>9}  {item.title[:80]}")
        if args.verbose:
            print(f"      {(item.body or '')[:200]}")
    return 0


async def cmd_observe(args) -> int:
    async with Runtime() as runtime:
        store = await runtime.open_db()
        async with Observer(store, config=runtime.settings) as observer:
            stats = await observer.run_once(limit=args.limit or 200)
    print(stats.summary())
    for error in stats.errors[:10]:
        print(f"  ! {error}")
    return 0


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #


async def _one_pass(runtime: Runtime, ingestor: Ingestor, pipeline: Pipeline,
                    publisher: Publisher | None, limit: int | None) -> dict:
    """Ingest, analyse, publish. The unit of work ``run`` and ``loop`` share."""
    started = datetime.now(timezone.utc)
    items = await ingestor.collect(limit=limit)
    log.info("collected %d articles", len(items))

    alerts = await pipeline.run_batch(items)
    stats = pipeline.stats_snapshot()

    if publisher is not None and alerts:
        report = await publisher.publish(alerts)
        log.info("published: %s", report.summary())
        stats["published"] = report.sent
        stats["suppressed"] = report.suppressed + report.below_threshold

    stats["seconds"] = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
    with contextlib.suppress(Exception):
        await runtime.store.save_run(stats)
    await runtime.flush_key_health()
    return stats


async def cmd_run(args) -> int:
    async with Runtime() as runtime:
        runtime.install_signals()
        store = await runtime.open_db()
        client = await runtime.open_llm()

        publisher = None
        if runtime.settings.telegram.configured and not args.no_publish:
            telegram = await runtime.open_telegram()
            publisher = Publisher(telegram, store, runtime.settings)

        stats = await _one_pass(
            runtime,
            Ingestor(config=runtime.settings),
            Pipeline(client, store, runtime.settings),
            publisher,
            args.limit,
        )

    print("── pass complete ──")
    for name, value in stats.items():
        print(f"  {name:<16} {value}")
    return 0


async def _pipeline_loop(runtime: Runtime, publisher: Publisher | None, limit: int | None) -> None:
    """Ingest and analyse forever, surviving a failed pass.

    A pass that throws must not end the loop: the usual cause is a provider or a
    feed being briefly unavailable, and a system that stops on the first outage is
    less useful than one that logs it and tries again in a minute.
    """
    store = await runtime.open_db()
    client = await runtime.open_llm()
    ingestor = Ingestor(config=runtime.settings)
    pipeline = Pipeline(client, store, runtime.settings)
    interval = max(30, runtime.settings.loop_interval_seconds)

    while not runtime.stop.is_set():
        try:
            stats = await _one_pass(runtime, ingestor, pipeline, publisher, limit)
            log.info(
                "pass: %s ingested, %s alerts, %s errors in %ss",
                stats.get("ingested", 0), stats.get("alerts", 0),
                stats.get("errors", 0), stats.get("seconds", 0),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("pipeline pass failed; retrying next interval")

        if not await runtime.sleep(interval):
            return


async def _observe_loop(runtime: Runtime) -> None:
    """Sample prices and resolve predictions forever, on its own cadence."""
    store = await runtime.open_db()
    async with Observer(store, config=runtime.settings) as observer:
        while not runtime.stop.is_set():
            try:
                stats = await observer.run_once()
                if stats.considered:
                    log.info("observe: %s", stats.summary())
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("observation pass failed; retrying next interval")

            if not await runtime.sleep(OBSERVE_INTERVAL):
                return


async def cmd_loop(args) -> int:
    async with Runtime() as runtime:
        runtime.install_signals()
        store = await runtime.open_db()

        publisher = None
        if runtime.settings.telegram.configured and not args.no_publish:
            telegram = await runtime.open_telegram()
            publisher = Publisher(telegram, store, runtime.settings)

        log.info("loop starting (pipeline every %ss, observer every %ss)",
                 runtime.settings.loop_interval_seconds, int(OBSERVE_INTERVAL))
        await _gather_until_stopped(
            runtime,
            _pipeline_loop(runtime, publisher, args.limit),
            _observe_loop(runtime),
        )
    return 0


# --------------------------------------------------------------------------- #
# the bot
# --------------------------------------------------------------------------- #


async def _build_bot(runtime: Runtime, stats_source=None) -> BotService:
    store = await runtime.open_db()
    telegram = await runtime.open_telegram()
    client = await runtime.open_llm()
    handler = CommandHandler(
        telegram, store, runtime.registry, runtime.settings, stats_source=stats_source
    )
    return BotService(telegram, handler)


async def cmd_bot(args) -> int:
    if not global_settings.telegram.configured:
        print("Telegram is not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_IDS)")
        return 1

    async with Runtime() as runtime:
        runtime.install_signals()
        service = await _build_bot(runtime)
        await service.start()
        log.info("bot polling; Ctrl-C to stop")
        await _gather_until_stopped(runtime, service.run())
        await service.stop()
    return 0


async def cmd_serve(args) -> int:
    """Pipeline, feedback loop and bot, as independent tasks."""
    async with Runtime() as runtime:
        runtime.install_signals()
        store = await runtime.open_db()

        publisher = None
        service = None
        if runtime.settings.telegram.configured:
            telegram = await runtime.open_telegram()
            publisher = Publisher(telegram, store, runtime.settings)
            if not args.no_bot:
                service = await _build_bot(runtime)
                await service.start()
        else:
            log.warning("Telegram not configured: analysing without publishing")

        tasks = [_pipeline_loop(runtime, publisher, args.limit), _observe_loop(runtime)]
        if service is not None:
            tasks.append(service.run())

        log.info("serve: %d subsystems running", len(tasks))
        await _gather_until_stopped(runtime, *tasks)

        if service is not None:
            await service.stop()
    return 0


async def _gather_until_stopped(runtime: Runtime, *coros) -> None:
    """Run coroutines together; the first crash stops the rest cleanly.

    ``return_exceptions=True`` would hide a subsystem dying silently, so failures
    are logged individually and the shutdown event is set for the others.
    """
    tasks = [asyncio.create_task(coro) for coro in coros]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            exc = task.exception()
            if exc is not None:
                log.error("subsystem stopped: %s: %s", type(exc).__name__, exc)
        runtime.stop.set()
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except asyncio.CancelledError:
        runtime.stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def cmd_status(args) -> int:
    async with Runtime() as runtime:
        store = await runtime.open_db()
        registry = KeyPoolRegistry(runtime.settings.failover)
        counts = await store.counts()
        print(_plain(format_status({}, counts, registry.snapshot())))
    return 0


def _plain(html_text: str) -> str:
    """Strip the Telegram markup the formatters emit, for terminal output."""
    import re
    from html import unescape

    text = re.sub(r"<a href=\"([^\"]+)\">([^<]*)</a>", r"\2 (\1)", html_text)
    return unescape(re.sub(r"<[^>]+>", "", text))


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Crypto event intelligence: ingest, analyse, alert, score.",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check config, keys, database and models")
    doctor.add_argument("--offline", action="store_true",
                        help="skip network probes (no model or price checks)")
    doctor.set_defaults(func=cmd_doctor)

    sub.add_parser("initdb", help="create tables and indexes").set_defaults(func=cmd_initdb)
    sub.add_parser("keys", help="key pool health (no secrets)").set_defaults(func=cmd_keys)
    sub.add_parser("status", help="row counts and key summary").set_defaults(func=cmd_status)

    ingest = sub.add_parser("ingest", help="show what the feeds return (does not store)")
    ingest.add_argument("-n", "--limit", type=int, default=None)
    ingest.add_argument("-v", "--verbose", action="store_true", help="include body text")
    ingest.set_defaults(func=cmd_ingest)

    observe = sub.add_parser("observe", help="one feedback pass")
    observe.add_argument("-n", "--limit", type=int, default=None)
    observe.set_defaults(func=cmd_observe)

    one = sub.add_parser("run", help="one full pass")
    one.add_argument("-n", "--limit", type=int, default=None)
    one.add_argument("--no-publish", action="store_true", help="analyse without sending alerts")
    one.set_defaults(func=cmd_run)

    loop = sub.add_parser("loop", help="pipeline + feedback, forever")
    loop.add_argument("-n", "--limit", type=int, default=None)
    loop.add_argument("--no-publish", action="store_true")
    loop.set_defaults(func=cmd_loop)

    sub.add_parser("bot", help="Telegram command bot only").set_defaults(func=cmd_bot)

    serve = sub.add_parser("serve", help="pipeline, feedback and bot together")
    serve.add_argument("-n", "--limit", type=int, default=None)
    serve.add_argument("--no-bot", action="store_true", help="publish but do not poll for commands")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    try:
        return asyncio.run(args.func(args))
    except KeyboardInterrupt:
        # The signal handler covers the graceful path; this is the fallback on
        # Windows, where add_signal_handler does not exist.
        print("\nstopped")
        return 130


if __name__ == "__main__":
    sys.exit(main())
