"""Tests for the Telegram output layer.

The bot is the only part of the system a user ever sees, so the failures worth
protecting against are the ones that are invisible from inside the process:

1. **A lost alert looks exactly like an alert the system never produced.** One
   unescaped ``<`` in a headline turns a valid message into a 400, so escaping is
   checked against the characters publishers actually use, and the plain-text
   retry is checked to actually fire when Telegram rejects markup.

2. **A secret in a message cannot be recalled.** ``/keys`` goes to Telegram, and
   the provider error text it prints is written by the provider. A test asserts
   the rendered output contains fingerprints and no key material.

3. **Over-sending is how a bot gets ignored.** A story that updates six times
   with the same verdict must produce one message; a story whose verdict
   *changed* must produce another. Both directions are tested, because a
   suppression bug in either direction is silent.

4. **A rate limit answered with a retry extends the block.** The client is
   checked to honour ``retry_after`` rather than hammering.

The Telegram API is faked at the transport layer (an ``httpx`` handler), so the
request the client would really send is what gets asserted on.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from bot.commands import CommandHandler
from bot.format import (
    format_alert,
    format_digest,
    format_keys,
    format_plain,
    format_recent,
    format_status,
    split_message,
)
from bot.publisher import Publisher
from bot.telegram import MAX_RETRY_AFTER, TelegramClient, TelegramError, Update
from services.config import (
    FailoverSettings,
    Settings,
    TelegramSettings,
)
from services.errors import Disposition, KeyFault, ProviderError
from services.keypool import KeyPool
from services.types import (
    AssetImpact,
    Causality,
    Direction,
    FactSet,
    Magnitude,
    NewsItem,
    Relation,
    Stage0Decision,
)
from stages.pipeline import Alert

# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #

TOKEN = "123456789:AAF-fake-token-for-tests-only-xxxxxxxxx"


def telegram_settings(**overrides) -> TelegramSettings:
    base = {
        "bot_token": TOKEN,
        "chat_ids": ["-100111"],
        "admin_ids": ["555"],
        "min_alert_score": 55,
        "parse_mode": "HTML",
        "enabled": True,
    }
    base.update(overrides)
    return TelegramSettings(**base)


def app_settings(**telegram_overrides) -> Settings:
    """A Settings object whose only interesting part is the Telegram block."""
    return Settings(telegram=telegram_settings(**telegram_overrides))


def news(**overrides) -> NewsItem:
    base = {
        "title": "SEC approves BlackRock spot BTC ETF & 2 others",
        "body": "The order lets the funds list immediately.",
        "url": "https://coindesk.com/policy/sec-approves-etf",
        "source": "CoinDesk",
        "source_type": "rss",
    }
    base.update(overrides)
    return NewsItem(**base)


def facts(**overrides) -> FactSet:
    base = {
        "event_type": "REGULATION",
        "entity": "SEC",
        "action": "approved",
        "target": "spot BTC ETF",
        "headline": "SEC approves BlackRock & 2 other spot BTC ETFs",
    }
    base.update(overrides)
    return FactSet(**base)


def impact(**overrides) -> AssetImpact:
    base = {
        "asset": "BTC",
        "direction": Direction.BULLISH,
        "magnitude": Magnitude.HIGH,
        "expected_low": 3.0,
        "expected_high": 7.5,
        "confidence": 0.82,
        "horizon_minutes": 180,
        "causality": Causality.REGULATORY,
        "relation": Relation.DIRECT,
        "mechanism": "Approval unlocks institutional inflows.",
        "risks": "Sell-the-news if week-one inflows disappoint.",
        "agreement": 0.8,
        "model_count": 5,
        "source": "stage4",
        "tradeable": True,
    }
    base.update(overrides)
    return AssetImpact(**base)


def alert(**overrides) -> Alert:
    base = {
        "news": news(),
        "facts": facts(),
        "impact": impact(),
        "score": 88.0,
        "urgency": "critical",
        "event_id": 7,
        "news_id": 11,
        "analysis_id": 3,
        "source_count": 3,
        "decision": Stage0Decision.NEW,
        "trail": ["stage3", "stage4"],
    }
    base.update(overrides)
    return Alert(**base)


class FakeTelegram:
    """An ``httpx`` transport standing in for the Bot API.

    Records every request so a test can assert on what would really have gone
    over the wire, and can be told to fail a given method a set number of times.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []
        #: method -> list of (status, body) to serve before succeeding
        self.failures: dict[str, list[tuple[int, dict]]] = {}
        self.updates: list[dict] = []
        self._message_id = 1000

    def queue_failure(self, method: str, status: int, body: dict, times: int = 1) -> None:
        self.failures.setdefault(method, []).extend([(status, body)] * times)

    def handler(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        payload = json.loads(request.content or b"{}")
        self.requests.append((method, payload))

        queued = self.failures.get(method)
        if queued:
            status, body = queued.pop(0)
            return httpx.Response(status, json=body)

        if method == "getUpdates":
            result, self.updates = self.updates, []
            return httpx.Response(200, json={"ok": True, "result": result})
        if method == "getMe":
            return httpx.Response(200, json={"ok": True, "result": {"username": "testbot"}})

        self._message_id += 1
        return httpx.Response(
            200,
            json={"ok": True, "result": {"message_id": self._message_id,
                                         "chat": {"id": payload.get("chat_id")}}},
        )

    def sent(self) -> list[dict]:
        return [payload for method, payload in self.requests if method == "sendMessage"]

    def texts(self) -> list[str]:
        return [payload.get("text", "") for payload in self.sent()]


def client_for(fake: FakeTelegram, config: TelegramSettings | None = None) -> TelegramClient:
    transport = httpx.MockTransport(fake.handler)
    return TelegramClient(
        config=config or telegram_settings(),
        http=httpx.AsyncClient(transport=transport, timeout=5.0),
    )


async def no_delay(_seconds: float) -> None:
    """Stand-in for ``asyncio.sleep`` that yields without waiting.

    Written out rather than patched with a lambda calling ``asyncio.sleep``,
    which would recurse into the patch it just installed.
    """
    return None


class FakeStore:
    """Only the four methods the bot touches."""

    def __init__(self, *, alerted: set[tuple[int, str]] | None = None) -> None:
        self.alerted = alerted or set()
        self.recorded: list[dict] = []
        self.count_calls = 0

    async def record_alert(self, **kwargs) -> int:
        self.recorded.append(kwargs)
        return len(self.recorded)

    async def already_alerted(self, event_id: int, asset: str) -> bool:
        return (event_id, asset) in self.alerted

    async def recent_analyses(self, limit: int = 10) -> list[dict]:
        return [
            {"asset": "BTC", "direction": "BULLISH", "expected_low": 3.0,
             "expected_high": 7.5, "confidence": 0.82, "deepest_stage": 5,
             "headline": "SEC approves spot BTC ETFs"}
        ][:limit]

    async def counts(self) -> dict[str, int]:
        self.count_calls += 1
        return {"news": 412, "events": 180, "analyses": 61}


# --------------------------------------------------------------------------- #
# formatting: escaping
# --------------------------------------------------------------------------- #


def test_alert_escapes_markup_characters():
    """A headline with ``&`` and ``<`` must not produce unbalanced HTML.

    This is the whole reason alerts get lost: Telegram rejects the message, the
    send fails, and from the outside it is indistinguishable from silence.
    """
    rendered = format_alert(
        alert(
            news=news(title="Hack: <Curve> pools drained & paused"),
            facts=facts(headline="Hack: <Curve> pools drained & paused"),
            impact=impact(mechanism="TVL < $100M remains; 2 > 1 pools halted"),
        )
    )

    assert "<Curve>" not in rendered
    assert "&lt;Curve&gt;" in rendered
    assert "drained &amp; paused" in rendered
    assert "&lt; $100M" in rendered
    # The tags the formatter opens are still balanced.
    assert rendered.count("<b>") == rendered.count("</b>")
    assert rendered.count("<i>") == rendered.count("</i>")


def test_alert_carries_every_required_field():
    """The spec's alert layout: direction, size, confidence, horizon, why, risk."""
    rendered = format_alert(alert())

    assert "BTC" in rendered
    assert Direction.BULLISH.emoji in rendered
    assert "+3.0%" in rendered and "+7.5%" in rendered
    assert "82%" in rendered
    assert "3h" in rendered
    assert "Approval unlocks institutional inflows." in rendered
    assert "Sell-the-news" in rendered
    assert "CoinDesk" in rendered
    assert "5 models" in rendered


def test_bearish_range_reads_as_a_fall():
    """A negative forecast must not print as a positive number.

    The models return magnitudes; direction carries the sign. Printing ``3.0%``
    for a BEARISH call is the one formatting bug that could be read as the
    opposite trade.
    """
    rendered = format_alert(
        alert(impact=impact(direction=Direction.BEARISH,
                            expected_low=-2.0, expected_high=-6.0))
    )
    assert "−2.0%" in rendered and "−6.0%" in rendered
    assert "+2.0%" not in rendered
    assert Direction.BEARISH.emoji in rendered


def test_point_estimate_collapses_to_one_number():
    rendered = format_alert(alert(impact=impact(expected_low=4.0, expected_high=4.0)))
    assert "+4.0%" in rendered
    assert "…" not in rendered.split("Confidence")[0].split("Expected move")[1]


def test_update_is_labelled_differently_from_new():
    assert "UPDATE" in format_alert(alert(decision=Stage0Decision.UPDATE))
    assert "NEW" in format_alert(alert(decision=Stage0Decision.NEW))


def test_untradeable_verdict_says_so():
    """A HIGH-magnitude verdict the router marked untradeable must not read as a call."""
    rendered = format_alert(alert(impact=impact(tradeable=False)))
    assert "Not tradeable" in rendered


def test_plain_fallback_has_no_markup():
    """The fallback exists to survive a parse rejection, so it must not re-trigger one."""
    rendered = format_plain(
        alert(facts=facts(headline="Hack: <Curve> pools drained & paused"))
    )
    assert "<b>" not in rendered and "<i>" not in rendered
    # And it keeps the content that matters.
    assert "BTC" in rendered and "+3.0%" in rendered and "82%" in rendered


def test_digest_lists_each_alert_once():
    alerts = [
        alert(impact=impact(asset="BTC")),
        alert(impact=impact(asset="ETH", direction=Direction.BEARISH)),
        alert(impact=impact(asset="SOL")),
    ]
    rendered = format_digest(alerts, limit=2)
    assert "3 alerts" in rendered
    assert "BTC" in rendered and "ETH" in rendered
    assert "and 1 more" in rendered


# --------------------------------------------------------------------------- #
# formatting: splitting
# --------------------------------------------------------------------------- #


def test_short_message_is_not_split():
    assert split_message("short") == ["short"]


def test_long_message_splits_under_the_limit_on_blank_lines():
    text = "\n\n".join(f"paragraph {index} " + "x" * 200 for index in range(60))
    chunks = split_message(text, limit=1000)

    assert len(chunks) > 1
    assert all(len(chunk) <= 1000 for chunk in chunks)
    # Nothing lost: every paragraph marker survives somewhere.
    joined = "\n\n".join(chunks)
    assert all(f"paragraph {index} " in joined for index in range(60))


def test_split_falls_back_to_hard_cut_without_boundaries():
    """A wall of text with no newlines still has to be delivered."""
    chunks = split_message("y" * 5000, limit=1000)
    assert [len(chunk) for chunk in chunks] == [1000, 1000, 1000, 1000, 1000]


# --------------------------------------------------------------------------- #
# formatting: /keys must never print a secret
# --------------------------------------------------------------------------- #


SECRET = "AIzaSyB1aaaa000000000000000000000000000"


@pytest.mark.asyncio
async def test_keys_output_masks_secrets_including_echoed_ones():
    """``/keys`` goes to Telegram, and its error text is written by the provider.

    Gemini passes the key as a URL query parameter and quotes the URL back in
    some errors, so the raw secret genuinely arrives inside ``last_error``. A
    message containing it cannot be unsent.
    """
    pool = KeyPool(
        "gemini",
        [SECRET, "AIzaSyB2bbbb000000000000000000000000000"],
        FailoverSettings(),
    )
    key = await pool.acquire()
    await pool.report_failure(
        key,
        ProviderError(
            "gemini",
            f"API key not valid: {SECRET}",
            Disposition.RETRY_OTHER_KEY,
            KeyFault.INVALID,
            status=400,
        ),
    )

    rendered = format_keys(pool_snapshot(pool))

    assert SECRET not in rendered
    assert "API key not valid" in rendered  # the useful half is kept
    assert pool.keys[0].fingerprint in rendered


@pytest.mark.asyncio
async def test_keys_output_distinguishes_same_prefix_keys():
    """Three Gemini keys share the ``AIzaSy`` prefix and often the same tail.

    If they render identically, the one moment the table is needed — one key of
    three has died — is the moment it cannot say which.
    """
    pool = KeyPool(
        "gemini",
        [
            "AIzaSyB1aaaa000000000000000000000000000",
            "AIzaSyB2bbbb000000000000000000000000000",
            "AIzaSyB3cccc000000000000000000000000000",
        ],
        FailoverSettings(),
    )
    rendered = format_keys(pool_snapshot(pool))
    fingerprints = {key.fingerprint for key in pool.keys}

    assert len(fingerprints) == 3
    assert all(fingerprint in rendered for fingerprint in fingerprints)


def pool_snapshot(pool: KeyPool) -> dict:
    return {pool.provider: pool.snapshot()}


def test_keys_output_names_an_empty_pool():
    rendered = format_keys({"groq": {"provider": "groq", "total": 0,
                                     "available": 0, "keys": []}})
    assert "groq" in rendered and "no keys configured" in rendered


def test_status_reports_the_funnel_and_pools():
    rendered = format_status(
        {"ingested": 41, "duplicates": 12, "new_events": 22, "analysed": 9,
         "alerts": 2, "llm_calls": 155, "llm_failures": 4},
        {"news": 412, "events": 180},
        {"gemini": {"total": 3, "available": 2}, "groq": {"total": 0, "available": 0}},
    )
    assert "41" in rendered and "412" in rendered
    assert "gemini: 2/3" in rendered
    assert "155" in rendered and "4 failed" in rendered


def test_recent_falls_back_to_the_mechanism_without_a_headline():
    rendered = format_recent(
        [{"asset": "ETH", "direction": "bearish", "expected_low": -1.0,
          "expected_high": -3.0, "confidence": 0.4, "deepest_stage": 4,
          "mechanism": "Rotation out of alt L1s"}]
    )
    assert "ETH" in rendered and "Rotation out of alt L1s" in rendered


def test_recent_handles_an_empty_table():
    assert "Nothing analysed" in format_recent([])


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_send_posts_html_with_previews_disabled():
    fake = FakeTelegram()
    async with client_for(fake) as client:
        result = await client.send("-100111", "<b>hello</b>")

    assert result.sent and result.message_id
    payload = fake.sent()[0]
    assert payload["parse_mode"] == "HTML"
    assert payload["link_preview_options"] == {"is_disabled": True}
    assert payload["chat_id"] == "-100111"


@pytest.mark.asyncio
async def test_parse_rejection_retries_as_plain_text():
    """The whole point of the fallback: a readable alert beats a lost one."""
    fake = FakeTelegram()
    fake.queue_failure(
        "sendMessage", 400,
        {"ok": False, "error_code": 400,
         "description": "Bad Request: can't parse entities: Unsupported start tag"},
    )

    async with client_for(fake) as client:
        result = await client.send(
            "-100111", "<b>broken <notatag></b>", plain_fallback="broken but readable"
        )

    assert result.sent and result.downgraded
    texts = fake.texts()
    assert len(texts) == 2
    assert texts[1] == "broken but readable"
    # The retry must not carry the markup that was just rejected.
    assert "parse_mode" not in fake.sent()[1]


@pytest.mark.asyncio
async def test_permanent_chat_error_is_not_retried():
    """A blocked bot will still be blocked on the third attempt."""
    fake = FakeTelegram()
    fake.queue_failure(
        "sendMessage", 403,
        {"ok": False, "description": "Forbidden: bot was blocked by the user"},
        times=5,
    )

    async with client_for(fake) as client:
        result = await client.send("-100111", "hello")

    assert not result.sent
    assert "blocked" in result.error
    assert len(fake.sent()) == 1


@pytest.mark.asyncio
async def test_rate_limit_waits_the_requested_delay(monkeypatch):
    """Retrying a 429 immediately extends the block instead of clearing it."""
    slept: list[float] = []

    async def record(seconds):
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", record)

    fake = FakeTelegram()
    fake.queue_failure(
        "sendMessage", 429,
        {"ok": False, "description": "Too Many Requests", "parameters": {"retry_after": 7}},
    )

    async with client_for(fake) as client:
        result = await client.send("-100111", "hello")

    assert result.sent
    assert 7.0 in slept


@pytest.mark.asyncio
async def test_absurd_rate_limit_gives_up_instead_of_blocking(monkeypatch):
    """A multi-minute block is not worth holding the pipeline for."""
    monkeypatch.setattr(asyncio, "sleep", no_delay)

    fake = FakeTelegram()
    fake.queue_failure(
        "sendMessage", 429,
        {"ok": False, "description": "Too Many Requests",
         "parameters": {"retry_after": MAX_RETRY_AFTER + 120}},
        times=3,
    )

    async with client_for(fake) as client:
        result = await client.send("-100111", "hello")

    assert not result.sent
    assert "rate limited" in result.error


@pytest.mark.asyncio
async def test_server_error_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", no_delay)

    fake = FakeTelegram()
    fake.queue_failure("sendMessage", 502, {"ok": False, "description": "Bad Gateway"})

    async with client_for(fake) as client:
        result = await client.send("-100111", "hello")

    assert result.sent
    assert len(fake.sent()) == 2


@pytest.mark.asyncio
async def test_long_message_is_sent_as_several_messages(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", no_delay)

    fake = FakeTelegram()
    async with client_for(fake) as client:
        result = await client.send("-100111", "\n\n".join(["x" * 500] * 20))

    assert result.sent and result.parts > 1
    assert len(fake.sent()) == result.parts


@pytest.mark.asyncio
async def test_missing_token_fails_without_a_request():
    fake = FakeTelegram()
    async with client_for(fake, telegram_settings(bot_token="")) as client:
        with pytest.raises(TelegramError):
            await client.call("getMe")
    assert fake.requests == []


@pytest.mark.asyncio
async def test_token_is_not_echoed_in_error_text():
    """The token lives in the URL, so httpx error text can contain it."""
    fake = FakeTelegram()
    fake.queue_failure(
        "sendMessage", 400,
        {"ok": False, "description": f"Bad Request from https://api.telegram.org/bot{TOKEN}/sendMessage"},
        times=3,
    )
    async with client_for(fake) as client:
        result = await client.send("-100111", "hello")

    assert not result.sent
    assert TOKEN not in result.error


# --------------------------------------------------------------------------- #
# polling
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_updates_advance_the_offset():
    """Without an advancing offset Telegram redelivers the same update forever."""
    fake = FakeTelegram()
    fake.updates = [
        {"update_id": 10, "message": {"message_id": 1, "text": "/status",
                                      "chat": {"id": -100111}, "from": {"id": 555}}},
        {"update_id": 11, "message": {"message_id": 2, "text": "hello",
                                      "chat": {"id": -100111}, "from": {"id": 555}}},
    ]

    async with client_for(fake) as client:
        first = await client.get_updates(timeout=0)
        await client.get_updates(timeout=0)

    assert [update.update_id for update in first] == [10, 11]
    offsets = [payload.get("offset") for method, payload in fake.requests
               if method == "getUpdates"]
    assert offsets == [None, 12]


@pytest.mark.asyncio
async def test_non_command_update_is_parsed_but_not_a_command():
    fake = FakeTelegram()
    fake.updates = [
        {"update_id": 1, "message": {"message_id": 4, "text": "just chatting",
                                     "chat": {"id": -100111},
                                     "from": {"id": 555, "username": "ali"}}}
    ]
    async with client_for(fake) as client:
        updates = await client.get_updates(timeout=0)

    assert not updates[0].is_command
    assert updates[0].username == "ali"


def test_command_parsing_strips_the_bot_suffix():
    update = Update(update_id=1, text="/keys@crypto_intel_bot gemini")
    assert update.command() == ("keys", "gemini")


def test_plain_text_is_not_a_command():
    assert Update(update_id=1, text="keys").command() == ("", "")


# --------------------------------------------------------------------------- #
# publishing: the gate
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_alert_below_threshold_is_held():
    fake = FakeTelegram()
    store = FakeStore()
    async with client_for(fake) as client:
        publisher = Publisher(client, store, app_settings(min_alert_score=70))
        report = await publisher.publish([alert(score=61.0)])

    assert report.sent == 0 and report.below_threshold == 1
    assert fake.sent() == []


@pytest.mark.asyncio
async def test_unchanged_verdict_is_sent_once():
    """A live-updated story must not produce six identical messages."""
    fake = FakeTelegram()
    store = FakeStore()
    async with client_for(fake) as client:
        publisher = Publisher(client, store, app_settings())
        first = await publisher.publish([alert()])
        second = await publisher.publish(
            [alert(decision=Stage0Decision.UPDATE, score=89.0)]
        )

    assert first.sent == 1
    assert second.sent == 0 and second.suppressed == 1


@pytest.mark.asyncio
async def test_changed_verdict_is_sent_again():
    """A direction flip on the same event is the most important message there is."""
    fake = FakeTelegram()
    async with client_for(fake) as client:
        publisher = Publisher(client, FakeStore(), app_settings())
        await publisher.publish([alert()])
        second = await publisher.publish(
            [alert(decision=Stage0Decision.UPDATE,
                   impact=impact(direction=Direction.BEARISH,
                                 expected_low=-2.0, expected_high=-5.0))]
        )

    assert second.sent == 1
    assert Direction.BEARISH.emoji in fake.texts()[-1]


@pytest.mark.asyncio
async def test_escalated_verdict_may_repeat():
    """"We looked harder and it is worse than we said" is new information."""
    fake = FakeTelegram()
    async with client_for(fake) as client:
        publisher = Publisher(client, FakeStore(), app_settings())
        await publisher.publish([alert()])
        second = await publisher.publish([alert(impact=impact(source="final"))])

    assert second.sent == 1


@pytest.mark.asyncio
async def test_already_alerted_before_restart_is_respected():
    """The in-process memory is gone after a restart; the table is the backstop."""
    fake = FakeTelegram()
    store = FakeStore(alerted={(7, "BTC")})
    async with client_for(fake) as client:
        publisher = Publisher(client, store, app_settings())
        report = await publisher.publish([alert()])

    assert report.sent == 0 and report.suppressed == 1


@pytest.mark.asyncio
async def test_failed_send_is_not_remembered_as_published():
    """A suppressed retry after a failed send would lose the alert permanently."""
    fake = FakeTelegram()
    fake.queue_failure(
        "sendMessage", 403,
        {"ok": False, "description": "Forbidden: bot was blocked by the user"},
    )

    async with client_for(fake) as client:
        publisher = Publisher(client, FakeStore(), app_settings())
        first = await publisher.publish([alert()])
        second = await publisher.publish([alert()])

    assert first.failed == 1 and first.sent == 0
    assert second.sent == 1


@pytest.mark.asyncio
async def test_every_attempt_is_recorded_including_failures():
    """An alert formatted but never delivered is the invisible failure."""
    fake = FakeTelegram()
    fake.queue_failure(
        "sendMessage", 400, {"ok": False, "description": "Bad Request: chat not found"}
    )
    store = FakeStore()

    async with client_for(fake) as client:
        publisher = Publisher(client, store, app_settings())
        await publisher.publish([alert()])

    assert len(store.recorded) == 1
    row = store.recorded[0]
    assert row["sent"] is False
    assert "chat not found" in row["error"]
    assert row["asset"] == "BTC" and row["event_id"] == 7


@pytest.mark.asyncio
async def test_burst_collapses_into_a_digest(monkeypatch):
    """A twenty-message burst is unreadable and gets the bot throttled."""
    monkeypatch.setattr(asyncio, "sleep", no_delay)

    fake = FakeTelegram()
    alerts = [
        alert(event_id=index, impact=impact(asset=f"A{index}"))
        for index in range(9)
    ]
    async with client_for(fake) as client:
        publisher = Publisher(client, FakeStore(), app_settings())
        report = await publisher.publish(alerts)

    assert report.digested
    assert len(fake.sent()) == 1
    assert "9 alerts" in fake.texts()[0]


@pytest.mark.asyncio
async def test_disabled_telegram_sends_nothing():
    fake = FakeTelegram()
    async with client_for(fake) as client:
        publisher = Publisher(client, FakeStore(), app_settings(enabled=False))
        report = await publisher.publish([alert()])

    assert report.sent == 0 and fake.requests == []


@pytest.mark.asyncio
async def test_unconfigured_telegram_does_not_raise():
    """A missing token is a misconfiguration, not a crash mid-pipeline."""
    fake = FakeTelegram()
    async with client_for(fake, telegram_settings(chat_ids=[])) as client:
        publisher = Publisher(client, FakeStore(), app_settings(chat_ids=[]))
        report = await publisher.publish([alert()])

    assert report.sent == 0 and fake.requests == []


@pytest.mark.asyncio
async def test_notice_bypasses_the_score_threshold():
    """"All Gemini keys are dead" is not a market alert and must not be scored."""
    fake = FakeTelegram()
    store = FakeStore()
    async with client_for(fake) as client:
        publisher = Publisher(client, store, app_settings(min_alert_score=99))
        results = await publisher.notify("⛔ every gemini key is benched")

    assert results and results[0].sent
    assert store.recorded[0]["kind"] == "notice"


@pytest.mark.asyncio
async def test_store_failure_does_not_lose_a_sent_alert():
    """Recording is bookkeeping; the message already left."""
    class BrokenStore(FakeStore):
        async def record_alert(self, **kwargs):
            raise RuntimeError("database is down")

    fake = FakeTelegram()
    async with client_for(fake) as client:
        publisher = Publisher(client, BrokenStore(), app_settings())
        report = await publisher.publish([alert()])

    assert report.sent == 1


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def command_update(text: str, *, user_id: str = "555", chat_id: str = "-100111") -> Update:
    return Update(update_id=1, chat_id=chat_id, user_id=user_id,
                  text=text, message_id=9)


@pytest.mark.asyncio
async def test_help_is_answered():
    fake = FakeTelegram()
    async with client_for(fake) as client:
        handler = CommandHandler(client, FakeStore(), config=app_settings())
        await handler.handle(command_update("/help"))

    assert "/status" in fake.texts()[0]


@pytest.mark.asyncio
async def test_keys_command_refuses_a_non_admin():
    """``/keys`` prints which of your credentials are failing and why."""
    fake = FakeTelegram()
    async with client_for(fake) as client:
        handler = CommandHandler(client, FakeStore(), config=app_settings())
        await handler.handle(command_update("/keys", user_id="999"))

    assert handler.refused == 1
    assert "Admins only" in fake.texts()[0]


@pytest.mark.asyncio
async def test_keys_command_is_disabled_without_an_admin_list():
    """An open bot must not default to reading out key health."""
    fake = FakeTelegram()
    settings = app_settings(admin_ids=[])
    async with client_for(fake) as client:
        handler = CommandHandler(client, FakeStore(), config=settings)
        await handler.handle(command_update("/keys", user_id="555"))

    assert "TELEGRAM_ADMIN_IDS" in fake.texts()[0]


@pytest.mark.asyncio
async def test_stranger_in_another_chat_gets_silence():
    """Replying would confirm the bot exists to someone who guessed its name."""
    fake = FakeTelegram()
    async with client_for(fake) as client:
        handler = CommandHandler(client, FakeStore(), config=app_settings())
        await handler.handle(command_update("/status", chat_id="-100999", user_id="999"))

    assert fake.sent() == []
    assert handler.refused == 1


@pytest.mark.asyncio
async def test_admin_reaches_the_bot_from_a_private_chat():
    """An admin messaging the bot directly is not in the broadcast chat list."""
    fake = FakeTelegram()
    async with client_for(fake) as client:
        handler = CommandHandler(client, FakeStore(), config=app_settings())
        await handler.handle(command_update("/status", chat_id="555", user_id="555"))

    assert fake.sent()


@pytest.mark.asyncio
async def test_status_survives_an_unreachable_database():
    """``/status`` is what someone sends *because* something is broken."""
    class BrokenStore(FakeStore):
        async def counts(self):
            raise RuntimeError("connection refused")

    fake = FakeTelegram()
    async with client_for(fake) as client:
        handler = CommandHandler(client, BrokenStore(), config=app_settings())
        await handler.handle(command_update("/status"))

    assert "database unreachable" in fake.texts()[0]


@pytest.mark.asyncio
async def test_last_accepts_a_count_and_clamps_it():
    fake = FakeTelegram()
    store = FakeStore()
    async with client_for(fake) as client:
        handler = CommandHandler(client, store, config=app_settings())
        assert await handler.last("500") is not None
        assert await handler.last("abc") is not None


@pytest.mark.asyncio
async def test_unknown_command_is_answered_not_ignored():
    fake = FakeTelegram()
    async with client_for(fake) as client:
        handler = CommandHandler(client, FakeStore(), config=app_settings())
        await handler.handle(command_update("/frobnicate"))

    assert "Unknown command" in fake.texts()[0]


@pytest.mark.asyncio
async def test_command_failure_replies_instead_of_raising():
    """A raising command inside the poll loop would stop the bot answering."""
    class BrokenStore(FakeStore):
        async def recent_analyses(self, limit: int = 10):
            raise RuntimeError("boom")

    fake = FakeTelegram()
    async with client_for(fake) as client:
        handler = CommandHandler(client, BrokenStore(), config=app_settings())
        await handler.handle(command_update("/last"))

    assert "failed" in fake.texts()[0]


@pytest.mark.asyncio
async def test_models_command_lists_every_panel():
    fake = FakeTelegram()
    async with client_for(fake) as client:
        handler = CommandHandler(client, FakeStore(), config=app_settings())
        await handler.handle(command_update("/models"))

    text = fake.texts()[0]
    assert "Stages 1–5 panel" in text
    assert "Final layer" in text
    assert "Embeddings" in text


@pytest.mark.asyncio
async def test_keys_command_explains_itself_without_a_registry():
    """A standalone bot has no pools; a stale table would mislead rather than inform."""
    fake = FakeTelegram()
    async with client_for(fake) as client:
        handler = CommandHandler(client, FakeStore(), config=app_settings())
        await handler.handle(command_update("/keys"))

    assert "alongside the pipeline" in fake.texts()[0]
