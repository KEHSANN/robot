"""Telegram Bot API client.

Only the four methods this system needs — ``sendMessage``, ``getUpdates``,
``getMe``, ``deleteWebhook`` — so there is no framework to work around when
Telegram behaves unhelpfully. Three ways it does, and what the client does about
each:

**Rate limits are per chat and per bot, and exceeding them returns 429 with a
``retry_after``.** Retrying immediately extends the block rather than clearing it,
so the client paces sends per chat and honours the delay Telegram asks for.

**A malformed entity rejects the whole message.** ``parse_mode=HTML`` means one
stray ``<`` in a headline turns a valid alert into a 400. The formatter escapes
everything, but escaping cannot anticipate every way a publisher writes a title,
so a parse rejection retries once as plain text: a plain alert is worth much more
than a lost one.

**Long polling ends in a timeout, not an error.** ``getUpdates`` holding a
connection open for 30 seconds is the normal case, so a read timeout there is
success with no updates — not a fault to log and back off from.

The token is treated as a secret throughout: it lives in the URL, so error text
is redacted before it is logged or stored.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from bot.format import MAX_MESSAGE, split_message
from services.config import TelegramSettings, settings as global_settings
from services.logging_setup import redact

log = logging.getLogger(__name__)

API_ROOT = "https://api.telegram.org"

#: Telegram allows about one message per second per chat before throttling.
#: A little over that keeps bursts from being queued behind a 429.
PER_CHAT_INTERVAL = 1.05

#: Cap on how long a 429 will be waited out inline. Longer than this and the send
#: is abandoned: holding the pipeline for two minutes to deliver one alert costs
#: more than the alert is worth.
MAX_RETRY_AFTER = 60.0

_PARSE_FAILURE_MARKERS = (
    "can't parse entities",
    "cant parse entities",
    "unsupported start tag",
    "unclosed tag",
    "wrong entity",
    "reserved and must be escaped",
)


class TelegramError(RuntimeError):
    """A Bot API call that failed in a way retrying will not fix."""

    def __init__(self, method: str, description: str, code: int | None = None) -> None:
        super().__init__(f"{method}: {description}")
        self.method = method
        self.description = description
        self.code = code

    @property
    def is_parse_failure(self) -> bool:
        lowered = self.description.lower()
        return any(marker in lowered for marker in _PARSE_FAILURE_MARKERS)

    @property
    def is_permanent(self) -> bool:
        """Whether the chat itself is the problem, so retrying is pointless."""
        lowered = self.description.lower()
        return any(
            marker in lowered
            for marker in (
                "chat not found",
                "bot was blocked",
                "bot was kicked",
                "user is deactivated",
                "not enough rights",
                "chat_id is empty",
                "group chat was upgraded",
            )
        )


@dataclass
class SendResult:
    """Outcome of one delivery attempt, for the alert audit row."""

    chat_id: str
    sent: bool
    message_id: int | None = None
    error: str = ""
    #: True when HTML was rejected and the plain-text retry went out instead.
    downgraded: bool = False
    parts: int = 1


@dataclass
class Update:
    """One ``getUpdates`` entry, flattened to what the command layer needs."""

    update_id: int
    chat_id: str = ""
    user_id: str = ""
    username: str = ""
    text: str = ""
    message_id: int | None = None
    raw: dict = field(default_factory=dict)

    @property
    def is_command(self) -> bool:
        return self.text.startswith("/")

    def command(self) -> tuple[str, str]:
        """``/keys@mybot gemini`` -> ``("keys", "gemini")``."""
        if not self.is_command:
            return "", ""
        head, _, tail = self.text.partition(" ")
        name = head[1:].split("@", 1)[0].lower()
        return name, tail.strip()


class TelegramClient:
    """Minimal Bot API client with pacing, retries and a plain-text fallback."""

    def __init__(
        self,
        config: TelegramSettings | None = None,
        http: httpx.AsyncClient | None = None,
        api_root: str = API_ROOT,
    ) -> None:
        self.config = config or global_settings.telegram
        self.api_root = api_root.rstrip("/")
        self._http = http
        self._owns_http = http is None
        #: chat id -> monotonic time of the last send, for per-chat pacing
        self._last_send: dict[str, float] = {}
        self._offset: int | None = None
        self.sent_count = 0
        self.failed_count = 0

    async def __aenter__(self) -> "TelegramClient":
        if self._http is None:
            self._http = httpx.AsyncClient(
                # Read timeout must exceed the long-poll window, or every poll
                # would abort just before Telegram answers.
                timeout=httpx.Timeout(20.0, read=75.0),
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            )
        return self

    async def __aexit__(self, *_exc) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            raise RuntimeError("TelegramClient must be used as an async context manager")
        return self._http

    @property
    def configured(self) -> bool:
        return bool(self.config.bot_token)

    # ------------------------------------------------------------ raw API call

    def _url(self, method: str) -> str:
        return f"{self.api_root}/bot{self.config.bot_token}/{method}"

    async def call(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        *,
        attempts: int = 3,
        timeout: float | None = None,
    ) -> Any:
        """Invoke one Bot API method, retrying transient failures.

        Raises :class:`TelegramError` for anything Telegram rejected on the
        merits, so the caller can decide between a fallback and giving up. A 429
        is retried here because the wait is short and known; a 5xx is retried
        because Telegram's own edge returns them under load.
        """
        if not self.configured:
            raise TelegramError(method, "TELEGRAM_BOT_TOKEN is not set")

        body = {key: value for key, value in (payload or {}).items() if value is not None}
        last_error = ""

        for attempt in range(1, max(1, attempts) + 1):
            try:
                response = await self.http.post(
                    self._url(method),
                    json=body,
                    timeout=timeout if timeout is not None else None,
                )
            except httpx.TimeoutException as exc:
                last_error = f"timeout: {type(exc).__name__}"
            except httpx.HTTPError as exc:
                last_error = redact(f"{type(exc).__name__}: {exc}")
            else:
                data = self._decode(method, response)
                if data is not None:
                    return data

                if response.status_code == 429:
                    delay = self._retry_after(response)
                    if delay > MAX_RETRY_AFTER:
                        raise TelegramError(
                            method, f"rate limited for {delay:.0f}s", 429
                        )
                    log.warning("telegram %s rate limited, waiting %.1fs", method, delay)
                    await asyncio.sleep(delay)
                    continue

                description = self._description(response)
                if response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}: {description}"
                else:
                    # 400/401/403 are verdicts, not weather.
                    raise TelegramError(method, description, response.status_code)

            if attempt < attempts:
                backoff = min(8.0, 0.5 * 2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                log.debug("telegram %s attempt %d failed (%s), retrying in %.1fs",
                          method, attempt, last_error, backoff)
                await asyncio.sleep(backoff)

        raise TelegramError(method, last_error or "no response")

    def _decode(self, method: str, response: httpx.Response) -> Any:
        """The ``result`` payload, or ``None`` when the call did not succeed."""
        try:
            data = response.json()
        except ValueError:
            return None
        if isinstance(data, dict) and data.get("ok"):
            return data.get("result")
        return None

    def _description(self, response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return redact(response.text[:300]) or f"HTTP {response.status_code}"
        if isinstance(data, dict):
            return redact(str(data.get("description") or data)[:300])
        return f"HTTP {response.status_code}"

    def _retry_after(self, response: httpx.Response) -> float:
        """How long Telegram asked us to wait, from the body or the header."""
        try:
            data = response.json()
        except ValueError:
            data = {}
        if isinstance(data, dict):
            parameters = data.get("parameters")
            if isinstance(parameters, dict) and parameters.get("retry_after"):
                try:
                    return max(1.0, float(parameters["retry_after"]))
                except (TypeError, ValueError):
                    pass
        header = response.headers.get("retry-after")
        if header:
            try:
                return max(1.0, float(header))
            except ValueError:
                pass
        return 3.0

    # ---------------------------------------------------------------- pacing

    async def _pace(self, chat_id: str) -> None:
        """Space sends to one chat, so a burst is not answered with a 429."""
        last = self._last_send.get(chat_id)
        now = time.monotonic()
        if last is not None:
            gap = PER_CHAT_INTERVAL - (now - last)
            if gap > 0:
                await asyncio.sleep(gap)
        self._last_send[chat_id] = time.monotonic()

    # ------------------------------------------------------------------ send

    async def send(
        self,
        chat_id: str,
        text: str,
        *,
        parse_mode: str | None = None,
        preview: bool = False,
        plain_fallback: str | None = None,
        reply_to: int | None = None,
        silent: bool = False,
    ) -> SendResult:
        """Deliver one message, splitting it and downgrading markup as needed.

        ``plain_fallback`` is the unmarked-up version of the same content. When
        Telegram rejects the markup it is sent instead, because a readable alert
        without bold text still tells the reader what happened.
        """
        mode = parse_mode if parse_mode is not None else (self.config.parse_mode or None)
        chunks = split_message(text, MAX_MESSAGE)
        message_id: int | None = None
        downgraded = False

        for index, chunk in enumerate(chunks):
            payload = {
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": mode,
                "link_preview_options": {"is_disabled": not preview},
                "disable_notification": silent or None,
                # Only the first chunk quotes the message it answers.
                "reply_to_message_id": reply_to if index == 0 else None,
            }

            await self._pace(chat_id)
            try:
                result = await self.call("sendMessage", payload)
            except TelegramError as exc:
                if exc.is_parse_failure and mode:
                    log.warning("telegram rejected %s markup, resending as text", mode)
                    downgraded = True
                    fallback = plain_fallback if plain_fallback and len(chunks) == 1 else chunk
                    retry = await self._send_plain(chat_id, fallback, reply_to if index == 0 else None)
                    if not retry.sent:
                        self.failed_count += 1
                        return retry
                    message_id = message_id or retry.message_id
                    continue

                self.failed_count += 1
                level = log.error if not exc.is_permanent else log.warning
                level("telegram send to %s failed: %s", chat_id, exc.description)
                return SendResult(
                    chat_id=chat_id, sent=False,
                    error=exc.description[:400], parts=len(chunks),
                )

            if isinstance(result, dict) and message_id is None:
                message_id = result.get("message_id")

        self.sent_count += 1
        return SendResult(
            chat_id=chat_id, sent=True, message_id=message_id,
            downgraded=downgraded, parts=len(chunks),
        )

    async def _send_plain(
        self, chat_id: str, text: str, reply_to: int | None
    ) -> SendResult:
        """The markup-free retry. Failing here means the send is genuinely lost."""
        await self._pace(chat_id)
        try:
            result = await self.call(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": text[:MAX_MESSAGE],
                    "link_preview_options": {"is_disabled": True},
                    "reply_to_message_id": reply_to,
                },
                attempts=2,
            )
        except TelegramError as exc:
            log.error("telegram plain retry to %s failed: %s", chat_id, exc.description)
            return SendResult(chat_id=chat_id, sent=False, error=exc.description[:400])

        message_id = result.get("message_id") if isinstance(result, dict) else None
        return SendResult(chat_id=chat_id, sent=True, message_id=message_id, downgraded=True)

    async def broadcast(self, text: str, **kwargs) -> list[SendResult]:
        """Send to every configured chat. One bad chat does not stop the rest."""
        results: list[SendResult] = []
        for chat_id in self.config.chat_ids:
            results.append(await self.send(chat_id, text, **kwargs))
        return results

    # ---------------------------------------------------------------- polling

    async def get_updates(self, timeout: int = 25, limit: int = 20) -> list[Update]:
        """Long-poll for commands.

        A timeout is the expected outcome of an idle poll, so it returns an empty
        list rather than raising — otherwise an idle bot would log an error every
        25 seconds.
        """
        try:
            result = await self.call(
                "getUpdates",
                {
                    "offset": self._offset,
                    "timeout": timeout,
                    "limit": limit,
                    # Only messages matter; asking for less means Telegram does
                    # not queue callback and inline traffic we would discard.
                    "allowed_updates": ["message"],
                },
                attempts=1,
                timeout=timeout + 20.0,
            )
        except TelegramError as exc:
            if "timeout" in exc.description.lower():
                return []
            raise

        updates = [self._to_update(entry) for entry in (result or [])]
        if updates:
            # Acknowledge everything received, including entries we ignore, or
            # one unparseable update would be redelivered forever.
            self._offset = max(update.update_id for update in updates) + 1
        return updates

    def _to_update(self, entry: dict) -> Update:
        message = entry.get("message") or {}
        chat = message.get("chat") or {}
        user = message.get("from") or {}
        return Update(
            update_id=int(entry.get("update_id", 0)),
            chat_id=str(chat.get("id", "")),
            user_id=str(user.get("id", "")),
            username=str(user.get("username") or user.get("first_name") or ""),
            text=str(message.get("text") or message.get("caption") or "").strip(),
            message_id=message.get("message_id"),
            raw=entry,
        )

    async def drop_pending_updates(self) -> None:
        """Skip whatever queued while the bot was down.

        Telegram holds updates for 24 hours. Answering a day-old ``/status`` on
        startup is noise, and replaying a day of commands can be worse.
        """
        try:
            result = await self.call("getUpdates", {"offset": -1, "timeout": 0}, attempts=1)
        except TelegramError as exc:
            log.debug("could not drain pending updates: %s", exc.description)
            return
        if result:
            self._offset = max(int(entry.get("update_id", 0)) for entry in result) + 1

    # ----------------------------------------------------------------- checks

    async def get_me(self) -> dict:
        """Verify the token. Used by ``doctor`` before anything else runs."""
        result = await self.call("getMe", attempts=2)
        return result if isinstance(result, dict) else {}

    async def delete_webhook(self) -> None:
        """Long polling and a webhook are mutually exclusive; polling wins here."""
        try:
            await self.call("deleteWebhook", {"drop_pending_updates": False}, attempts=1)
        except TelegramError as exc:
            log.debug("deleteWebhook: %s", exc.description)


__all__ = [
    "API_ROOT",
    "MAX_RETRY_AFTER",
    "PER_CHAT_INTERVAL",
    "SendResult",
    "TelegramClient",
    "TelegramError",
    "Update",
]
