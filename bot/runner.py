"""The command-polling loop.

Separate from :mod:`bot.commands` because the loop's concerns are all about
staying up rather than about what any command means. Three of them:

**A poll failure must not end the loop.** Telegram's edge has outages, and a bot
that exits on the first one stops answering ``/status`` exactly when someone wants
to ask why nothing is arriving. Failures back off and retry, and the backoff
resets on the first successful poll.

**Only one poller may run per token.** Telegram answers a second ``getUpdates``
by cancelling the first, so two processes polling one token starve each other. The
loop therefore deletes any webhook on startup and expects to be the only poller;
running the bot twice is an operator error the log calls out.

**Commands run sequentially.** They are cheap, they mostly read, and answering
them in order means a slow ``/status`` cannot interleave its reply with another
command's.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random

from bot.commands import CommandHandler
from bot.telegram import TelegramClient, TelegramError

log = logging.getLogger(__name__)

#: How long each long-poll holds the connection open. Long enough that an idle
#: bot makes a handful of requests a minute, short enough that shutdown is quick.
POLL_TIMEOUT = 25

MAX_BACKOFF = 60.0


class BotService:
    """Polls for commands until cancelled."""

    def __init__(self, client: TelegramClient, handler: CommandHandler) -> None:
        self.client = client
        self.handler = handler
        self.polls = 0
        self.errors = 0
        self._stop = asyncio.Event()

    async def start(self, *, drop_pending: bool = True) -> None:
        """Prepare to poll: drop a webhook, and skip whatever queued while down."""
        await self.client.delete_webhook()
        if drop_pending:
            await self.client.drop_pending_updates()

        try:
            me = await self.client.get_me()
        except TelegramError as exc:
            log.error("telegram token rejected: %s", exc.description)
            raise
        log.info("bot @%s ready", me.get("username", "?"))

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        """The loop. Returns when :meth:`stop` is called or the task is cancelled."""
        backoff = 1.0

        while not self._stop.is_set():
            try:
                updates = await self.client.get_updates(timeout=POLL_TIMEOUT)
            except asyncio.CancelledError:
                raise
            except TelegramError as exc:
                self.errors += 1
                if "conflict" in exc.description.lower():
                    # Telegram's own words for "someone else is polling this
                    # token". Retrying forever would flap between the two.
                    log.error(
                        "another process is polling this bot token — "
                        "stop it before this one can answer commands"
                    )
                log.warning("poll failed: %s (retry in %.0fs)", exc.description, backoff)
                await self._sleep(backoff)
                backoff = min(MAX_BACKOFF, backoff * 2) * random.uniform(0.9, 1.1)
                continue
            except Exception:
                self.errors += 1
                log.exception("poll raised (retry in %.0fs)", backoff)
                await self._sleep(backoff)
                backoff = min(MAX_BACKOFF, backoff * 2)
                continue

            self.polls += 1
            backoff = 1.0

            for update in updates:
                if self._stop.is_set():
                    break
                try:
                    await self.handler.handle(update)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A command that fails is a bug to fix, not a reason to stop
                    # answering the next one.
                    log.exception("handling update %s failed", update.update_id)

        log.info("bot stopped after %d polls (%d errors, %d commands)",
                 self.polls, self.errors, self.handler.handled)

    async def _sleep(self, seconds: float) -> None:
        """Sleep, but wake immediately on shutdown."""
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)


__all__ = ["MAX_BACKOFF", "POLL_TIMEOUT", "BotService"]
