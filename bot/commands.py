"""Bot commands.

A small, deliberately read-only surface: the commands report what the system is
doing and what state its keys are in. Nothing here changes the pipeline's
behaviour, because a chat message is a weak place to put an irreversible action —
the one exception is ``/reset``, which clears cooldowns and is guarded by the
admin list.

Authorisation is by ``TELEGRAM_ADMIN_IDS``. It matters more than it looks: ``/keys``
prints key fingerprints and provider error text, which tells anyone reading it
which of your credentials are failing and why. An empty admin list means anyone
who can reach the bot can run any command, so the sensitive commands refuse to
run at all in that case rather than defaulting to open.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from bot.format import (
    esc,
    format_keys,
    format_recent,
    format_status,
)
from bot.telegram import TelegramClient, TelegramError, Update
from database.base import Store
from services import models as models_module
from services.config import Settings, settings as global_settings
from services.keypool import KeyPoolRegistry

log = logging.getLogger(__name__)

HELP = """<b>Crypto event intelligence</b>

/status — pipeline counters, stored rows, key pools
/last [n] — the newest verdicts (default 5, max 20)
/keys — per-key health for every provider 🔒
/models — which model runs at which stage
/ping — check the bot is alive
/help — this message

🔒 = admin only. Alerts arrive automatically; there is nothing to subscribe to."""

#: Commands that expose credential state or change it. Refused when no admin list
#: is configured — an open bot must not be able to read out key health.
_SENSITIVE = {"keys", "reset"}


class CommandHandler:
    """Answers commands. One instance per bot process."""

    def __init__(
        self,
        client: TelegramClient,
        store: Store | None = None,
        registry: KeyPoolRegistry | None = None,
        config: Settings | None = None,
        stats_source=None,
    ) -> None:
        self.settings = config or global_settings
        self.client = client
        self.store = store
        self.registry = registry
        #: Callable returning the live pipeline stats dict, when a pipeline is
        #: running in this process. ``/status`` degrades to stored counts without.
        self.stats_source = stats_source
        self.started = datetime.now(timezone.utc)
        self.handled = 0
        self.refused = 0

    # ------------------------------------------------------------------- auth

    def is_admin(self, user_id: str) -> bool:
        admins = self.settings.telegram.admin_ids
        return bool(admins) and str(user_id) in {str(admin) for admin in admins}

    def _authorised(self, update: Update, command: str) -> tuple[bool, str]:
        """Whether this user may run this command, and why not if they may not."""
        admins = self.settings.telegram.admin_ids

        if command in _SENSITIVE:
            if not admins:
                return False, (
                    "🔒 This command is disabled because TELEGRAM_ADMIN_IDS is "
                    "empty. Set it to your Telegram user id to enable it."
                )
            if not self.is_admin(update.user_id):
                return False, "🔒 Admins only."
            return True, ""

        # Non-sensitive commands are open to the configured chats. A stranger who
        # found the bot's username gets nothing.
        allowed = {str(chat) for chat in self.settings.telegram.chat_ids}
        if allowed and update.chat_id and update.chat_id not in allowed:
            if not self.is_admin(update.user_id):
                return False, ""  # silence: do not confirm the bot exists
        return True, ""

    # ---------------------------------------------------------------- dispatch

    async def handle(self, update: Update) -> None:
        """Route one update. Never raises: a bad command must not stop polling."""
        if not update.is_command or not update.chat_id:
            return

        command, argument = update.command()
        allowed, refusal = self._authorised(update, command)
        if not allowed:
            self.refused += 1
            log.warning("refused /%s from user %s in chat %s",
                        command, update.user_id or "?", update.chat_id)
            if refusal:
                await self._reply(update, refusal)
            return

        self.handled += 1
        try:
            text = await self._run(command, argument, update)
        except TelegramError:
            raise
        except Exception as exc:
            log.exception("command /%s failed", command)
            text = f"⚠️ <code>/{esc(command)}</code> failed: {esc(type(exc).__name__)}"

        if text:
            await self._reply(update, text)

    async def _run(self, command: str, argument: str, update: Update) -> str:
        if command in ("start", "help"):
            return HELP
        if command == "ping":
            return "🏓 alive"
        if command == "status":
            return await self.status()
        if command == "keys":
            return self.keys()
        if command == "last":
            return await self.last(argument)
        if command == "models":
            return self.models()
        if command == "reset":
            return await self.reset()
        return (
            f"Unknown command <code>/{esc(command)}</code>. "
            "Send /help for the list."
        )

    async def _reply(self, update: Update, text: str) -> None:
        await self.client.send(update.chat_id, text, reply_to=update.message_id)

    # ---------------------------------------------------------------- commands

    async def status(self) -> str:
        stats: dict = {}
        if self.stats_source is not None:
            try:
                live = self.stats_source()
                stats = live if isinstance(live, dict) else live.as_dict()
            except Exception:
                log.exception("stats source failed")

        counts: dict = {}
        if self.store is not None:
            try:
                counts = await self.store.counts()
            except Exception:
                log.exception("counts failed")
                counts = {"(database unreachable)": 0}

        keys = self.registry.snapshot() if self.registry is not None else {}
        return format_status(stats, counts, keys, started=self.started)

    def keys(self) -> str:
        """Per-key health.

        Reads the live registry when the pipeline shares this process. A bot
        running standalone has no registry — and the stored table is a snapshot
        of a *different* process's pools, which would be misleading rather than
        merely stale, so it says so instead of guessing.
        """
        if self.registry is None:
            return (
                "🔑 Key health is only available when the bot runs alongside the "
                "pipeline (<code>run.py bot --with-pipeline</code>). "
                "Use <code>run.py keys</code> on the host for a standalone check."
            )
        return format_keys(self.registry.snapshot())

    async def last(self, argument: str) -> str:
        limit = 5
        if argument:
            digits = "".join(character for character in argument if character.isdigit())
            if digits:
                limit = max(1, min(20, int(digits)))

        if self.store is None:
            return "No database configured."
        rows = await self.store.recent_analyses(limit)
        return format_recent(rows, limit=limit)

    def models(self) -> str:
        """Which model serves which role, so a wrong slug is visible in chat.

        Worth having in the bot rather than only in ``doctor``: the model slugs
        are overridable by environment variable, and a typo there does not fail
        loudly — it fails as one panel member timing out on every article.
        """
        lines = ["🧠 <b>Model assignment</b>", "", "<b>Stages 1–5 panel</b>"]
        lines += [
            f"• <code>{esc(spec.id)}</code> <i>{esc(spec.provider)}</i>"
            for spec in models_module.PANEL
        ]

        lines += ["", "<b>Final layer</b> (critical events only)"]
        lines += [
            f"• <code>{esc(spec.id)}</code> <i>{esc(spec.provider)}</i>"
            for spec in models_module.FINAL_PANEL
        ]

        lines += ["", "<b>Stage 0 fact extraction</b> (first that answers)"]
        lines += [
            f"• <code>{esc(spec.id)}</code> <i>{esc(spec.provider)}</i>"
            for spec in models_module.EXTRACTORS
        ]

        embed = models_module.EMBED_MODEL
        lines += [
            "",
            f"<b>Embeddings</b>: <code>{esc(embed.id)}</code> "
            f"<i>{esc(embed.provider)}</i>",
        ]
        return "\n".join(lines)

    async def reset(self) -> str:
        """Clear key cooldowns.

        The one state-changing command, because the situation it exists for is
        real: a key was rotated in the provider console and the pool is still
        benching it on the old key's failures.
        """
        if self.registry is None:
            return "No key registry in this process."
        await self.registry.reset()
        log.warning("key pools reset by admin command")
        return "🔄 Key cooldowns cleared. All configured keys are healthy again."


__all__ = ["HELP", "CommandHandler"]
