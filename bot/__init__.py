"""Telegram output.

The layering: :mod:`bot.format` turns a verdict into text, :mod:`bot.telegram`
gets text to Telegram, :mod:`bot.publisher` decides which verdicts are worth
sending, and :mod:`bot.commands` / :mod:`bot.runner` answer questions about the
system. Formatting is deliberately free of I/O so an alert's layout can be tested
without a network or a token.
"""

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
from bot.publisher import PublishReport, Publisher
from bot.runner import BotService
from bot.telegram import SendResult, TelegramClient, TelegramError, Update

__all__ = [
    "BotService",
    "CommandHandler",
    "PublishReport",
    "Publisher",
    "SendResult",
    "TelegramClient",
    "TelegramError",
    "Update",
    "format_alert",
    "format_digest",
    "format_keys",
    "format_plain",
    "format_recent",
    "format_status",
    "split_message",
]
