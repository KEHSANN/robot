"""Logging setup.

Keys must never reach the logs, so the formatter redacts anything that looks
like a provider secret even if a stray f-string interpolates one.

:func:`redact` is public because logs are not the only way text leaves the
process: provider error bodies get stored on the key's health record and shown by
``/keys``, and those need the same treatment.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys

from services.config import settings

#: Prefixes used by the three providers, plus the Telegram bot token shape.
#:
#: The Telegram pattern deliberately does not start with ``\b``. A bot token
#: reaches this function almost exclusively inside an API URL —
#: ``https://api.telegram.org/bot<TOKEN>/sendMessage`` — where the token is
#: preceded directly by the letters ``bot``, so there is no word boundary to
#: match and a ``\b`` anchor would skip the one place the token actually appears.
#: The negative lookbehind is there instead, so a longer digit run is left alone
#: rather than being matched from the middle.
_SECRET_RE = re.compile(
    r"(AIza[0-9A-Za-z_\-]{10,}"
    r"|gsk_[0-9A-Za-z]{10,}"
    r"|nvapi-[0-9A-Za-z_\-]{10,}"
    r"|AQ\.[0-9A-Za-z_\-]{10,}"
    r"|(?<![0-9])\d{8,10}:[A-Za-z0-9_\-]{30,})"
)


def mask_secret(secret: str) -> str:
    """Mask a credential so it stays identifiable but unusable.

    The visible ends are not enough on their own. Every Gemini key begins with
    ``AIzaSy`` and a pool's keys are usually issued minutes apart in the same
    console, so prefix-plus-tail can print identically for three different keys —
    which makes the health table useless at the one moment it is needed, when one
    key of three has died. The short digest is what guarantees two keys never
    share a mask; the tail is kept because provider consoles list keys by suffix,
    so it is what lets someone find the failing key to rotate it.
    """
    if not secret:
        return "<empty>"
    tag = hashlib.blake2s(secret.encode("utf-8", "replace"), digest_size=2).hexdigest()
    head = secret[:6] if len(secret) > 12 else secret[:2]
    return f"{head}…{secret[-4:]}#{tag}"


def redact(text: str) -> str:
    """Mask anything shaped like a provider secret, keeping it identifiable.

    Uses the same mask as :func:`services.keypool.fingerprint`, so a redacted log
    line and a key's health record name the same key the same way.
    """
    return _SECRET_RE.sub(lambda match: mask_secret(match.group(0)), text)


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup_logging(level: str | None = None) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        RedactingFormatter(
            fmt="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, (level or settings.log_level).upper(), logging.INFO))

    # httpx logs a line per request at INFO, which drowns out the pipeline.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


__all__ = ["setup_logging", "RedactingFormatter", "mask_secret", "redact"]
