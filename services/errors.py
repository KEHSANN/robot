"""Failure classification for provider calls.

The single most important decision in a multi-key setup is whether a failure is
the *key's* fault. Get it wrong in one direction and a malformed prompt burns
through every key you own; get it wrong in the other and a dead key is retried
forever.

Three providers report the same conditions differently, and not always with an
honest status code — Gemini in particular returns HTTP 400 for an invalid API
key — so classification looks at the body as well as the status.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum


class Disposition(str, Enum):
    """What the caller should do next."""

    #: The key is implicated. Bench it and try the next key in the pool.
    RETRY_OTHER_KEY = "retry_other_key"
    #: A provider-side blip unrelated to the key. Briefly retry the same key.
    RETRY_SAME_KEY = "retry_same_key"
    #: The request itself is wrong (bad model id, malformed body, oversized
    #: prompt). No key will fix it, so fail immediately instead of burning the
    #: pool.
    FATAL = "fatal"


class KeyFault(str, Enum):
    """How the key should be penalised, if at all."""

    NONE = "none"
    #: Rejected as invalid/revoked. Benched for a long time, then probed once.
    INVALID = "invalid"
    #: Rate limited. Short cooldown, honouring Retry-After when supplied.
    RATE_LIMITED = "rate_limited"
    #: Quota or credit exhausted. Long cooldown; these usually reset daily.
    QUOTA = "quota"
    #: Provider-side error while using this key. Brief cooldown.
    TRANSIENT = "transient"


@dataclass
class ProviderError(Exception):
    """A normalised provider failure."""

    provider: str
    message: str
    disposition: Disposition
    fault: KeyFault = KeyFault.NONE
    status: int | None = None
    retry_after: float | None = None
    model: str | None = None

    def __str__(self) -> str:  # pragma: no cover - display only
        bits = [self.provider]
        if self.model:
            bits.append(self.model)
        if self.status is not None:
            bits.append(f"HTTP {self.status}")
        bits.append(self.disposition.value)
        if self.fault is not KeyFault.NONE:
            bits.append(f"fault={self.fault.value}")
        return f"[{' '.join(bits)}] {self.message}"

    @property
    def is_key_fault(self) -> bool:
        return self.fault is not KeyFault.NONE


# --------------------------------------------------------------------------- #
# body sniffing
# --------------------------------------------------------------------------- #

# Gemini answers an invalid key with 400 + API_KEY_INVALID, so a bare status
# code is not enough to tell a bad key from a bad request.
_INVALID_KEY_PATTERNS = (
    "api key not valid",
    "api_key_invalid",
    "api key expired",
    "invalid api key",
    "invalid_api_key",
    "invalid authentication",
    "incorrect api key",
    "unauthorized",
    "permission_denied",
    "permission denied",
    "authentication_error",
    "no access to model",
    "consumer has been suspended",
)

_QUOTA_PATTERNS = (
    "quota",
    "resource_exhausted",
    "insufficient_quota",
    "billing",
    "credit balance",
    "out of credits",
    "exceeded your current",
    "free tier",
    "spend limit",
)

_RATE_LIMIT_PATTERNS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "requests per",
    "tokens per",
    "please try again in",
)

# A wrong/retired model id must never be treated as a key problem.
_BAD_REQUEST_PATTERNS = (
    "model_not_found",
    "model not found",
    "does not exist",
    "decommissioned",
    "is not supported",
    "unsupported",
    "invalid_request_error",
    "context_length_exceeded",
    "string too long",
    "token limit",
    "too large",
    "invalid json",
    "invalid value at",
    "unknown name",
)

_OVERLOADED_PATTERNS = (
    "overloaded",
    "unavailable",
    "capacity",
    "try again later",
    "internal error",
    "internal server error",
    "bad gateway",
    "timeout",
    "deadline",
)


def _contains(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(needle in haystack for needle in needles)


_RETRY_SECONDS_RE = re.compile(r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|s|m)?", re.I)


def parse_retry_after(headers: dict[str, str] | None, body: str) -> float | None:
    """Extract a retry delay from a Retry-After header or the message body.

    Groq embeds the delay in prose ("Please try again in 2.34s") and Gemini uses
    a RetryInfo detail block, so both are checked when the header is absent.
    """
    if headers:
        lowered = {key.lower(): value for key, value in headers.items()}
        raw = lowered.get("retry-after") or lowered.get("x-ratelimit-reset-requests")
        if raw:
            try:
                return max(0.0, float(str(raw).rstrip("smSM")))
            except ValueError:
                pass

    match = _RETRY_SECONDS_RE.search(body)
    if match:
        value = float(match.group(1))
        unit = (match.group(2) or "s").lower()
        if unit == "ms":
            return value / 1000.0
        if unit == "m":
            return value * 60.0
        return value

    # Gemini: {"error": {"details": [{"@type": ".../RetryInfo",
    #                                "retryDelay": "31s"}]}}
    if "retrydelay" in body.lower():
        delay = re.search(r'"retryDelay"\s*:\s*"?([0-9.]+)s"?', body)
        if delay:
            return float(delay.group(1))
    return None


def extract_provider_message(body: str) -> str:
    """Pull the human-readable error out of a provider's JSON envelope."""
    snippet = body.strip()
    if not snippet:
        return "empty response body"
    try:
        parsed = json.loads(snippet)
    except (ValueError, TypeError):
        return snippet[:400]

    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("status") or error.get("code")
            if message:
                return str(message)[:400]
        if isinstance(error, str):
            return error[:400]
        for key in ("message", "detail", "title", "warnings"):
            if parsed.get(key):
                return str(parsed[key])[:400]
    return snippet[:400]


def classify_http(
    provider: str,
    status: int,
    body: str,
    headers: dict[str, str] | None = None,
    model: str | None = None,
) -> ProviderError:
    """Turn a non-2xx provider response into a :class:`ProviderError`."""
    lowered = (body or "").lower()
    message = extract_provider_message(body or "")
    retry_after = parse_retry_after(headers, body or "")

    def error(
        disposition: Disposition,
        fault: KeyFault = KeyFault.NONE,
        *,
        delay: float | None = None,
    ) -> ProviderError:
        return ProviderError(
            provider=provider,
            message=message,
            disposition=disposition,
            fault=fault,
            status=status,
            retry_after=delay if delay is not None else retry_after,
            model=model,
        )

    if status in (401, 407):
        return error(Disposition.RETRY_OTHER_KEY, KeyFault.INVALID)

    if status == 403:
        # Could be a revoked key or a key without access to this model. Both are
        # key-scoped, so another key is worth trying.
        return error(Disposition.RETRY_OTHER_KEY, KeyFault.INVALID)

    if status == 429:
        fault = KeyFault.QUOTA if _contains(lowered, _QUOTA_PATTERNS) else KeyFault.RATE_LIMITED
        return error(Disposition.RETRY_OTHER_KEY, fault)

    if status in (402, 413) or status == 451:
        if status == 413:
            return error(Disposition.FATAL)  # payload too large: our fault
        return error(Disposition.RETRY_OTHER_KEY, KeyFault.QUOTA)

    if status == 400:
        # Gemini reports an invalid key here, so the body decides.
        if _contains(lowered, _INVALID_KEY_PATTERNS):
            return error(Disposition.RETRY_OTHER_KEY, KeyFault.INVALID)
        if _contains(lowered, _QUOTA_PATTERNS):
            return error(Disposition.RETRY_OTHER_KEY, KeyFault.QUOTA)
        return error(Disposition.FATAL)

    if status == 404:
        # Wrong model id or wrong base URL — no other key can help.
        return error(Disposition.FATAL)

    if status in (408, 409, 425):
        return error(Disposition.RETRY_SAME_KEY, KeyFault.TRANSIENT)

    if status in (422, 501):
        return error(Disposition.FATAL)

    if status >= 500:
        return error(Disposition.RETRY_SAME_KEY, KeyFault.TRANSIENT)

    if status in (405, 406, 415):
        return error(Disposition.FATAL)

    # Unmapped 4xx: assume the request is at fault rather than spending keys.
    if 400 <= status < 500:
        if _contains(lowered, _RATE_LIMIT_PATTERNS):
            return error(Disposition.RETRY_OTHER_KEY, KeyFault.RATE_LIMITED)
        return error(Disposition.FATAL)

    return error(Disposition.RETRY_SAME_KEY, KeyFault.TRANSIENT)


def classify_exception(provider: str, exc: BaseException, model: str | None = None) -> ProviderError:
    """Turn a transport-level exception into a :class:`ProviderError`."""
    name = type(exc).__name__
    text = f"{name}: {exc}".strip()
    lowered = text.lower()

    # Network/timeout faults are provider- or route-scoped, not key-scoped, but
    # a key that keeps timing out still deserves a brief cooldown so traffic
    # shifts elsewhere.
    if "timeout" in lowered or "timedout" in name.lower():
        return ProviderError(
            provider=provider,
            message=text,
            disposition=Disposition.RETRY_SAME_KEY,
            fault=KeyFault.TRANSIENT,
            model=model,
        )

    if any(token in lowered for token in ("connect", "dns", "ssl", "socket", "network", "proxy", "resolve")):
        return ProviderError(
            provider=provider,
            message=text,
            disposition=Disposition.RETRY_SAME_KEY,
            fault=KeyFault.TRANSIENT,
            model=model,
        )

    if _contains(lowered, _OVERLOADED_PATTERNS):
        return ProviderError(
            provider=provider,
            message=text,
            disposition=Disposition.RETRY_SAME_KEY,
            fault=KeyFault.TRANSIENT,
            model=model,
        )

    return ProviderError(
        provider=provider,
        message=text,
        disposition=Disposition.RETRY_SAME_KEY,
        fault=KeyFault.TRANSIENT,
        model=model,
    )


__all__ = [
    "Disposition",
    "KeyFault",
    "ProviderError",
    "classify_http",
    "classify_exception",
    "extract_provider_message",
    "parse_retry_after",
]
