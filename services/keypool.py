"""Multi-key failover pools.

Each provider (gemini / groq / nvidia) gets a pool of API keys ordered by
priority. Selection always rescans from key #1, so a healthy first key keeps
serving every request and the later keys exist purely as fallbacks — rather than
round-robin, which would spread load across keys that may have different quotas.

A key moves through three states:

    HEALTHY   usable now
    COOLING   temporarily benched until `available_at` (rate limit, quota, 5xx)
    DEAD      rejected as invalid; benched for a long window, then probed once

Cooldowns grow exponentially per consecutive failure of that key and honour a
provider-supplied Retry-After. Jitter is applied because the pipeline fires five
models in parallel and they would otherwise retry in lockstep.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from services.config import (
    PROVIDER_ENV_PREFIX,
    FailoverSettings,
    read_key_pool,
    settings as global_settings,
)
from services.errors import KeyFault, ProviderError
from services.logging_setup import mask_secret, redact


class KeyStatus(str, Enum):
    HEALTHY = "healthy"
    COOLING = "cooling"
    DEAD = "dead"


class NoKeysConfigured(RuntimeError):
    """The provider has no API keys at all."""


class AllKeysUnavailable(RuntimeError):
    """Every key is benched and none freed up inside the wait budget."""

    def __init__(self, provider: str, detail: str, retry_in: float | None = None):
        super().__init__(f"[{provider}] no key available: {detail}")
        self.provider = provider
        self.retry_in = retry_in


def fingerprint(secret: str) -> str:
    """A log-safe identifier for a key: enough to tell keys apart, not to use one.

    Shares its implementation with the log redactor so one key is named the same
    way in a log line, in the health table and in a ``/keys`` reply.
    """
    return mask_secret(secret)


def safe_error_text(error: ProviderError, secret: str = "") -> str:
    """Provider error text safe to store, display and send.

    ``last_error`` does not stay inside the process: it is persisted with the
    health record and printed by ``/keys``, which goes to Telegram. Providers do
    echo the rejected credential back — Gemini takes the key as a query parameter
    and quotes the whole URL in some errors — so the known key shapes are masked,
    then the key's own secret in case its format is not one of them.
    """
    text = redact(str(error))
    if secret and secret in text:
        text = text.replace(secret, fingerprint(secret))
    return text[:500]


@dataclass
class ManagedKey:
    """One API key plus its health record."""

    provider: str
    index: int  # 1-based priority; 1 is tried first
    secret: str = field(repr=False)

    status: KeyStatus = KeyStatus.HEALTHY
    #: monotonic deadline before which this key must not be used
    available_at: float = 0.0
    consecutive_failures: int = 0

    success_count: int = 0
    failure_count: int = 0
    rate_limited_count: int = 0
    quota_count: int = 0
    invalid_count: int = 0
    total_latency: float = 0.0
    last_error: str | None = None
    last_used_wall: float | None = None
    in_flight: int = 0

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.secret)

    @property
    def label(self) -> str:
        return f"{self.provider}#{self.index}"

    def cooldown_remaining(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        return max(0.0, self.available_at - now)

    def is_available(self, now: float | None = None) -> bool:
        return self.status is not KeyStatus.DEAD and self.cooldown_remaining(now) <= 0.0

    def is_on_probation(self, now: float | None = None) -> bool:
        """A dead key whose bench time has elapsed and may be probed once."""
        return self.status is KeyStatus.DEAD and self.cooldown_remaining(now) <= 0.0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.success_count if self.success_count else 0.0

    def snapshot(self) -> dict:
        return {
            "provider": self.provider,
            "index": self.index,
            "fingerprint": self.fingerprint,
            "status": self.status.value,
            "cooldown_remaining": round(self.cooldown_remaining(), 1),
            "success": self.success_count,
            "failure": self.failure_count,
            "rate_limited": self.rate_limited_count,
            "quota": self.quota_count,
            "invalid": self.invalid_count,
            "consecutive_failures": self.consecutive_failures,
            "avg_latency": round(self.avg_latency, 3),
            "last_error": self.last_error,
            "last_used_wall": self.last_used_wall,
        }

    def restore(self, row: dict) -> bool:
        """Reload a health record written by a previous process.

        Worth doing: without it every restart re-learns that a revoked key is
        revoked by spending a live request on it, and a deploy loop turns that into
        a request per deploy per dead key.

        Three things this deliberately does not do.

        **It does not read a secret.** The persisted row has only a fingerprint —
        the key material is never written to the database — so the match is by
        fingerprint and a row that does not match is refused rather than applied to
        whichever key happened to be at that index.

        **It does not restore ``available_at``.** That is a monotonic deadline,
        meaningful only inside the process that set it; carried across a restart it
        would be an arbitrary number. The remaining cooldown is rebuilt against
        this process's clock instead.

        **It does not extend a dead key's bench time.** A key that died an hour ago
        comes back eligible for its probation probe immediately, because the point
        of probation is to notice a key that has been fixed — and re-serving the
        full cooldown on every restart would mean a restarted process never
        rediscovers a rotated credential.
        """
        if str(row.get("fingerprint") or "") != self.fingerprint:
            return False

        status = str(row.get("status") or "").upper()
        if status in KeyStatus.__members__:
            self.status = KeyStatus[status]

        remaining = float(row.get("cooldown_remaining") or 0.0)
        if remaining > 0 and self.status is KeyStatus.COOLING:
            self.available_at = time.monotonic() + remaining
        else:
            self.available_at = 0.0

        self.success_count = int(row.get("success") or 0)
        self.failure_count = int(row.get("failure") or 0)
        self.rate_limited_count = int(row.get("rate_limited") or 0)
        self.quota_count = int(row.get("quota") or 0)
        self.invalid_count = int(row.get("invalid") or 0)
        self.consecutive_failures = int(row.get("consecutive_failures") or 0)
        self.total_latency = float(row.get("avg_latency") or 0.0) * self.success_count
        self.last_error = row.get("last_error") or None
        self.last_used_wall = row.get("last_used_wall")
        return True


#: Signature of the optional observer notified whenever a key's health changes.
HealthObserver = Callable[[ManagedKey], None]


class KeyPool:
    """A priority-ordered, health-aware pool of API keys for one provider."""

    def __init__(
        self,
        provider: str,
        secrets: Sequence[str],
        config: FailoverSettings | None = None,
        observer: HealthObserver | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or global_settings.failover
        self._observer = observer
        self.keys: list[ManagedKey] = [
            ManagedKey(provider=provider, index=index, secret=secret)
            for index, secret in enumerate(secrets, start=1)
        ]
        self._lock = asyncio.Lock()

    # -- introspection ----------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def configured(self) -> bool:
        return bool(self.keys)

    def available_count(self) -> int:
        now = time.monotonic()
        return sum(1 for key in self.keys if key.is_available(now))

    def snapshot(self) -> dict:
        return {
            "provider": self.provider,
            "total": len(self.keys),
            "available": self.available_count(),
            "keys": [key.snapshot() for key in self.keys],
        }

    # -- selection --------------------------------------------------------- #

    def _select(self, now: float) -> ManagedKey | None:
        """First usable key by priority.

        Healthy keys win over probation keys regardless of index: a revoked key
        #1 must not outrank a working key #2. Within each group, lowest index
        wins, and ties break toward the key carrying the least in-flight load so
        five parallel model calls do not all pile onto one key.
        """
        usable = [key for key in self.keys if key.is_available(now)]
        if usable:
            return min(usable, key=lambda k: (k.in_flight, k.index))

        probation = [key for key in self.keys if key.is_on_probation(now)]
        if probation:
            return min(probation, key=lambda k: k.index)
        return None

    def _earliest_deadline(self, now: float) -> float | None:
        deadlines = [key.available_at for key in self.keys if key.available_at > now]
        return min(deadlines) if deadlines else None

    async def acquire(self, wait: bool = True) -> ManagedKey:
        """Reserve the highest-priority usable key.

        Callers must pair this with exactly one :meth:`report_success` or
        :meth:`report_failure` so the in-flight counter unwinds.
        """
        if not self.keys:
            raise NoKeysConfigured(
                f"No API keys configured for '{self.provider}'. Set "
                f"{PROVIDER_ENV_PREFIX.get(self.provider, self.provider.upper())}_API_KEYS "
                f"(comma separated) or {PROVIDER_ENV_PREFIX.get(self.provider, self.provider.upper())}_API_KEY_1..N."
            )

        budget = self.config.max_wait_for_key_seconds if wait else 0.0
        deadline = time.monotonic() + budget

        while True:
            async with self._lock:
                now = time.monotonic()
                key = self._select(now)
                if key is not None:
                    key.in_flight += 1
                    key.last_used_wall = time.time()
                    return key
                next_free = self._earliest_deadline(now)

            remaining = deadline - time.monotonic()
            if remaining <= 0 or next_free is None:
                detail = ", ".join(
                    f"#{k.index} {k.status.value} ({k.cooldown_remaining():.0f}s)"
                    for k in self.keys
                )
                retry_in = None if next_free is None else max(0.0, next_free - time.monotonic())
                raise AllKeysUnavailable(self.provider, detail, retry_in)

            # Sleep until the soonest key frees up, bounded by our budget.
            await asyncio.sleep(min(max(next_free - time.monotonic(), 0.05), remaining))

    # -- reporting --------------------------------------------------------- #

    async def report_success(self, key: ManagedKey, latency: float | None = None) -> None:
        async with self._lock:
            key.in_flight = max(0, key.in_flight - 1)
            key.success_count += 1
            key.consecutive_failures = 0
            key.last_error = None
            key.available_at = 0.0
            if latency is not None:
                key.total_latency += latency
            if key.status is not KeyStatus.HEALTHY:
                # A probed dead key that works is fully reinstated.
                key.status = KeyStatus.HEALTHY
                self._notify(key)

    async def report_failure(self, key: ManagedKey, error: ProviderError) -> None:
        """Apply a penalty to ``key`` based on the classified failure."""
        async with self._lock:
            key.in_flight = max(0, key.in_flight - 1)

            if error.fault is KeyFault.NONE:
                # The request was at fault (bad model id, malformed prompt).
                # Penalising the key here is what burns a whole pool on one typo.
                return

            key.failure_count += 1
            key.consecutive_failures += 1
            key.last_error = safe_error_text(error, key.secret)

            if error.fault is KeyFault.INVALID:
                key.invalid_count += 1
                key.status = KeyStatus.DEAD
                key.available_at = time.monotonic() + self.config.dead_retry_seconds
            else:
                if error.fault is KeyFault.RATE_LIMITED:
                    key.rate_limited_count += 1
                elif error.fault is KeyFault.QUOTA:
                    key.quota_count += 1
                key.status = KeyStatus.COOLING
                key.available_at = time.monotonic() + self._cooldown_for(key, error)

            self._notify(key)

    def _cooldown_for(self, key: ManagedKey, error: ProviderError) -> float:
        cfg = self.config
        attempt = max(1, key.consecutive_failures)

        if error.fault is KeyFault.QUOTA:
            base = cfg.quota_cooldown_seconds
        else:
            # Exponential backoff per consecutive failure of this specific key.
            base = cfg.cooldown_base_seconds * (2 ** (attempt - 1))

        # A provider-supplied Retry-After is authoritative when it asks for more
        # time than we computed; asking for less is ignored to avoid hot-looping.
        if error.retry_after:
            base = max(base, float(error.retry_after))

        ceiling = max(cfg.cooldown_max_seconds, cfg.quota_cooldown_seconds)
        base = min(base, ceiling)
        return base * random.uniform(0.85, 1.15)

    def _notify(self, key: ManagedKey) -> None:
        if self._observer is None:
            return
        try:
            self._observer(key)
        except Exception:  # pragma: no cover - observers must never break calls
            pass

    # -- maintenance ------------------------------------------------------- #

    async def reset(self) -> None:
        """Clear all cooldowns. Used by the `keys --reset` command."""
        async with self._lock:
            for key in self.keys:
                key.status = KeyStatus.HEALTHY
                key.available_at = 0.0
                key.consecutive_failures = 0


class KeyPoolRegistry:
    """Builds and owns one :class:`KeyPool` per provider."""

    def __init__(
        self,
        config: FailoverSettings | None = None,
        observer: HealthObserver | None = None,
        providers: Iterable[str] | None = None,
    ) -> None:
        self.config = config or global_settings.failover
        self._pools: dict[str, KeyPool] = {}
        for provider in providers or PROVIDER_ENV_PREFIX:
            prefix = PROVIDER_ENV_PREFIX.get(provider, provider.upper())
            self._pools[provider] = KeyPool(
                provider, read_key_pool(prefix), self.config, observer
            )

    def get(self, provider: str) -> KeyPool:
        try:
            return self._pools[provider]
        except KeyError:
            raise NoKeysConfigured(f"Unknown provider '{provider}'") from None

    @property
    def providers(self) -> list[str]:
        return list(self._pools)

    def configured_providers(self) -> list[str]:
        return [name for name, pool in self._pools.items() if pool.configured]

    def missing_providers(self) -> list[str]:
        return [name for name, pool in self._pools.items() if not pool.configured]

    def snapshot(self) -> dict:
        return {name: pool.snapshot() for name, pool in self._pools.items()}

    def restore(self, rows: Iterable[dict]) -> int:
        """Apply persisted health records to whichever keys they belong to.

        Rows are matched by provider and fingerprint, so a pool whose keys were
        rotated since the rows were written simply ignores them — which is the
        behaviour that matters, since a rotated key must start out healthy rather
        than inheriting the dead status of the key it replaced.
        """
        restored = 0
        for row in rows:
            pool = self._pools.get(str(row.get("provider") or ""))
            if pool is None:
                continue
            for key in pool.keys:
                if key.restore(row):
                    restored += 1
                    break
        return restored

    async def reset(self) -> None:
        for pool in self._pools.values():
            await pool.reset()


__all__ = [
    "AllKeysUnavailable",
    "KeyPool",
    "KeyPoolRegistry",
    "KeyStatus",
    "ManagedKey",
    "NoKeysConfigured",
    "fingerprint",
    "safe_error_text",
]
