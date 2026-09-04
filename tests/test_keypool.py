"""Multi-key failover: what happens when a key breaks mid-pipeline.

Every stage fires five models in parallel, so a pool that mishandles one bad key
does not degrade — it takes the whole cycle down. Four properties matter more than
the rest, and each one has a failure mode that is invisible in production:

**Priority, not round-robin.** A healthy key #1 serves every request; #2 and #3
exist only as fallbacks. Round-robin would look identical on a good day and then
spread traffic onto keys with smaller quotas on a bad one.

**A request-side error must never cost a key.** One wrong model id classified as a
key fault benches all three keys in three calls, and the outage looks like a
provider problem for as long as it takes to read the logs.

**Cooldowns grow, and Retry-After only extends them.** Honouring a shorter
Retry-After turns a rate limit into a hot loop against the provider.

**Secrets stay out of everything that leaves the process.** ``last_error`` is
persisted and printed by ``/keys``, which goes to Telegram; a provider that echoes
the rejected key would publish it.

Cooldowns here are milliseconds so the waiting paths can be exercised for real
rather than mocked. Jitter is pinned only where a test asserts an exact delay.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from services import keypool as keypool_module
from services.config import FailoverSettings
from services.errors import Disposition, KeyFault, ProviderError
from services.keypool import (
    AllKeysUnavailable,
    KeyPool,
    KeyPoolRegistry,
    KeyStatus,
    NoKeysConfigured,
    fingerprint,
    safe_error_text,
)
from services.logging_setup import redact

# Realistic shapes: the redaction patterns are prefix-based, so a placeholder like
# "key1" would pass tests that a real key would fail.
GEMINI_KEYS = (
    "AIzaSyB1first0000000000000000000000000",
    "AIzaSyB2second000000000000000000000000",
    "AIzaSyB3third0000000000000000000000000",
)


def failover(**overrides) -> FailoverSettings:
    """Fast settings: real sleeps stay in the tens of milliseconds."""
    values = {
        "cooldown_base_seconds": 0.04,
        "cooldown_max_seconds": 2.0,
        "quota_cooldown_seconds": 0.6,
        "dead_retry_seconds": 0.05,
        "max_wait_for_key_seconds": 1.0,
    }
    values.update(overrides)
    return FailoverSettings(**values)


def pool(*secrets: str, **overrides) -> KeyPool:
    return KeyPool("gemini", secrets or GEMINI_KEYS, failover(**overrides))


def fault(kind: KeyFault, retry_after: float | None = None) -> ProviderError:
    disposition = (
        Disposition.FATAL if kind is KeyFault.NONE else Disposition.RETRY_OTHER_KEY
    )
    return ProviderError(
        provider="gemini",
        message=f"simulated {kind.value}",
        disposition=disposition,
        fault=kind,
        retry_after=retry_after,
    )


@pytest.fixture
def no_jitter(monkeypatch):
    """Pin the cooldown jitter so exact delays can be asserted."""
    monkeypatch.setattr(keypool_module.random, "uniform", lambda low, high: 1.0)


async def use(p: KeyPool, error: ProviderError | None = None) -> int:
    """One full request cycle. Returns the index of the key that served it."""
    key = await p.acquire()
    if error is None:
        await p.report_success(key, latency=0.01)
    else:
        await p.report_failure(key, error)
    return key.index


# --------------------------------------------------------------------------- #
# priority ordering: KEY 1 -> KEY 2 -> KEY 3
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_healthy_first_key_serves_every_request():
    """Not round-robin. A working key #1 keeps the traffic."""
    p = pool()
    served = [await use(p) for _ in range(6)]
    assert served == [1, 1, 1, 1, 1, 1]
    assert p.keys[1].success_count == 0
    assert p.keys[2].success_count == 0


@pytest.mark.asyncio
async def test_cooling_first_key_falls_over_to_the_second():
    p = pool()
    assert await use(p, fault(KeyFault.RATE_LIMITED)) == 1
    assert p.keys[0].status is KeyStatus.COOLING

    assert await use(p) == 2
    assert await use(p) == 2  # and stays there while #1 is benched


@pytest.mark.asyncio
async def test_traffic_returns_to_the_first_key_once_it_recovers():
    """The fallback is temporary: priority reasserts itself, no manual reset."""
    p = pool(cooldown_base_seconds=0.03)
    await use(p, fault(KeyFault.RATE_LIMITED))
    assert await use(p) == 2

    await asyncio.sleep(0.09)
    assert await use(p) == 1
    assert p.keys[0].is_available()


@pytest.mark.asyncio
async def test_second_failure_walks_down_to_the_third_key():
    p = pool(cooldown_base_seconds=1.0)
    assert await use(p, fault(KeyFault.RATE_LIMITED)) == 1
    assert await use(p, fault(KeyFault.QUOTA)) == 2
    assert await use(p) == 3


@pytest.mark.asyncio
async def test_dead_first_key_loses_to_a_healthy_second():
    """A revoked key #1 must not outrank a working key #2, before or after probation."""
    p = pool(dead_retry_seconds=0.03)
    await use(p, fault(KeyFault.INVALID))
    assert p.keys[0].status is KeyStatus.DEAD

    assert await use(p) == 2

    await asyncio.sleep(0.06)
    assert p.keys[0].is_on_probation()
    assert await use(p) == 2  # probation still loses to healthy


@pytest.mark.asyncio
async def test_dead_key_is_probed_when_nothing_healthy_is_left():
    """Probation is the last resort, and it beats failing the request outright."""
    p = pool(dead_retry_seconds=0.03, quota_cooldown_seconds=5.0)
    await use(p, fault(KeyFault.INVALID))          # #1 dead, back in 0.03s
    await use(p, fault(KeyFault.QUOTA))            # #2 out for 5s
    await use(p, fault(KeyFault.QUOTA))            # #3 out for 5s

    await asyncio.sleep(0.06)
    key = await p.acquire()
    assert key.index == 1
    assert key.status is KeyStatus.DEAD  # a probe, not a reinstatement


@pytest.mark.asyncio
async def test_successful_probe_reinstates_a_dead_key():
    p = pool(dead_retry_seconds=0.03, quota_cooldown_seconds=5.0)
    await use(p, fault(KeyFault.INVALID))
    await use(p, fault(KeyFault.QUOTA))
    await use(p, fault(KeyFault.QUOTA))

    await asyncio.sleep(0.06)
    assert await use(p) == 1
    assert p.keys[0].status is KeyStatus.HEALTHY
    assert p.keys[0].cooldown_remaining() == 0.0
    assert p.keys[0].consecutive_failures == 0


@pytest.mark.asyncio
async def test_parallel_calls_spread_instead_of_queueing():
    """Five models fire at once; they should not all pile onto key #1."""
    p = pool()
    first, second, third = [await p.acquire() for _ in range(3)]
    assert [first.index, second.index, third.index] == [1, 2, 3]

    # With every key equally loaded, priority takes over again.
    fourth = await p.acquire()
    assert fourth.index == 1
    assert p.keys[0].in_flight == 2


@pytest.mark.asyncio
async def test_in_flight_unwinds_on_both_outcomes():
    """A leaked counter would make a healthy key look permanently busy."""
    p = pool()
    await use(p)
    await use(p, fault(KeyFault.TRANSIENT))
    assert [key.in_flight for key in p.keys] == [0, 0, 0]


# --------------------------------------------------------------------------- #
# penalties: whose fault was it
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_request_side_error_does_not_penalise_the_key():
    """A bad model id is our bug. Charging it to the keys burns the pool on a typo."""
    p = pool()
    for _ in range(4):
        assert await use(p, fault(KeyFault.NONE)) == 1

    key = p.keys[0]
    assert key.status is KeyStatus.HEALTHY
    assert key.failure_count == 0
    assert key.consecutive_failures == 0
    assert key.cooldown_remaining() == 0.0
    assert p.available_count() == 3


@pytest.mark.asyncio
async def test_cooldown_doubles_with_each_consecutive_failure(no_jitter):
    p = pool(cooldown_base_seconds=0.5, cooldown_max_seconds=100.0)
    key = p.keys[0]

    delays = []
    for _ in range(3):
        acquired = await p.acquire()
        assert acquired is key, "the same key must be retried to grow its backoff"
        await p.report_failure(key, fault(KeyFault.TRANSIENT))
        delays.append(key.cooldown_remaining())
        key.available_at = 0.0  # let it be picked again without waiting
        key.status = KeyStatus.HEALTHY

    assert [round(d, 2) for d in delays] == [0.5, 1.0, 2.0]


@pytest.mark.asyncio
async def test_success_resets_the_backoff(no_jitter):
    p = pool(cooldown_base_seconds=0.5, cooldown_max_seconds=100.0)
    key = p.keys[0]

    for _ in range(3):
        await p.report_failure(await p.acquire(), fault(KeyFault.TRANSIENT))
        key.available_at = 0.0
        key.status = KeyStatus.HEALTHY

    await p.report_success(key)
    await p.report_failure(await p.acquire(), fault(KeyFault.TRANSIENT))
    assert round(key.cooldown_remaining(), 2) == 0.5


@pytest.mark.asyncio
async def test_retry_after_wins_when_it_asks_for_longer(no_jitter):
    p = pool(cooldown_base_seconds=0.04, cooldown_max_seconds=100.0)
    await p.report_failure(await p.acquire(), fault(KeyFault.RATE_LIMITED, retry_after=30.0))
    assert round(p.keys[0].cooldown_remaining(), 1) == 30.0


@pytest.mark.asyncio
async def test_retry_after_is_ignored_when_it_asks_for_less(no_jitter):
    """Trusting a shorter Retry-After is how a rate limit becomes a hot loop."""
    p = pool(cooldown_base_seconds=5.0, cooldown_max_seconds=100.0)
    await p.report_failure(await p.acquire(), fault(KeyFault.RATE_LIMITED, retry_after=0.01))
    assert round(p.keys[0].cooldown_remaining(), 1) == 5.0


@pytest.mark.asyncio
async def test_quota_gets_a_longer_bench_than_a_rate_limit(no_jitter):
    """Quotas reset on a daily boundary; retrying in 20s just wastes the attempt."""
    p = pool(cooldown_base_seconds=1.0, quota_cooldown_seconds=60.0, cooldown_max_seconds=100.0)
    await p.report_failure(await p.acquire(), fault(KeyFault.RATE_LIMITED))
    await p.report_failure(await p.acquire(), fault(KeyFault.QUOTA))

    assert round(p.keys[0].cooldown_remaining(), 1) == 1.0
    assert round(p.keys[1].cooldown_remaining(), 1) == 60.0
    assert p.keys[1].quota_count == 1


@pytest.mark.asyncio
async def test_cooldown_is_capped(no_jitter):
    """A provider asking for an hour must not bench a key past the ceiling."""
    p = pool(cooldown_base_seconds=1.0, cooldown_max_seconds=10.0, quota_cooldown_seconds=10.0)
    await p.report_failure(await p.acquire(), fault(KeyFault.RATE_LIMITED, retry_after=3600.0))
    assert p.keys[0].cooldown_remaining() <= 10.0


@pytest.mark.asyncio
async def test_cooldowns_are_jittered():
    """Without jitter the five parallel calls retry in lockstep and re-collide."""
    delays = set()
    for _ in range(12):
        p = pool(cooldown_base_seconds=10.0, cooldown_max_seconds=100.0)
        await p.report_failure(await p.acquire(), fault(KeyFault.RATE_LIMITED))
        delays.add(round(p.keys[0].cooldown_remaining(), 4))

    assert len(delays) > 1
    assert all(8.4 <= delay <= 11.6 for delay in delays)


# --------------------------------------------------------------------------- #
# exhaustion
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_acquire_waits_for_a_key_to_come_back():
    p = pool(cooldown_base_seconds=0.05, max_wait_for_key_seconds=2.0)
    for _ in range(3):
        await p.report_failure(await p.acquire(), fault(KeyFault.RATE_LIMITED))

    started = time.monotonic()
    key = await p.acquire()
    assert key.index in (1, 2, 3)
    assert time.monotonic() - started >= 0.04


@pytest.mark.asyncio
async def test_all_keys_benched_raises_with_a_retry_hint():
    p = pool(cooldown_base_seconds=5.0)
    for _ in range(3):
        await p.report_failure(await p.acquire(), fault(KeyFault.RATE_LIMITED))

    with pytest.raises(AllKeysUnavailable) as excinfo:
        await p.acquire(wait=False)

    error = excinfo.value
    assert error.provider == "gemini"
    assert error.retry_in is not None and 0 < error.retry_in <= 6.0
    assert "cooling" in str(error)


@pytest.mark.asyncio
async def test_wait_budget_is_respected():
    """A caller must not block past its budget waiting for a long cooldown."""
    p = pool(cooldown_base_seconds=30.0, max_wait_for_key_seconds=0.12)
    for _ in range(3):
        await p.report_failure(await p.acquire(), fault(KeyFault.RATE_LIMITED))

    started = time.monotonic()
    with pytest.raises(AllKeysUnavailable):
        await p.acquire()
    assert time.monotonic() - started < 1.0


@pytest.mark.asyncio
async def test_empty_pool_names_the_variable_to_set():
    empty = KeyPool("gemini", [], failover())
    assert not empty.configured
    with pytest.raises(NoKeysConfigured) as excinfo:
        await empty.acquire()
    assert "GEMINI_API_KEYS" in str(excinfo.value)


@pytest.mark.asyncio
async def test_reset_clears_every_cooldown():
    p = pool()
    await p.report_failure(await p.acquire(), fault(KeyFault.INVALID))
    await p.report_failure(await p.acquire(), fault(KeyFault.QUOTA))
    assert p.available_count() == 1

    await p.reset()
    assert p.available_count() == 3
    assert all(key.status is KeyStatus.HEALTHY for key in p.keys)


# --------------------------------------------------------------------------- #
# secrets must not leave the process
# --------------------------------------------------------------------------- #


def test_fingerprint_reveals_only_the_ends():
    secret = GEMINI_KEYS[0]
    printed = fingerprint(secret)

    assert secret not in printed
    assert len(printed) < len(secret) / 2
    assert printed.startswith("AIzaSy")


def test_fingerprint_still_distinguishes_keys():
    """Masking is useless for ops if every key in the pool prints the same.

    These three share both a prefix and a tail, which is what real keys issued
    from one console look like.
    """
    prints = {fingerprint(secret) for secret in GEMINI_KEYS}
    assert len(prints) == 3


def test_short_secret_is_masked_too():
    assert "cdef" not in fingerprint("abcdefghij")
    assert fingerprint("") == "<empty>"


def test_fingerprint_matches_the_log_redactor():
    """One key, one name — in a log line, in /keys and in the health table."""
    secret = GEMINI_KEYS[2]
    assert redact(f"rejected {secret} at 09:00") == f"rejected {fingerprint(secret)} at 09:00"


@pytest.mark.asyncio
async def test_snapshot_contains_no_secret():
    """/keys and the persisted health record are built from this."""
    p = pool()
    await p.report_failure(await p.acquire(), fault(KeyFault.INVALID))
    await p.report_success(await p.acquire())

    dumped = json.dumps(p.snapshot(), ensure_ascii=False)
    for secret in GEMINI_KEYS:
        assert secret not in dumped
    assert fingerprint(GEMINI_KEYS[0]) in dumped


@pytest.mark.asyncio
async def test_stored_error_scrubs_a_key_the_provider_echoed_back():
    """Gemini takes the key as a query parameter and quotes the URL on failure."""
    secret = GEMINI_KEYS[0]
    p = pool()
    echoed = ProviderError(
        provider="gemini",
        message=f"API key not valid: /v1beta/models?key={secret}",
        disposition=Disposition.RETRY_OTHER_KEY,
        fault=KeyFault.INVALID,
        status=400,
    )
    await p.report_failure(await p.acquire(), echoed)

    stored = p.keys[0].last_error
    assert stored and secret not in stored
    assert "API key not valid" in stored
    assert secret not in json.dumps(p.snapshot())


def test_safe_error_text_masks_other_providers_keys():
    """A pooled Gemini key is not the only secret that can appear in a body."""
    for secret in ("gsk_" + "a" * 40, "nvapi-" + "b" * 40, "AQ." + "c" * 40):
        text = safe_error_text(
            ProviderError(
                provider="groq",
                message=f"rejected {secret}",
                disposition=Disposition.RETRY_OTHER_KEY,
                fault=KeyFault.INVALID,
            )
        )
        assert secret not in text


def test_key_repr_hides_the_secret():
    """A ManagedKey lands in tracebacks and debug logs."""
    p = pool()
    assert GEMINI_KEYS[0] not in repr(p.keys[0])


# --------------------------------------------------------------------------- #
# registry: reading the pools out of the environment
# --------------------------------------------------------------------------- #


@pytest.fixture
def clean_env(monkeypatch):
    for prefix in ("GEMINI", "GROQ", "NVIDIA"):
        monkeypatch.delenv(f"{prefix}_API_KEYS", raising=False)
        monkeypatch.delenv(f"{prefix}_API_KEY", raising=False)
        for index in range(1, 21):
            monkeypatch.delenv(f"{prefix}_API_KEY_{index}", raising=False)
    return monkeypatch


def test_registry_builds_one_pool_per_provider(clean_env):
    clean_env.setenv("GEMINI_API_KEYS", ", ".join(GEMINI_KEYS))
    clean_env.setenv("GROQ_API_KEY_1", "gsk_" + "1" * 40)
    clean_env.setenv("GROQ_API_KEY_2", "gsk_" + "2" * 40)

    registry = KeyPoolRegistry(failover())

    assert len(registry.get("gemini")) == 3
    assert len(registry.get("groq")) == 2
    assert sorted(registry.configured_providers()) == ["gemini", "groq"]
    assert registry.missing_providers() == ["nvidia"]


def test_registry_merges_the_three_spellings_in_priority_order(clean_env):
    clean_env.setenv("NVIDIA_API_KEYS", "nvapi-aaaa000000000000000000000000000000")
    clean_env.setenv("NVIDIA_API_KEY_2", "nvapi-bbbb000000000000000000000000000000")
    clean_env.setenv("NVIDIA_API_KEY", "nvapi-cccc000000000000000000000000000000")

    keys = KeyPoolRegistry(failover()).get("nvidia").keys
    assert [key.index for key in keys] == [1, 2, 3]
    assert [key.secret[6:10] for key in keys] == ["aaaa", "bbbb", "cccc"]


def test_registry_deduplicates_a_repeated_key(clean_env):
    """Two copies of one key are one point of failure, and must not look like two."""
    clean_env.setenv("GEMINI_API_KEYS", f"{GEMINI_KEYS[0]},{GEMINI_KEYS[0]}")
    clean_env.setenv("GEMINI_API_KEY", GEMINI_KEYS[0])

    assert len(KeyPoolRegistry(failover()).get("gemini")) == 1


def test_registry_rejects_an_unknown_provider(clean_env):
    with pytest.raises(NoKeysConfigured):
        KeyPoolRegistry(failover()).get("openai")


@pytest.mark.asyncio
async def test_health_observer_sees_state_changes(clean_env):
    """The observer is what persists cooldowns across a restart."""
    seen: list[tuple[int, str]] = []
    p = KeyPool(
        "gemini", GEMINI_KEYS, failover(dead_retry_seconds=0.03, quota_cooldown_seconds=5.0),
        observer=lambda key: seen.append((key.index, key.status.value)),
    )

    await p.report_failure(await p.acquire(), fault(KeyFault.INVALID))   # #1 dead
    await p.report_success(await p.acquire())                            # #2, no change
    await p.report_failure(await p.acquire(), fault(KeyFault.QUOTA))     # #2 out
    await p.report_failure(await p.acquire(), fault(KeyFault.QUOTA))     # #3 out

    await asyncio.sleep(0.06)
    assert await use(p) == 1        # #1 probed and reinstated

    assert (1, "dead") in seen
    assert (1, "healthy") in seen
    assert seen.count((2, "healthy")) == 0  # a success on a healthy key notifies nothing


@pytest.mark.asyncio
async def test_a_broken_observer_never_breaks_a_request():
    def explode(key):
        raise RuntimeError("health persistence is down")

    p = KeyPool("gemini", GEMINI_KEYS, failover(), observer=explode)
    await p.report_failure(await p.acquire(), fault(KeyFault.RATE_LIMITED))
    assert p.keys[0].status is KeyStatus.COOLING
    assert await use(p) == 2
