"""Provider transports and the failover-aware LLM client.

One client serves all three providers. Gemini uses its native
``generateContent`` API; Groq and NVIDIA share an OpenAI-compatible
``/chat/completions`` transport. Every request goes through
:class:`services.keypool.KeyPool`, so a failing key is benched and the next key
picks the request up without the caller noticing.

Failures never raise into the stage code — they come back as
``LLMResult(ok=False)``. That matters for consensus: five models voting means
four successes and one outage should still produce a verdict, and the stage needs
to *see* that only four answered.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from services.config import PROVIDER_BASE_URL, Settings, settings as global_settings
from services.errors import (
    Disposition,
    KeyFault,
    ProviderError,
    classify_exception,
    classify_http,
)
from services.jsonparse import extract_json_object, strip_reasoning
from services.keypool import (
    AllKeysUnavailable,
    KeyPoolRegistry,
    ManagedKey,
    NoKeysConfigured,
)
from services.models import EMBED_MODEL, ModelSpec

log = logging.getLogger(__name__)


@dataclass
class LLMResult:
    """Outcome of one logical model call, successful or not."""

    model: ModelSpec
    ok: bool = False
    data: dict | None = None
    text: str = ""
    error: str | None = None
    latency: float = 0.0
    attempts: int = 0
    key_index: int | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: Every classified failure encountered while trying to satisfy this call.
    failures: list[str] = field(default_factory=list)

    @property
    def model_id(self) -> str:
        return self.model.id

    @property
    def provider(self) -> str:
        return self.model.provider

    @property
    def truncated(self) -> bool:
        return (self.finish_reason or "").upper() in ("MAX_TOKENS", "LENGTH")

    def as_record(self) -> dict:
        """Row shape for the ``stage_results`` table."""
        return {
            "model_id": self.model.id,
            "provider": self.model.provider,
            "key_index": self.key_index,
            "ok": self.ok,
            "latency_ms": int(self.latency * 1000),
            "attempts": self.attempts,
            "error": self.error,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "raw": self.data if self.data is not None else {"text": self.text[:4000]},
        }


@dataclass
class _Request:
    """A provider-shaped HTTP request, before a key is attached."""

    url: str
    body: dict
    #: Header builder, applied once a key is chosen.
    auth: Callable[[str], dict[str, str]]


def _bearer(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


def _goog_key(secret: str) -> dict[str, str]:
    return {"x-goog-api-key": secret}


# --------------------------------------------------------------------------- #
# request builders
# --------------------------------------------------------------------------- #

def _build_openai_chat(
    spec: ModelSpec,
    system: str,
    user: str,
    *,
    json_mode: bool,
    max_tokens: int,
    temperature: float,
    token_field: str = "max_tokens",
) -> _Request:
    base = PROVIDER_BASE_URL[spec.provider].rstrip("/")
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    body: dict[str, Any] = {
        "model": spec.id,
        "messages": messages,
        "temperature": temperature,
        token_field: max_tokens,
        "stream": False,
    }
    if json_mode and spec.supports_json_mode:
        body["response_format"] = {"type": "json_object"}
    return _Request(url=f"{base}/chat/completions", body=body, auth=_bearer)


def _build_gemini_generate(
    spec: ModelSpec,
    system: str,
    user: str,
    *,
    json_mode: bool,
    max_tokens: int,
    temperature: float,
) -> _Request:
    base = PROVIDER_BASE_URL[spec.provider].rstrip("/")
    prompt = user
    body: dict[str, Any] = {
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }

    if system:
        if spec.supports_system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        else:
            # Gemma over the Gemini API rejects systemInstruction outright.
            prompt = f"{system}\n\n---\n\n{user}"

    body["contents"] = [{"role": "user", "parts": [{"text": prompt}]}]

    if json_mode and spec.supports_json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"

    return _Request(
        url=f"{base}/v1beta/models/{spec.id}:generateContent",
        body=body,
        auth=_goog_key,
    )


def _build_gemini_embed(
    spec: ModelSpec, text: str, *, dim: int | None, task_type: str | None
) -> _Request:
    base = PROVIDER_BASE_URL[spec.provider].rstrip("/")
    body: dict[str, Any] = {
        "model": f"models/{spec.id}",
        "content": {"parts": [{"text": text}]},
    }
    if task_type:
        body["taskType"] = task_type
    if dim:
        body["outputDimensionality"] = dim
    return _Request(
        url=f"{base}/v1beta/models/{spec.id}:embedContent",
        body=body,
        auth=_goog_key,
    )


# --------------------------------------------------------------------------- #
# response readers
# --------------------------------------------------------------------------- #

def _read_openai_chat(payload: dict) -> tuple[str, str | None, dict]:
    choices = payload.get("choices") or []
    if not choices:
        return "", None, payload.get("usage") or {}
    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, list):  # some gateways return content parts
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not content:
        # Reasoning models occasionally put everything in `reasoning` and leave
        # content empty when they run out of tokens.
        content = message.get("reasoning") or message.get("reasoning_content") or ""

    return content or "", choices[0].get("finish_reason"), payload.get("usage") or {}


def _read_gemini_generate(payload: dict) -> tuple[str, str | None, dict]:
    feedback = payload.get("promptFeedback") or {}
    block_reason = feedback.get("blockReason")

    candidates = payload.get("candidates") or []
    if not candidates:
        reason = block_reason or "no candidates returned"
        raise ContentRejected(str(reason))

    candidate = candidates[0]
    finish = candidate.get("finishReason")
    parts = ((candidate.get("content") or {}).get("parts")) or []
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))

    if not text and finish and finish.upper() in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"):
        raise ContentRejected(f"blocked by provider: {finish}")

    usage = payload.get("usageMetadata") or {}
    return text, finish, {
        "prompt_tokens": usage.get("promptTokenCount"),
        "completion_tokens": usage.get("candidatesTokenCount"),
    }


class ContentRejected(Exception):
    """Provider returned 200 but refused to answer (safety filter, empty result).

    Not a key fault and not retryable — the consensus layer just sees one fewer
    vote.
    """


def _usage_tokens(usage: dict) -> tuple[int | None, int | None]:
    prompt = usage.get("prompt_tokens") or usage.get("promptTokenCount")
    completion = usage.get("completion_tokens") or usage.get("candidatesTokenCount")
    return (
        int(prompt) if isinstance(prompt, (int, float)) else None,
        int(completion) if isinstance(completion, (int, float)) else None,
    )


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #

class LLMClient:
    """Failover-aware client for every configured provider."""

    def __init__(
        self,
        registry: KeyPoolRegistry | None = None,
        config: Settings | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = config or global_settings
        self.registry = registry or KeyPoolRegistry(self.settings.failover)
        self._http = http
        self._owns_http = http is None
        self._semaphore = asyncio.Semaphore(self.settings.max_concurrent_llm_calls)

    async def __aenter__(self) -> "LLMClient":
        if self._http is None:
            self._http = httpx.AsyncClient(
                # The client-level timeout is the generous one; each request then
                # narrows it to what that model should need. Setting the cheap
                # panel's timeout here instead would cap the heavy models, and
                # httpx cannot widen a per-request timeout beyond the client's.
                timeout=httpx.Timeout(self._heavy_timeout),
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
                headers={"content-type": "application/json"},
            )
        return self

    @property
    def _heavy_timeout(self) -> float:
        return max(self.settings.http_timeout, self.settings.heavy_http_timeout)

    def _timeout_for(self, spec: ModelSpec) -> float:
        """How long to wait for one model, rather than one number for all of them.

        A single global timeout has to be either too short for the final panel or
        too long for the cheap one, and both failures are expensive in different
        ways.

        Too short is the worse of the two, and it is not obvious why: a timeout is
        classified as a *transient* fault, so it puts the key on cooldown. A
        550B reasoning model emitting 4096 tokens legitimately takes minutes, so a
        90-second limit would cool the NVIDIA key on every critical event — the
        failover would then report a healthy key as failing, for a reason that has
        nothing to do with the key.

        Too long merely wastes wall-clock on a stuck cheap model, which the
        pipeline can absorb because the panel votes with a quorum.
        """
        if spec.reasoning or spec.max_output_tokens >= 3000:
            return self._heavy_timeout
        return self.settings.http_timeout

    async def __aexit__(self, *_exc) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            raise RuntimeError("LLMClient must be used as an async context manager")
        return self._http

    # -- public API -------------------------------------------------------- #

    async def complete_json(
        self,
        spec: ModelSpec,
        system: str,
        user: str,
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        """Call ``spec`` and parse its reply as a JSON object."""
        result = await self.complete_text(
            spec,
            system,
            user,
            json_mode=True,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        if not result.ok:
            return result

        data = extract_json_object(result.text)
        if data is None:
            result.ok = False
            detail = "truncated before valid JSON" if result.truncated else "no JSON object in response"
            result.error = f"unparseable response ({detail})"
            log.debug("%s returned unparseable output: %r", spec.label, result.text[:300])
            return result

        result.data = data
        return result

    async def complete_text(
        self,
        spec: ModelSpec,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        max_tokens = max_output_tokens or spec.max_output_tokens
        temp = spec.temperature if temperature is None else temperature

        if spec.provider == "gemini":
            def build(variant: int) -> _Request:
                return _build_gemini_generate(
                    spec, system, user,
                    json_mode=json_mode, max_tokens=max_tokens, temperature=temp,
                )
            reader = _read_gemini_generate
            variants = 1
        else:
            def build(variant: int) -> _Request:
                # Some newer OpenAI-compatible deployments retired `max_tokens`
                # in favour of `max_completion_tokens`; swap on a 400 about it.
                token_field = "max_tokens" if variant == 0 else "max_completion_tokens"
                return _build_openai_chat(
                    spec, system, user,
                    json_mode=json_mode, max_tokens=max_tokens, temperature=temp,
                    token_field=token_field,
                )
            reader = _read_openai_chat
            variants = 2

        return await self._execute(spec, build, reader, variants=variants)

    async def embed(
        self,
        text: str,
        *,
        spec: ModelSpec | None = None,
        dim: int | None = None,
    ) -> list[float] | None:
        """Embed ``text``, returning ``None`` if every key and variant failed."""
        spec = spec or EMBED_MODEL
        dim = dim or self.settings.stage0.embed_dim
        payload = text.strip()[: self.settings.stage0.embed_max_chars]
        if not payload:
            return None

        holder: dict[str, list[float]] = {}

        def build(variant: int) -> _Request:
            # Drop the optional fields one at a time if the model rejects them.
            if variant == 0:
                return _build_gemini_embed(spec, payload, dim=dim, task_type="SEMANTIC_SIMILARITY")
            if variant == 1:
                return _build_gemini_embed(spec, payload, dim=dim, task_type=None)
            return _build_gemini_embed(spec, payload, dim=None, task_type=None)

        def reader(response_payload: dict) -> tuple[str, str | None, dict]:
            embedding = response_payload.get("embedding") or {}
            values = embedding.get("values") or embedding.get("value")
            if not values and isinstance(response_payload.get("embeddings"), list):
                first = response_payload["embeddings"][0] or {}
                values = first.get("values")
            if not values:
                raise ContentRejected("embedding response contained no values")
            holder["vector"] = [float(v) for v in values]
            return "", None, {}

        result = await self._execute(spec, build, reader, variants=3)
        if not result.ok:
            log.warning("embedding failed: %s", result.error)
            return None

        vector = holder.get("vector")
        if vector and dim and len(vector) != dim:
            log.warning(
                "embedding dimension mismatch: model returned %d, EMBED_DIM=%d",
                len(vector), dim,
            )
        return vector

    # -- the failover loop ------------------------------------------------- #

    async def _execute(
        self,
        spec: ModelSpec,
        build: Callable[[int], _Request],
        reader: Callable[[dict], tuple[str, str | None, dict]],
        *,
        variants: int = 1,
    ) -> LLMResult:
        """Run one logical call, failing over across keys and body variants."""
        cfg = self.settings.failover
        result = LLMResult(model=spec)
        started = time.monotonic()

        try:
            pool = self.registry.get(spec.provider)
        except NoKeysConfigured as exc:
            result.error = str(exc)
            return result

        variant = 0
        async with self._semaphore:
            while result.attempts < cfg.max_attempts:
                result.attempts += 1
                request = build(variant)

                try:
                    key = await pool.acquire()
                except (AllKeysUnavailable, NoKeysConfigured) as exc:
                    result.error = str(exc)
                    result.failures.append(str(exc))
                    result.latency = time.monotonic() - started
                    return result

                outcome = await self._attempt(spec, pool, key, request, reader, cfg)
                if outcome.ok:
                    outcome.attempts = result.attempts
                    outcome.failures = result.failures
                    outcome.latency = time.monotonic() - started
                    return outcome

                result.failures.extend(outcome.failures)
                result.error = outcome.error
                result.key_index = outcome.key_index

                if outcome.finish_reason == "__fatal__":
                    # Request-shaped problem. Try the next body variant if we
                    # have one; otherwise there is nothing a new key can fix.
                    if variant + 1 < variants:
                        variant += 1
                        continue
                    break
                if outcome.finish_reason == "__rejected__":
                    break  # provider answered but refused; not retryable

            result.latency = time.monotonic() - started
            if result.error is None:
                result.error = f"exhausted {result.attempts} attempt(s) with no usable key"
            return result

    async def _attempt(
        self,
        spec: ModelSpec,
        pool,
        key: ManagedKey,
        request: _Request,
        reader: Callable[[dict], tuple[str, str | None, dict]],
        cfg,
    ) -> LLMResult:
        """One key, one body. Reports the key's health exactly once."""
        outcome = LLMResult(model=spec, key_index=key.index)
        same_key_budget = cfg.same_key_retries

        while True:
            headers = request.auth(key.secret)
            started = time.monotonic()
            error: ProviderError | None = None

            try:
                response = await self.http.post(
                    request.url,
                    json=request.body,
                    headers=headers,
                    timeout=self._timeout_for(spec),
                )
            except Exception as exc:  # transport-level
                error = classify_exception(spec.provider, exc, spec.id)
            else:
                latency = time.monotonic() - started
                if response.is_success:
                    try:
                        payload = response.json()
                    except ValueError:
                        error = ProviderError(
                            provider=spec.provider,
                            message="provider returned non-JSON body",
                            disposition=Disposition.RETRY_SAME_KEY,
                            fault=KeyFault.TRANSIENT,
                            status=response.status_code,
                            model=spec.id,
                        )
                    else:
                        try:
                            text, finish, usage = reader(payload)
                        except ContentRejected as exc:
                            # The key worked; the model declined to answer.
                            await pool.report_success(key, latency)
                            outcome.ok = False
                            outcome.error = f"content rejected: {exc}"
                            outcome.finish_reason = "__rejected__"
                            outcome.failures.append(outcome.error)
                            return outcome

                        await pool.report_success(key, latency)
                        outcome.ok = True
                        outcome.text = strip_reasoning(text) if spec.reasoning else text
                        outcome.finish_reason = finish
                        outcome.latency = latency
                        outcome.prompt_tokens, outcome.completion_tokens = _usage_tokens(usage)
                        return outcome
                else:
                    error = classify_http(
                        spec.provider,
                        response.status_code,
                        response.text,
                        dict(response.headers),
                        spec.id,
                    )

            assert error is not None

            # A transient provider blip is worth one immediate retry on the same
            # key before we shift traffic to a lower-priority key.
            if error.disposition is Disposition.RETRY_SAME_KEY and same_key_budget > 0:
                same_key_budget -= 1
                await asyncio.sleep(0.75 * (cfg.same_key_retries - same_key_budget))
                continue

            await pool.report_failure(key, error)
            outcome.error = str(error)
            outcome.failures.append(str(error))
            if error.disposition is Disposition.FATAL:
                outcome.finish_reason = "__fatal__"
            log.debug("%s key #%d failed: %s", spec.label, key.index, error)
            return outcome

    # -- convenience ------------------------------------------------------- #

    async def fan_out(
        self,
        specs: Sequence[ModelSpec],
        system: str,
        user: str,
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[LLMResult]:
        """Ask every model the same question in parallel.

        Never raises: a model that fails comes back as ``ok=False`` so the caller
        can count how many actually voted.
        """
        tasks = [
            self.complete_json(
                spec, system, user,
                max_output_tokens=max_output_tokens, temperature=temperature,
            )
            for spec in specs
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[LLMResult] = []
        for spec, item in zip(specs, gathered):
            if isinstance(item, BaseException):
                log.exception("unexpected error calling %s", spec.label, exc_info=item)
                results.append(
                    LLMResult(model=spec, ok=False, error=f"{type(item).__name__}: {item}")
                )
            else:
                results.append(item)
        return results


__all__ = ["ContentRejected", "LLMClient", "LLMResult"]
