"""LLM client used by Stage 1, Stage 2 and the final stage.

Uses only Python's standard library so the project works on a bare server. It
supports OpenAI-compatible APIs (OpenAI, Groq, NVIDIA) and Google Gemini.

``LLMClient`` supports a single provider for direct use. For Stage 2 a client
can carry a ``fallback`` client (e.g. Gemini primary, OpenAI fallback) so the
pipeline keeps working even if one provider is unreachable.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from .config import Settings


def _post_json(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    *,
    timeout: float = 60.0,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def openai_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    system: str | None = None,
    max_tokens: int = 700,
    temperature: float = 0.2,
    timeout: float = 60.0,
) -> str:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    data = _post_json(
        f"{base_url.rstrip('/')}/chat/completions",
        {"Authorization": f"Bearer {api_key}"},
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    return data["choices"][0]["message"]["content"].strip()


def gemini_completion(
    *,
    api_key: str,
    model: str,
    prompt: str,
    system: str | None = None,
    max_tokens: int = 700,
    temperature: float = 0.2,
    timeout: float = 60.0,
) -> str:
    """Call the Google Gemini ``generateContent`` endpoint."""
    parts: list[dict[str, str]] = []
    if system:
        parts.append({"text": system})
    parts.append({"text": prompt})
    data = _post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {"x-goog-api-key": api_key},
        {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        },
        timeout=timeout,
    )
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


@dataclass
class LLMClient:
    """Provider-agnostic completion client with optional fallback."""

    provider: str = ""
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    system: str = ""
    max_tokens: int = 700
    temperature: float = 0.2
    timeout: float = 60.0
    fallback: "LLMClient | None" = None

    @classmethod
    def from_settings(cls, settings: Settings, *, provider: str | None = None) -> "LLMClient":
        selected = (provider or settings.resolved_llm_provider or "").lower()
        if not selected:
            return cls()
        if selected == "gemini":
            if not settings.gemini_api_key:
                return cls()
            return cls(
                provider="gemini",
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                timeout=settings.request_timeout,
            )
        if selected == "groq":
            if not settings.groq_api_key:
                return cls()
            return cls(
                provider="groq",
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                base_url="https://api.groq.com/openai/v1",
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                timeout=settings.request_timeout,
            )
        if selected == "nvidia":
            if not settings.nvidia_api_key:
                return cls()
            return cls(
                provider="nvidia",
                api_key=settings.nvidia_api_key,
                model=settings.nvidia_model,
                base_url="https://integrate.api.nvidia.com/v1",
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                timeout=settings.request_timeout,
            )
        if selected == "openai":
            if not settings.openai_api_key:
                return cls()
            return cls(
                provider="openai",
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                base_url=settings.openai_base_url,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                timeout=settings.request_timeout,
            )
        return cls()

    @classmethod
    def stage2_client(cls, settings: Settings) -> "LLMClient":
        """Gemini primary, OpenAI fallback for Stage 2 narrative generation."""
        primary = cls.from_settings(settings, provider="gemini")
        fallback = cls.from_settings(settings, provider="openai")
        if primary.configured and fallback.configured:
            primary.fallback = fallback
            return primary
        if primary.configured:
            return primary
        if fallback.configured:
            return fallback
        return cls()

    @classmethod
    def final_client(cls, settings: Settings) -> "LLMClient":
        """NVIDIA client for the final Stage (summary/panel)."""
        return cls.from_settings(settings, provider="nvidia")

    @property
    def configured(self) -> bool:
        if self.provider and self.api_key and self.model:
            return True
        return self.fallback is not None and self.fallback.configured

    def _complete_direct(self, prompt: str, *, system: str | None = None) -> str | None:
        if not (self.provider and self.api_key and self.model):
            return None
        active_system = system or self.system
        try:
            if self.provider == "gemini":
                return gemini_completion(
                    api_key=self.api_key,
                    model=self.model,
                    prompt=prompt,
                    system=active_system or None,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout,
                )
            return openai_completion(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                prompt=prompt,
                system=active_system or None,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
            )
        except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError, OSError):
            return None

    def complete(self, prompt: str, *, system: str | None = None) -> str | None:
        """Return completion text from primary, then fallback, or ``None``."""
        result = self._complete_direct(prompt, system=system)
        if result is not None:
            return result
        if self.fallback is not None:
            return self.fallback.complete(prompt, system=system)
        return None
