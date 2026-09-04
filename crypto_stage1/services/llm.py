"""LLM client used by Stage 1 and Stage 2.

Uses only Python's standard library so the project works on a bare server. It
supports OpenAI-compatible APIs (OpenAI, Groq, NVIDIA) and Google Gemini.

If no usable key is configured the client returns ``None``; callers then fall
back to deterministic rules.
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
    """Provider-agnostic completion client."""

    provider: str = ""
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    system: str = ""
    max_tokens: int = 700
    temperature: float = 0.2
    timeout: float = 60.0

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMClient":
        provider = settings.resolved_llm_provider
        if not provider:
            return cls()
        if provider == "gemini":
            return cls(
                provider="gemini",
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                timeout=settings.request_timeout,
            )
        if provider == "groq":
            return cls(
                provider="groq",
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                base_url="https://api.groq.com/openai/v1",
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                timeout=settings.request_timeout,
            )
        if provider == "nvidia":
            return cls(
                provider="nvidia",
                api_key=settings.nvidia_api_key,
                model=settings.nvidia_model,
                base_url="https://integrate.api.nvidia.com/v1",
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                timeout=settings.request_timeout,
            )
        return cls(
            provider="openai",
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
        )

    @property
    def configured(self) -> bool:
        return bool(self.provider and self.api_key and self.model)

    def complete(self, prompt: str, *, system: str | None = None) -> str | None:
        """Return completion text, or ``None`` if not usable / on failure."""
        if not self.configured:
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
