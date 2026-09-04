"""Embedding provider used by Stage 0.

``EmbeddingService`` is a deterministic hashing fallback that produces a
stable vector without network access. A real OpenAI-compatible or Gemini
backend can be supplied through configuration; the Stage 0 pipeline only needs
the ``embed`` method.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping

from .config import Settings

_DIM = 128


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


def _stable_token_features(text: str) -> list[str]:
    """Return lowercase tokens plus simple character n-grams for a text."""
    tokens = (text or "").lower().split()
    features = list(tokens)
    for token in tokens:
        if len(token) >= 3:
            for size in (2, 3):
                for index in range(len(token) - size + 1):
                    features.append(token[index:index + size])
    return features


def hashed_vector(text: str, *, dimensions: int = _DIM) -> list[float]:
    """Produce a stable, deterministic feature vector from ``text``.

    Features are lowercase tokens plus character n-grams. Each feature is
    hashed into a bucket so the vector is deterministic, L2 normalised and
    ``cosine_similarity`` stays in ``[0, 1]``.
    """
    vector = [0.0] * dimensions
    for feature in _stable_token_features(text):
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


class OpenAIEmbedder:
    """Embedding backend for OpenAI-compatible ``/embeddings`` APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        data = _post_json(
            f"{self.base_url.rstrip('/')}/embeddings",
            {"Authorization": f"Bearer {self.api_key}"},
            {"model": self.model, "input": text},
            timeout=self.timeout,
        )
        return [float(value) for value in data["data"][0]["embedding"]]


class GeminiEmbedder:
    """Embedding backend for the Google Gemini ``embedContent`` API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        data = _post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent",
            {"x-goog-api-key": self.api_key},
            {"content": {"parts": [{"text": text}]}},
            timeout=self.timeout,
        )
        return [float(value) for value in data["embedding"]["values"]]


class EmbeddingService:
    """Small wrapper around an embedding backend.

    If ``backend`` is omitted, ``hashed_vector`` is used. ``backend`` can be a
    callable or an object exposing ``embed(text)``.
    """

    def __init__(
        self,
        backend: Callable[[str], list[float]] | object | None = None,
        *,
        model_name: str = "fallback-hash-v1",
    ) -> None:
        self.backend = backend
        self.model_name = model_name

    @classmethod
    def from_settings(cls, settings: Settings) -> "EmbeddingService":
        provider = settings.resolved_embedding_provider
        if provider == "openai" and settings.openai_api_key:
            return cls(
                backend=OpenAIEmbedder(
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                    model=settings.active_embedding_model,
                    timeout=settings.request_timeout,
                ),
                model_name=settings.active_embedding_model,
            )
        if provider == "gemini" and settings.gemini_api_key:
            return cls(
                backend=GeminiEmbedder(
                    api_key=settings.gemini_api_key,
                    model=settings.active_embedding_model,
                    timeout=settings.request_timeout,
                ),
                model_name=settings.active_embedding_model,
            )
        return cls(model_name="fallback-hash-v1")

    def embed(self, text: str) -> list[float]:
        if self.backend is None:
            return hashed_vector(text)
        if callable(self.backend):
            return list(self.backend(text))
        embedder = getattr(self.backend, "embed", None)
        if embedder is None:
            raise TypeError("embedding backend must be callable or expose .embed(text)")
        try:
            return list(embedder(text))
        except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError, OSError):
            return hashed_vector(text)
