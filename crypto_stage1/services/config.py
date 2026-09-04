"""Dependency-free configuration for crypto_stage1.

Reads provider keys from environment variables and (optionally) a local ``.env``
file. The project intentionally avoids requiring ``python-dotenv`` so these
checks/green tests run on a bare Python install.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


def load_dotenv(path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Load ``KEY=VALUE`` lines from ``path`` into ``os.environ``.

    Existing environment variables win, matching ``python-dotenv`` semantics.
    Returns the parsed mapping.
    """
    target = Path(path or ".env")
    if not target.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        loaded[key] = value
        if key and key not in os.environ:
            os.environ[key] = value
    return loaded


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


@dataclass
class Settings:
    """Runtime settings for the pipeline.

    ``provider`` is resolved lazily by ``LLMClient``/``EmbeddingService`` so this
    dataclass has no hidden import-time behaviour.
    """

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    nvidia_api_key: str = ""
    nvidia_model: str = "meta/llama-3.1-8b-instruct"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "text-embedding-004"

    llm_provider: str = ""
    embedding_provider: str = ""
    embedding_model: str = ""
    database_url: str = ""

    request_timeout: float = 60.0
    max_tokens: int = 700
    temperature: float = 0.2

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        mapping = env if env is not None else os.environ
        load_dotenv()

        def get(*names: str, default: str = "") -> str:
            for name in names:
                value = str(mapping.get(name, "") or "").strip()
                if value:
                    return value
            return default

        openai_base = get("OPENAI_BASE_URL", default="https://api.openai.com/v1")
        return cls(
            openai_api_key=get("OPENAI_API_KEY"),
            openai_base_url=openai_base,
            openai_model=get("OPENAI_MODEL", default="gpt-4o-mini"),
            openai_embedding_model=get(
                "OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small"
            ),
            groq_api_key=get("GROQ_API_KEY"),
            groq_model=get("GROQ_MODEL", default="llama-3.3-70b-versatile"),
            nvidia_api_key=get("NVIDIA_API_KEY"),
            nvidia_model=get("NVIDIA_MODEL", default="meta/llama-3.1-8b-instruct"),
            gemini_api_key=get("GEMINI_API_KEY"),
            gemini_model=get("GEMINI_MODEL", default="gemini-2.0-flash"),
            gemini_embedding_model=get(
                "GEMINI_EMBEDDING_MODEL", default="text-embedding-004"
            ),
            llm_provider=get("LLM_PROVIDER").lower(),
            embedding_provider=get("EMBEDDING_PROVIDER").lower(),
            embedding_model=get("EMBEDDING_MODEL"),
            database_url=get("DATABASE_URL"),
            request_timeout=float(get("REQUEST_TIMEOUT", default="60") or "60"),
            max_tokens=int(get("MAX_TOKENS", default="700") or "700"),
            temperature=float(get("TEMPERATURE", default="0.2") or "0.2"),
        )

    @property
    def resolved_llm_provider(self) -> str:
        if self.llm_provider:
            return self.llm_provider
        if self.openai_api_key:
            return "openai"
        if self.groq_api_key:
            return "groq"
        if self.nvidia_api_key:
            return "nvidia"
        if self.gemini_api_key:
            return "gemini"
        return ""

    @property
    def has_llm_key(self) -> bool:
        return bool(self.resolved_llm_provider)

    @property
    def resolved_embedding_provider(self) -> str:
        if self.embedding_provider:
            return self.embedding_provider
        if self.embedding_model:
            return "openai"
        if self.openai_api_key:
            return "openai"
        if self.gemini_api_key:
            return "gemini"
        return ""

    @property
    def active_llm_model(self) -> str:
        provider = self.resolved_llm_provider
        if provider == "openai":
            return self.openai_model
        if provider == "groq":
            return self.groq_model
        if provider == "nvidia":
            return self.nvidia_model
        if provider == "gemini":
            return self.gemini_model
        return ""

    @property
    def active_embedding_model(self) -> str:
        if self.embedding_model:
            return self.embedding_model
        provider = self.resolved_embedding_provider
        if provider == "gemini":
            return self.gemini_embedding_model
        return self.openai_embedding_model
