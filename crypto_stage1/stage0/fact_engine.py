"""Lightweight fact extraction for crypto event narratives.

The real system delegates most fact extraction to a model. The functions here
provide deterministic, dependency-free fallbacks (ticker extraction, address-
like token detection and sentence chunking) so Stage 0 has a testable shape
until a provider is plugged in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_TICKER_RE = re.compile(r"\b([A-Z]{2,10})\b")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_ADDRESS_RE = re.compile(r"\b(?:0x[a-fA-F0-9]{6,}|(?:bc1|[13])[a-zA-HJ-NP-Z0-9]{20,})\b")


@dataclass
class Fact:
    """A single extracted fact."""

    key: str
    value: Any
    source: str = "rule"
    confidence: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
        }


def split_sentences(text: str) -> list[str]:
    """Split into sentences on common punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [part.strip() for part in parts if part.strip()]


def extract_upper_tokens(text: str) -> list[str]:
    """Return distinct uppercase tokens, useful as candidate tickers."""
    return list(dict.fromkeys(_TICKER_RE.findall(text or "")))


def extract_addresses(text: str) -> list[str]:
    """Return distinct crypto-address-like tokens."""
    return list(dict.fromkeys(_ADDRESS_RE.findall(text or "")))


def extract_facts(text: str) -> list[Fact]:
    """Extract deterministic facts from ``text``."""
    facts: list[Fact] = []
    tokens = extract_upper_tokens(text)
    if tokens:
        facts.append(Fact(key="tickers", value=tokens, source="rule"))

    addresses = extract_addresses(text)
    if addresses:
        facts.append(Fact(key="addresses", value=addresses, source="rule"))

    sentences = split_sentences(text)
    if sentences:
        facts.append(Fact(key="sentences", value=sentences[:20], source="rule"))

    return facts
