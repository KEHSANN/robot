"""Tolerant JSON extraction from model output.

Even with JSON mode enabled, the panel models wrap their answers in different
ways: Qwen's reasoning variants emit a ``<think>`` block first, several models
fence the payload in triple backticks, and a few prepend a sentence of prose.
Rather than discard those responses (which would silently shrink the vote count
in every consensus stage), we dig the JSON out.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: Reasoning scratchpads emitted by Qwen-style models before the real answer.
_THINK_BLOCK_RE = re.compile(
    r"<(think|thinking|reasoning|scratchpad)>.*?</\1>", re.DOTALL | re.IGNORECASE
)
#: An unterminated reasoning block — the model ran out of tokens mid-thought.
_OPEN_THINK_RE = re.compile(r"<(think|thinking|reasoning|scratchpad)>", re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def strip_reasoning(text: str) -> str:
    """Remove ``<think>`` style blocks, closed or unterminated."""
    cleaned = _THINK_BLOCK_RE.sub(" ", text)
    match = _OPEN_THINK_RE.search(cleaned)
    if match:
        # Unterminated block: anything after the opening tag is scratch work,
        # but a JSON object may still follow it, so keep both sides and let the
        # balanced-scan below find the payload.
        cleaned = cleaned[: match.start()] + " " + cleaned[match.end() :]
    return cleaned


def _iter_balanced_candidates(text: str):
    """Yield every balanced ``{...}`` / ``[...]`` slice, longest first.

    Scanning respects string literals and escapes so a brace inside a quoted
    value cannot terminate the object early.
    """
    openers = {"{": "}", "[": "]"}
    spans: list[tuple[int, int]] = []

    for start, char in enumerate(text):
        if char not in openers:
            continue
        closer = openers[char]
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == char:
                depth += 1
            elif current == closer:
                depth -= 1
                if depth == 0:
                    spans.append((start, index + 1))
                    break

    for start, end in sorted(spans, key=lambda s: s[1] - s[0], reverse=True):
        yield text[start:end]


def _try_load(candidate: str) -> Any | None:
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except ValueError:
        pass
    # One cheap repair pass: trailing commas are by far the most common defect.
    repaired = _TRAILING_COMMA_RE.sub(r"\1", candidate)
    if repaired != candidate:
        try:
            return json.loads(repaired)
        except ValueError:
            pass
    return None


def extract_json(text: str | None) -> Any | None:
    """Best-effort parse of a model response into Python data.

    Returns ``None`` when nothing JSON-shaped is present, which callers treat as
    a failed vote rather than an exception.
    """
    if not text:
        return None

    parsed = _try_load(text)
    if parsed is not None:
        return parsed

    cleaned = strip_reasoning(text)

    for fenced in _FENCE_RE.findall(cleaned):
        parsed = _try_load(fenced)
        if parsed is not None:
            return parsed

    parsed = _try_load(cleaned)
    if parsed is not None:
        return parsed

    for candidate in _iter_balanced_candidates(cleaned):
        parsed = _try_load(candidate)
        if parsed is not None:
            return parsed
    return None


def extract_json_object(text: str | None) -> dict | None:
    """Like :func:`extract_json` but only accepts an object at the top level."""
    parsed = extract_json(text)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                return item
    return None


# --------------------------------------------------------------------------- #
# coercion helpers — models are inconsistent about types
# --------------------------------------------------------------------------- #

_TRUE_WORDS = {"true", "yes", "y", "1", "keep", "relevant", "bullish", "worth"}
_FALSE_WORDS = {"false", "no", "n", "0", "remove", "drop", "irrelevant", "none"}


def as_bool(value: Any, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_WORDS:
            return True
        if lowered in _FALSE_WORDS:
            return False
    return default


def as_float(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            return float(match.group(0))
    return default


def as_int(value: Any, default: int | None = None) -> int | None:
    result = as_float(value, None)
    return default if result is None else int(round(result))


def as_probability(value: Any, default: float | None = None) -> float | None:
    """Normalise a confidence to 0..1, accepting ``0.87``, ``87`` or ``"87%"``."""
    result = as_float(value, None)
    if result is None:
        return default
    if result > 1.0:
        result = result / 100.0
    return max(0.0, min(1.0, result))


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in re.split(r"[,\n;]", value)]
        return [p for p in parts if p]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                for key in ("name", "asset", "symbol", "value", "text", "claim"):
                    if isinstance(item.get(key), str) and item[key].strip():
                        out.append(item[key].strip())
                        break
            elif item is not None:
                out.append(str(item))
        return out
    return [str(value)]


__all__ = [
    "as_bool",
    "as_float",
    "as_int",
    "as_probability",
    "as_str_list",
    "extract_json",
    "extract_json_object",
    "strip_reasoning",
]
