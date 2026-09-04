"""LLM-based voter for Stage 1 consensus.

The voter asks the configured model to choose one of the known event ids. It
never crashes the pipeline: any invalid/missing answer falls back to the Stage 0
assignment with a ``rule`` source.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from services.llm import LLMClient

from .schemas import Stage0Output, Stage1Decision

_SYSTEM = (
    "You choose the most appropriate event cluster for a news record. "
    "Respond only with a JSON object."
)


def _json_from_text(text: str) -> dict:
    # Strip markdown code fences if a model wraps the answer.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def build_event_candidates(records: Iterable[Stage0Output]) -> list[tuple[str, str]]:
    """Return ``[(event_id, label)]`` in first-seen order."""
    seen: dict[str, str] = {}
    for record in records:
        seen.setdefault(record.event_id, record.event_label)
    return list(seen.items())


def llm_event_decision(
    client: LLMClient,
    record: Stage0Output,
    candidates: list[tuple[str, str]],
) -> Stage1Decision:
    """Ask ``client`` to vote on ``record``'s event assignment."""
    candidate_lines = "".join(
        f"- {event_id}: {label}\n" for event_id, label in candidates
    )
    prompt = (
        f"Record id: {record.record_id}\n"
        f"Title: {record.normalised_title}\n"
        f"Stage 0 event: {record.event_id}\n"
        f"Similarity score: {record.similarity_score:.3f}\n"
        f"Facts: {record.facts}\n\n"
        f"Known events:\n{candidate_lines}\n\n"
        'Reply with JSON: {"event_id": "<one known event id>", '
        '"confidence": 0.0-1.0, "reasoning": "short reason"}'
    )
    text = client.complete(prompt, system=_SYSTEM)
    if not text:
        return Stage1Decision(
            record_id=record.record_id,
            event_id=record.event_id,
            confidence=record.similarity_score or 1.0,
            reasoning="llm unavailable, inherited stage0 assignment",
            model=client.model or "rule",
        )

    try:
        payload = _json_from_text(text)
    except (json.JSONDecodeError, ValueError):
        return Stage1Decision(
            record_id=record.record_id,
            event_id=record.event_id,
            confidence=record.similarity_score or 1.0,
            reasoning="llm returned invalid json, inherited stage0 assignment",
            model=client.model or "rule",
        )

    known = {event_id for event_id, _ in candidates}
    event_id = str(payload.get("event_id", "")).strip()
    if event_id not in known:
        event_id = record.event_id
    confidence = float(payload.get("confidence", record.similarity_score or 1.0))
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(payload.get("reasoning", "")).strip() or "llm consensus"
    return Stage1Decision(
        record_id=record.record_id,
        event_id=event_id,
        confidence=confidence,
        reasoning=reasoning,
        model=client.model or "llm",
    )
