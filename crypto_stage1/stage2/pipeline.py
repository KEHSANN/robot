"""Stage 2: generated narrative/output pipeline.

Turns the Stage 1 consensus into a publishable narrative. With a configured
LLM, Stage 2 asks the model for a concise, source-aware summary and risk
label. Without a model it uses a deterministic template so the pipeline remains
runnable everywhere.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from services.llm import LLMClient

_NUMERIC_RE = re.compile(r"[-+]?\d*\.?\d+")


@dataclass
class Stage2Input:
    """Input for Stage 2."""

    record_id: str
    event_id: str
    confidence: float
    votes: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    normalised_title: str = ""
    event_label: str = ""
    facts: list[dict[str, Any]] = field(default_factory=list)
    source: str = ""


@dataclass
class Stage2Output:
    """Output produced by Stage 2."""

    record_id: str
    event_id: str
    narrative: str
    risk_label: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_stage2_input(record: Any) -> Stage2Input:
    if isinstance(record, Stage2Input):
        return record
    if isinstance(record, dict):
        return Stage2Input(**record)
    if hasattr(record, "__dict__"):
        return Stage2Input(**vars(record))
    return Stage2Input(**asdict(record))


def _deterministic_narrative(record: Stage2Input) -> str:
    event = record.event_label or record.event_id
    title = record.normalised_title or record.record_id
    if record.confidence >= 0.75:
        confidence_label = "high confidence"
    elif record.confidence >= 0.5:
        confidence_label = "moderate confidence"
    else:
        confidence_label = "low confidence"
    return (
        f"{title} was assigned to event {record.event_id} "
        f"({event}) with {confidence_label} "
        f"({record.confidence:.2f}). {record.rationale.strip('.')}."
    )


def _deterministic_risk(record: Stage2Input) -> str:
    if record.confidence >= 0.75:
        return "high-confidence"
    if record.confidence >= 0.5:
        return "moderate"
    return "low-confidence"


def _risk_from_text(text: str, fallback: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ("high", "major", "critical", "severe", "strong")):
        return "high-confidence"
    if any(word in lowered for word in ("moderate", "medium")):
        return "moderate"
    if any(word in lowered for word in ("low", "weak")):
        return "low-confidence"
    return fallback


def _extract_text_value(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return text.strip()
    candidate = payload.get("narrative") or payload.get("text") or payload.get("summary")
    return str(candidate or text).strip()


class Stage2Pipeline:
    """Stage 2 orchestrator producing narratives."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client

    def _llm_prompt(self, record: Stage2Input) -> str:
        facts = record.facts or []
        return (
            f"Record:\n- title: {record.normalised_title}\n"
            f"- event id: {record.event_id}\n- event label: {record.event_label}\n"
            f"- confidence: {record.confidence:.3f}\n- rationale: {record.rationale}\n"
            f"- facts: {facts}\n- source: {record.source}\n\n"
            'Reply with JSON: {"narrative": "<1-3 sentences>", '
            '"risk_label": "high-confidence|moderate|low-confidence"}'
        )

    def run(self, record: Stage2Input) -> Stage2Output:
        fallback_narrative = _deterministic_narrative(record)
        fallback_risk = _deterministic_risk(record)

        narrative = fallback_narrative
        risk_label = fallback_risk
        used_model = False

        if self.client is not None and self.client.configured:
            text = self.client.complete(
                self._llm_prompt(record),
                system="You summarise crypto event intelligence for a risk dashboard.",
            )
            if text:
                narrative = _extract_text_value(text)
                risk_label = _risk_from_text(text, fallback_risk)
                used_model = True

        return Stage2Output(
            record_id=record.record_id,
            event_id=record.event_id,
            narrative=narrative,
            risk_label=risk_label,
            confidence=record.confidence,
            metadata={
                "used_model": used_model,
                "votes": dict(record.votes),
                "source": record.source,
            },
        )


def run_stage2(records: Iterable[Any], *, client: LLMClient | None = None) -> list[Stage2Output]:
    """Run Stage 2 over an iterable of ``Stage2Input`` or mapping objects."""
    pipeline = Stage2Pipeline(client=client)
    outputs: list[Stage2Output] = []
    for record in records:
        output = pipeline.run(_as_stage2_input(record))
        outputs.append(output)
    return outputs
