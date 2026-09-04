"""Final stage: NVIDIA-powered market summary.

This is the last stage of the pipeline. When an NVIDIA API key is configured it
asks NVIDIA for a concise market event summary/panel. Without a key it produces
a deterministic summary so end-to-end execution still works on a bare server.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from services.llm import LLMClient


@dataclass
class FinalInput:
    """Stage 2 record used by the final stage."""

    record_id: str
    event_id: str
    narrative: str
    risk_label: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalOutput:
    """Output of the final market summary stage."""

    panel: str
    summary: str
    records: int
    high_confidence: int
    moderate: int
    low_confidence: int
    used_model: bool
    provider: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_final_input(record: Any) -> FinalInput:
    if isinstance(record, FinalInput):
        return record
    if isinstance(record, dict):
        return FinalInput(**record)
    if hasattr(record, "__dict__"):
        return FinalInput(**vars(record))
    return FinalInput(**asdict(record))


def _count_risk(records: list[FinalInput]) -> tuple[int, int, int]:
    high = sum(1 for r in records if r.risk_label == "high-confidence")
    moderate = sum(1 for r in records if r.risk_label == "moderate")
    low = sum(1 for r in records if r.risk_label == "low-confidence")
    return high, moderate, low


def _deterministic_panel(records: list[FinalInput]) -> tuple[str, str]:
    high, moderate, low = _count_risk(records)
    summary = (
        f"Processed {len(records)} events: "
        f"{high} high-confidence, {moderate} moderate, {low} low-confidence."
    )
    lines = [summary]
    for record in records[:10]:
        lines.append(f"- {record.event_id} [{record.risk_label}] {record.narrative}")
    return lines[0], "\n".join(lines)


class FinalPipeline:
    """NVIDIA-powered final panel when available."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client

    def run(self, records: list[FinalInput]) -> FinalOutput:
        panel, summary = _deterministic_panel(records)
        used_model = False
        provider = ""

        if self.client is not None and self.client.configured:
            prompt = (
                "Create a concise crypto event market summary panel from these Stage 2 records.\n\n"
                + "\n".join(
                    f"- event {r.event_id}: {r.narrative} (risk: {r.risk_label}, confidence {r.confidence:.2f})"
                    for r in records
                )
                + "\n\nReturn JSON: {\"panel\": \"<2-4 sentences>\"}"
            )
            text = self.client.complete(
                prompt,
                system="You are a crypto risk analyst producing a concise panel.",
            )
            if text:
                panel = text.strip()
                summary = text.strip()
                used_model = True
                provider = self.client.provider or "nvidia"

        high, moderate, low = _count_risk(records)
        return FinalOutput(
            panel=panel,
            summary=summary,
            records=len(records),
            high_confidence=high,
            moderate=moderate,
            low_confidence=low,
            used_model=used_model,
            provider=provider,
            metadata={"provider": provider},
        )


def run_final(
    records: Iterable[Any],
    *,
    client: LLMClient | None = None,
) -> FinalOutput:
    """Run the final stage on Stage 2 output records."""
    pipeline = FinalPipeline(client=client)
    return pipeline.run([_as_final_input(record) for record in records])
