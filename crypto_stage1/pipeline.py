"""Top-level orchestrator for Stage 0 -> Stage 1 -> Stage 2 -> Final.

Run a JSONL feed of raw records with:
    python3 main.py --input feed.jsonl --output out.jsonl

Stage assignments:
- Stage 0: Gemini Embedding 2 (`gemini-embedding-002`) when a Gemini key exists.
- Stage 2: Gemini primary, OpenAI fallback.
- Final: NVIDIA.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

from services.config import Settings
from services.embedding_service import EmbeddingService
from services.llm import LLMClient
from stage0.pipeline import Stage0Record, Stage0Result, run_stage0
from stage1.runner import run_stage1
from stage1.schemas import Stage1Result
from stage2.pipeline import Stage2Input, run_stage2
from stage3.pipeline import run_final


def _record_from_mapping(mapping: dict[str, Any]) -> Stage0Record:
    title = str(mapping.get("title") or mapping.get("normalised_title") or "")
    body = str(mapping.get("body") or mapping.get("text") or mapping.get("content") or "")
    source = str(mapping.get("source") or mapping.get("feed") or "")
    return Stage0Record(
        record_id=str(
            mapping.get("record_id")
            or mapping.get("id")
            or mapping.get("url")
            or f"record-{len(title) + len(body)}"
        ),
        title=title,
        body=body,
        source=source,
        raw=mapping,
    )


def _build_stage2_input(stage0: Stage0Result, stage1: Stage1Result) -> Stage2Input:
    return Stage2Input(
        record_id=stage0.record_id,
        event_id=stage1.event_id,
        confidence=stage1.confidence,
        votes=stage1.votes,
        rationale=stage1.rationale,
        normalised_title=stage0.normalised_title,
        event_label=stage0.event_label,
        facts=stage0.facts,
        source=stage0.source,
    )


def run_pipeline(
    records: Iterable[Any],
    *,
    settings: Settings | None = None,
    use_model_voter: bool = True,
    include_final: bool = True,
) -> list[dict[str, Any]]:
    """Run the full pipeline and return Stage 2 (+ Final) outputs as dicts."""
    active = settings or Settings.from_env()
    embedding = EmbeddingService.from_settings(active)
    stage1_model = LLMClient.from_settings(active)
    stage2_client = LLMClient.stage2_client(active)
    final_client = LLMClient.final_client(active)

    stage0_records = [_record_from_mapping(item) for item in records]
    stage0_results = run_stage0(stage0_records, embedding_service=embedding)
    stage1_results = run_stage1(
        stage0_results,
        model_client=stage1_model,
        use_model_voter=use_model_voter,
    )
    stage2_inputs = [
        _build_stage2_input(stage0_raw, stage1_raw)
        for stage0_raw, stage1_raw in zip(stage0_results, stage1_results)
    ]
    stage2_outputs = run_stage2(stage2_inputs, client=stage2_client)
    outputs: list[dict[str, Any]] = [asdict(output) for output in stage2_outputs]

    if include_final:
        final = run_final(stage2_outputs, client=final_client)
        final_dict = asdict(final)
        final_dict["stage"] = "final"
        outputs.append(final_dict)

    return outputs
