"""Run a small end-to-end demo through Stage 0, Stage 1 and Stage 2.

Usage:
    cd ~/crypto_stage1
    python3 run_demo.py
"""

from __future__ import annotations

import json

from services.embedding_service import EmbeddingService
from stage0.pipeline import Stage0Record, run_stage0
from stage1.runner import run_stage1
from stage2.pipeline import run_stage2


def main() -> int:
    records = [
        Stage0Record(
            record_id="r1",
            title="Bitcoin ETF approved",
            body="SEC approves spot bitcoin ETF.",
            source="demo",
        ),
        Stage0Record(
            record_id="r2",
            title="Bitcoin ETF approved",
            body="SEC approves spot bitcoin ETF.",
            source="demo",
        ),
        Stage0Record(
            record_id="r3",
            title="Ethereum upgrade",
            body="Ethereum activates next upgrade.",
            source="demo",
        ),
    ]

    stage0_results = run_stage0(records, embedding_service=EmbeddingService())
    stage1_results = run_stage1(stage0_results)

    stage2_inputs = [
        {
            "record_id": item.record_id,
            "event_id": item.event_id,
            "confidence": item.confidence,
            "votes": item.votes,
            "rationale": item.rationale,
        }
        for item in stage1_results
    ]
    stage2_outputs = run_stage2(stage2_inputs)

    for item in stage2_outputs:
        print(json.dumps(item.__dict__, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
