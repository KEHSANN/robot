"""End-to-end shape test through Stage 0, Stage 1 and Stage 2."""

from __future__ import annotations

import unittest

from services.embedding_service import EmbeddingService
from stage0.pipeline import run_stage0
from stage0.pipeline import Stage0Record
from stage1.runner import run_stage1
from stage2.pipeline import run_stage2


class StagePipelineTest(unittest.TestCase):
    def test_stage0_assigns_events(self) -> None:
        embedding = EmbeddingService()
        records = [
            Stage0Record(record_id="r1", title="Bitcoin ETF approved", body="SEC approves spot bitcoin ETF."),
            Stage0Record(record_id="r2", title="Bitcoin ETF approved", body="SEC approves spot bitcoin ETF."),
            Stage0Record(record_id="r3", title="Ethereum upgrade", body="Ethereum activates next upgrade."),
        ]
        results = run_stage0(records, embedding_service=embedding)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].event_id, "r1")
        self.assertEqual(results[1].event_id, "r1")
        self.assertNotEqual(results[1].event_id, "r2")
        self.assertTrue(results[1].deduplicated)

        stage1 = run_stage1(results)
        self.assertEqual(len(stage1), 3)
        self.assertEqual(stage1[0].event_id, "r1")

        stage2_inputs = [
            {
                "record_id": item.record_id,
                "event_id": item.event_id,
                "confidence": item.confidence,
                "votes": item.votes,
                "rationale": item.rationale,
            }
            for item in stage1
        ]
        outputs = run_stage2(stage2_inputs)
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(output.narrative for output in outputs))


if __name__ == "__main__":
    unittest.main()
