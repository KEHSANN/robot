"""Tests for Stage 1 majority consensus."""

from __future__ import annotations

import unittest

from stage1.consensus import majority_consensus
from stage1.schemas import Stage1Decision


class ConsensusTests(unittest.TestCase):
    def test_majority_wins(self) -> None:
        decisions = [
            Stage1Decision(record_id="r1", event_id="e1", confidence=0.9),
            Stage1Decision(record_id="r1", event_id="e1", confidence=0.8),
            Stage1Decision(record_id="r1", event_id="e2", confidence=0.7),
        ]
        result = majority_consensus(decisions)
        self.assertEqual(result.event_id, "e1")
        self.assertAlmostEqual(result.confidence, 0.85)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            majority_consensus([])


if __name__ == "__main__":
    unittest.main()
