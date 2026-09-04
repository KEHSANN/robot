"""Tests for Stage 0 similarity primitives.

This test intentionally runs with ``unittest`` so it works without pytest:
    python -m unittest discover -s tests -p '*_test.py' -v
"""

from __future__ import annotations

import unittest

from stage0.similarity import cosine_similarity, text_similarity
from stage0.normalizer import normalize_text
from stage0.dedup import DedupIndex, content_hash


class SimilarityTests(unittest.TestCase):
    def test_cosine_identical(self) -> None:
        self.assertEqual(cosine_similarity([1, 0, 0], [1, 0, 0]), 1.0)

    def test_cosine_orthogonal(self) -> None:
        self.assertEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_cosine_empty(self) -> None:
        self.assertEqual(cosine_similarity([], []), 0.0)

    def test_cosine_negative_tokens_clamped(self) -> None:
        self.assertEqual(cosine_similarity([1, 0], [-1, 0]), 0.0)

    def test_text_similarity_same_words(self) -> None:
        self.assertEqual(text_similarity("bitcoin etf approval", "bitcoin etf approval"), 1.0)

    def test_text_similarity_related(self) -> None:
        score = text_similarity("bitcoin etf", "bitcoin spot etf")
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)


class NormalizerTests(unittest.TestCase):
    def test_normalise_deterministic(self) -> None:
        self.assertEqual(
            normalize_text("  Bitcoin <b>ETF</b>  "),
            "bitcoin etf",
        )

    def test_normalise_preserves_casing_option(self) -> None:
        self.assertEqual(normalize_text("BTC", lowercase=False), "BTC")


class DedupTests(unittest.TestCase):
    def test_first_record_wins(self) -> None:
        index = DedupIndex()
        first_duplicate, first_canonical = index.add("r1", "same token")
        second_duplicate, second_canonical = index.add("r2", "same token")
        self.assertFalse(first_duplicate)
        self.assertIsNone(first_canonical)
        self.assertTrue(second_duplicate)
        self.assertEqual(second_canonical, "r1")

    def test_content_hash_stable(self) -> None:
        self.assertEqual(content_hash("abc"), content_hash("abc"))
        self.assertNotEqual(content_hash("abc"), content_hash("abd"))


if __name__ == "__main__":
    unittest.main()
