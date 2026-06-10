# pattern: Functional Core
"""Tests for free heuristic audit metrics (AC6.1, AC6.2, AC6.5)."""

import math
import pytest
from src.validation.free_audit import heuristic_error_rate, lead_similarity


class TestHeuristicErrorRate:
    """Tests for error_rate metric."""

    def test_exact_agreement(self):
        """All heuristic == dateline labels should give 0.0 error rate."""
        heuristic = [True, False, True, False, True]
        dateline = [True, False, True, False, True]
        assert heuristic_error_rate(heuristic, dateline) == 0.0

    def test_exact_disagreement(self):
        """All heuristic != dateline labels should give 1.0 error rate."""
        heuristic = [True, False, True, False]
        dateline = [False, True, False, True]
        assert heuristic_error_rate(heuristic, dateline) == 1.0

    def test_partial_disagreement(self):
        """2 out of 4 disagreement should give 0.5 error rate."""
        heuristic = [True, False, True, False]
        dateline = [True, True, False, False]
        assert heuristic_error_rate(heuristic, dateline) == 0.5

    def test_none_values_filtered(self):
        """None values in heuristic or dateline should be excluded from calculation."""
        heuristic = [True, None, True, False]
        dateline = [True, True, False, False]
        # Only compare (True, True) agree, (True, False) disagree, (False, False) agree
        # (None, True) is excluded. So 1 disagreement out of 3 = 1/3 error rate
        assert heuristic_error_rate(heuristic, dateline) == 1/3

    def test_none_on_dateline_filtered(self):
        """None values on dateline side should be excluded."""
        heuristic = [True, False, True, False]
        dateline = [True, None, False, False]
        # Only compare (True, True) agree, (False, False) agree, (True, False) disagree
        # (False, None) is excluded. So 1 disagreement out of 3 = 1/3 error rate
        assert heuristic_error_rate(heuristic, dateline) == 1/3

    def test_empty_pairs(self):
        """Empty or all-none pairs should return NaN."""
        assert math.isnan(heuristic_error_rate([], []))
        assert math.isnan(heuristic_error_rate([None, None], [None, None]))


class TestLeadSimilarity:
    """Tests for lead paragraph similarity metric."""

    def test_identical_texts(self):
        """Identical texts should have similarity = 1.0."""
        texts = ["hello world test", "hello world test", "hello world test"]
        apis = ["hello world test", "hello world test", "hello world test"]
        sim = lead_similarity(texts, apis)
        assert sim == 1.0

    def test_empty_similarity(self):
        """Completely different texts should have low similarity."""
        texts = ["abcdefgh"]
        apis = ["xyzzzzz"]
        sim = lead_similarity(texts, apis)
        assert 0.0 <= sim < 0.3

    def test_high_similarity_different_case(self):
        """Case differences should normalize to high similarity."""
        texts = ["Hello World Test"]
        apis = ["hello world test"]
        sim = lead_similarity(texts, apis)
        assert sim == 1.0  # Both normalize to same

    def test_high_similarity_with_punctuation(self):
        """Punctuation removal should lead to high similarity."""
        texts = ["Hello-World!!"]
        apis = ["helloworld"]
        sim = lead_similarity(texts, apis)
        assert sim == 1.0  # Both normalize to "helloworld"

    def test_mean_of_multiple(self):
        """Multiple pairs should return mean similarity."""
        texts = ["abc", "xyz"]
        apis = ["abc", "xyz"]
        sim = lead_similarity(texts, apis)
        # Both are 1.0, so mean is 1.0
        assert sim == 1.0

    def test_mean_with_mixed_similarity(self):
        """Mixed similarities should return correct mean."""
        texts = ["hello", "world"]
        apis = ["hello", "xyz12345"]
        sim = lead_similarity(texts, apis)
        # First is 1.0 (identical after norm), second is much lower
        # Should be (1.0 + low) / 2, so < 0.6
        assert 0.3 < sim < 0.7

    def test_empty_similarity_list(self):
        """Empty lists should return NaN."""
        sim = lead_similarity([], [])
        assert math.isnan(sim)

    def test_none_values_in_text(self):
        """None values should be treated as empty strings in normalization."""
        texts = [None, "hello"]
        apis = ["", "hello"]
        sim = lead_similarity(texts, apis)
        # First: None normalizes to "", apis is "", so 1.0
        # Second: "hello" normalizes to "hello", apis is "hello", so 1.0
        # Mean is 1.0
        assert sim == 1.0
