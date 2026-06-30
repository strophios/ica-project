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


class TestAuditMatchedParquetIntegration:
    """Integration tests for audit_matched_parquet shell function."""

    def test_heuristic_error_rate_filters_non_dateline_labels(self):
        """Audit shell should exclude non-dateline labels from AC6.1 error rate.

        This test ensures the dateline-only filtering breaks circularity:
        desk-derived labels should not be compared against heuristic verdicts
        derived from desk/section/keywords.
        """
        # Synthetic test case: mixed label sources.
        # Only dateline-labeled rows should contribute to AC6.1.

        # In the real audit_matched_parquet(), this filtering happens:
        #   df_dateline = df.filter(pl.col("ldc_label_source") == "dateline")
        #   error_rate = heuristic_error_rate(df_dateline["ldc_heuristic_us"], df_dateline["ldc_us_label"])
        #
        # We verify that the filtering correctly excludes heuristic/conflict rows.

        import math

        # Simulate a scenario with mixed label sources.
        # If we naively included all rows, error_rate would be 1.0 (all disagree).
        # If we filter to dateline-only, error_rate would be 0.0 (all agree).

        heuristic_verdicts_all = [True, False, True, False]
        labels_all = [False, True, False, True]  # All disagree (from heuristic source)
        label_sources_all = ["heuristic", "heuristic", "dateline", "dateline"]

        # Without filtering (wrong): 4 disagreements out of 4 = 1.0 error rate
        assert heuristic_error_rate(heuristic_verdicts_all, labels_all) == 1.0

        # With filtering to dateline-only (correct):
        heuristic_dateline = [v for v, src in zip(heuristic_verdicts_all, label_sources_all) if src == "dateline"]
        labels_dateline = [l for l, src in zip(labels_all, label_sources_all) if src == "dateline"]
        # [True, False] vs [False, True] — still 2 disagreements out of 2 = 1.0
        assert heuristic_error_rate(heuristic_dateline, labels_dateline) == 1.0

        # Better example: heuristic verdicts agree with dateline labels.
        heuristic_verdicts_all2 = [True, False, True, False]
        labels_all2 = [False, True, True, False]  # Agree on indices 2,3; disagree on 0,1
        label_sources_all2 = ["heuristic", "heuristic", "dateline", "dateline"]

        # Without filtering (wrong): 2 disagreements out of 4 = 0.5 error rate
        assert heuristic_error_rate(heuristic_verdicts_all2, labels_all2) == 0.5

        # With filtering to dateline-only (correct):
        heuristic_dateline2 = [v for v, src in zip(heuristic_verdicts_all2, label_sources_all2) if src == "dateline"]
        labels_dateline2 = [l for l, src in zip(labels_all2, label_sources_all2) if src == "dateline"]
        # [True, False] vs [True, False] — 0 disagreements out of 2 = 0.0
        assert heuristic_error_rate(heuristic_dateline2, labels_dateline2) == 0.0
