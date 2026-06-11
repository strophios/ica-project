# pattern: Functional Core (tests)
"""
Tests for `src.validation.doca_recall` (DoCA-matched recall diagnostic).

Tests cover:
- Recall calculation over DoCA-matched rows (doca_id non-null)
- Topic-skew caveat string in the report
- Synthetic scored_df rows with doca_id + us_score columns
"""

import polars as pl
import pytest

from src.validation.doca_recall import doca_recall, DOCA_TOPIC_SKEW_CAVEAT


class TestDocaRecall:
    """Test doca_recall(scored_df, threshold) -> {recall, n}"""

    def test_perfect_recall_all_scored_above_threshold(self):
        """All DoCA-matched rows with us_score >= threshold.
        Recall = 1.0."""
        df = pl.DataFrame({
            "doca_id": ["d1", "d2", "d3"],
            "us_score": [0.9, 0.8, 0.7],
        })
        result = doca_recall(df, threshold=0.5)
        assert result["recall"] == pytest.approx(1.0)
        assert result["n"] == 3

    def test_zero_recall_none_scored_above_threshold(self):
        """No DoCA-matched rows with us_score >= threshold.
        Recall = 0.0."""
        df = pl.DataFrame({
            "doca_id": ["d1", "d2", "d3"],
            "us_score": [0.3, 0.2, 0.1],
        })
        result = doca_recall(df, threshold=0.5)
        assert result["recall"] == pytest.approx(0.0)
        assert result["n"] == 3

    def test_partial_recall(self):
        """2 of 4 DoCA rows above threshold.
        Recall = 2/4 = 0.5."""
        df = pl.DataFrame({
            "doca_id": ["d1", "d2", "d3", "d4"],
            "us_score": [0.9, 0.6, 0.3, 0.1],
        })
        result = doca_recall(df, threshold=0.5)
        assert result["recall"] == pytest.approx(0.5)
        assert result["n"] == 4

    def test_edge_case_threshold_boundary(self):
        """us_score exactly equal to threshold should count as hit."""
        df = pl.DataFrame({
            "doca_id": ["d1", "d2", "d3"],
            "us_score": [0.5, 0.5, 0.5],
        })
        result = doca_recall(df, threshold=0.5)
        assert result["recall"] == pytest.approx(1.0)
        assert result["n"] == 3

    def test_empty_dataframe(self):
        """Empty scored_df returns NaN recall and n=0."""
        df = pl.DataFrame({
            "doca_id": [],
            "us_score": [],
        })
        result = doca_recall(df, threshold=0.5)
        assert result["n"] == 0
        # recall should be NaN or 0.0 depending on implementation
        # (empty set has no positives to recall)

    def test_no_doca_matched_rows(self):
        """Rows with doca_id=null should be ignored."""
        df = pl.DataFrame({
            "doca_id": [None, "d1", None, "d2"],
            "us_score": [0.9, 0.3, 0.8, 0.6],
        })
        # Only d1 and d2 are counted; d1 is below threshold
        result = doca_recall(df, threshold=0.5)
        assert result["recall"] == pytest.approx(0.5)
        assert result["n"] == 2

    def test_mixed_null_and_valid_doca_ids(self):
        """Only rows with non-null doca_id are included."""
        df = pl.DataFrame({
            "doca_id": ["d1", None, "d2"],
            "us_score": [0.9, 0.7, 0.3],
        })
        # d1 and d2 included; only d1 > 0.5
        result = doca_recall(df, threshold=0.5)
        assert result["recall"] == pytest.approx(0.5)
        assert result["n"] == 2

    def test_topic_skew_caveat_in_report(self):
        """The module-level DOCA_TOPIC_SKEW_CAVEAT string should be present."""
        assert isinstance(DOCA_TOPIC_SKEW_CAVEAT, str)
        assert "topic-skew" in DOCA_TOPIC_SKEW_CAVEAT.lower() or "doca" in DOCA_TOPIC_SKEW_CAVEAT.lower()
        # Caveat should mention that DoCA is skewed toward US protests
        assert len(DOCA_TOPIC_SKEW_CAVEAT) > 20

    def test_different_thresholds(self):
        """Varying threshold should affect recall correctly."""
        df = pl.DataFrame({
            "doca_id": ["d1", "d2", "d3", "d4"],
            "us_score": [0.95, 0.75, 0.45, 0.05],
        })
        # At 0.9: only d1 -> recall = 1/4 = 0.25
        r1 = doca_recall(df, threshold=0.9)
        assert r1["recall"] == pytest.approx(0.25)

        # At 0.5: d1, d2 -> recall = 2/4 = 0.5
        r2 = doca_recall(df, threshold=0.5)
        assert r2["recall"] == pytest.approx(0.5)

        # At 0.0: all -> recall = 1.0
        r3 = doca_recall(df, threshold=0.0)
        assert r3["recall"] == pytest.approx(1.0)
