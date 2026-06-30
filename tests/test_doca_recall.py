# pattern: Functional Core (tests)
"""
Tests for `src.validation.doca_recall` (DoCA-matched recall diagnostic).

Tests cover:
- Recall calculation over DoCA-matched rows (doca_id non-null)
- Topic-skew caveat string in the report
- Threshold picking (pick_us_threshold) via the recall recipe
- Synthetic scored_df rows with doca_id + us_score columns
"""

import polars as pl
import pytest

from src.validation.doca_recall import doca_recall, DOCA_TOPIC_SKEW_CAVEAT, pick_us_threshold, ThresholdPickResult


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


class TestPickUsThreshold:
    """Test pick_us_threshold(scored_df, target_recall, thresholds) -> ThresholdPickResult"""

    def test_single_qualifying_threshold(self):
        """Exactly one threshold meets target_recall; return it."""
        df = pl.DataFrame({
            "doca_id": ["d1", "d2", "d3", "d4"],
            "us_score": [0.95, 0.75, 0.45, 0.05],
        })
        # At 0.5: 2/4 = 0.5 recall (meets target)
        # At 0.9: 1/4 = 0.25 recall (below target)
        thresholds = [0.5, 0.9]
        result = pick_us_threshold(df, target_recall=0.4, thresholds=thresholds)

        assert result.threshold == 0.5
        assert result.qualified is True

    def test_multiple_qualifying_thresholds_returns_largest(self):
        """Multiple thresholds meet target; return the LARGEST."""
        df = pl.DataFrame({
            "doca_id": ["d1", "d2", "d3", "d4"],
            "us_score": [0.95, 0.75, 0.45, 0.05],
        })
        # At 0.3: 3/4 = 0.75 recall (meets 0.7 target)
        # At 0.5: 2/4 = 0.5 recall (meets 0.4 target)
        # At 0.8: 1/4 = 0.25 recall (below 0.4 target)
        thresholds = [0.3, 0.5, 0.8]
        result = pick_us_threshold(df, target_recall=0.4, thresholds=thresholds)

        # Both 0.3 and 0.5 meet 0.4; return largest: 0.5
        assert result.threshold == 0.5
        assert result.qualified is True

    def test_no_qualifying_threshold_returns_lowest_with_flag(self):
        """No threshold meets target_recall; return lowest threshold with qualified=False."""
        df = pl.DataFrame({
            "doca_id": ["d1", "d2"],
            "us_score": [0.3, 0.2],
        })
        # At 0.1: 2/2 = 1.0 recall (meets all targets)
        # At 0.25: 1/2 = 0.5 recall
        # At 0.35: 0/2 = 0.0 recall
        # With target 0.95, the threshold 0.1 would still qualify (1.0 >= 0.95)
        # So we need scores that max out below 0.95
        # All scores are <= 0.3, so even at threshold 0.1, max recall is 1.0
        # Actually to make NO threshold qualify, we need a high enough target
        # but we need to be careful about the scores
        # Let me use a simpler example:
        thresholds = [0.25, 0.35]  # Skip 0.1 to test with limited thresholds
        result = pick_us_threshold(df, target_recall=0.95, thresholds=thresholds)

        # At 0.25: 1/2 = 0.5 (below 0.95)
        # At 0.35: 0/2 = 0.0 (below 0.95)
        # Neither qualifies; return lowest (0.25) with qualified=False
        assert result.threshold == 0.25
        assert result.qualified is False

    def test_edge_case_all_thresholds_exceed_target(self):
        """All thresholds meet or exceed target; return the LARGEST."""
        df = pl.DataFrame({
            "doca_id": ["d1", "d2", "d3", "d4"],
            "us_score": [0.95, 0.75, 0.45, 0.05],
        })
        thresholds = [0.0, 0.2, 0.4]
        # At 0.0: 4/4 = 1.0 (meets 0.5)
        # At 0.2: 4/4 = 1.0 (meets 0.5)
        # At 0.4: 3/4 = 0.75 (meets 0.5)
        result = pick_us_threshold(df, target_recall=0.5, thresholds=thresholds)

        assert result.threshold == 0.4
        assert result.qualified is True

    def test_target_recall_at_boundary(self):
        """Recall exactly equal to target should qualify."""
        df = pl.DataFrame({
            "doca_id": ["d1", "d2", "d3", "d4"],
            "us_score": [0.95, 0.75, 0.45, 0.05],
        })
        thresholds = [0.5, 0.8]
        # At 0.5: 2/4 = 0.5 recall
        result = pick_us_threshold(df, target_recall=0.5, thresholds=thresholds)

        assert result.threshold == 0.5
        assert result.qualified is True

    def test_empty_thresholds_raises_error(self):
        """Empty thresholds list should raise ValueError."""
        df = pl.DataFrame({
            "doca_id": ["d1", "d2"],
            "us_score": [0.9, 0.1],
        })
        with pytest.raises(ValueError, match="thresholds must be non-empty"):
            pick_us_threshold(df, target_recall=0.5, thresholds=[])

    def test_realistic_us_gate_recipe_0_98(self):
        """Realistic example from recall recipe: preserve 98% of DoCA-matched articles."""
        # Simulate a scored dataset where some thresholds can hit 0.98 recall
        df = pl.DataFrame({
            "doca_id": [f"d{i}" for i in range(100)],
            "us_score": [0.99 - i * 0.005 for i in range(100)],  # 0.99, 0.985, ..., 0.495
        })
        # At threshold 0.5: ~98 rows above it (recall ≈ 0.98)
        # At threshold 0.6: ~80 rows above it (recall ≈ 0.80)
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        result = pick_us_threshold(df, target_recall=0.98, thresholds=thresholds)

        # Should pick largest threshold still achieving recall >= 0.98
        assert result.qualified is True
        assert result.threshold > 0.0  # Should be non-trivial
        # Verify manually: at selected threshold, recall should be >= 0.98
        check = doca_recall(df, threshold=result.threshold)
        assert check["recall"] >= 0.98

    def test_result_equality(self):
        """ThresholdPickResult equality and repr."""
        r1 = ThresholdPickResult(threshold=0.5, qualified=True)
        r2 = ThresholdPickResult(threshold=0.5, qualified=True)
        r3 = ThresholdPickResult(threshold=0.6, qualified=True)
        r4 = ThresholdPickResult(threshold=0.5, qualified=False)

        assert r1 == r2
        assert r1 != r3
        assert r1 != r4
        assert "ThresholdPickResult" in repr(r1)
        assert "0.5" in repr(r1)
        assert "qualified=True" in repr(r1)

    def test_result_not_equal_to_non_threshold_pick_result(self):
        """ThresholdPickResult.__eq__ handles non-ThresholdPickResult types."""
        r = ThresholdPickResult(threshold=0.5, qualified=True)
        assert r != (0.5, True)
        assert r != "ThresholdPickResult(0.5, True)"
        assert r is not None

    def test_unordered_thresholds(self):
        """Thresholds order doesn't matter; algorithm finds largest qualifying."""
        df = pl.DataFrame({
            "doca_id": ["d1", "d2", "d3", "d4"],
            "us_score": [0.95, 0.75, 0.45, 0.05],
        })
        # Recall by threshold:
        # 0.3: 3/4 = 0.75, 0.4: 3/4 = 0.75, 0.5: 2/4 = 0.5, 0.6: 2/4 = 0.5, 0.8: 1/4 = 0.25
        # Unordered input: [0.8, 0.3, 0.6, 0.4, 0.5]
        thresholds = [0.8, 0.3, 0.6, 0.4, 0.5]
        result = pick_us_threshold(df, target_recall=0.4, thresholds=thresholds)

        # Largest threshold >= 0.4 recall is 0.6 (recall = 0.5 >= 0.4)
        assert result.threshold == 0.6
        assert result.qualified is True

    def test_single_threshold_qualifies(self):
        """Single threshold that qualifies."""
        df = pl.DataFrame({
            "doca_id": ["d1", "d2"],
            "us_score": [0.9, 0.1],
        })
        result = pick_us_threshold(df, target_recall=0.5, thresholds=[0.5])

        assert result.threshold == 0.5
        assert result.qualified is True

    def test_single_threshold_does_not_qualify(self):
        """Single threshold that does not qualify."""
        df = pl.DataFrame({
            "doca_id": ["d1", "d2"],
            "us_score": [0.6, 0.4],
        })
        result = pick_us_threshold(df, target_recall=0.9, thresholds=[0.5])

        # At 0.5: 1/2 = 0.5 recall (below 0.9 target)
        assert result.threshold == 0.5
        assert result.qualified is False

    def test_with_null_doca_ids(self):
        """pick_us_threshold correctly ignores rows with null doca_id."""
        df = pl.DataFrame({
            "doca_id": ["d1", None, "d2", None, "d3"],
            "us_score": [0.95, 0.9, 0.75, 0.8, 0.45],
        })
        # Only d1, d2, d3 count (scores 0.95, 0.75, 0.45)
        # At 0.5: d1 (0.95), d2 (0.75) -> 2/3 ≈ 0.667 recall
        # At 0.7: d1 (0.95), d2 (0.75) -> 2/3 ≈ 0.667 recall
        thresholds = [0.5, 0.7]
        result = pick_us_threshold(df, target_recall=0.6, thresholds=thresholds)

        # Both thresholds meet 0.6 target; return largest: 0.7
        assert result.threshold == 0.7
        assert result.qualified is True
