"""Tests for src.validation.matched_operating_point (pure comparison logic)."""

from __future__ import annotations

import numpy as np
import pytest

from src.validation.matched_operating_point import recall_at_matched_precision, roc_auc


def _separable_scores(n_pos=50, n_neg=50, seed=0):
    rng = np.random.default_rng(seed)
    pos_scores = rng.normal(loc=2.0, scale=1.0, size=n_pos)
    neg_scores = rng.normal(loc=-2.0, scale=1.0, size=n_neg)
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(bool)
    return scores, labels


class TestRecallAtMatchedPrecision:
    def test_perfect_separation_yields_high_recall_at_high_precision(self):
        scores, labels = _separable_scores()
        result = recall_at_matched_precision(scores, labels, target_precision=0.95)
        assert result["precision"] >= 0.9
        assert result["recall"] > 0.5

    def test_scale_invariance_monotonic_transform(self):
        """A monotonic rescaling of scores (e.g. calibration) must not change the result."""
        scores, labels = _separable_scores()
        rescaled = 3.0 * scores + 10.0  # affine, monotonic increasing
        r1 = recall_at_matched_precision(scores, labels, target_precision=0.8)
        r2 = recall_at_matched_precision(rescaled, labels, target_precision=0.8)
        assert r1["precision"] == pytest.approx(r2["precision"])
        assert r1["recall"] == pytest.approx(r2["recall"])

    def test_target_precision_out_of_range_raises(self):
        scores, labels = _separable_scores()
        with pytest.raises(ValueError, match="target_precision"):
            recall_at_matched_precision(scores, labels, target_precision=1.5)
        with pytest.raises(ValueError, match="target_precision"):
            recall_at_matched_precision(scores, labels, target_precision=-0.1)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            recall_at_matched_precision(
                np.array([0.1, 0.2, 0.3]), np.array([True, False]), target_precision=0.5
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            recall_at_matched_precision(np.array([]), np.array([]), target_precision=0.5)

    def test_returns_achieved_precision_close_to_target(self):
        scores, labels = _separable_scores(n_pos=200, n_neg=200, seed=1)
        result = recall_at_matched_precision(scores, labels, target_precision=0.6)
        # Achieved precision should be reasonably close (curve is discrete, not exact).
        assert abs(result["precision"] - 0.6) < 0.3

    def test_all_positive_labels_still_returns_a_point(self):
        scores = np.array([0.1, 0.5, 0.9, 0.3])
        labels = np.array([True, True, True, True])
        result = recall_at_matched_precision(scores, labels, target_precision=1.0)
        assert result["precision"] == pytest.approx(1.0)


class TestRocAuc:
    def test_perfect_separation_is_one(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        assert roc_auc(scores, labels) == pytest.approx(1.0)

    def test_random_scores_near_half(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=1000)
        labels = rng.integers(0, 2, size=1000).astype(bool)
        auc = roc_auc(scores, labels)
        assert 0.4 < auc < 0.6

    def test_inverted_ranking_is_near_zero(self):
        scores = np.array([0.9, 0.8, 0.2, 0.1])  # negatives score HIGH
        labels = np.array([0, 0, 1, 1])
        assert roc_auc(scores, labels) == pytest.approx(0.0)
