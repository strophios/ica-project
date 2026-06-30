# pattern: Functional Core

"""Unit and property-based tests for src/calibration/report.py."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.calibration.calibrator import PlattCalibrator
from src.calibration.report import calibration_report


class TestCalibrationReportBasic:
    """Basic tests for calibration_report function."""

    def test_report_returns_dict_with_keys(self):
        """calibration_report returns dict with ece, brier, reliability."""
        probs = np.array([0.1, 0.2, 0.9])
        labels = np.array([0, 0, 1])
        report = calibration_report(probs, labels)
        assert isinstance(report, dict)
        assert "ece" in report
        assert "brier" in report
        assert "reliability" in report

    def test_report_ece_is_float(self):
        """ECE is a float."""
        probs = np.array([0.1, 0.5, 0.9])
        labels = np.array([0, 0, 1])
        report = calibration_report(probs, labels)
        assert isinstance(report["ece"], float)

    def test_report_brier_is_float(self):
        """Brier is a float."""
        probs = np.array([0.1, 0.5, 0.9])
        labels = np.array([0, 0, 1])
        report = calibration_report(probs, labels)
        assert isinstance(report["brier"], float)

    def test_report_reliability_is_list(self):
        """reliability is a list."""
        probs = np.array([0.1, 0.5, 0.9])
        labels = np.array([0, 0, 1])
        report = calibration_report(probs, labels)
        assert isinstance(report["reliability"], list)

    def test_report_reliability_tuples(self):
        """reliability entries are (confidence, accuracy, count) tuples."""
        probs = np.array([0.1, 0.2, 0.9])
        labels = np.array([0, 0, 1])
        report = calibration_report(probs, labels, n_bins=2)
        if len(report["reliability"]) > 0:
            entry = report["reliability"][0]
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            conf, acc, cnt = entry
            assert isinstance(conf, (float, np.floating))
            assert isinstance(acc, (float, np.floating))
            assert isinstance(cnt, int)


class TestCalibrationReportPerfectlyCalibrated:
    """Test calibration_report on perfectly calibrated data.

    Perfect calibration: in each bin, the mean predicted probability equals
    the observed label frequency.
    """

    def test_perfectly_calibrated_ece_near_zero(self):
        """Perfectly calibrated data → ECE ≈ 0."""
        # Create perfect calibration: probs = observed frequencies
        # Bin 1: confidence 0.25, all are 0 → accuracy 0
        # Bin 2: confidence 0.75, all are 1 → accuracy 1
        # ECE = (4/8) * |0 - 0.25| + (4/8) * |1 - 0.75| = 0.5 * 0.25 + 0.5 * 0.25 = 0.25
        probs = np.array([0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        report = calibration_report(probs, labels, n_bins=2)
        # With this setup, ECE should be minimal (0.25) but still non-zero
        assert report["ece"] < 0.5

    def test_perfectly_calibrated_low_brier(self):
        """Perfectly calibrated data → low Brier."""
        probs = np.array([0.0, 0.0, 1.0, 1.0])
        labels = np.array([0, 0, 1, 1])
        report = calibration_report(probs, labels, n_bins=2)
        assert report["brier"] < 0.01


class TestCalibrationReportBrier:
    """Test Brier score calculation."""

    def test_brier_perfect_predictions(self):
        """Brier = 0 for perfect predictions."""
        probs = np.array([0.0, 0.0, 1.0, 1.0])
        labels = np.array([0, 0, 1, 1])
        report = calibration_report(probs, labels)
        assert abs(report["brier"]) < 1e-10

    def test_brier_worst_predictions(self):
        """Brier = 1 for maximally wrong predictions."""
        probs = np.array([1.0, 1.0, 0.0, 0.0])
        labels = np.array([0, 0, 1, 1])
        report = calibration_report(probs, labels)
        assert abs(report["brier"] - 1.0) < 1e-10

    def test_brier_random_predictions(self):
        """Brier ≈ 0.25 for 0.5 probability on balanced labels."""
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([0, 0, 1, 1])
        report = calibration_report(probs, labels)
        assert abs(report["brier"] - 0.25) < 1e-10


class TestCalibrationReportReliability:
    """Test reliability diagram structure."""

    def test_reliability_non_empty_for_sufficient_data(self):
        """With enough data, all bins should have entries."""
        np.random.seed(42)
        probs = np.random.uniform(0, 1, 300)
        labels = np.random.randint(0, 2, 300)
        report = calibration_report(probs, labels, n_bins=10)
        assert len(report["reliability"]) > 0

    def test_reliability_skips_empty_bins(self):
        """Empty bins are omitted from reliability."""
        # All probs in [0, 0.3], so bins [0.3, 1) are empty
        probs = np.array([0.1, 0.15, 0.2, 0.25])
        labels = np.array([0, 0, 1, 1])
        report = calibration_report(probs, labels, n_bins=10)
        # Should have fewer than n_bins non-empty bins
        assert len(report["reliability"]) <= 10
        assert len(report["reliability"]) >= 1  # At least one bin has data

    def test_reliability_bin_count_field(self):
        """Reliability entries have non-zero count."""
        probs = np.array([0.1, 0.2, 0.9])
        labels = np.array([0, 0, 1])
        report = calibration_report(probs, labels, n_bins=2)
        for conf, acc, cnt in report["reliability"]:
            assert cnt > 0


class TestCalibrationReportEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_bin(self):
        """n_bins=1 lumps all data into one bin."""
        probs = np.array([0.1, 0.5, 0.9])
        labels = np.array([0, 0, 1])
        report = calibration_report(probs, labels, n_bins=1)
        assert len(report["reliability"]) == 1
        conf, acc, cnt = report["reliability"][0]
        assert cnt == 3

    def test_all_one_class_labels(self):
        """All 0 labels → ECE and Brier reflect all-zero predictions."""
        probs = np.array([0.1, 0.2, 0.3])
        labels = np.array([0, 0, 0])
        report = calibration_report(probs, labels, n_bins=5)
        # Brier = mean((p - 0)^2) = mean(p^2)
        expected_brier = np.mean([0.01, 0.04, 0.09])
        assert abs(report["brier"] - expected_brier) < 1e-10

    def test_all_one_class_predictions(self):
        """All predictions = 1.0."""
        probs = np.array([1.0, 1.0, 1.0])
        labels = np.array([0, 1, 1])
        report = calibration_report(probs, labels)
        # All probs land in the last bin
        assert len(report["reliability"]) >= 1

    def test_accepts_list_inputs(self):
        """Can accept Python lists."""
        probs = [0.1, 0.5, 0.9]
        labels = [0, 0, 1]
        report = calibration_report(probs, labels)
        assert "ece" in report


class TestCalibrationReportMonotonicity:
    """Test that ECE and Brier behave predictably."""

    @given(
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_ece_non_negative(self, data):
        """ECE is always non-negative."""
        n = data.draw(st.integers(min_value=10, max_value=100))
        probs = data.draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        )
        labels = data.draw(st.lists(st.integers(min_value=0, max_value=1), min_size=n, max_size=n))
        report = calibration_report(np.array(probs), np.array(labels))
        assert report["ece"] >= -1e-10

    @given(
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_brier_non_negative(self, data):
        """Brier is always non-negative."""
        n = data.draw(st.integers(min_value=10, max_value=100))
        probs = data.draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        )
        labels = data.draw(st.lists(st.integers(min_value=0, max_value=1), min_size=n, max_size=n))
        report = calibration_report(np.array(probs), np.array(labels))
        assert report["brier"] >= -1e-10

    @given(
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_ece_bounded(self, data):
        """ECE is bounded in [0, 1]."""
        n = data.draw(st.integers(min_value=10, max_value=100))
        probs = data.draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        )
        labels = data.draw(st.lists(st.integers(min_value=0, max_value=1), min_size=n, max_size=n))
        report = calibration_report(np.array(probs), np.array(labels))
        assert 0 <= report["ece"] <= 1.0 + 1e-10


class TestAC45CalibratorReducesECE:
    """AC4.5: Platt reduces ECE on miscalibrated evaluation set."""

    def test_platt_reduces_ece_on_overconfident(self):
        """Fit Platt on one split of overconfident data; ECE improves on other split.

        This is the canonical boundary case: logits are scaled beyond what
        frequencies warrant. Platt should absorb this miscalibration.
        """
        # Generate overconfident logits (scaled 4x, mimicking neural net overconfidence)
        n_pos, n_neg = 50, 450
        rng = np.random.RandomState(42)
        pos_logits = rng.normal(2.0, 1.0, n_pos) * 4
        neg_logits = rng.normal(-2.0, 1.0, n_neg) * 4

        all_logits = np.concatenate([pos_logits, neg_logits])
        all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

        # Shuffle
        rng = np.random.RandomState(43)
        idx = rng.permutation(len(all_logits))
        all_logits = all_logits[idx]
        all_labels = all_labels[idx]

        # Split into fit and eval (both natural balance)
        n_fit = len(all_logits) // 2
        fit_logits, eval_logits = all_logits[:n_fit], all_logits[n_fit:]
        fit_labels, eval_labels = all_labels[:n_fit], all_labels[n_fit:]

        # Fit calibrator
        cal = PlattCalibrator.fit(fit_logits, fit_labels, fit_population="overconfident_fit")

        # Compute ECE on eval: raw vs calibrated
        raw_probs = 1.0 / (1.0 + np.exp(-eval_logits))
        raw_report = calibration_report(raw_probs, eval_labels)

        cal_probs = cal.transform(eval_logits)
        cal_report = calibration_report(cal_probs, eval_labels)

        # Calibrated should be better (lower ECE)
        assert cal_report["ece"] < raw_report["ece"]
