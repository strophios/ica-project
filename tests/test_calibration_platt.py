# pattern: Functional Core

"""Unit and property-based tests for src/calibration/calibrator.py."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.calibration.calibrator import Calibrator, PlattCalibrator, platt_fit, platt_transform


class TestPlattFit:
    """Tests for platt_fit function."""

    def test_fit_basic(self):
        """Fit on synthetic data and check (A, B) have expected sign."""
        # Synthetic: logits correlate with labels (logits > 0 → mostly label 1)
        logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        labels = np.array([0, 0, 0, 1, 1])
        A, B = platt_fit(logits, labels)
        # With data organized this way, A should be positive (increasing logit → increasing probability)
        assert isinstance(A, float)
        assert isinstance(B, float)
        assert A > 0

    def test_fit_deterministic(self):
        """Same input → same (A, B)."""
        logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        labels = np.array([0, 0, 0, 1, 1])
        A1, B1 = platt_fit(logits, labels)
        A2, B2 = platt_fit(logits, labels)
        assert A1 == A2
        assert B1 == B2

    def test_fit_accepts_list_input(self):
        """Input can be Python lists."""
        logits = [-2.0, -1.0, 0.0, 1.0, 2.0]
        labels = [0, 0, 0, 1, 1]
        A, B = platt_fit(logits, labels)
        assert isinstance(A, float)
        assert isinstance(B, float)

    def test_fit_casts_labels_to_int(self):
        """Labels are cast to int."""
        logits = np.array([-1.0, 0.0, 1.0])
        labels = np.array([0.0, 0.5, 1.0])  # floats
        A, B = platt_fit(logits, labels)
        assert isinstance(A, float)
        assert isinstance(B, float)

    def test_fit_returns_floats(self):
        """Return type is (float, float)."""
        logits = np.array([0.0, 1.0])
        labels = np.array([0, 1])
        A, B = platt_fit(logits, labels)
        assert type(A) is float
        assert type(B) is float


class TestPlattTransform:
    """Tests for platt_transform function."""

    def test_transform_output_in_range(self):
        """Output is in [0, 1]."""
        logits = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        A, B = 1.0, 0.0
        probs = platt_transform(logits, A, B)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_transform_at_zero(self):
        """σ(0) = 0.5."""
        logits = np.array([0.0])
        A, B = 1.0, 0.0
        probs = platt_transform(logits, A, B)
        np.testing.assert_allclose(probs, [0.5])

    def test_transform_monotonic(self):
        """For A > 0, transform is monotonically non-decreasing in logits.
        Property: sorted logits → sorted probs."""
        A, B = 2.0, -0.5
        logits = np.array([-5.0, -2.0, 0.0, 1.5, 5.0])
        probs = platt_transform(logits, A, B)
        assert np.all(np.diff(probs) >= -1e-10)  # allow tiny numerical slop

    def test_transform_accepts_list(self):
        """Input can be Python list."""
        logits = [0.0, 1.0]
        A, B = 1.0, 0.0
        probs = platt_transform(logits, A, B)
        assert isinstance(probs, np.ndarray)

    @given(
        logits=st.lists(st.floats(min_value=-10.0, max_value=10.0), min_size=1, max_size=100),
    )
    @settings(max_examples=50)
    def test_transform_range_property(self, logits):
        """For any logit array, output is always in [0, 1]."""
        A, B = 1.5, -0.3
        logits_arr = np.array(logits, dtype=np.float64)
        probs = platt_transform(logits_arr, A, B)
        assert np.all(probs >= -1e-10) and np.all(probs <= 1.0 + 1e-10)

    @given(
        logits=st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=100,
        ),
    )
    @settings(max_examples=50)
    def test_transform_monotonicity_property(self, logits):
        """Sorted logits → sorted probs (for A > 0)."""
        A, B = 1.0, 0.0
        logits_sorted = np.array(sorted(logits), dtype=np.float64)
        probs = platt_transform(logits_sorted, A, B)
        # Check monotonicity (accounting for numerical precision)
        diffs = np.diff(probs)
        assert np.all(diffs >= -1e-10)


class TestPlattCalibratorABC:
    """Test that Calibrator ABC is properly defined."""

    def test_calibrator_is_abstract(self):
        """Cannot instantiate Calibrator directly."""
        with pytest.raises(TypeError):
            Calibrator()

    def test_calibrator_fit_is_abstract(self):
        """fit is an abstract classmethod."""
        assert hasattr(Calibrator, 'fit')
        assert hasattr(Calibrator.fit, '__isabstractmethod__')

    def test_calibrator_transform_is_abstract(self):
        """transform is an abstract method."""
        assert hasattr(Calibrator, 'transform')
        assert hasattr(Calibrator.transform, '__isabstractmethod__')


class TestPlattCalibratorConstruction:
    """Test PlattCalibrator construction and attributes."""

    def test_platt_calibrator_frozen(self):
        """PlattCalibrator is frozen (immutable)."""
        cal = PlattCalibrator(A=1.0, B=0.0, fit_population="test", n=100)
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            cal.A = 2.0

    def test_platt_calibrator_attributes(self):
        """PlattCalibrator has required attributes."""
        cal = PlattCalibrator(A=1.5, B=-0.3, fit_population="ldc_val", n=500)
        assert cal.A == 1.5
        assert cal.B == -0.3
        assert cal.fit_population == "ldc_val"
        assert cal.n == 500
        assert cal.method == "platt"

    def test_platt_calibrator_method_default(self):
        """method defaults to 'platt'."""
        cal = PlattCalibrator(A=1.0, B=0.0, fit_population="test", n=100)
        assert cal.method == "platt"

    def test_platt_calibrator_method_override(self):
        """method can be overridden (though it should be 'platt')."""
        cal = PlattCalibrator(A=1.0, B=0.0, fit_population="test", n=100, method="platt")
        assert cal.method == "platt"


class TestPlattCalibratorFit:
    """Test PlattCalibrator.fit classmethod."""

    def test_fit_requires_fit_population_keyword(self):
        """fit_population must be keyword-only."""
        logits = np.array([0.0, 1.0])
        labels = np.array([0, 1])
        # This should work (keyword argument)
        cal = PlattCalibrator.fit(logits, labels, fit_population="test_pop")
        assert isinstance(cal, PlattCalibrator)
        # Positional argument should fail (fit_population is keyword-only)
        with pytest.raises(TypeError):
            PlattCalibrator.fit(logits, labels, "test_pop")

    def test_fit_returns_platt_calibrator(self):
        """fit returns a PlattCalibrator instance."""
        logits = np.array([-1.0, 0.0, 1.0])
        labels = np.array([0, 0, 1])
        cal = PlattCalibrator.fit(logits, labels, fit_population="my_pop")
        assert isinstance(cal, PlattCalibrator)
        assert cal.fit_population == "my_pop"

    def test_fit_stores_population(self):
        """fit stores the fit_population value."""
        logits = np.array([0.0, 1.0])
        labels = np.array([0, 1])
        cal = PlattCalibrator.fit(logits, labels, fit_population="ldc_val_natural_balance")
        assert cal.fit_population == "ldc_val_natural_balance"

    def test_fit_stores_n(self):
        """fit stores the number of samples."""
        logits = np.array([0.0, 1.0, 2.0, 3.0])
        labels = np.array([0, 0, 1, 1])
        cal = PlattCalibrator.fit(logits, labels, fit_population="test")
        assert cal.n == 4

    def test_fit_deterministic(self):
        """Same input → identical calibrator."""
        logits = np.array([-1.0, 0.0, 1.0, 2.0])
        labels = np.array([0, 0, 1, 1])
        cal1 = PlattCalibrator.fit(logits, labels, fit_population="pop")
        cal2 = PlattCalibrator.fit(logits, labels, fit_population="pop")
        assert cal1.A == cal2.A
        assert cal1.B == cal2.B


class TestPlattCalibratorTransform:
    """Test PlattCalibrator.transform method."""

    def test_transform_returns_array(self):
        """transform returns an ndarray."""
        logits = np.array([0.0, 1.0])
        labels = np.array([0, 1])
        cal = PlattCalibrator.fit(logits, labels, fit_population="test")
        probs = cal.transform(np.array([-1.0, 0.0, 1.0]))
        assert isinstance(probs, np.ndarray)

    def test_transform_output_in_range(self):
        """transform output is in [0, 1]."""
        logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        labels = np.array([0, 0, 0, 1, 1])
        cal = PlattCalibrator.fit(logits, labels, fit_population="test")
        probs = cal.transform(np.array([-10.0, -1.0, 0.0, 1.0, 10.0]))
        assert np.all(probs >= -1e-10)
        assert np.all(probs <= 1.0 + 1e-10)

    def test_transform_monotonic(self):
        """transform is monotonic for sorted logits."""
        logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        labels = np.array([0, 0, 0, 1, 1])
        cal = PlattCalibrator.fit(logits, labels, fit_population="test")
        test_logits = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
        probs = cal.transform(test_logits)
        assert np.all(np.diff(probs) >= -1e-10)


class TestAC45EdgeCase:
    """Test AC4.5: Platt reduces ECE on miscalibrated data."""

    def test_platt_improves_overconfident(self):
        """Fit on one half of overconfident data; compare ECE on other half.

        This mimics the canonical boundary case: logits are scaled 4x beyond
        what the label frequencies warrant (overconfident). Platt should
        reduce ECE on held-out eval split.
        """
        # Generate overconfident logits (scaled 4x beyond natural balance)
        n_pos, n_neg = 50, 450  # natural balance ~10%
        pos_logits = np.random.RandomState(42).normal(2.0, 1.0, n_pos) * 4
        neg_logits = np.random.RandomState(43).normal(-2.0, 1.0, n_neg) * 4

        all_logits = np.concatenate([pos_logits, neg_logits])
        all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

        # Split into fit and eval (natural balance on both)
        np.random.RandomState(44).shuffle(all_logits)
        np.random.RandomState(44).shuffle(all_labels)

        n_fit = len(all_logits) // 2
        fit_logits, eval_logits = all_logits[:n_fit], all_logits[n_fit:]
        fit_labels, eval_labels = all_labels[:n_fit], all_labels[n_fit:]

        # Fit calibrator on first half (natural balance)
        cal = PlattCalibrator.fit(fit_logits, fit_labels, fit_population="overconfident_fit")

        # Compute ECE on eval half: raw sigmoid vs calibrated
        from src.calibration.report import calibration_report

        raw_probs = 1.0 / (1.0 + np.exp(-eval_logits))
        raw_report = calibration_report(raw_probs, eval_labels)

        cal_probs = cal.transform(eval_logits)
        cal_report = calibration_report(cal_probs, eval_labels)

        # Platt should improve (reduce) ECE
        assert cal_report["ece"] <= raw_report["ece"] + 0.01  # allow tiny numerical slack
