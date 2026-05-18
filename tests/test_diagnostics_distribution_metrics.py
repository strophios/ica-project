# pattern: Functional Core

"""Unit + property-based tests for src/diagnostics/distribution_metrics.py."""

from __future__ import annotations

import keras
import numpy as np
import pytest
import tensorflow as tf
from hypothesis import given, settings
from hypothesis import strategies as st

from src.cca_config import DiagnosticsConfig
from src.diagnostics.distribution_metrics import (
    PredictionFracAboveMetric,
    PredictionMeanMetric,
    PredictionStdMetric,
    make_distribution_metrics,
)


def _logits(*vals):
    return tf.constant([[v] for v in vals], dtype=tf.float32)


def _sigmoid(a):
    return 1.0 / (1.0 + np.exp(-np.asarray(a, dtype=np.float64)))


class TestPredictionMeanMetric:
    def test_name_default(self):
        assert PredictionMeanMetric().name == "pred_dist/mean"

    def test_mean_of_sigmoid(self):
        m = PredictionMeanMetric()
        m.update_state(None, _logits(0.0, 0.0))  # sigmoid(0)=0.5
        assert float(m.result()) == pytest.approx(0.5, rel=1e-5)

    def test_accumulates_across_batches(self):
        m = PredictionMeanMetric()
        m.update_state(None, _logits(0.0))            # 0.5
        m.update_state(None, _logits(1e9, 1e9))       # ~1.0, ~1.0
        # mean of [0.5, 1.0, 1.0]
        assert float(m.result()) == pytest.approx(2.5 / 3.0, rel=1e-4)

    def test_reset_state(self):
        m = PredictionMeanMetric()
        m.update_state(None, _logits(1e9))
        m.reset_state()
        assert float(m.result()) == 0.0

    def test_from_config_roundtrip(self):
        # Head cloning relies on cls.from_config(m.get_config()).
        m = PredictionMeanMetric()
        clone = PredictionMeanMetric.from_config(m.get_config())
        assert clone.name == m.name
        clone.update_state(None, _logits(0.0))
        assert float(clone.result()) == pytest.approx(0.5, rel=1e-5)

    def test_head_prefix_clone_pattern(self):
        # Exactly what ClassificationHead.__init__ does.
        m = PredictionMeanMetric()
        cfg = m.get_config()
        cfg["name"] = f"cca_{cfg['name']}"
        clone = PredictionMeanMetric.from_config(cfg)
        assert clone.name == "cca_pred_dist/mean"


class TestPredictionFracAboveMetric:
    def test_name_default(self):
        assert PredictionFracAboveMetric().name == "pred_dist/frac_above_0.5"

    def test_fraction_above_half(self):
        m = PredictionFracAboveMetric()
        # sigmoid>0.5 iff logit>0. Two of four above.
        m.update_state(None, _logits(2.0, 3.0, -1.0, -2.0))
        assert float(m.result()) == pytest.approx(0.5, rel=1e-5)

    def test_all_below(self):
        m = PredictionFracAboveMetric()
        m.update_state(None, _logits(-5.0, -5.0))
        assert float(m.result()) == 0.0

    def test_reset_state(self):
        m = PredictionFracAboveMetric()
        m.update_state(None, _logits(5.0))
        m.reset_state()
        assert float(m.result()) == 0.0


class TestPredictionStdMetric:
    def test_name_default(self):
        assert PredictionStdMetric().name == "pred_dist/std"

    def test_zero_std_constant_input(self):
        m = PredictionStdMetric()
        m.update_state(None, _logits(0.0, 0.0, 0.0))  # all 0.5
        assert float(m.result()) == pytest.approx(0.0, abs=1e-5)

    def test_zero_std_large_constant_logits(self):
        # Regression test for float32 catastrophic cancellation.
        # Large constant logits → sigmoid ≈ 1.0 (exactly equal in float32).
        # E[s²] ≈ E[s]² numerically, but the subtraction must not yield float32 noise.
        m = PredictionStdMetric()
        m.update_state(None, _logits(30.0, 30.0, 30.0, 30.0))
        assert float(m.result()) == pytest.approx(0.0, abs=1e-5)

    def test_matches_numpy_population_std(self):
        m = PredictionStdMetric()
        logits = [2.0, -1.0, 0.5, -3.0, 1.0]
        m.update_state(None, _logits(*logits))
        expected = float(np.std(_sigmoid(logits)))  # population std (ddof=0)
        assert float(m.result()) == pytest.approx(expected, rel=1e-4, abs=1e-5)

    def test_accumulates_across_batches(self):
        m = PredictionStdMetric()
        m.update_state(None, _logits(2.0, -1.0))
        m.update_state(None, _logits(0.5, -3.0, 1.0))
        expected = float(np.std(_sigmoid([2.0, -1.0, 0.5, -3.0, 1.0])))
        assert float(m.result()) == pytest.approx(expected, rel=1e-4, abs=1e-5)

    def test_reset_state(self):
        m = PredictionStdMetric()
        m.update_state(None, _logits(2.0, -3.0))
        m.reset_state()
        assert float(m.result()) == 0.0


_logit_lists = st.lists(
    st.floats(min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    min_size=1, max_size=64,
)


class TestDistributionMetricProperties:
    @given(_logit_lists)
    @settings(max_examples=50, deadline=None)
    def test_mean_in_unit_interval(self, logits):
        m = PredictionMeanMetric()
        m.update_state(None, _logits(*logits))
        assert 0.0 <= float(m.result()) <= 1.0

    @given(_logit_lists)
    @settings(max_examples=50, deadline=None)
    def test_frac_in_unit_interval(self, logits):
        m = PredictionFracAboveMetric()
        m.update_state(None, _logits(*logits))
        assert 0.0 <= float(m.result()) <= 1.0

    @given(_logit_lists)
    @settings(max_examples=50, deadline=None)
    def test_std_non_negative_and_matches_numpy(self, logits):
        m = PredictionStdMetric()
        m.update_state(None, _logits(*logits))
        r = float(m.result())
        assert r >= 0.0
        assert r == pytest.approx(float(np.std(_sigmoid(logits))), rel=1e-3, abs=1e-4)


class TestMakeDistributionMetrics:
    def test_default_config_builds_all_three_in_order(self):
        metrics = make_distribution_metrics(DiagnosticsConfig())
        assert [m.name for m in metrics] == [
            "pred_dist/mean", "pred_dist/std", "pred_dist/frac_above_0.5"
        ]

    def test_disabled_returns_empty(self):
        metrics = make_distribution_metrics(
            DiagnosticsConfig(enable_prediction_distribution=False)
        )
        assert metrics == []

    def test_subset_and_order_preserved(self):
        cfg = DiagnosticsConfig(prediction_summary_stats=("frac_above_0.5", "mean"))
        metrics = make_distribution_metrics(cfg)
        assert [m.name for m in metrics] == [
            "pred_dist/frac_above_0.5", "pred_dist/mean"
        ]

    def test_fresh_instances_each_call(self):
        cfg = DiagnosticsConfig()
        a = make_distribution_metrics(cfg)
        b = make_distribution_metrics(cfg)
        assert all(x is not y for x, y in zip(a, b))

    def test_instances_are_metric_subclasses(self):
        for m in make_distribution_metrics(DiagnosticsConfig()):
            assert isinstance(m, keras.metrics.Metric)
