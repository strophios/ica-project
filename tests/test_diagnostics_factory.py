# pattern: Functional Core

"""Tests for src/diagnostics/factory.py:build_trackers."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
import keras as _keras

from src.cca_config import DiagnosticsConfig
from src.diagnostics.factory import build_trackers


def _group_fn(var):
    return var.name.split("/", 1)[0]


def _vars(*names):
    return [tf.Variable(tf.zeros([1]), name=n) for n in names]


class _StubHead:
    """Synthetic stand-in: build_trackers only touches `.loss_fn`."""

    def __init__(self, loss_fn):
        self.loss_fn = loss_fn


class TestBuildTrackersGradientCategory:
    def test_returns_diagnostic_bundle_shape(self):
        bundle = build_trackers(
            DiagnosticsConfig(),
            group_fn=_group_fn,
            heads={},
            trainable_variables=_vars("cca/w"),
        )
        assert set(bundle.keys()) == {"per_step", "periodic"}
        assert set(bundle["per_step"].keys()) == {
            "gradient", "loss_component", "batch_target"
        }
        assert bundle["periodic"] == []

    def test_grad_norm_trackers_per_group_times_agg(self):
        bundle = build_trackers(
            DiagnosticsConfig(gradient_norm_aggregations=("max", "mean")),
            group_fn=_group_fn,
            heads={},
            trainable_variables=_vars("cca/w1", "cca/w2", "extra/w"),
        )
        names = sorted(t.name for t in bundle["per_step"]["gradient"])
        # 2 groups {cca, extra} x 2 aggs + 1 overflow tracker
        assert "grad_norm/cca/max" in names
        assert "grad_norm/cca/mean" in names
        assert "grad_norm/extra/max" in names
        assert "grad_norm/extra/mean" in names
        assert "grad_overflow_rate" in names
        assert len(bundle["per_step"]["gradient"]) == 5

    def test_groups_are_sorted_and_deduped(self):
        bundle = build_trackers(
            DiagnosticsConfig(gradient_norm_aggregations=("mean",)),
            group_fn=_group_fn,
            heads={},
            trainable_variables=_vars("b/w", "a/w", "b/w2", "a/w2"),
        )
        grad_norm_names = [
            t.name for t in bundle["per_step"]["gradient"]
            if t.name.startswith("grad_norm/")
        ]
        assert grad_norm_names == ["grad_norm/a/mean", "grad_norm/b/mean"]

    def test_disable_gradient_norms(self):
        bundle = build_trackers(
            DiagnosticsConfig(enable_gradient_norms=False),
            group_fn=_group_fn,
            heads={},
            trainable_variables=_vars("cca/w"),
        )
        names = [t.name for t in bundle["per_step"]["gradient"]]
        assert all(not n.startswith("grad_norm/") for n in names)
        assert "grad_overflow_rate" in names  # overflow still on

    def test_disable_overflow_proxy(self):
        bundle = build_trackers(
            DiagnosticsConfig(enable_overflow_proxy=False),
            group_fn=_group_fn,
            heads={},
            trainable_variables=_vars("cca/w"),
        )
        names = [t.name for t in bundle["per_step"]["gradient"]]
        assert "grad_overflow_rate" not in names
        assert any(n.startswith("grad_norm/") for n in names)


class TestBuildTrackersBatchTarget:
    def test_one_balance_tracker_per_head(self):
        heads = {"cca": _StubHead(None), "immig": _StubHead(None)}
        bundle = build_trackers(
            DiagnosticsConfig(enable_loss_components=False),
            group_fn=_group_fn,
            heads=heads,
            trainable_variables=_vars("cca/w"),
        )
        names = sorted(t.name for t in bundle["per_step"]["batch_target"])
        assert names == ["cca/positive_fraction", "immig/positive_fraction"]

    def test_disable_batch_balance(self):
        heads = {"cca": _StubHead(None)}
        bundle = build_trackers(
            DiagnosticsConfig(enable_batch_balance=False, enable_loss_components=False),
            group_fn=_group_fn,
            heads=heads,
            trainable_variables=_vars("cca/w"),
        )
        assert bundle["per_step"]["batch_target"] == []

    def test_periodic_empty_regardless_of_enable_flag(self):
        # periodic is a permanently-empty forward-compat slot. The factory
        # never populates it; enable_prediction_distribution gates the
        # per-head distribution metrics (Phase 5), which ride the head
        # metric path, NOT this bundle.
        heads = {"cca": _StubHead(None)}
        bundle = build_trackers(
            DiagnosticsConfig(enable_prediction_distribution=True,
                              enable_loss_components=False),
            group_fn=_group_fn,
            heads=heads,
            trainable_variables=_vars("cca/w"),
        )
        assert bundle["periodic"] == []


class _LossWithIntermediates(_keras.losses.Loss):
    def call(self, y_true, y_pred, return_intermediates=False):
        return tf.constant(0.0)


class _LossWithoutIntermediates(_keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.constant(0.0)


class TestFLPUComponentKeysSync:
    def test_flpu_component_keys_in_sync_with_loss_intermediates(self):
        # Drift guard: factory.py deliberately duplicates the component keys
        # ("positive_risk", "negative_risk", "correction_triggered") rather
        # than importing from loss.py (preserves the "factory as functional
        # core does not import the production machinery" invariant). This test
        # recovers drift-prevention at the test boundary.
        from src.diagnostics.factory import _FLPU_COMPONENT_KEYS
        from src.loss_functions.loss import FLPULoss

        loss = FLPULoss(prior=0.1)
        # Build a minimal valid batch: 1 positive, 1 unlabeled
        y_true = np.array([1.0, 0.0], dtype="float32")
        y_pred = np.array([0.5, -0.5], dtype="float32").reshape(-1, 1)

        _, components = loss.call(y_true, y_pred, return_intermediates=True)

        assert set(components.keys()) == set(_FLPU_COMPONENT_KEYS)


class TestBuildTrackersLossComponentGuard:
    def test_all_heads_supporting_yield_three_trackers_each(self):
        heads = {
            "cca": _StubHead(_LossWithIntermediates()),
            "immig": _StubHead(_LossWithIntermediates()),
        }
        bundle = build_trackers(
            DiagnosticsConfig(),
            group_fn=_group_fn,
            heads=heads,
            trainable_variables=_vars("cca/w"),
        )
        names = sorted(t.name for t in bundle["per_step"]["loss_component"])
        assert names == [
            "cca/correction_triggered/mean",
            "cca/negative_risk/mean",
            "cca/positive_risk/mean",
            "immig/correction_triggered/mean",
            "immig/negative_risk/mean",
            "immig/positive_risk/mean",
        ]

    def test_zero_supporting_raises(self):
        heads = {"cca": _StubHead(_LossWithoutIntermediates())}
        with pytest.raises(ValueError, match="return_intermediates"):
            build_trackers(
                DiagnosticsConfig(),
                group_fn=_group_fn,
                heads=heads,
                trainable_variables=_vars("cca/w"),
            )

    def test_loss_fn_none_treated_as_unsupported(self):
        heads = {"cca": _StubHead(None)}
        with pytest.raises(ValueError, match="return_intermediates"):
            build_trackers(
                DiagnosticsConfig(),
                group_fn=_group_fn,
                heads=heads,
                trainable_variables=_vars("cca/w"),
            )

    def test_partial_support_warns_and_skips_unsupported(self):
        heads = {
            "cca": _StubHead(_LossWithIntermediates()),
            "bce_head": _StubHead(_LossWithoutIntermediates()),
        }
        with pytest.warns(UserWarning, match="bce_head"):
            bundle = build_trackers(
                DiagnosticsConfig(),
                group_fn=_group_fn,
                heads=heads,
                trainable_variables=_vars("cca/w"),
            )
        names = sorted(t.name for t in bundle["per_step"]["loss_component"])
        assert names == [
            "cca/correction_triggered/mean",
            "cca/negative_risk/mean",
            "cca/positive_risk/mean",
        ]

    def test_disable_loss_components_skips_guard_entirely(self):
        # enable_loss_components=False → no guard, no raise even with
        # zero supporting losses.
        heads = {"cca": _StubHead(_LossWithoutIntermediates())}
        bundle = build_trackers(
            DiagnosticsConfig(enable_loss_components=False),
            group_fn=_group_fn,
            heads=heads,
            trainable_variables=_vars("cca/w"),
        )
        assert bundle["per_step"]["loss_component"] == []
