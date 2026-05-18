# pattern: Functional Core

"""Tests for src/diagnostics/factory.py:build_trackers."""

from __future__ import annotations

import pytest
import tensorflow as tf

from src.cca_config import DiagnosticsConfig
from src.diagnostics.factory import DiagnosticBundle, build_trackers


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
