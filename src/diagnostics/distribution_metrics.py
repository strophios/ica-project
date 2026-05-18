# pattern: Mixed (unavoidable)
# Reason: Keras Metric subclasses must hold tf.Variable state for cross-batch
# aggregation; they cannot be pure functions. The statistic arithmetic is
# pure, but the surrounding Metric protocol (update_state + result +
# reset_state over persistent state vars) is inherently stateful.

"""Per-head prediction-distribution metrics for Tier 5.

These are ordinary keras.metrics.Metric subclasses (standard
update_state(y_true, y_pred) signature; y_true ignored). They are passed
into ClassificationHead(metrics=...) alongside make_cca_metrics() and ride
the head's metric_objs path — computed for both the train and val phases
per epoch, no extra forward pass, no metric pollution. They do NOT go
through DiagnosticBundle or LayerLRModel.train_step dispatch.

Supersedes the original design's PeriodicDiagnostic/DiagnosticsCallback
subsystem (see tier5-design.md supersession note).
"""

from __future__ import annotations

import keras
from keras import ops

from src.cca_config import DiagnosticsConfig

__all__ = [
    "PredictionMeanMetric",
    "PredictionStdMetric",
    "PredictionFracAboveMetric",
    "make_distribution_metrics",
]


class PredictionMeanMetric(keras.metrics.Metric):
    """Running mean of sigmoid(logits) since last reset."""

    def __init__(self, name="pred_dist/mean", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self._total = self.add_variable(shape=(), initializer="zeros", name="total")
        self._count = self.add_variable(shape=(), initializer="zeros", name="count")

    def update_state(self, y_true, y_pred, sample_weight=None):
        s = ops.sigmoid(ops.cast(y_pred, "float32"))
        self._total.assign_add(ops.sum(s))
        self._count.assign_add(ops.cast(ops.size(s), self._count.dtype))

    def result(self):
        return ops.divide_no_nan(self._total, self._count)

    def reset_state(self):
        self._total.assign(0.0)
        self._count.assign(0.0)


class PredictionFracAboveMetric(keras.metrics.Metric):
    """Fraction of sigmoid(logits) strictly greater than 0.5
    (equivalently, fraction of logits > 0)."""

    def __init__(self, name="pred_dist/frac_above_0.5", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self._above = self.add_variable(shape=(), initializer="zeros", name="above")
        self._count = self.add_variable(shape=(), initializer="zeros", name="count")

    def update_state(self, y_true, y_pred, sample_weight=None):
        s = ops.sigmoid(ops.cast(y_pred, "float32"))
        above = ops.cast(s > 0.5, "float32")
        self._above.assign_add(ops.sum(above))
        self._count.assign_add(ops.cast(ops.size(s), self._count.dtype))

    def result(self):
        return ops.divide_no_nan(self._above, self._count)

    def reset_state(self):
        self._above.assign(0.0)
        self._count.assign(0.0)


class PredictionStdMetric(keras.metrics.Metric):
    """Population std (ddof=0) of sigmoid(logits) over all samples seen
    since last reset: sqrt(max(E[s^2] - E[s]^2, 0))."""

    def __init__(self, name="pred_dist/std", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self._sum = self.add_variable(shape=(), initializer="zeros", name="sum")
        self._sum_sq = self.add_variable(shape=(), initializer="zeros", name="sum_sq")
        self._count = self.add_variable(shape=(), initializer="zeros", name="count")

    def update_state(self, y_true, y_pred, sample_weight=None):
        s = ops.sigmoid(ops.cast(y_pred, "float32"))
        self._sum.assign_add(ops.sum(s))
        self._sum_sq.assign_add(ops.sum(s * s))
        self._count.assign_add(ops.cast(ops.size(s), self._count.dtype))

    def result(self):
        mean = ops.divide_no_nan(self._sum, self._count)
        mean_sq = ops.divide_no_nan(self._sum_sq, self._count)
        var = ops.maximum(mean_sq - mean * mean, 0.0)
        return ops.sqrt(var)

    def reset_state(self):
        self._sum.assign(0.0)
        self._sum_sq.assign(0.0)
        self._count.assign(0.0)


_STAT_TO_METRIC = {
    "mean": PredictionMeanMetric,
    "std": PredictionStdMetric,
    "frac_above_0.5": PredictionFracAboveMetric,
}


def make_distribution_metrics(
    config: DiagnosticsConfig,
) -> list[keras.metrics.Metric]:
    """Construct the configured per-head prediction-distribution metrics.

    Returns [] when config.enable_prediction_distribution is False.
    Otherwise one fresh metric per entry of config.prediction_summary_stats
    (order preserved). Fresh instances each call — ClassificationHead clones
    them per head anyway; callers should not share the list across heads
    (mirrors make_cca_metrics).
    """
    if not config.enable_prediction_distribution:
        return []
    return [_STAT_TO_METRIC[stat]() for stat in config.prediction_summary_stats]
