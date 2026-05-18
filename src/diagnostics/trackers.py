# pattern: Mixed (unavoidable)
# Reason: Keras Metric subclasses must hold tf.Variable state for cross-step
# aggregation; they cannot be pure functions. The aggregation arithmetic is
# pure, but the surrounding Metric protocol (update_state + result +
# reset_state over persistent state vars) is inherently stateful.

"""Per-step diagnostic trackers for Tier 5.

Concrete keras.metrics.Metric subclasses observed inside
LayerLRModel.train_step. Categories ('gradient', 'loss_component',
'batch_target') are enforced by registration in src/diagnostics/factory.py,
not by inheritance.
"""

from __future__ import annotations

import keras
import tensorflow as tf

__all__ = [
    "PerGroupGradNormTracker",
    "GradientFiniteTracker",
    "LossComponentTracker",
    "BatchLabelBalanceTracker",
]

_VALID_AGGREGATIONS = ("max", "mean")


def _norm_of_gradient(grad: tf.Tensor | tf.IndexedSlices) -> tf.Tensor:
    """L2 norm of a gradient, handling sparse IndexedSlices.

    Returns the L2 norm of the dense-equivalent gradient. For IndexedSlices,
    duplicate indices are additively combined (as the optimizer applies them),
    so the returned norm matches the magnitude the optimizer uses.
    """
    if isinstance(grad, tf.IndexedSlices):
        # IndexedSlices with duplicate indices: sum rows with same index before norm
        summed = tf.math.unsorted_segment_sum(
            grad.values, grad.indices, grad.dense_shape[0]
        )
        return tf.norm(summed)
    return tf.norm(grad)


class PerGroupGradNormTracker(keras.metrics.Metric):
    """Aggregated L2 norm of gradients whose variable belongs to a named group.

    aggregation='mean' is the running mean of per-variable norms across steps;
    aggregation='max' is the running max.

    Frozen-encoder note: if the group has no variables in the trainable set
    (e.g., 'encoder' under freeze_encoder=True), update_state is a no-op and
    result() returns 0.0. The tracker reports zero because nothing was
    computed, NOT because computed gradients were zero.
    """

    def __init__(self, group_name: str, aggregation: str, **kwargs):
        if not group_name:
            raise ValueError("group_name must be a non-empty string")
        if aggregation not in _VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {_VALID_AGGREGATIONS}, got {aggregation!r}"
            )
        super().__init__(name=f"grad_norm/{group_name}/{aggregation}", **kwargs)
        self.group_name = group_name
        self.aggregation = aggregation
        self._total = self.add_variable(shape=(), initializer="zeros", name="total")
        self._count = self.add_variable(shape=(), initializer="zeros", name="count")
        self._running_max = self.add_variable(
            shape=(), initializer="zeros", name="running_max"
        )

    def update_state(self, gradients, variables, group_fn):
        in_group_norms = []
        for grad, var in zip(gradients, variables):
            if grad is None:
                continue
            if group_fn(var) != self.group_name:
                continue
            in_group_norms.append(_norm_of_gradient(grad))

        if not in_group_norms:
            return  # empty group is a no-op (e.g., frozen encoder)

        norms = tf.stack(in_group_norms)
        if self.aggregation == "mean":
            self._total.assign_add(tf.reduce_sum(norms))
            self._count.assign_add(tf.cast(tf.size(norms), self._count.dtype))
        else:  # "max"
            self._running_max.assign(
                tf.maximum(self._running_max, tf.reduce_max(norms))
            )

    def result(self):
        if self.aggregation == "mean":
            return tf.math.divide_no_nan(self._total, self._count)
        return self._running_max

    def reset_state(self):
        self._total.assign(0.0)
        self._count.assign(0.0)
        self._running_max.assign(0.0)


class GradientFiniteTracker(keras.metrics.Metric):
    """Stub for Task 4."""
    pass


class LossComponentTracker(keras.metrics.Metric):
    """Stub for Task 5."""
    pass


class BatchLabelBalanceTracker(keras.metrics.Metric):
    """Stub for Task 6."""
    pass
