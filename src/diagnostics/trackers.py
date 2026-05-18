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


def _gradient_is_finite(grad: tf.Tensor | tf.IndexedSlices) -> tf.Tensor:
    """True iff all elements of the gradient are finite."""
    values = grad.values if isinstance(grad, tf.IndexedSlices) else grad
    return tf.reduce_all(tf.math.is_finite(values))


class GradientFiniteTracker(keras.metrics.Metric):
    """Rate at which a training step contains any non-finite gradient.

    A step counts as 'overflow' if at least one non-None gradient contains
    NaN or Inf. Under local float32 this is effectively a constant 0.0; under
    mixed_float16 it is the active diagnostic that observes LossScaleOptimizer
    dynamic-loss-scaling floor behavior (Tier 5 level-2 acceptance criterion).
    """

    def __init__(self, **kwargs):
        super().__init__(name="grad_overflow_rate", **kwargs)
        self._overflow_steps = self.add_variable(
            shape=(), initializer="zeros", name="overflow_steps"
        )
        self._total_steps = self.add_variable(
            shape=(), initializer="zeros", name="total_steps"
        )

    def update_state(self, gradients, variables=None, group_fn=None):
        # variables/group_fn accepted for uniform gradient-category signature.
        del variables, group_fn
        any_nonfinite = tf.constant(False)
        for grad in gradients:
            if grad is None:
                continue
            any_nonfinite = tf.logical_or(
                any_nonfinite, tf.logical_not(_gradient_is_finite(grad))
            )
        self._overflow_steps.assign_add(
            tf.cast(any_nonfinite, self._overflow_steps.dtype)
        )
        self._total_steps.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self._overflow_steps, self._total_steps)

    def reset_state(self):
        self._overflow_steps.assign(0.0)
        self._total_steps.assign(0.0)


class LossComponentTracker(keras.metrics.Metric):
    """Aggregated scalar loss-component value across steps.

    Reads components_dict[component_key] per update. Raises KeyError if absent
    (Layer-2 mismatch signal — the factory should have ensured the tracker
    subscribes only to keys the loss emits). aggregation='mean' is the running
    mean of the scalar; 'max' is the running max (running_max starts at 0.0).
    """

    def __init__(
        self, head_name: str, component_key: str, aggregation: str, **kwargs
    ):
        if not head_name:
            raise ValueError("head_name must be a non-empty string")
        if not component_key:
            raise ValueError("component_key must be a non-empty string")
        if aggregation not in _VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {_VALID_AGGREGATIONS}, got {aggregation!r}"
            )
        super().__init__(name=f"{head_name}/{component_key}/{aggregation}", **kwargs)
        self.head_name = head_name
        self.component_key = component_key
        self.aggregation = aggregation
        self._total = self.add_variable(shape=(), initializer="zeros", name="total")
        self._count = self.add_variable(shape=(), initializer="zeros", name="count")
        self._running_max = self.add_variable(
            shape=(), initializer="zeros", name="running_max"
        )

    def update_state(self, components_dict):
        if self.component_key not in components_dict:
            raise KeyError(
                f"LossComponentTracker {self.name!r} expects key "
                f"{self.component_key!r}; got keys {list(components_dict.keys())}"
            )
        value = tf.cast(components_dict[self.component_key], self._total.dtype)
        if self.aggregation == "mean":
            self._total.assign_add(value)
            self._count.assign_add(1.0)
        else:
            self._running_max.assign(tf.maximum(self._running_max, value))

    def result(self):
        if self.aggregation == "mean":
            return tf.math.divide_no_nan(self._total, self._count)
        return self._running_max

    def reset_state(self):
        self._total.assign(0.0)
        self._count.assign(0.0)
        self._running_max.assign(0.0)


class BatchLabelBalanceTracker(keras.metrics.Metric):
    """Stub for Task 6."""
    pass
