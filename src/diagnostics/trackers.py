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
