"""
`LayerLRModel`: a `keras.Model` subclass that applies per-variable
learning-rate multipliers during training.

The use case: discriminative fine-tuning of a transformer backbone plus
classification heads, where different parts of the model should train at
different effective learning rates. The head typically wants full LR;
upper encoder layers want a fraction of that (e.g., 0.95); lower layers
a smaller fraction still; embeddings often the smallest or zero.

Rather than configuring multiple independent optimizers (heavy) or
resorting to a fully custom training loop (loses `fit()` integration),
this class wraps a single base optimizer and scales gradients per
variable before application. The scaling is mathematically equivalent
to a per-variable LR, since for the standard update rule
`var -= lr * grad`, multiplying `grad` by `m` is the same as
multiplying `lr` by `m`.

See `docs/notes/tier2-design.md` Piece 2 for the full design reasoning,
including the trade-offs against multiple-optimizer approaches and the
`trainable=False` vs. `multiplier=0` composition.

Example
-------

    # In Keras 3, var.name is the short name ("kernel", "bias"); var.path
    # is the full hierarchical path ("cca_head/intermediate_dense/kernel").
    # For layer-identity-based grouping, use var.path.

    def group_of(var):
        path = var.path
        if "cca_head" in path:
            return "head"
        if "roberta_layer_11" in path:
            return "encoder_top"
        return "encoder_lower"

    model = LayerLRModel(
        inputs=inputs,
        outputs=outputs,
        group_fn=group_of,
        multipliers={"head": 1.0, "encoder_top": 0.3, "encoder_lower": 0.1},
    )
    model.compile(optimizer=keras.optimizers.AdamW(1e-3), loss=...)
    model.fit(...)
"""

from typing import Callable, Optional

import keras
import tensorflow as tf


class LayerLRModel(keras.Model):
    """
    A `keras.Model` that applies per-variable learning-rate multipliers.

    See module docstring for context. Key behaviors:

    - `group_fn(var) -> str` assigns every trainable variable to a group.
    - `multipliers` is a dict of `group_name -> float` specifying the
      LR multiplier for each group. Missing groups default to 1.0
      (i.e., "train normally").
    - During `train_step`, gradients are scaled by the per-variable
      multiplier before being applied by the base optimizer.

    Parameters
    ----------
    group_fn : Callable[[tf.Variable], str]
        Maps each trainable variable to a group name. Typical
        implementation inspects `variable.path` (the hierarchical
        layer-qualified name in Keras 3). Required.
    multipliers : dict[str, float] or None
        Group name → multiplier. Missing entries default to 1.0.
        Can be mutated at training time (e.g., by a callback) via
        `set_multiplier`.
    *args, **kwargs
        Forwarded to `keras.Model.__init__`. Supports the usual
        functional API (`inputs=..., outputs=...`) as well as
        subclassing (pass neither).

    Notes
    -----
    `trainable=False` on any layer or variable is respected naturally —
    `self.trainable_variables` already excludes non-trainable variables,
    so they don't participate in the scaled-gradient update. Use
    `trainable=False` for variables permanently frozen in this run
    (compute savings); use `multiplier=0` for variables that will
    toggle freeze status mid-training (cleaner state management).
    """

    def __init__(
        self,
        *args,
        group_fn: Optional[Callable[[tf.Variable], str]] = None,
        multipliers: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if group_fn is None:
            # Default: everything is in a single group named "default",
            # effectively making LayerLRModel behave like a plain Model
            # with a uniform multiplier (which is 1.0 unless overridden).
            group_fn = lambda v: "default"  # noqa: E731
        self.group_fn = group_fn
        self.multipliers = dict(multipliers) if multipliers is not None else {}

    def get_multiplier(self, variable) -> float:
        """Return the LR multiplier for a given trainable variable.

        Variables whose group is not in `self.multipliers` return 1.0
        (the "train normally" default), so an unconfigured model
        behaves identically to a plain `keras.Model`.
        """
        group = self.group_fn(variable)
        return self.multipliers.get(group, 1.0)

    def set_multiplier(self, group_name: str, value: float) -> None:
        """Update a group's multiplier.

        Called by callbacks for gradual-unfreezing schedules. Does not
        trigger re-compilation or any optimizer-state changes; the new
        multiplier takes effect on the next `train_step` call.
        """
        self.multipliers[group_name] = value

    def train_step(self, data):
        """One training step with per-variable LR multipliers applied.

        Mirrors Keras's default `train_step` structure with one change:
        between computing gradients and applying them, we scale each
        gradient by the multiplier for its variable's group.

        `compute_loss` already aggregates the compile-time loss with
        any `add_loss` contributions from endpoint layers, so both
        paths (dict-of-losses via `compile`, `add_loss` inside heads)
        are covered.

        `sample_weight` is plumbed through even though we don't
        currently use it; `None` is a no-op for both `compute_loss`
        and `compute_metrics`, and having the hook in place means
        adding hard-negative mining or per-sample weighting later is
        a data-pipeline change, not a train_step change.

        Parameters
        ----------
        data : tuple
            A batch produced by the Dataset. Typically `(x, y)` or
            `(x, y, sample_weight)`; unpacked via Keras's helper.

        Returns
        -------
        dict
            Metric name → scalar value, forwarded to `fit()`'s
            progress bar, callbacks, and TensorBoard logging.
        """
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(
                x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
            )

        gradients = tape.gradient(loss, self.trainable_variables)
        # Scale each gradient by its variable's multiplier. Preserve
        # `None` gradients (variables that don't affect the loss this
        # step — can happen with multi-head models when a head isn't
        # active on every batch). `apply_gradients` skips `None`
        # entries automatically; filtering them out here would
        # misalign the gradient list with `self.trainable_variables`.
        scaled = [
            self.get_multiplier(w) * g if g is not None else None
            for w, g in zip(self.trainable_variables, gradients)
        ]
        self.optimizer.apply_gradients(zip(scaled, self.trainable_variables))

        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)
