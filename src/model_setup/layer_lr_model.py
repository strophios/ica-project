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

from typing import TYPE_CHECKING, Callable, Optional

import keras
import tensorflow as tf

if TYPE_CHECKING:
    from src.diagnostics.factory import DiagnosticBundle
    from src.model_setup.heads import ClassificationHead


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
        diagnostic_trackers: "Optional[DiagnosticBundle]" = None,
        diagnostic_head_refs: "Optional[list[ClassificationHead]]" = None,
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
        self._diagnostic_trackers = diagnostic_trackers
        self._head_refs_by_name = {
            h.name: h for h in (diagnostic_head_refs or [])
        }

    @property
    def metrics(self):
        base = super().metrics
        if self._diagnostic_trackers is None:
            return base
        extra = []
        for category in self._diagnostic_trackers["per_step"].values():
            extra.extend(category)
        return base + extra

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

    def _dispatch_diagnostics(self, gradients, y):
        """Dispatch gradient/loss-component/batch-target observations to trackers.

        Called from train_step with raw (pre-scaling) gradients and targets,
        so trackers observe what was computed, not what was applied.
        """
        per_step = self._diagnostic_trackers["per_step"]
        for tracker in per_step["gradient"]:
            tracker.update_state(
                gradients, self.trainable_variables, self.group_fn
            )
        for tracker in per_step["loss_component"]:
            head = self._head_refs_by_name[tracker.head_name]
            tracker.update_state(head.last_components)
        for tracker in per_step["batch_target"]:
            tracker.update_state(y)

    def train_step(self, data):
        """One training step with per-variable LR multipliers applied.

        Mirrors stock `keras.Model.train_step` (Keras 3, TensorFlow
        backend) faithfully and inserts one extra step: between
        gradient computation and optimizer apply, scale each gradient
        by the multiplier for its variable's group.

        Implementation notes (worth being explicit because the Keras
        "Customizing fit() with TensorFlow" guide elides several of
        these and an earlier version of this method did too — the Tier
        2 review surfaced the gap):

        - **`_compute_loss` over `compute_loss`.** `_compute_loss` is
          the internal wrapper that handles `compute_loss` overrides
          missing the `training=` kwarg (added in Keras 3.3). Stock
          `train_step` uses it for backward compatibility with older
          subclass overrides; using it here means `LayerLRModel`
          subclasses that override `compute_loss` with the pre-3.3
          signature continue to work.

        - **`_loss_tracker.update_state`.** The loss tracker is the
          metric named `"loss"` that Keras automatically registers
          when you `compile()`. Stock train_step calls
          `_loss_tracker.update_state(loss, sample_weight=batch_size)`
          so that `history.history["loss"]`, the `fit()` progress
          bar, TensorBoard's training-loss curve, and any callback
          reading the training-side `loss` log all reflect actual
          values. Without this call, *training still happens* (we
          still backprop), but the loss reported to all of those
          surfaces is silently 0. Note: passing `batch_size` as
          `sample_weight` is what makes the per-epoch mean correct
          when batch sizes vary (e.g., a partial last batch).

        - **`optimizer.scale_loss`.** When the optimizer is a
          `LossScaleOptimizer` (used here under `mixed_float16` on
          the cluster — see `run_cca_classification.py`), `scale_loss`
          multiplies the loss by the dynamic loss-scale factor before
          backprop, protecting fp16 gradients from underflow. For a
          plain optimizer, `scale_loss` is a no-op identity. Calling
          it unconditionally matches stock train_step and ensures the
          mixed-precision wrap is real rather than silently skipped.

        - **Compile-time loss + `add_loss` contributions.** Both flow
          through `_compute_loss` automatically (it aggregates the
          configured compile-time loss with `self.losses` from
          endpoint-layer `add_loss` calls). Both training paths the
          codebase uses — `compile(loss=...)` for the L/U classifier,
          `add_loss` inside `ClassificationHead` for the FLPU CCA
          classifier — are therefore covered without conditional
          logic here.

        `sample_weight` is plumbed through to `_compute_loss` and
        `compute_metrics` even though we don't currently use it (no
        per-sample weighting in any current training pipeline);
        `None` is a no-op for both, and having the hook in place
        means adding hard-negative mining or per-sample weighting
        later is a data-pipeline change, not a train_step change.

        Parameters
        ----------
        data : tuple
            A batch produced by the Dataset. Typically `(x, y)` or
            `(x, y, sample_weight)`; unpacked via Keras's helper.
            Endpoint-mode batches (single dict, no labels) also
            work — `keras.utils.unpack_x_y_sample_weight` returns
            `(x, None, None)` for those, and the head's `add_loss`
            provides the loss internally.

        Returns
        -------
        dict
            Metric name → scalar value, forwarded to `fit()`'s
            progress bar, callbacks, and TensorBoard logging. Both
            the loss tracker and any compile-time / layer-tracked
            metrics are surfaced here.
        """
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self._compute_loss(
                x=x,
                y=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
                training=True,
            )
            # Update the loss tracker so history.history["loss"], the
            # fit progress bar, and TensorBoard reflect actual loss
            # rather than zero. Stock train_step weights by batch size
            # so a partial last batch contributes proportionally to the
            # epoch mean.
            self._loss_tracker.update_state(
                loss,
                sample_weight=tf.shape(
                    next(t for t in tf.nest.flatten(x) if t is not None)
                )[0],
            )
            # Loss scaling for LossScaleOptimizer (no-op for plain
            # optimizers). MUST happen inside the GradientTape context
            # so the scaled loss is what tape.gradient sees.
            if self.optimizer is not None:
                loss = self.optimizer.scale_loss(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        # Tier 5: read-only diagnostic observation of the COMPUTED
        # gradients (before per-variable multiplier scaling) plus loss
        # components and targets. No-op when diagnostics aren't configured.
        if self._diagnostic_trackers is not None:
            self._dispatch_diagnostics(gradients, y)
        # Scale each gradient by its variable's multiplier. Two
        # subtleties handled here:
        #
        # (1) Preserve `None` gradients — variables that don't
        #     affect the loss this step (e.g., multi-head models
        #     where a head isn't active on every batch).
        #     `apply_gradients` skips `None` entries automatically,
        #     but filtering them out here would misalign the list
        #     with `self.trainable_variables`.
        #
        # (2) Sparse gradients (`tf.IndexedSlices`) require
        #     `tf.math.scalar_mul` — Python's `float * tensor`
        #     dispatches to the tensor's `__rmul__`, which is
        #     undefined for `IndexedSlices`. `Embedding` layers
        #     produce sparse gradients (only the looked-up rows are
        #     non-zero), so any model with a trainable embedding
        #     hits this path. `tf.math.scalar_mul` handles both
        #     dense tensors and `IndexedSlices` correctly.
        scaled = [
            tf.math.scalar_mul(self.get_multiplier(w), g) if g is not None else None
            for w, g in zip(self.trainable_variables, gradients)
        ]
        self.optimizer.apply_gradients(zip(scaled, self.trainable_variables))

        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)
