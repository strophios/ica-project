"""
Classification head Layers for the multi-head ICA classifier.

This module defines the head Layers that sit on top of the shared DAPT
backbone. Each head is a `keras.layers.Layer` subclass, which is a more
structured abstraction than the inline-head construction the original
`classification_setup.py` used (retired in Tier 2 Piece 4c). The upside
is that heads are:

- **Self-contained.** Each head owns its sub-layers, weights, and
  optionally its own loss. Adding a new head is an additive change
  rather than a refactor.
- **Reusable.** The same head class can be instantiated multiple times
  with different configs (different hidden_dim, different loss_fn,
  different name) without duplicating code.
- **Serializable.** Keras tracks sub-layers automatically when they are
  set as attributes on the Layer, so save/load works correctly.
- **Composable into multi-head models.** The outer assembly function
  can call each head on the shared backbone features and route each
  head's output (and optionally targets) independently.

This module currently exposes:
  - `ClassificationHead`: a simple head for single-task binary
    classification (CCA, immigrant-involvement, and any future
    same-shape head). Supports both standard and endpoint modes — see
    docstring.

Planned (not yet implemented):
  - `CombinedClassificationHead`: a head that takes shared backbone
    features *and* the logits from the component heads as inputs.
    Exists for the combined-ICA prediction. Pending an open design
    decision about gradient flow through component-head logits
    (see `docs/notes/tier2-design.md` Piece 1, "Open decision").
"""

import inspect
import math

import keras


class ClassificationHead(keras.layers.Layer):
    """
    Binary classification head that sits on top of shared backbone features.

    Architecture (matches the pre-Tier-2 `classification_setup.py` single-
    head shape — that file has been retired in Piece 4c, but weights
    saved under the old single-head code are compatible because the
    layer ordering and Dense widths haven't changed):

        features (batch, hidden_dim)
          → Dropout(rate=dropout)
          → Dense(hidden_dim, activation="relu")
          → Dropout(rate=dropout)
          → Dense(1, activation=None)   # logits, shape (batch, 1)

    Supports two modes:

    **Standard mode** (`loss_fn=None`): the head produces logits and the
    outer model's `compile(loss=...)` is responsible for the loss. Use
    for heads whose loss is a plain per-output loss (e.g., BCE for a
    well-balanced US/not-US head, if we keep that head).

    **Endpoint mode** (`loss_fn=<Loss>`): the head produces logits *and*
    — if targets are provided at call time — computes the loss internally
    via `self.add_loss(loss_fn(targets, logits))`. Use for heads whose
    loss depends on internal state or needs access to inputs the outer
    model would not normally provide to `compile`-time losses (FLPU needs
    this because the targets drive the per-sample mask split; ALUM
    eventually will need this because its loss depends on perturbed
    embeddings from inside the model).

    In endpoint mode the outer `compile()` call must NOT pass a loss for
    this head's output, or Keras will double-count.

    Parameters
    ----------
    hidden_dim : int
        Width of the intermediate Dense layer. Typically matches the
        backbone's hidden_dim (768 for RoBERTa-base).
    dropout : float, default 0.1
        Dropout rate applied before and after the intermediate Dense.
    loss_fn : keras.losses.Loss or None, default None
        If provided, activates endpoint mode; the head will call
        `add_loss(loss_fn(targets, logits))` when targets are passed
        to `call`. If None, operates in standard mode.
    metrics : list of keras.metrics.Metric or None, default None
        Per-head metric instances. When `targets` is supplied at call
        time (training/eval), each metric's `update_state(targets,
        logits)` is invoked, and Keras's automatic Layer-attribute
        tracking surfaces them via `model.metrics`. The metric
        objects are *renamed* to be prefixed with this head's `name`
        (e.g., `BinaryAccuracy()` becomes `cca_binary_accuracy` for a
        head named `"cca"`) so multi-head models don't collide on
        metric names. Renaming uses `m.from_config(...)` so the
        original metric instances passed in are not mutated.
        Symmetric with `loss_fn`: both fire only when targets are
        provided, both are part of the endpoint-layer pattern.
    name : str
        Required keyword-only parameter. Passed to
        `keras.layers.Layer.__init__`. Appears in `model.summary()`
        output and is used for serialization. Must be non-None to
        prevent accidental Keras auto-generated name collisions across
        multiple heads.
    loss_weight : float, default 1.0
        Scalarization weight (lambda) applied at loss registration —
        `add_loss(loss_weight * loss)`. See
        `docs/design-plans/2026-08-18-stage4-joint-finetune.md`
        "Components" item 1 for the why.

    Notes
    -----
    The `features` input to `call` is expected to already be the shared
    CLS-token representation from the backbone — i.e., shape
    `(batch, hidden_dim)`, not the full sequence `(batch, seq_len, hidden_dim)`.
    The assembly function that wires the backbone and heads together is
    responsible for the CLS-token slice.
    """

    def __init__(
        self,
        hidden_dim,
        dropout=0.1,
        loss_fn=None,
        metrics=None,
        *,
        name,
        expose_loss_components=False,
        loss_weight=1.0,
    ):
        if name is None:
            raise ValueError(
                "ClassificationHead requires an explicit name; name=None "
                "would fall back to Keras auto-generated names "
                "(e.g., 'classification_head_1') which collide silently "
                "across heads in a multi-head model."
            )
        if (
            not isinstance(loss_weight, (int, float))
            or isinstance(loss_weight, bool)
            or not math.isfinite(float(loss_weight))
            or float(loss_weight) <= 0
        ):
            raise ValueError(
                f"ClassificationHead {name!r}: loss_weight must be a finite "
                f"positive number; got {loss_weight!r} "
                f"(type {type(loss_weight).__name__})."
            )
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight

        self.expose_loss_components = expose_loss_components
        self.last_components = None
        if expose_loss_components and loss_fn is not None:
            if "return_intermediates" not in inspect.signature(
                loss_fn.call
            ).parameters:
                raise ValueError(
                    f"ClassificationHead {name!r} was constructed with "
                    f"expose_loss_components=True but its loss "
                    f"{type(loss_fn).__name__} does not accept a "
                    f"`return_intermediates` parameter. Loss-component "
                    f"harvest requires an FLPU-style loss."
                )

        # Per-head metrics. Each metric is renamed to be prefixed with
        # this head's name to avoid collisions when a multi-head model
        # uses the same metric type on multiple heads (e.g.,
        # `BinaryAccuracy()` on both cca and immig heads would otherwise
        # produce two metrics both named "binary_accuracy" in
        # `model.metrics`). The renaming uses `from_config` to clone
        # rather than mutating the originals — callers' Metric instances
        # are unchanged.
        #
        # The renamed metrics are stored as a list attribute; Keras 3's
        # tracker picks up Metric instances inside attribute lists and
        # exposes them via `Layer.metrics`, which propagates to
        # `Model.metrics` for fit/evaluate logging.
        self.metric_objs = []
        if metrics is not None:
            for m in metrics:
                config = m.get_config()
                if not config["name"].startswith(f"{self.name}_"):
                    config["name"] = f"{self.name}_{config['name']}"
                self.metric_objs.append(m.__class__.from_config(config))

        # Sub-layers are constructed here and stored as attributes. Keras
        # tracks sub-layers automatically via attribute assignment on a
        # Layer, so nothing further is needed to register weights for
        # checkpointing or backprop. Explicit names yield hierarchical
        # names of the form "<head_name>/<sublayer_name>" in model
        # summaries, which is important once we have multiple head
        # instances and a backbone in the same Model.
        self.dropout_1 = keras.layers.Dropout(rate=dropout, name="pre_dense_dropout")
        self.dense = keras.layers.Dense(
            units=self.hidden_dim, activation="relu", name="intermediate_dense"
        )
        self.dropout_2 = keras.layers.Dropout(rate=dropout, name="post_dense_dropout")
        self.logits = keras.layers.Dense(units=1, activation=None, name="logits")

    def call(self, features, targets=None):
        """
        Forward pass.

        Parameters
        ----------
        features : tensor, shape (batch, hidden_dim)
            The shared backbone features (CLS-token representation).
        targets : tensor or None, shape (batch,) or (batch, 1)
            Target labels for this head's task. Only used in endpoint
            mode. When `targets is not None` and `self.loss_fn is not
            None`, the head calls
            `self.add_loss(self.loss_weight * loss_fn(targets, logits))`.

        Returns
        -------
        logits : tensor, shape (batch, 1)
            Per-sample logits. The caller is responsible for any sigmoid/
            threshold handling downstream.
        """

        out = self.dropout_1(features)
        out = self.dense(out)
        out = self.dropout_2(out)
        logits = self.logits(out)

        # Endpoint-layer pattern: if this head was configured with a
        # loss_fn and/or metrics, and is receiving targets (i.e.,
        # training/eval time, not inference), register the loss and
        # update each metric's state. Keras aggregates losses via
        # `model.losses` and metrics via `model.metrics`. The caller
        # should NOT pass a compile-time loss for this output when
        # operating in endpoint mode (Keras would double-count); for
        # the same reason, callers don't need to pass `compile(metrics=...)`
        # for head-internal metrics.
        if targets is not None:
            if self.loss_fn is not None:
                if self.expose_loss_components:
                    loss, components = self.loss_fn.call(
                        targets, logits, return_intermediates=True
                    )
                    # last_components stays UNSCALED — it diagnoses FLPU
                    # internals, not the multi-head mixing (loss_weight).
                    self.last_components = components
                    self.add_loss(self.loss_weight * loss)
                else:
                    self.add_loss(self.loss_weight * self.loss_fn(targets, logits))
            for metric in self.metric_objs:
                metric.update_state(targets, logits)
        return logits
