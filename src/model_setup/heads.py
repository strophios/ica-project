"""
Classification head Layers for the multi-head ICA classifier.

This module defines the head Layers that sit on top of the shared DAPT
backbone. Each head is a `keras.layers.Layer` subclass, which is a more
structured abstraction than the inline-head construction the original
`classification_setup.py` used. The upside is that heads are:

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

import keras


class ClassificationHead(keras.layers.Layer):
    """
    Binary classification head that sits on top of shared backbone features.

    Architecture (matches the original `classification_setup.py` single-head
    shape, so the weights learned under the old code are compatible):

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
    name : str or None
        Passed to `keras.layers.Layer.__init__`. Appears in
        `model.summary()` output and is used for serialization.

    Notes
    -----
    The `features` input to `call` is expected to already be the shared
    CLS-token representation from the backbone — i.e., shape
    `(batch, hidden_dim)`, not the full sequence `(batch, seq_len, hidden_dim)`.
    The assembly function that wires the backbone and heads together is
    responsible for the CLS-token slice.
    """

    def __init__(self, hidden_dim, dropout=0.1, loss_fn=None, name=None):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.loss_fn = loss_fn

        # Sub-layers are constructed here and stored as attributes. Keras
        # tracks sub-layers automatically via attribute assignment on a
        # Layer, so nothing further is needed to register weights for
        # checkpointing or backprop. Explicit names yield hierarchical
        # names of the form "<head_name>/<sublayer_name>" in model
        # summaries, which is important once we have multiple head
        # instances and a backbone in the same Model.
        self.dropout_1 = keras.layers.Dropout(
            rate=dropout, name="pre_dense_dropout"
        )
        self.dense = keras.layers.Dense(
            units=self.hidden_dim, activation="relu", name="intermediate_dense"
        )
        self.dropout_2 = keras.layers.Dropout(
            rate=dropout, name="post_dense_dropout"
        )
        self.logits = keras.layers.Dense(
            units=1, activation=None, name="logits"
        )

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
            None`, the head calls `self.add_loss(loss_fn(targets, logits))`.

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
        # loss_fn and is receiving targets (i.e., training/eval time,
        # not inference), register the loss with Keras's internal
        # aggregator via `add_loss`. The outer Model picks these up
        # automatically; the caller should NOT pass a compile-time
        # loss for this output when operating in endpoint mode.
        if targets is not None and self.loss_fn is not None:
            self.add_loss(self.loss_fn(targets, logits))
        return logits
