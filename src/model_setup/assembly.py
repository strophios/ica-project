"""
Assembly: wire backbone + heads into full Keras models for the
classification stack.

Two functions, intentionally separate:

  - `build_endpoint_model`: multi-input training model. Targets are
    `keras.Input`s; head's `add_loss` handles loss internally.
    Returns a `LayerLRModel` so per-layer LR / unfreezing is
    forward-compatible.

  - `build_inference_model`: single-input model with no targets and
    no `add_loss` path. Returns a regular `keras.Model`.

The two-function split is the natural shape of the endpoint-layer
pattern: the training and inference models have *different input
signatures* (targets vs. no targets), so they can't be the same
functional graph. The choice is deliberate — see Piece 4b in
`docs/notes/tier2-design.md` for the full reasoning, including
the empirical finding (verified in
`scripts/experiment_endpoint_inference_evaluate.py`) that Pattern A
— sharing head Layer instances across the two graphs in-process —
is safe in Keras 3.

Usage:

    backbone = load_dapt_backbone(config.DAPT_BACKBONE_WEIGHTS)
    cca_head = ClassificationHead(
        hidden_dim=backbone.hidden_dim,
        loss_fn=FLPULoss(prior=0.02),
        name="cca",
    )
    train_model = build_endpoint_model(
        backbone=backbone,
        heads={"cca": cca_head},
        seq_length=128,
        freeze_encoder=True,
    )
    # Pattern A: same heads dict, weights shared by Python identity
    inf_model = build_inference_model(
        backbone=backbone,
        heads={"cca": cca_head},
        seq_length=128,
    )
"""

from __future__ import annotations

import keras

from src.model_setup.layer_lr_model import LayerLRModel
from src.diagnostics.factory import build_trackers


def _default_group_fn(variable):
    """
    Default `group_fn` for `LayerLRModel`: extract the first path
    component of `variable.path`.

    Gives groups like `"roberta_backbone"`, `"cca"`, `"immig"` — the
    natural per-component groups for backbone-vs-head discriminative
    LR. Not sufficient for *per-encoder-layer* discriminative LR
    (e.g., layer 11 vs. layer 0); for that, callers should provide a
    custom `group_fn`.
    """
    return variable.path.split("/")[0]


def build_endpoint_model(
    backbone,
    heads,
    seq_length,
    target_dtype="float32",
    freeze_encoder=False,
    layer_multipliers=None,
    group_fn=None,
    diagnostics=None,
):
    """
    Wire backbone + heads into a multi-input endpoint-mode training
    model, returned as a `LayerLRModel`.

    Inputs to the resulting model:
      - `token_ids`: `(batch, seq_length)` int32
      - `padding_mask`: `(batch, seq_length)` int32
      - one target Input per head, named `"<head>_targets"`: e.g.,
        for `heads = {"cca": cca_head}`, an Input named
        `"cca_targets"` of dtype `target_dtype`, shape `()` (per-
        sample scalar — produces a `(batch,)` tensor at runtime,
        matching the `ClassifierPreprocessor`'s output).

    The `_targets` suffix on target Input names exists because Keras
    requires unique op names within a Functional graph, and the head
    Layer itself produces an op named `head_name` (e.g., `"cca"`).
    A same-named target Input would collide. Preprocessor users
    should configure `label_keys={"<head>_targets": "<source_col>"}`
    so the preprocessor's output-dict keys match these Input names
    for `.fit()`'s name-based routing.

    Outputs: a dict keyed by head name (no suffix), with each
    head's logit tensor. Output names align with head names so
    `compile(loss={head_name: ...})`-style dict-valued routing
    works idiomatically.

    Args:
        backbone: already-loaded backbone (typically from
            `load_dapt_backbone`). Must accept dict input
            `{"token_ids", "padding_mask"}` and return a
            `(batch, seq, hidden)` tensor.
        heads: `dict[str, ClassificationHead]`. Head names are used
            as the model output names; the corresponding target
            `keras.Input`s use the suffixed name `"<head>_targets"`
            (see the function docstring above for why). Preprocessor
            users configure `label_keys` to emit dict keys matching
            those `_targets` names. The endpoint contract: each
            head's `targets` argument is the corresponding
            `"<head>_targets"` `keras.Input`.
        seq_length: int.
        target_dtype: dtype for target inputs. Must match the
            preprocessor's `target_dtype` (default `"float32"` here
            and there). Default `"float32"`.
        freeze_encoder: if True, sets `backbone.trainable = False`
            (a real freeze — backbone variables are excluded from
            `trainable_variables`, not just zero-multiplier-scaled).
        layer_multipliers: optional `dict[str, float]` for
            `LayerLRModel`. Default `None` = empty = all multipliers
            1.0 (behaviorally equivalent to `keras.Model`).
        group_fn: optional `Callable[[Variable], str]` for
            `LayerLRModel`. Default: `_default_group_fn` (first path
            component).

    Returns:
        A `LayerLRModel` ready to compile (without a `loss` argument
        — heads handle loss via `add_loss`) and fit.
    """

    # Forward-compat boundary-inventory check: enforce unique head names
    # at the call site as well as construction-site (ClassificationHead
    # now requires explicit name). Dict structurally prevents duplicates
    # today, but a future API change (e.g., heads as list of pairs)
    # could allow them — this assertion is the forward-compat guard.
    names = list(heads.keys())
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(
            f"build_endpoint_model requires unique head names; got "
            f"duplicates: {duplicates}"
        )

    # When diagnostics is enabled, assert that dict keys match head.name.
    # _dispatch_diagnostics relies on _head_refs_by_name[tracker.head_name]
    # keyed by head.name, and tracker.head_name derives from the heads-dict
    # key. These must remain equal for diagnostic head-ref lookup to work.
    if diagnostics is not None:
        for k, h in heads.items():
            if k != h.name:
                raise ValueError(
                    f"build_endpoint_model: heads dict key {k!r} != "
                    f"head.name {h.name!r}; diagnostic head-ref lookup "
                    f"requires them equal"
                )

    if freeze_encoder:
        backbone.trainable = False

    # Create inputs
    token_ids = keras.Input(shape=(seq_length,), dtype="int32", name="token_ids")
    padding_mask = keras.Input(shape=(seq_length,), dtype="int32", name="padding_mask")

    target_inputs = dict()

    for head_name in heads.keys():
        target_inputs[f"{head_name}_targets"] = keras.Input(
            shape=(), dtype=target_dtype, name=f"{head_name}_targets"
        )

    # Build the model
    backbone_out = backbone({"token_ids": token_ids, "padding_mask": padding_mask})
    cls_features = backbone_out[:, 0, :]

    outputs = dict()
    for head_name, head in heads.items():
        outputs[head_name] = head(
            cls_features, targets=target_inputs[f"{head_name}_targets"]
        )

    all_inputs = {"token_ids": token_ids, "padding_mask": padding_mask, **target_inputs}

    # Tier 5 diagnostics. The constituent-variable gather MUST happen here:
    # after the head-call loop (so head/backbone variables are realized) AND
    # after the freeze_encoder block above (so backbone.trainable is already
    # False and frozen-encoder builds enumerate only head groups). build_trackers
    # uses this list ONLY for group enumeration; train_step's runtime dispatch
    # uses the model's own self.trainable_variables.
    diagnostic_trackers = None
    diagnostic_head_refs = None
    if diagnostics is not None:
        constituent_trainable = list(backbone.trainable_variables)
        for _h in heads.values():
            constituent_trainable.extend(_h.trainable_variables)
        diagnostic_trackers = build_trackers(
            diagnostics,
            group_fn=group_fn or _default_group_fn,
            heads=heads,
            trainable_variables=constituent_trainable,
        )
        diagnostic_head_refs = list(heads.values())

    return LayerLRModel(
        inputs=all_inputs,
        outputs=outputs,
        group_fn=group_fn or _default_group_fn,
        multipliers=layer_multipliers or {},
        diagnostic_trackers=diagnostic_trackers,
        diagnostic_head_refs=diagnostic_head_refs,
    )


def build_inference_model(backbone, heads, seq_length):
    """
    Wire backbone + heads into a single-input inference model,
    returned as a regular `keras.Model`.

    Inputs to the resulting model:
      - `token_ids`: `(batch, seq_length)` int32
      - `padding_mask`: `(batch, seq_length)` int32

    No target inputs. Heads are called with `targets=None`, which
    skips the `add_loss` path inside the head.

    Outputs: a dict keyed by head name, same as `build_endpoint_model`.

    **Pattern A caveat (in-process flow)**: when this function is
    called with the same `heads` dict that was passed to
    `build_endpoint_model` in the same process, weights are
    physically shared via Python identity — fitting the training
    model also trains the inference model's weights. Use the
    inference model for `predict()` in this scenario.

    `evaluate()` on the shared-instance inference model is empirically
    safe in Keras 3 (Keras 3 filters losses by graph reachability,
    so the head's stale training-graph `add_loss` tensor doesn't
    contaminate the inference model's loss aggregation — verified in
    `scripts/experiment_endpoint_inference_evaluate.py`). But the
    operational rule "predict only on the shared-instance inference
    model" is simpler than reasoning about Keras-version-dependent
    dependency-filtering logic. For full-method support (evaluate,
    fit), build a fresh inference model from fresh head instances
    and load weights by name (Pattern 2 — what the eval script does).

    Args:
        backbone: already-loaded backbone. Same contract as
            `build_endpoint_model`.
        heads: `dict[str, ClassificationHead]`.
        seq_length: int.

    Returns:
        A `keras.Model` ready for `predict()`.
    """

    # Create inputs
    token_ids = keras.Input(shape=(seq_length,), dtype="int32", name="token_ids")
    padding_mask = keras.Input(shape=(seq_length,), dtype="int32", name="padding_mask")

    # Build the model
    backbone_out = backbone({"token_ids": token_ids, "padding_mask": padding_mask})
    cls_features = backbone_out[:, 0, :]

    outputs = dict()
    for head_name, head in heads.items():
        outputs[head_name] = head(cls_features)

    all_inputs = {"token_ids": token_ids, "padding_mask": padding_mask}

    return keras.Model(inputs=all_inputs, outputs=outputs)
