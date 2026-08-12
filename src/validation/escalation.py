# pattern: Functional Core
"""Fine-tuning escalation knobs and decision logic (AC3.3).

Provides grouping logic for per-layer discriminative learning rates and
a decision function to evaluate whether to escalate from frozen-probe
to unfreezing/fine-tuning based on transfer learning performance.
"""

from __future__ import annotations


# Non-layer backbone components (keras_hub RobertaBackbone): the token/position
# embeddings and their layer norm. These carry no shared substring with the
# numbered transformer blocks or with each other beyond this "embeddings"
# prefix -- there is no single marker that identifies "any backbone param" the
# way the pre-2026-07-27 "roberta" substring check assumed (see NAMING NOTE
# below). Always classified "encoder_frozen": unfreezing embeddings is out of
# scope for the top-N-transformer-blocks scheme this function implements.
_ENCODER_NON_LAYER_PREFIXES = ("embeddings",)


def top_n_group_fn(n_top: int, n_layers: int = 12, layer_prefix: str = "transformer_layer"):
    """Create a group function for per-layer LR scaling with top-N unfreezing.

    Groups variables into three categories for LayerLRModel's multiplier-scaling:
    - "encoder_top": the top n_top backbone transformer blocks (unfrozen in
      escalation mode)
    - "encoder_frozen": all other backbone params -- the remaining transformer
      blocks AND the non-layer embeddings/embeddings_layer_norm params
    - "head": classification head (always trained)

    The top N layers are identified by checking if any layer name in the set
    {layer_prefix}_{n_layers-1} ... {layer_prefix}_{n_layers-n_top} appears
    in the variable's path.

    NAMING NOTE (found empirically 2026-07-27, via the first real
    encoder-unfreeze smoke run against the actual DAPT backbone -- see
    scripts/rel_unfreeze_smoke.py): keras_hub's `RobertaBackbone` names its
    transformer blocks `transformer_layer_{i}`, NOT `roberta_layer_{i}` --
    the prior default was a documentation-only naming guess, never checked
    against a real backbone (the US-filter's real unfreeze run this
    machinery was "proven" on is still an operator-gated pending item; only
    the frozen-probe default path, which never calls this function, had
    actually run for real). The smoke run's `grad_norm/encoder_top/*`
    tracker was silently ABSENT (not zero -- absent) under the old default,
    because `top_n_group_fn`'s `groups = sorted({{group_fn(v) for v in
    trainable_variables}})` never produced an "encoder_top" group at all: no
    real variable path ever matched `roberta_layer_*`, and none matched the
    old "roberta" substring fallback either (no real backbone param contains
    that substring anywhere), so every backbone variable silently fell
    through to "head" and trained at full LR -- a correctness bug, not just
    a cosmetic one, affecting every un-run escalation config that predates
    this fix (`run_us_classification.py`'s escalation branch included).

    Args:
        n_top: Number of top layers to unfreeze (0 <= n_top <= n_layers).
        n_layers: Total number of backbone transformer blocks (default 12
            for RoBERTa-base).
        layer_prefix: The transformer-block variable-path prefix (default
            "transformer_layer", matching keras_hub's RobertaBackbone).
            Exposed for forward-compat with other backbones/keras_hub
            versions, not because this project uses more than one today.

    Returns:
        A group function (var -> str) for use with LayerLRModel(group_fn=...).
    """
    # Pre-compute the set of top-layer names to check.
    # If n_layers=12, n_top=2: check for "transformer_layer_11" and "transformer_layer_10"
    top_indices = [n_layers - 1 - i for i in range(n_top)]
    top_layer_names = {f"{layer_prefix}_{idx}" for idx in top_indices}

    def group_of(var):
        """Group a variable based on whether it's in the top N layers."""
        path = var.path

        # Check if any top-layer name appears in the path, using boundary matching
        # to avoid substring collisions (e.g., transformer_layer_1 matching
        # transformer_layer_11). Boundary patterns: "{layer_name}/" or path
        # segment equality after splitting on "/".
        path_segments = path.split("/")
        if any(
            layer_name in path_segments or f"{layer_name}/" in path
            for layer_name in top_layer_names
        ):
            return "encoder_top"

        # Any other transformer block (not in the top N).
        if layer_prefix in path:
            return "encoder_frozen"

        # Non-layer backbone params (embeddings, embeddings_layer_norm).
        if path_segments[0].startswith(_ENCODER_NON_LAYER_PREFIXES):
            return "encoder_frozen"

        # Everything else (head vars)
        return "head"

    return group_of


DEFAULT_ESCALATION_MULTIPLIERS: dict[str, float] = {
    "head": 1.0,
    "encoder_top": 0.1,
    "encoder_frozen": 0.0,
}


def per_layer_group_fn(n_top: int, n_layers: int = 12, layer_prefix: str = "transformer_layer"):
    """Like `top_n_group_fn`, but each unfrozen layer gets its OWN group.

    Groups: "encoder_layer_{idx}" for each of the top `n_top` transformer
    blocks (so each can carry a distinct LR multiplier -- ULMFiT-style graded
    discriminative rates), "encoder_frozen" for the rest of the backbone
    (lower blocks + embeddings), "head" for everything else. Same
    boundary-safe path matching as `top_n_group_fn` (segment equality, so
    transformer_layer_1 never matches transformer_layer_11).
    """
    top_indices = [n_layers - 1 - i for i in range(n_top)]
    name_to_group = {f"{layer_prefix}_{idx}": f"encoder_layer_{idx}" for idx in top_indices}

    def group_of(var):
        path = var.path
        path_segments = path.split("/")
        for layer_name, group in name_to_group.items():
            if layer_name in path_segments or f"{layer_name}/" in path:
                return group
        if layer_prefix in path:
            return "encoder_frozen"
        if path_segments[0].startswith(_ENCODER_NON_LAYER_PREFIXES):
            return "encoder_frozen"
        return "head"

    return group_of


def graded_multipliers(
    n_top: int, n_layers: int = 12, base: float = 0.1, decay: float = 0.5
) -> dict[str, float]:
    """Geometric per-layer LR multipliers for graded top-N unfreezing.

    Top layer gets `base`, each layer below it `base * decay^k` (ULMFiT-style
    discriminative rates: highest at the top, decaying with depth). Keys match
    `per_layer_group_fn`'s group names; "head" is 1.0, "encoder_frozen" 0.0.
    Example (n_top=3, base=0.1, decay=0.5): layer 11 -> 0.1, 10 -> 0.05,
    9 -> 0.025.
    """
    if n_top < 1:
        raise ValueError(f"graded multipliers need n_top >= 1, got {n_top}")
    if not (0.0 < decay <= 1.0) or base <= 0.0:
        raise ValueError(f"need base > 0 and 0 < decay <= 1, got base={base} decay={decay}")
    out: dict[str, float] = {"head": 1.0, "encoder_frozen": 0.0}
    for k in range(n_top):
        out[f"encoder_layer_{n_layers - 1 - k}"] = base * (decay**k)
    return out


def _validate_per_layer_keys(layer_multipliers: dict, n_top: int, n_layers: int) -> None:
    """Raise if per-layer multiplier keys don't exactly cover the top-N groups.

    Load-bearing defense: `LayerLRModel.get_multiplier` DEFAULTS missing groups
    to 1.0 ("train normally"), so a typo'd or missing per-layer key would
    silently train that layer at FULL learning rate. Exact-cover validation
    makes that impossible.
    """
    expected = {f"encoder_layer_{n_layers - 1 - k}" for k in range(n_top)}
    provided = {k for k in layer_multipliers if k.startswith("encoder_layer_")}
    if provided != expected:
        raise ValueError(
            f"per-layer multipliers must cover exactly the top-{n_top} groups: "
            f"missing={sorted(expected - provided)} unexpected={sorted(provided - expected)}"
        )


# Real Keras sub-layer names for the non-layer backbone components (see
# _ENCODER_NON_LAYER_PREFIXES above for the corresponding group-fn prefix
# match) -- distinct from that prefix tuple because `frozen_sublayer_names`
# needs exact `backbone.get_layer(name)` lookup names, not a prefix.
_NON_LAYER_BACKBONE_SUBLAYER_NAMES = ("embeddings", "embeddings_layer_norm")


def frozen_sublayer_names(unfreeze_top_n: int, n_layers: int = 12) -> tuple[str, ...]:
    """Pure: real backbone sub-layer NAMES to hard-freeze (Capability 1,
    docs/notes/branched-encoder-strategy.md "Hard freezing is now a
    requirement"), given `unfreeze_top_n` unfrozen top transformer blocks.

    Companion to `top_n_group_fn`'s grouping (the embeddings pair plus every
    transformer block below the top N are "encoder_frozen") but returns
    layer NAMES for `backbone.get_layer(name).trainable = False`, not a
    per-Variable group function. Exists because `LayerLRModel`'s zero
    gradient multiplier alone does not freeze a variable: AdamW's decoupled
    weight decay still shrinks multiplier=0 "frozen" variables every step
    regardless of gradient scale (docs/notes/encoder-unfreeze-strategy.md,
    2026-07-29 finding — measured ~2.3e-3 max-abs drift over 5 epochs).

    Args:
        unfreeze_top_n: number of top transformer blocks left trainable
            (must be in [0, n_layers]).
        n_layers: total backbone transformer blocks (default 12 for
            RoBERTa-base).
    """
    if not (0 <= unfreeze_top_n <= n_layers):
        raise ValueError(f"unfreeze_top_n must be in [0, {n_layers}]; got {unfreeze_top_n}")
    n_frozen_layers = n_layers - unfreeze_top_n
    return _NON_LAYER_BACKBONE_SUBLAYER_NAMES + tuple(
        f"transformer_layer_{i}" for i in range(n_frozen_layers)
    )


def escalation_build_kwargs(
    unfreeze_top_n: int,
    layer_multipliers: dict | None = None,
    n_layers: int = 12,
    hard_freeze: bool = False,
) -> dict:
    """Pure: derive the `build_endpoint_model` kwargs for the encoder-unfreeze
    escalation branch.

    Extracted from the inline branch in `run_us_classification.py:154-179` (that
    script's own branch is left untouched — this is the semantics a NEW caller
    (`run_relevance_text.py`) mirrors, made independently testable):

    - `unfreeze_top_n > 0` (unfreezing path): `freeze_encoder=False`, a
      `group_fn` from `top_n_group_fn(unfreeze_top_n, n_layers=n_layers)`, and
      `layer_multipliers` — the caller-supplied dict if given, else
      `DEFAULT_ESCALATION_MULTIPLIERS`. When `hard_freeze=True`, also
      includes `hard_freeze_names` (from `frozen_sublayer_names`) so
      `build_endpoint_model` sets `trainable=False` on the sub-branch layers
      instead of relying on the zero gradient multiplier alone.
    - `unfreeze_top_n == 0` (frozen-probe path, the default): just
      `{"freeze_encoder": True}` — no `group_fn`/`layer_multipliers`/
      `hard_freeze_names` keys, matching `build_endpoint_model`'s own
      defaults for those params (a byte-identical no-op relative to not
      touching them at all). `hard_freeze` is ignored on this path — the
      frozen-probe encoder is already fully frozen via `freeze_encoder=True`,
      so hard-freezing it too would be a no-op; treating it as a no-op here
      (rather than raising) means the CLI's hard-freeze-on-by-default
      posture for new runs doesn't force `--no-hard-freeze` on a plain
      frozen-probe smoke run (mirrors the existing "frozen path ignores
      layer_multipliers" precedent below).

    Returns a dict meant to be merged into the full `build_endpoint_model` call
    (backbone/heads/seq_length/diagnostics are the caller's responsibility).
    """
    if unfreeze_top_n > 0:
        # Grouping is inferred from the multipliers dict's keys so that a
        # config sidecar (which records only the dict) fully determines the
        # run's behavior on reload: per-layer keys ("encoder_layer_*") select
        # per-layer grouping (graded rates); otherwise the flat
        # encoder_top/encoder_frozen scheme.
        if layer_multipliers and any(k.startswith("encoder_layer_") for k in layer_multipliers):
            _validate_per_layer_keys(layer_multipliers, unfreeze_top_n, n_layers)
            result = {
                "freeze_encoder": False,
                "group_fn": per_layer_group_fn(unfreeze_top_n, n_layers=n_layers),
                "layer_multipliers": dict(layer_multipliers),
            }
        else:
            result = {
                "freeze_encoder": False,
                "group_fn": top_n_group_fn(unfreeze_top_n, n_layers=n_layers),
                "layer_multipliers": (
                    dict(layer_multipliers)
                    if layer_multipliers
                    else dict(DEFAULT_ESCALATION_MULTIPLIERS)
                ),
            }
        if hard_freeze:
            result["hard_freeze_names"] = frozen_sublayer_names(unfreeze_top_n, n_layers=n_layers)
        return result
    return {"freeze_encoder": True}


def escalation_decision(
    baseline_f1: float, transfer_f1: float, margin: float = 0.1
) -> dict[str, bool | str]:
    """Decide whether to escalate from frozen probe to unfreezing.

    Escalates (returns True) when the transfer F1 drops more than `margin`
    below the baseline F1. This is used to decide if fine-tuning on the
    transfer task is necessary.

    Args:
        baseline_f1: F1 score of the frozen probe on in-distribution data.
        transfer_f1: F1 score of the frozen probe on transfer (pre-1986) data.
        margin: Acceptable performance gap (default 0.1). Escalate when
                baseline_f1 - transfer_f1 > margin.

    Returns:
        Dict with keys:
        - escalate (bool): whether to fine-tune
        - rationale (str): human-readable explanation
    """
    gap = baseline_f1 - transfer_f1
    escalate = gap > margin

    rationale = (
        f"transfer F1 {transfer_f1:.3f} vs baseline {baseline_f1:.3f} "
        f"(gap {gap:.3f} {'>' if escalate else '<='} margin {margin})"
    )

    return {
        "escalate": escalate,
        "rationale": rationale,
    }
