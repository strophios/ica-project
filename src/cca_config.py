"""
Run configuration for the CCA classifier (Tier 3 Piece 3 — I4).

This module is the single source of truth for the **architectural and
research-dimension parameters** of a CCA training run. It exists to
solve the train/eval coupling problem (I4 from the Tier 2 review):
training and eval scripts independently set values like `seq_length`,
`text_key`, head configuration, and FLPU prior; without a shared
config, drift between the two scripts is silent and bites at predict
time as wrong-answers-from-the-wrong-architecture.

The pattern: each training run is identified by a `RunConfig`
instance (a frozen dataclass). At training time the script imports
`DEFAULT_CCA_CONFIG` (or derives a variant via
`dataclasses.replace`); the config drives all coupling-relevant
values; the config is serialized to a JSON sidecar alongside the
weights at the end of training. At eval time the script loads the
config from the sidecar and constructs the inference model with
the exact same values.

See `docs/notes/tier3-design.md` Piece 3 for the design framing,
including the "wrapped vs. flat" forward-compat reasoning behind
the dataclass nesting (`loss`, `lr_schedule`, `ratio_batch`,
`optimizer` are wrapped sub-configs to keep type-discrimination
migrations non-breaking when alternative types are introduced).

What's IN here: parameters that must agree between train and eval
(`seq_length`, `text_key`, `target_dtype`, head names + shapes +
loss config) plus parameters we expect to vary across HP-search
experiments (`epochs`, backbone choice, Ratio Batch ratios, LR
schedule params, optimizer params).

What's NOT in here: script-local operational choices (BATCH_SIZE,
callbacks), train/predict mode (a call-site choice per pinned
question #3), metrics (monitoring choice, not load-bearing for
predict), DAPT input/cache paths (environment-dependent, in
`src/config.py`).

Example usage::

    import dataclasses
    import src.cca_config as cca_config

    # Use the canonical config
    run_config = cca_config.DEFAULT_CCA_CONFIG

    # Or derive a variant for an experiment
    run_config = dataclasses.replace(
        cca_config.DEFAULT_CCA_CONFIG,
        epochs=10,
        heads=(dataclasses.replace(
            cca_config.DEFAULT_CCA_CONFIG.heads[0],
            loss=dataclasses.replace(
                cca_config.DEFAULT_CCA_CONFIG.heads[0].loss,
                prior=0.05,
            ),
        ),),
    )

    # Drive preprocessor / head / assembly construction
    seq_length = run_config.seq_length
    label_keys = run_config.label_keys      # derived
    head_names = run_config.head_names      # derived

    # After fit, save sidecar alongside weights
    sidecar_path = cca_config.config_path_for_weights(weights_path)
    run_config.to_json(sidecar_path)

    # In eval script, load from sidecar
    run_config = cca_config.RunConfig.from_json(
        cca_config.config_path_for_weights(weights_path)
    )
    run_config.validate_against_backbone(backbone)
"""

from __future__ import annotations

import dataclasses
import json
import math
import warnings
from pathlib import Path
from typing import Any

import keras  # for keras.backend.standardize_dtype in target_dtype validation

import src.config as config


# ---------------------------------------------------------------------------
# Loss configs
# ---------------------------------------------------------------------------
# `FLPULossConfig` is wrapped under `HeadConfig.loss` (rather than as
# flat `flpu_prior` / `flpu_kiryo_clawback` fields directly on
# HeadConfig). When the planned ALUM piece (pinned question #1)
# lands, we'll add `ALUMLossConfig` and widen the annotation to
# `loss: FLPULossConfig | ALUMLossConfig | ...`. Old saved sidecars
# (with `loss: {prior: 0.02, kiryo_clawback: false}`) still
# deserialize cleanly because they're valid `FLPULossConfig`
# payloads. See `docs/notes/tier3-design.md` Piece 3 reasoning for
# the wrapped-vs-flat forward-compat framing.


@dataclasses.dataclass(frozen=True)
class FLPULossConfig:
    """FLPU loss configuration (Kiryo nnPU + Lin focal CE)."""

    prior: float
    kiryo_clawback: bool = False
    # nnPNU PU<->PN mixing weight for reliable negatives (0 = pure nnPU). Used by
    # the relevance head; FLPULoss guards eta>0 against kiryo_clawback=True.
    nnpnu_eta: float = 0.0

    def __post_init__(self):
        if not isinstance(self.prior, (int, float)):
            raise ValueError(
                f"FLPULossConfig.prior must be a number; "
                f"got {self.prior!r} (type {type(self.prior).__name__})."
            )
        if not (0.0 < float(self.prior) < 1.0):
            raise ValueError(
                f"FLPULossConfig.prior must be in (0, 1); got {self.prior}."
            )
        if not isinstance(self.kiryo_clawback, bool):
            raise ValueError(
                f"FLPULossConfig.kiryo_clawback must be a bool; "
                f"got {self.kiryo_clawback!r} "
                f"(type {type(self.kiryo_clawback).__name__})."
            )
        if not (0.0 <= float(self.nnpnu_eta) <= 1.0):
            raise ValueError(
                f"FLPULossConfig.nnpnu_eta must be in [0, 1]; got {self.nnpnu_eta}."
            )
        if float(self.nnpnu_eta) > 0.0 and self.kiryo_clawback:
            raise ValueError(
                "FLPULossConfig: nnpnu_eta>0 with kiryo_clawback is not supported "
                "(see FLPULoss / docs/notes/pinned-questions.md)."
            )

    @classmethod
    def _from_dict(
        cls, payload: dict, _source: str = "<dict>"
    ) -> FLPULossConfig:
        kwargs = _filter_known_fields(cls, payload, _source=_source)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Head config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class HeadConfig:
    """Configuration for a single ClassificationHead.

    Holds the architectural-shape parameters (name, source column,
    hidden_dim) plus the head-internal loss configuration. Will
    become a discriminated union (or grow a `type` field) when
    `CombinedClassificationHead` lands; for now strictly
    ClassificationHead-shaped.
    """

    name: str
    source_column: str
    hidden_dim: int
    loss: FLPULossConfig

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"HeadConfig.name must be a non-empty string; "
                f"got {self.name!r}."
            )
        if "/" in self.name:
            raise ValueError(
                f"HeadConfig.name must not contain '/'; got {self.name!r}. "
                f"'/' is the Keras variable-path separator that "
                f"_default_group_fn in src/model_setup/assembly.py splits "
                f"on to group variables by head; a name containing '/' "
                f"would silently mis-group, breaking discriminative LR."
            )
        if not isinstance(self.source_column, str) or not self.source_column:
            raise ValueError(
                f"HeadConfig.source_column must be a non-empty string; "
                f"got {self.source_column!r}."
            )
        if not isinstance(self.hidden_dim, int) or self.hidden_dim <= 0:
            raise ValueError(
                f"HeadConfig.hidden_dim must be a positive int; "
                f"got {self.hidden_dim!r} "
                f"(type {type(self.hidden_dim).__name__})."
            )
        if not isinstance(self.loss, FLPULossConfig):
            raise ValueError(
                f"HeadConfig.loss must be a FLPULossConfig instance "
                f"(currently the only supported loss type); got "
                f"{type(self.loss).__name__}."
            )

    @classmethod
    def _from_dict(
        cls, payload: dict, _source: str = "<dict>"
    ) -> HeadConfig:
        if "loss" not in payload:
            raise ValueError(
                f"HeadConfig payload at {_source} is missing required "
                f"field 'loss'."
            )
        loss = FLPULossConfig._from_dict(
            payload["loss"], _source=f"{_source}.loss"
        )
        kwargs = _filter_known_fields(cls, payload, _source=_source)
        kwargs["loss"] = loss
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Ratio Batch config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RatioBatchConfig:
    """Per-split positive-class oversampling weights for the data
    pipeline's `sample_from_datasets`.

    Each value is the *positive-class weight* for that split; the
    unlabeled weight is implicitly `1 - <pos>`. Defaults
    (0.1 train, 0.5 val/test) match the current script behavior:
    1:9 positive:unlabeled in training, 1:1 in validation/test.

    Pre-Tier-3 the train ratio was a script-local hardcode (1/10);
    promoting it here makes it sweepable for the Tier-2-pinned
    "Ratio Batch sensitivity" empirical-check item.
    """

    train_pos: float = 0.1
    val_pos: float = 0.5
    test_pos: float = 0.5

    def __post_init__(self):
        for name, val in (
            ("train_pos", self.train_pos),
            ("val_pos", self.val_pos),
            ("test_pos", self.test_pos),
        ):
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"RatioBatchConfig.{name} must be a number; "
                    f"got {val!r} (type {type(val).__name__})."
                )
            if not (0.0 < float(val) < 1.0):
                raise ValueError(
                    f"RatioBatchConfig.{name} must be in (0, 1); got {val}."
                )

    @classmethod
    def _from_dict(
        cls, payload: dict, _source: str = "<dict>"
    ) -> RatioBatchConfig:
        kwargs = _filter_known_fields(cls, payload, _source=_source)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Resolved steps (Tier 4 Piece 2 — I4)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ResolvedSteps:
    """LR schedule step counts resolved from LRScheduleConfig
    factors against a concrete steps_per_epoch.

    Populated by `LRScheduleConfig.with_resolved` at training
    time; `steps_per_epoch` is recorded for provenance. See
    docs/notes/tier4-design.md Piece 2 for the rationale.
    """

    warmup_steps: int
    decay_steps: int
    steps_per_epoch: int

    def __post_init__(self):
        for name, val in (
            ("warmup_steps", self.warmup_steps),
            ("decay_steps", self.decay_steps),
            ("steps_per_epoch", self.steps_per_epoch),
        ):
            if not isinstance(val, int) or isinstance(val, bool):
                raise ValueError(
                    f"ResolvedSteps.{name} must be an int; got "
                    f"{val!r} (type {type(val).__name__})."
                )
            if val <= 0:
                raise ValueError(
                    f"ResolvedSteps.{name} must be > 0; got {val}."
                )

    @classmethod
    def _from_dict(cls, payload: dict, _source: str = "<dict>") -> "ResolvedSteps":
        kwargs = _filter_known_fields(cls, payload, _source=_source)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# LR schedule config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LRScheduleConfig:
    """CosineDecay-with-warmup learning rate schedule parameters.

    Parameters drive the construction of
    `keras.optimizers.schedules.CosineDecay` at training time.
    Schedule type is implicitly CosineDecay; will be renamed to
    `CosineDecayConfig` when alternative schedule types
    (ExponentialDecay etc.) are introduced.

    `warmup_steps_factor` is the fraction of one epoch's steps used
    for warmup; `decay_steps_factor` is the multiple of one epoch's
    steps over which decay happens. `decay_alpha` is the `alpha`
    argument to CosineDecay (final-LR / warmup_target).
    """

    initial_lr: float = 1e-4
    warmup_target: float = 1e-3
    decay_alpha: float = 1e-1
    warmup_steps_factor: float = 0.25
    decay_steps_factor: float = 3.0
    resolved: "ResolvedSteps | None" = None

    def __post_init__(self):
        for name, val, must_be_positive in (
            ("initial_lr", self.initial_lr, True),
            ("warmup_target", self.warmup_target, True),
            ("decay_alpha", self.decay_alpha, False),  # >=0
            ("warmup_steps_factor", self.warmup_steps_factor, True),
            ("decay_steps_factor", self.decay_steps_factor, True),
        ):
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"LRScheduleConfig.{name} must be a number; "
                    f"got {val!r} (type {type(val).__name__})."
                )
            if must_be_positive and float(val) <= 0:
                raise ValueError(
                    f"LRScheduleConfig.{name} must be > 0; got {val}."
                )
            if not must_be_positive and float(val) < 0:
                raise ValueError(
                    f"LRScheduleConfig.{name} must be >= 0; got {val}."
                )

        if self.resolved is not None and not isinstance(
            self.resolved, ResolvedSteps
        ):
            raise ValueError(
                f"LRScheduleConfig.resolved must be a ResolvedSteps "
                f"instance or None; got {self.resolved!r} (type "
                f"{type(self.resolved).__name__})."
            )

    def with_resolved(self, steps_per_epoch: int) -> "LRScheduleConfig":
        """Return a new LRScheduleConfig instance with the resolved
        field populated.

        Computes warmup_steps = floor(warmup_steps_factor *
        steps_per_epoch) and decay_steps = floor(decay_steps_factor
        * steps_per_epoch), matching the math.floor pattern used at
        src/run_cca_classification.py:214 for steps_per_epoch itself.

        Existing behavior at lines 286-291 of run_cca_classification.py
        multiplied factors by steps_per_epoch without flooring,
        passing floats to Keras's CosineDecay (which coerces).
        Switching to explicit floor here makes the resolved values
        deterministic integers — the numerical effect is tiny (e.g.,
        571.75 → 571) and the project doesn't rely on byte-exact
        reproduction.
        """
        # Input validation duplicated from ResolvedSteps.__post_init__
        # (intentional): this boundary check produces a clearer error
        # message at the public-API level before attempting the
        # ResolvedSteps construction. Invariant: any change to the
        # input rules must be applied to both places.
        if not isinstance(steps_per_epoch, int) or isinstance(
            steps_per_epoch, bool
        ):
            raise ValueError(
                f"steps_per_epoch must be a positive int; got "
                f"{steps_per_epoch!r} (type "
                f"{type(steps_per_epoch).__name__})."
            )
        if steps_per_epoch <= 0:
            raise ValueError(
                f"steps_per_epoch must be > 0; got {steps_per_epoch}."
            )

        resolved = ResolvedSteps(
            warmup_steps=math.floor(
                self.warmup_steps_factor * steps_per_epoch
            ),
            decay_steps=math.floor(
                self.decay_steps_factor * steps_per_epoch
            ),
            steps_per_epoch=steps_per_epoch,
        )
        return dataclasses.replace(self, resolved=resolved)

    @classmethod
    def _from_dict(
        cls, payload: dict, _source: str = "<dict>"
    ) -> LRScheduleConfig:
        # Reconstruct the nested ResolvedSteps if present.
        resolved_payload = payload.get("resolved")
        if resolved_payload is None:
            resolved = None
        elif isinstance(resolved_payload, dict):
            resolved = ResolvedSteps._from_dict(
                resolved_payload, _source=f"{_source}.resolved"
            )
        else:
            raise ValueError(
                f"Expected 'resolved' in {_source} to be a dict or "
                f"null; got {type(resolved_payload).__name__}."
            )

        kwargs = _filter_known_fields(cls, payload, _source=_source)
        kwargs["resolved"] = resolved
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Optimizer config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    """AdamW-specific optimizer parameters.

    Wrapped in a sub-config (rather than as a flat `weight_decay`
    field on RunConfig) for forward-compat: when alternative
    optimizer types (SGD with momentum, etc.) are introduced, this
    becomes a discriminated union — old sidecars deserialize
    cleanly with a default `type="adamw"` discriminator. Likely
    renamed `AdamWOptimizerConfig` at that point.
    """

    weight_decay: float = 5e-3

    def __post_init__(self):
        if not isinstance(self.weight_decay, (int, float)):
            raise ValueError(
                f"OptimizerConfig.weight_decay must be a number; "
                f"got {self.weight_decay!r} "
                f"(type {type(self.weight_decay).__name__})."
            )
        if float(self.weight_decay) < 0:
            raise ValueError(
                f"OptimizerConfig.weight_decay must be >= 0; "
                f"got {self.weight_decay}."
            )

    @classmethod
    def _from_dict(
        cls, payload: dict, _source: str = "<dict>"
    ) -> OptimizerConfig:
        kwargs = _filter_known_fields(cls, payload, _source=_source)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Diagnostics config (Tier 5 Phase 2)
# ---------------------------------------------------------------------------

# Valid per-step gradient-norm aggregations a run may request. Deliberately
# duplicates src.diagnostics.trackers._VALID_AGGREGATIONS rather than
# importing it: cca_config describes runs declaratively and must not import
# the machinery it configures (FLPULossConfig does not import FLPULoss,
# OptimizerConfig does not import the optimizer, etc.). The two literals are
# pinned equal by TestDiagnosticsAggregationConstantSync in
# tests/test_cca_config.py — change both together.
_VALID_GRADIENT_AGGREGATIONS = ("max", "mean")
_VALID_SUMMARY_STATS = ("mean", "std", "frac_above_0.5")


@dataclasses.dataclass(frozen=True)
class DiagnosticsConfig:
    """Tier 5 diagnostic instrumentation configuration.

    Embedded as the (defaulted) last field of RunConfig. A pre-Tier-5
    sidecar lacking the 'diagnostics' key reconstructs as DiagnosticsConfig()
    (all enabled) via RunConfig's default_factory — see RunConfig._from_dict.

    Group names are NOT stored here; they are derived from the model at
    factory build time by walking trainable_variables with group_fn.
    """

    enable_gradient_norms: bool = True
    enable_overflow_proxy: bool = True
    enable_loss_components: bool = True
    enable_batch_balance: bool = True
    enable_prediction_distribution: bool = True
    gradient_norm_aggregations: tuple[str, ...] = ("max", "mean")
    prediction_summary_stats: tuple[str, ...] = ("mean", "std", "frac_above_0.5")

    def __post_init__(self):
        for field_name in (
            "enable_gradient_norms",
            "enable_overflow_proxy",
            "enable_loss_components",
            "enable_batch_balance",
            "enable_prediction_distribution",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise ValueError(
                    f"DiagnosticsConfig.{field_name} must be bool; "
                    f"got {type(value).__name__}."
                )

        if (
            not isinstance(self.gradient_norm_aggregations, tuple)
            or len(self.gradient_norm_aggregations) == 0
        ):
            raise ValueError(
                "DiagnosticsConfig.gradient_norm_aggregations must be a "
                f"non-empty tuple; got {self.gradient_norm_aggregations!r}."
            )
        for agg in self.gradient_norm_aggregations:
            if agg not in _VALID_GRADIENT_AGGREGATIONS:
                raise ValueError(
                    "DiagnosticsConfig.gradient_norm_aggregations entries "
                    f"must be in {_VALID_GRADIENT_AGGREGATIONS}; got {agg!r}."
                )

        if (
            not isinstance(self.prediction_summary_stats, tuple)
            or len(self.prediction_summary_stats) == 0
        ):
            raise ValueError(
                "DiagnosticsConfig.prediction_summary_stats must be a "
                f"non-empty tuple; got {self.prediction_summary_stats!r}."
            )
        for stat in self.prediction_summary_stats:
            if stat not in _VALID_SUMMARY_STATS:
                raise ValueError(
                    "DiagnosticsConfig.prediction_summary_stats entries must "
                    f"be in {_VALID_SUMMARY_STATS}; got {stat!r}."
                )

    @classmethod
    def _from_dict(
        cls, payload: dict, _source: str = "<dict>"
    ) -> "DiagnosticsConfig":
        kwargs = _filter_known_fields(cls, payload, _source=_source)
        # dataclasses.asdict + json serializes tuples as arrays; they
        # deserialize as lists. Coerce back so __post_init__'s tuple
        # checks (and equality with default tuples) hold.
        for tuple_field in ("gradient_norm_aggregations", "prediction_summary_stats"):
            if tuple_field in kwargs and isinstance(kwargs[tuple_field], list):
                kwargs[tuple_field] = tuple(kwargs[tuple_field])
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Run config (the top-level container)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Full training-run configuration for a CCA classifier.

    See module docstring for the high-level pattern; see
    `docs/notes/tier3-design.md` Piece 3 for the design reasoning.

    Validation hierarchy: each sub-config validates its own fields
    in its own `__post_init__`; this top-level `__post_init__`
    validates the RunConfig-owned fields and the cross-object
    invariants (head name uniqueness). Construction order is
    bottom-up — sub-config `__post_init__` runs first, so by the
    time RunConfig's `__post_init__` runs every sub-config is
    self-valid.

    External-context invariants (e.g., `head.hidden_dim` matching
    an actual loaded backbone) live in `validate_against_backbone`
    and are called explicitly at script-level after the backbone
    is loaded.
    """

    seq_length: int
    text_key: str
    target_dtype: str
    heads: tuple[HeadConfig, ...]
    epochs: int
    backbone_weights_path: str
    ratio_batch: RatioBatchConfig
    lr_schedule: LRScheduleConfig
    optimizer: OptimizerConfig
    diagnostics: DiagnosticsConfig = dataclasses.field(
        default_factory=DiagnosticsConfig
    )
    # Encoder-unfreeze escalation knobs (docs/notes/encoder-unfreeze-strategy.md).
    # Mirror UsRunConfig's fields/validation exactly (src/us_config.py) — RunConfig
    # was the one config still missing them (UsRunConfig already carried them).
    # Back-compat: an older sidecar lacking these keys reconstructs via
    # _filter_known_fields's default-fallback (each field has a default here),
    # landing on the frozen-probe defaults below — no manual payload.get() needed,
    # unlike UsRunConfig.from_json's hand-written back-compat (that module's
    # from_json predates _filter_known_fields's generic default-fallback).
    freeze_encoder: bool = True
    unfreeze_top_n: int = 0
    layer_multipliers: dict | None = None
    # Hard-freeze knob (docs/notes/branched-encoder-strategy.md "Hard
    # freezing is now a requirement"): when True (and unfreeze_top_n > 0),
    # the trainer sets `trainable=False` on the backbone sub-layers below
    # the unfrozen top-N block, replacing the drift-prone zero-multiplier
    # "freeze" (AdamW's decoupled weight decay still updates multiplier=0
    # variables every step -- encoder-unfreeze-strategy.md, 2026-07-29
    # finding). Back-compat default False: every sidecar written before this
    # knob existed was multiplier-frozen, not trainable=False -- False is
    # the historically-accurate reconstruction, not a "safe" default choice.
    hard_freeze: bool = False
    # Training-time random seed for keras.utils.set_random_seed. Independent
    # of the seed=200 polars .sample() split seed in src/data_setup/data.py,
    # which this field does NOT affect and must not change (see that
    # module). Default 200 matches the hardcode every trainer used before
    # this knob existed, so pre-existing sidecars back-compat-default
    # correctly.
    seed: int = 200

    def __post_init__(self):
        # --- Self-consistency: own fields ---------------------------------
        if not isinstance(self.seq_length, int) or self.seq_length <= 0:
            raise ValueError(
                f"RunConfig.seq_length must be a positive int; "
                f"got {self.seq_length!r}."
            )
        if not isinstance(self.text_key, str) or not self.text_key:
            raise ValueError(
                f"RunConfig.text_key must be a non-empty string; "
                f"got {self.text_key!r}."
            )
        # target_dtype: validated as a Keras-recognized dtype string,
        # mirroring the Piece 1 ClassifierPreprocessor.__init__ check.
        try:
            keras.backend.standardize_dtype(self.target_dtype)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"RunConfig.target_dtype must be a valid Keras dtype "
                f"string; got {self.target_dtype!r}. "
                f"Underlying error: {e}"
            ) from e
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError(
                f"RunConfig.epochs must be a positive int; "
                f"got {self.epochs!r}."
            )
        if (
            not isinstance(self.backbone_weights_path, str)
            or not self.backbone_weights_path
        ):
            raise ValueError(
                f"RunConfig.backbone_weights_path must be a non-empty "
                f"string; got {self.backbone_weights_path!r}."
            )
        # heads: must be a tuple (frozen dataclass + JSON tuple/list
        # ergonomics) of HeadConfig instances, non-empty.
        if not isinstance(self.heads, tuple):
            raise ValueError(
                f"RunConfig.heads must be a tuple of HeadConfig; "
                f"got {type(self.heads).__name__} (use a tuple even "
                f"for a single head)."
            )
        if len(self.heads) == 0:
            raise ValueError("RunConfig.heads must be non-empty.")
        for i, head in enumerate(self.heads):
            if not isinstance(head, HeadConfig):
                raise ValueError(
                    f"RunConfig.heads[{i}] must be a HeadConfig; "
                    f"got {type(head).__name__}."
                )

        # --- Cross-object: head names unique ------------------------------
        head_names = [h.name for h in self.heads]
        if len(set(head_names)) != len(head_names):
            duplicates = sorted(
                {n for n in head_names if head_names.count(n) > 1}
            )
            raise ValueError(
                f"RunConfig.heads contains duplicate head names "
                f"{duplicates}. Head names must be unique within a "
                f"RunConfig (used as call-site routing keys for "
                f"compile(loss={{...}}) and dict outputs)."
            )

        # --- Sub-config types (defense-in-depth) --------------------------
        # Sub-configs validate their own internal fields in their own
        # __post_init__. Here we just confirm the sub-config types are
        # right — catches an attempt to pass, e.g., a dict instead of
        # a RatioBatchConfig instance.
        if not isinstance(self.ratio_batch, RatioBatchConfig):
            raise ValueError(
                f"RunConfig.ratio_batch must be a RatioBatchConfig; "
                f"got {type(self.ratio_batch).__name__}."
            )
        if not isinstance(self.lr_schedule, LRScheduleConfig):
            raise ValueError(
                f"RunConfig.lr_schedule must be a LRScheduleConfig; "
                f"got {type(self.lr_schedule).__name__}."
            )
        if not isinstance(self.optimizer, OptimizerConfig):
            raise ValueError(
                f"RunConfig.optimizer must be a OptimizerConfig; "
                f"got {type(self.optimizer).__name__}."
            )
        if not isinstance(self.diagnostics, DiagnosticsConfig):
            raise ValueError(
                f"RunConfig.diagnostics must be a DiagnosticsConfig; "
                f"got {type(self.diagnostics).__name__}."
            )

        # --- Escalation knobs (mirrors UsRunConfig's validation exactly) ---
        if not isinstance(self.freeze_encoder, bool):
            raise ValueError(
                f"RunConfig.freeze_encoder must be a bool; "
                f"got {type(self.freeze_encoder).__name__}."
            )
        # unfreeze_top_n: RoBERTa-base has 12 layers; max unfreeze is top 12 (all layers)
        if not isinstance(self.unfreeze_top_n, int) or self.unfreeze_top_n < 0 or self.unfreeze_top_n > 12:
            raise ValueError(
                f"RunConfig.unfreeze_top_n must be in [0, 12] (RoBERTa-base has 12 layers); "
                f"got {self.unfreeze_top_n!r}."
            )
        if self.layer_multipliers is not None and not isinstance(self.layer_multipliers, dict):
            raise ValueError(
                f"RunConfig.layer_multipliers must be a dict or None; "
                f"got {type(self.layer_multipliers).__name__}."
            )
        if not isinstance(self.hard_freeze, bool):
            raise ValueError(
                f"RunConfig.hard_freeze must be a bool; "
                f"got {type(self.hard_freeze).__name__}."
            )
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError(
                f"RunConfig.seed must be a non-negative int; got {self.seed!r}."
            )

    # ----------------------------------------------------------------------
    # Derived properties — drive preprocessor / assembly construction
    # ----------------------------------------------------------------------

    @property
    def label_keys(self) -> dict[str, str]:
        """`label_keys` for `ClassifierPreprocessor`: maps endpoint-
        mode target Input names (`{head.name}_targets`) to source
        columns from the dataset."""
        return {f"{h.name}_targets": h.source_column for h in self.heads}

    @property
    def head_names(self) -> tuple[str, ...]:
        """The head names in declared order."""
        return tuple(h.name for h in self.heads)

    @property
    def expected_columns(self) -> set[str]:
        """The dataset columns the preprocessor expects to see —
        `text_key` plus every head's source column. Useful for
        asserting against a dataset's element_spec to catch
        config-vs-data mismatches at script-level (the Piece 1
        `__call__` check fires on this same invariant at trace
        time)."""
        return {self.text_key, *(h.source_column for h in self.heads)}

    # ----------------------------------------------------------------------
    # External-context validation
    # ----------------------------------------------------------------------

    def validate_against_backbone(self, backbone) -> None:
        """Verify the loaded backbone is compatible with this run
        config. Currently checks that every head's `hidden_dim`
        matches the backbone's `hidden_dim`.

        This is defense-in-depth on top of the Piece 2 shape-mismatch
        check (which fires at weight-load time): catching the
        mismatch *before* the load attempt produces a clearer error
        message, and lets us distinguish "wrong backbone for this
        config" from "weight file shape doesn't match anything."

        Raises:
            ValueError: if any `head.hidden_dim != backbone.hidden_dim`.
        """
        backbone_hidden = getattr(backbone, "hidden_dim", None)
        if backbone_hidden is None:
            raise ValueError(
                "Cannot validate against backbone: backbone has no "
                "`hidden_dim` attribute. validate_against_backbone "
                "expects a keras_hub Backbone or compatible object."
            )
        mismatches = [
            (h.name, h.hidden_dim, backbone_hidden)
            for h in self.heads
            if h.hidden_dim != backbone_hidden
        ]
        if mismatches:
            details = "; ".join(
                f"head {n!r} declares hidden_dim={hd}, "
                f"backbone has hidden_dim={bd}"
                for n, hd, bd in mismatches
            )
            raise ValueError(
                f"RunConfig hidden_dim mismatch with backbone: "
                f"{details}. The head's intermediate Dense layer "
                f"width should match the backbone's output hidden "
                f"dim; see `docs/notes/tier3-design.md` Piece 3 for "
                f"context."
            )

    # ----------------------------------------------------------------------
    # JSON serialization
    # ----------------------------------------------------------------------

    def to_json(self, path: Path | str) -> None:
        """Serialize this RunConfig to a JSON sidecar at `path`.

        Uses `dataclasses.asdict` for recursive conversion (nested
        dataclasses → dicts, tuples → lists). Writes with
        `indent=2` so sidecars are readable as documentation
        artifacts.
        """
        path = Path(path)
        payload = dataclasses.asdict(self)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> RunConfig:
        """Load a RunConfig from a JSON sidecar at `path`.

        Forward-compat behavior:
          - **Unknown top-level fields** (a future schema added a
            field this version doesn't know about): ignored with a
            warning. The config still loads.
          - **Missing required fields** (the sidecar lacks a field
            this version requires): fails loud with a clear error
            naming the missing field — schema regression rather
            than evolution, which we want to catch.
          - **Unknown fields in nested sub-configs**: same handling
            recursively.

        Type discrimination is structural for now (FLPULossConfig
        is the only loss type, etc.). When discriminated unions
        land — see "Open / deferred" in
        `docs/notes/tier3-design.md` Piece 3 — `from_json` will
        dispatch on the `type` field added at that point, with old
        sidecars defaulting to the existing type.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"RunConfig sidecar not found at {path}. Train a "
                f"model with the current training script (which "
                f"writes the sidecar alongside weights), or use "
                f"`python -m src.cca_config write_default <weights_path>` "
                f"to write DEFAULT_CCA_CONFIG as a sidecar for an "
                f"existing weights file."
            )
        with open(path) as f:
            payload = json.load(f)
        return cls._from_dict(payload, _source=str(path))

    @classmethod
    def _from_dict(cls, payload: dict, _source: str = "<dict>") -> RunConfig:
        """Reconstruct a RunConfig from its dict form (the
        deserialized JSON). Handles nested sub-config
        reconstruction, unknown-field warnings, and missing-field
        errors.

        `_source` is for error message context (e.g., the file
        path the dict came from).
        """
        # Reconstruct nested sub-configs first (bottom-up).
        if "heads" not in payload:
            raise ValueError(
                f"RunConfig sidecar at {_source} is missing required "
                f"field 'heads'."
            )
        heads = tuple(
            HeadConfig._from_dict(h, _source=f"{_source}.heads[{i}]")
            for i, h in enumerate(payload["heads"])
        )

        sub_configs = {}
        for sub_field, sub_cls in (
            ("ratio_batch", RatioBatchConfig),
            ("lr_schedule", LRScheduleConfig),
            ("optimizer", OptimizerConfig),
        ):
            if sub_field not in payload:
                raise ValueError(
                    f"RunConfig sidecar at {_source} is missing "
                    f"required field {sub_field!r}."
                )
            sub_configs[sub_field] = sub_cls._from_dict(
                payload[sub_field], _source=f"{_source}.{sub_field}"
            )

        # Filter top-level fields and warn on unknown.
        kwargs = _filter_known_fields(cls, payload, _source=_source)

        # Replace nested-dataclass slots with reconstructed instances.
        kwargs["heads"] = heads
        kwargs.update(sub_configs)

        # diagnostics is optional (back-compat): absent or null → let the
        # RunConfig default_factory produce DiagnosticsConfig(); present
        # dict → reconstruct. Distinct from the strict sub-config loop
        # above, which raises on missing keys.
        diag_payload = payload.get("diagnostics")
        if diag_payload is not None:
            kwargs["diagnostics"] = DiagnosticsConfig._from_dict(
                diag_payload, _source=f"{_source}.diagnostics"
            )
        else:
            kwargs.pop("diagnostics", None)  # ensure default_factory fires

        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Sub-config _from_dict methods
# ---------------------------------------------------------------------------
# Attached after the classes are defined to keep the dataclass
# definitions clean. Each method handles its own dataclass's
# unknown-field warnings + missing-required-field errors.


def _filter_known_fields(
    cls, payload: dict, _source: str = "<dict>"
) -> dict[str, Any]:
    """Filter a dict to only the fields the dataclass `cls` knows
    about. Warns on unknown fields (forward-compat: future schemas
    that added fields don't break this version) and raises on
    missing required fields (no default in the dataclass)."""
    known = {f.name: f for f in dataclasses.fields(cls)}
    payload_keys = set(payload.keys())
    unknown = payload_keys - set(known.keys())
    if unknown:
        warnings.warn(
            f"Unknown field(s) {sorted(unknown)} in {cls.__name__} "
            f"payload at {_source}; ignoring. (Forward-compat: this "
            f"may be a sidecar from a newer schema with fields "
            f"this version doesn't know about.)",
            stacklevel=3,
        )
    kwargs = {k: v for k, v in payload.items() if k in known}

    # Check for missing required fields (those without defaults).
    missing_required = [
        name
        for name, field in known.items()
        if name not in kwargs
        and field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    ]
    if missing_required:
        raise ValueError(
            f"{cls.__name__} payload at {_source} is missing "
            f"required field(s) {sorted(missing_required)}."
        )
    return kwargs


# ---------------------------------------------------------------------------
# Default canonical config
# ---------------------------------------------------------------------------
# The starting point for the current canonical CCA training run.
# Experimental variants derive from this via `dataclasses.replace`.
# When the immigration head and combined ICA head land, additional
# DEFAULT_*_CONFIG instances will live in this module alongside this
# one (or the module may grow into a `cca_configs/` package).


DEFAULT_CCA_CONFIG = RunConfig(
    seq_length=128,
    text_key="headline_with_lead",
    target_dtype="float32",
    heads=(
        HeadConfig(
            name="cca",
            source_column="cca_label",
            hidden_dim=768,  # roberta_base_en's hidden dim
            loss=FLPULossConfig(prior=0.02, kiryo_clawback=False),
        ),
    ),
    epochs=7,
    backbone_weights_path=str(config.DAPT_BACKBONE_WEIGHTS),
    ratio_batch=RatioBatchConfig(),  # 0.1 / 0.5 / 0.5 defaults
    lr_schedule=LRScheduleConfig(),
    optimizer=OptimizerConfig(),
)


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------


def config_path_for_weights(weights_path: Path | str) -> Path:
    """Derive the run config sidecar path from a weights file path.

    Convention: replace `.weights.h5` suffix with `.config.json`.
    For unusual paths that don't end in `.weights.h5`, append
    `.config.json` to the filename as a graceful fallback.

    Examples:
        >>> config_path_for_weights('cca_classifier/cca.weights.h5')
        PosixPath('cca_classifier/cca.config.json')
        >>> config_path_for_weights('foo/bar.h5')
        PosixPath('foo/bar.h5.config.json')
    """
    weights_path = Path(weights_path)
    name = weights_path.name
    suffix = ".weights.h5"
    if name.endswith(suffix):
        new_name = name[: -len(suffix)] + ".config.json"
    else:
        new_name = name + ".config.json"
    return weights_path.with_name(new_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# `python -m src.cca_config write_default <weights_path>` writes
# DEFAULT_CCA_CONFIG as a sidecar for the given weights file —
# useful for ad-hoc test setups, generating a sidecar for a weights
# file that predates the sidecar discipline, etc.
#
# `python -m src.cca_config show <config_path>` pretty-prints an
# existing sidecar's contents.


def _cli_write_default(weights_path: Path) -> None:
    sidecar = config_path_for_weights(weights_path)
    DEFAULT_CCA_CONFIG.to_json(sidecar)
    print(f"Wrote DEFAULT_CCA_CONFIG sidecar: {sidecar}")


def _cli_show(config_path: Path) -> None:
    cfg = RunConfig.from_json(config_path)
    payload = dataclasses.asdict(cfg)
    print(json.dumps(payload, indent=2))


def _cli_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.cca_config",
        description="CCA run config CLI helpers.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    write_parser = sub.add_parser(
        "write_default",
        help="Write DEFAULT_CCA_CONFIG as a sidecar for a weights file.",
    )
    write_parser.add_argument(
        "weights_path",
        type=Path,
        help="Path to the weights file (e.g., cca.weights.h5). "
        "Sidecar is written at the derived path "
        "(.weights.h5 -> .config.json).",
    )

    show_parser = sub.add_parser(
        "show",
        help="Pretty-print the contents of an existing sidecar.",
    )
    show_parser.add_argument(
        "config_path",
        type=Path,
        help="Path to a .config.json sidecar to display.",
    )

    args = parser.parse_args(argv)

    if args.cmd == "write_default":
        _cli_write_default(args.weights_path)
    elif args.cmd == "show":
        _cli_show(args.config_path)
    else:
        parser.error(f"unknown subcommand: {args.cmd}")
        return 2  # unreachable; argparse exits

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_cli_main())
