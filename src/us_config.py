# pattern: Mixed (pure config dataclasses; to_json/from_json are the sidecar I/O seam)
"""US/not-US run configuration.

Parallel to cca_config.RunConfig but with NO FLPU/prior/nnPU coupling: the US
filter is plain supervised PN with BCE. This is a deliberate "separate for now"
config that REUSES every shared sub-config (LRScheduleConfig, OptimizerConfig,
DiagnosticsConfig, ResolvedSteps) and MIRRORS RunConfig's property surface
(label_keys, expected_columns, validate_against_backbone, to_json/from_json).
Convergence path: when the multi-head config is built, CCA + US unify via a
loss-type discriminated union on the head config; this module merges in mechanically.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import keras

import src.config as config
from src.cca_config import (
    LRScheduleConfig,
    OptimizerConfig,
    DiagnosticsConfig,
)

# Re-exported so UsRunConfig's module surface mirrors RunConfig's (consumers
# import the sidecar-path helper from the same module as the config they use).
from src.cca_config import config_path_for_weights as config_path_for_weights


@dataclasses.dataclass(frozen=True)
class UsHeadConfig:
    """Configuration for the US binary classification head.

    Simple head config with no loss specification (loss is BCE, defined
    at head construction time). Mirrors HeadConfig structure for eventual
    unification.
    """

    name: str
    source_column: str
    hidden_dim: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("UsHeadConfig.name must be a non-empty string")
        if "/" in self.name:
            raise ValueError("UsHeadConfig.name must not contain '/'")
        if not isinstance(self.source_column, str) or not self.source_column:
            raise ValueError("UsHeadConfig.source_column must be a non-empty string")
        if not isinstance(self.hidden_dim, int) or self.hidden_dim <= 0:
            raise ValueError("UsHeadConfig.hidden_dim must be a positive int")


@dataclasses.dataclass(frozen=True)
class UsRunConfig:
    """US filter run configuration (BCE, no FLPU/prior coupling).

    Frozen dataclass capturing the architectural and research-dimension
    parameters of a US filter training run. Intended to be serialized to
    a JSON sidecar alongside weights for deterministic eval-time model
    reconstruction.

    Fields:
    - seq_length: RoBERTa tokenizer input length
    - text_key: source DataFrame column for text input
    - target_dtype: Keras dtype for target tensors (validated)
    - head: UsHeadConfig with name, source_column, hidden_dim
    - epochs: training epochs
    - backbone_weights_path: path to DAPT checkpoint
    - lr_schedule: LRScheduleConfig with warmup/decay parameters
    - optimizer: OptimizerConfig (AdamW weight_decay)
    - diagnostics: DiagnosticsConfig (enable_loss_components must be False)
    - freeze_encoder: bool; if False, enables per-layer LR scaling (default True)
    - unfreeze_top_n: int; number of top RoBERTa layers to unfreeze for
      fine-tuning. Used with freeze_encoder=False. Validated >= 0 (default 0).
    - layer_multipliers: dict | None; custom per-group LR multipliers for
      LayerLRModel. When unfreeze_top_n > 0, passed to build_endpoint_model
      with sensible defaults if None. Validated as dict when not None (default None).
    """

    seq_length: int
    text_key: str
    target_dtype: str
    head: UsHeadConfig
    epochs: int
    backbone_weights_path: str
    lr_schedule: LRScheduleConfig
    optimizer: OptimizerConfig
    diagnostics: DiagnosticsConfig = dataclasses.field(
        default_factory=lambda: DiagnosticsConfig(enable_loss_components=False)
    )
    freeze_encoder: bool = True
    unfreeze_top_n: int = 0
    layer_multipliers: dict | None = None

    def __post_init__(self):
        if not isinstance(self.seq_length, int) or self.seq_length <= 0:
            raise ValueError(
                f"UsRunConfig.seq_length must be a positive int; "
                f"got {self.seq_length!r}."
            )
        if not isinstance(self.text_key, str) or not self.text_key:
            raise ValueError(
                f"UsRunConfig.text_key must be a non-empty string; "
                f"got {self.text_key!r}."
            )
        # target_dtype: validated as a Keras-recognized dtype string,
        # mirroring RunConfig's check.
        try:
            keras.backend.standardize_dtype(self.target_dtype)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"UsRunConfig.target_dtype must be a valid Keras dtype "
                f"string; got {self.target_dtype!r}. "
                f"Underlying error: {e}"
            ) from e
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError(
                f"UsRunConfig.epochs must be a positive int; "
                f"got {self.epochs!r}."
            )
        if not isinstance(self.backbone_weights_path, str) or not self.backbone_weights_path:
            raise ValueError(
                f"UsRunConfig.backbone_weights_path must be a non-empty "
                f"string; got {self.backbone_weights_path!r}."
            )
        if not isinstance(self.head, UsHeadConfig):
            raise ValueError(
                f"UsRunConfig.head must be a UsHeadConfig; "
                f"got {type(self.head).__name__}."
            )
        # BCE has no loss components; loss-component diagnostics must be disabled.
        if self.diagnostics.enable_loss_components:
            raise ValueError(
                "US filter uses BCE (no loss components); set "
                "DiagnosticsConfig.enable_loss_components=False"
            )
        # Sub-config types (defense-in-depth)
        if not isinstance(self.lr_schedule, LRScheduleConfig):
            raise ValueError(
                f"UsRunConfig.lr_schedule must be a LRScheduleConfig; "
                f"got {type(self.lr_schedule).__name__}."
            )
        if not isinstance(self.optimizer, OptimizerConfig):
            raise ValueError(
                f"UsRunConfig.optimizer must be a OptimizerConfig; "
                f"got {type(self.optimizer).__name__}."
            )
        if not isinstance(self.diagnostics, DiagnosticsConfig):
            raise ValueError(
                f"UsRunConfig.diagnostics must be a DiagnosticsConfig; "
                f"got {type(self.diagnostics).__name__}."
            )
        # Escalation knobs validation
        if not isinstance(self.freeze_encoder, bool):
            raise ValueError(
                f"UsRunConfig.freeze_encoder must be a bool; "
                f"got {type(self.freeze_encoder).__name__}."
            )
        if not isinstance(self.unfreeze_top_n, int) or self.unfreeze_top_n < 0:
            raise ValueError(
                f"UsRunConfig.unfreeze_top_n must be a non-negative int; "
                f"got {self.unfreeze_top_n!r}."
            )
        if self.layer_multipliers is not None and not isinstance(self.layer_multipliers, dict):
            raise ValueError(
                f"UsRunConfig.layer_multipliers must be a dict or None; "
                f"got {type(self.layer_multipliers).__name__}."
            )

    @property
    def label_keys(self) -> dict[str, str]:
        """`label_keys` for `ClassifierPreprocessor`: maps endpoint-
        mode target Input name to source column from the dataset."""
        return {f"{self.head.name}_targets": self.head.source_column}

    @property
    def expected_columns(self) -> set[str]:
        """Expected columns in the input dataset: text_key + head source_column."""
        return {self.text_key, self.head.source_column}

    def validate_against_backbone(self, backbone) -> None:
        """Validate that the head's hidden_dim matches the backbone."""
        if self.head.hidden_dim != backbone.hidden_dim:
            raise ValueError(
                f"UsRunConfig head hidden_dim {self.head.hidden_dim} != "
                f"backbone.hidden_dim {backbone.hidden_dim}"
            )

    def to_json(self, path: Path | str) -> None:
        """Serialize config to JSON sidecar."""
        payload = dataclasses.asdict(self)
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: Path | str) -> "UsRunConfig":
        """Deserialize config from JSON sidecar.

        Delegates sub-config reconstruction to cca_config's existing
        _from_dict classmethods: LRScheduleConfig._from_dict rebuilds the
        nested ResolvedSteps, OptimizerConfig._from_dict, and
        DiagnosticsConfig._from_dict coerces the JSON-array tuple fields
        back to tuples. This is robust to the populated `resolved` field
        (present in a sidecar written after with_resolved) and to future
        sub-config field additions.

        Back-compat: older sidecars (pre-escalation knobs) are missing
        freeze_encoder, unfreeze_top_n, and layer_multipliers fields.
        from_json defaults these to True, 0, and None respectively.
        """
        payload = json.loads(Path(path).read_text())
        return cls(
            seq_length=payload["seq_length"],
            text_key=payload["text_key"],
            target_dtype=payload["target_dtype"],
            head=UsHeadConfig(**payload["head"]),
            epochs=payload["epochs"],
            backbone_weights_path=payload["backbone_weights_path"],
            lr_schedule=LRScheduleConfig._from_dict(payload["lr_schedule"]),
            optimizer=OptimizerConfig._from_dict(payload["optimizer"]),
            diagnostics=DiagnosticsConfig._from_dict(payload["diagnostics"]),
            freeze_encoder=payload.get("freeze_encoder", True),
            unfreeze_top_n=payload.get("unfreeze_top_n", 0),
            layer_multipliers=payload.get("layer_multipliers", None),
        )


DEFAULT_US_CONFIG = UsRunConfig(
    seq_length=128,
    text_key="headline_with_lead",
    target_dtype="float32",
    head=UsHeadConfig(name="us", source_column="us_label", hidden_dim=768),
    epochs=7,
    backbone_weights_path=str(config.DAPT_BACKBONE_WEIGHTS),
    lr_schedule=LRScheduleConfig(),
    optimizer=OptimizerConfig(),
)
