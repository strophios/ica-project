"""
Tests for `src.cca_config` (Tier 3 Piece 3 — I4: train/eval config
coupling).

Test classes mirror the validation hierarchy: each dataclass's
self-consistency checks have their own TestConstructionValidation
section; cross-object invariants (head names unique) live with
RunConfig. Plus derived-property correctness, JSON round-trip,
forward-compat handling, backbone validation, and the path helper.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from src.cca_config import (
    FLPULossConfig,
    HeadConfig,
    RatioBatchConfig,
    LRScheduleConfig,
    OptimizerConfig,
    RunConfig,
    DEFAULT_CCA_CONFIG,
    config_path_for_weights,
    ResolvedSteps,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _valid_head(
    name="cca", source_column="cca_label", hidden_dim=768, prior=0.02, loss_weight=1.0
):
    """Build a valid HeadConfig for use in tests that need one."""
    return HeadConfig(
        name=name,
        source_column=source_column,
        hidden_dim=hidden_dim,
        loss=FLPULossConfig(prior=prior),
        loss_weight=loss_weight,
    )


def _valid_run_config(**overrides):
    """Build a valid RunConfig for use in tests, with optional
    field overrides via dataclasses.replace."""
    base = RunConfig(
        seq_length=128,
        text_key="text",
        target_dtype="float32",
        heads=(_valid_head(),),
        epochs=3,
        backbone_weights_path="/tmp/fake_backbone.weights.h5",
        ratio_batch=RatioBatchConfig(),
        lr_schedule=LRScheduleConfig(),
        optimizer=OptimizerConfig(),
    )
    return dataclasses.replace(base, **overrides) if overrides else base


class _FakeBackbone:
    """Minimal stand-in for a keras_hub Backbone — just exposes
    a `hidden_dim` attribute, which is all
    `validate_against_backbone` looks at."""

    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim


# ---------------------------------------------------------------------------
# TestConstructionValidation: per-dataclass self-consistency
# ---------------------------------------------------------------------------
# Each dataclass owns its own `__post_init__` validating its own
# fields. Tests are organized per-dataclass to mirror that hierarchy.


class TestFLPULossConfigValidation:
    def test_valid_construction(self):
        FLPULossConfig(prior=0.02)  # no exception
        FLPULossConfig(prior=0.5, kiryo_clawback=True)

    def test_prior_at_zero_rejected(self):
        with pytest.raises(ValueError, match="prior"):
            FLPULossConfig(prior=0.0)

    def test_prior_at_one_rejected(self):
        with pytest.raises(ValueError, match="prior"):
            FLPULossConfig(prior=1.0)

    def test_prior_negative_rejected(self):
        with pytest.raises(ValueError, match="prior"):
            FLPULossConfig(prior=-0.1)

    def test_prior_above_one_rejected(self):
        with pytest.raises(ValueError, match="prior"):
            FLPULossConfig(prior=1.5)

    def test_prior_non_numeric_rejected(self):
        with pytest.raises(ValueError, match="prior"):
            FLPULossConfig(prior="0.02")  # type: ignore[arg-type]

    def test_kiryo_clawback_non_bool_rejected(self):
        with pytest.raises(ValueError, match="kiryo_clawback"):
            FLPULossConfig(prior=0.02, kiryo_clawback="yes")  # type: ignore[arg-type]


class TestHeadConfigValidation:
    def test_valid_construction(self):
        _valid_head()  # no exception

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            HeadConfig(
                name="",
                source_column="cca_label",
                hidden_dim=768,
                loss=FLPULossConfig(prior=0.02),
            )

    def test_non_string_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            HeadConfig(
                name=42,  # type: ignore[arg-type]
                source_column="cca_label",
                hidden_dim=768,
                loss=FLPULossConfig(prior=0.02),
            )

    def test_empty_source_column_rejected(self):
        with pytest.raises(ValueError, match="source_column"):
            HeadConfig(
                name="cca",
                source_column="",
                hidden_dim=768,
                loss=FLPULossConfig(prior=0.02),
            )

    def test_zero_hidden_dim_rejected(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            HeadConfig(
                name="cca",
                source_column="cca_label",
                hidden_dim=0,
                loss=FLPULossConfig(prior=0.02),
            )

    def test_negative_hidden_dim_rejected(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            HeadConfig(
                name="cca",
                source_column="cca_label",
                hidden_dim=-1,
                loss=FLPULossConfig(prior=0.02),
            )

    def test_non_int_hidden_dim_rejected(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            HeadConfig(
                name="cca",
                source_column="cca_label",
                hidden_dim=128.0,  # type: ignore[arg-type]
                loss=FLPULossConfig(prior=0.02),
            )

    def test_non_flpu_loss_rejected(self):
        # Cross-validation: HeadConfig validates the loss is a
        # FLPULossConfig (currently the only supported type).
        # When ALUM lands this widens to a discriminated union.
        with pytest.raises(ValueError, match="loss"):
            HeadConfig(
                name="cca",
                source_column="cca_label",
                hidden_dim=768,
                loss={"prior": 0.02},  # type: ignore[arg-type]
            )

    def test_rejects_name_containing_slash(self):
        """Head names with '/' collide with Keras's variable-path
        separator used by _default_group_fn (assembly.py:54-65)."""
        with pytest.raises(ValueError, match="/"):
            HeadConfig(
                name="cca/v2",
                source_column="cca_label",
                hidden_dim=768,
                loss=FLPULossConfig(prior=0.02),
            )


class TestHeadConfigLossWeightValidation:
    """`HeadConfig.loss_weight` (the stage-4 joint-finetune lambda knob;
    docs/design-plans/2026-08-18-stage4-joint-finetune.md "Components" item
    1). Default 1.0 = the historical (pre-knob) behavior."""

    def test_default_loss_weight_is_one(self):
        assert _valid_head().loss_weight == 1.0

    def test_custom_loss_weight_accepted(self):
        assert _valid_head(loss_weight=0.25).loss_weight == 0.25

    def test_loss_weight_zero_rejected(self):
        with pytest.raises(ValueError, match="loss_weight"):
            _valid_head(loss_weight=0.0)

    def test_loss_weight_negative_rejected(self):
        with pytest.raises(ValueError, match="loss_weight"):
            _valid_head(loss_weight=-0.5)

    def test_loss_weight_bool_rejected(self):
        # bool is a subclass of int in Python; must be explicitly excluded
        # (mirrors RunConfig.seed's bool guard).
        with pytest.raises(ValueError, match="loss_weight"):
            _valid_head(loss_weight=True)

    def test_loss_weight_inf_rejected(self):
        with pytest.raises(ValueError, match="loss_weight"):
            _valid_head(loss_weight=float("inf"))

    def test_loss_weight_nan_rejected(self):
        with pytest.raises(ValueError, match="loss_weight"):
            _valid_head(loss_weight=float("nan"))

    def test_loss_weight_non_numeric_rejected(self):
        with pytest.raises(ValueError, match="loss_weight"):
            _valid_head(loss_weight="0.5")  # type: ignore[arg-type]


class TestRatioBatchConfigValidation:
    def test_valid_defaults(self):
        rb = RatioBatchConfig()
        assert rb.train_pos == 0.1
        assert rb.val_pos == 0.5
        assert rb.test_pos == 0.5

    def test_valid_explicit(self):
        RatioBatchConfig(train_pos=0.05, val_pos=0.5, test_pos=0.5)

    def test_train_pos_at_zero_rejected(self):
        with pytest.raises(ValueError, match="train_pos"):
            RatioBatchConfig(train_pos=0.0)

    def test_train_pos_at_one_rejected(self):
        with pytest.raises(ValueError, match="train_pos"):
            RatioBatchConfig(train_pos=1.0)

    def test_val_pos_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="val_pos"):
            RatioBatchConfig(val_pos=1.5)


class TestLRScheduleConfigValidation:
    def test_valid_defaults(self):
        lr = LRScheduleConfig()
        assert lr.initial_lr == 1e-4
        assert lr.warmup_target == 1e-3

    def test_initial_lr_zero_rejected(self):
        with pytest.raises(ValueError, match="initial_lr"):
            LRScheduleConfig(initial_lr=0.0)

    def test_initial_lr_negative_rejected(self):
        with pytest.raises(ValueError, match="initial_lr"):
            LRScheduleConfig(initial_lr=-1e-5)

    def test_warmup_target_zero_rejected(self):
        with pytest.raises(ValueError, match="warmup_target"):
            LRScheduleConfig(warmup_target=0.0)

    def test_decay_alpha_negative_rejected(self):
        with pytest.raises(ValueError, match="decay_alpha"):
            LRScheduleConfig(decay_alpha=-0.1)

    def test_decay_alpha_zero_accepted(self):
        # decay_alpha is allowed to be zero (final LR = 0)
        LRScheduleConfig(decay_alpha=0.0)


class TestOptimizerConfigValidation:
    def test_valid_default(self):
        OptimizerConfig()  # weight_decay=5e-3

    def test_weight_decay_zero_accepted(self):
        OptimizerConfig(weight_decay=0.0)

    def test_weight_decay_negative_rejected(self):
        with pytest.raises(ValueError, match="weight_decay"):
            OptimizerConfig(weight_decay=-0.001)

    def test_weight_decay_non_numeric_rejected(self):
        with pytest.raises(ValueError, match="weight_decay"):
            OptimizerConfig(weight_decay="5e-3")  # type: ignore[arg-type]


class TestRunConfigValidation:
    def test_valid_construction(self):
        _valid_run_config()  # no exception

    def test_seq_length_zero_rejected(self):
        with pytest.raises(ValueError, match="seq_length"):
            _valid_run_config(seq_length=0)

    def test_seq_length_negative_rejected(self):
        with pytest.raises(ValueError, match="seq_length"):
            _valid_run_config(seq_length=-1)

    def test_empty_text_key_rejected(self):
        with pytest.raises(ValueError, match="text_key"):
            _valid_run_config(text_key="")

    def test_invalid_target_dtype_rejected(self):
        with pytest.raises(ValueError, match="target_dtype"):
            _valid_run_config(target_dtype="flot32")

    def test_negative_epochs_rejected(self):
        with pytest.raises(ValueError, match="epochs"):
            _valid_run_config(epochs=-1)

    def test_zero_epochs_rejected(self):
        with pytest.raises(ValueError, match="epochs"):
            _valid_run_config(epochs=0)

    def test_empty_backbone_path_rejected(self):
        with pytest.raises(ValueError, match="backbone_weights_path"):
            _valid_run_config(backbone_weights_path="")

    def test_empty_heads_rejected(self):
        with pytest.raises(ValueError, match="heads"):
            _valid_run_config(heads=())

    def test_heads_as_list_rejected(self):
        # heads must be a tuple, not a list (frozen dataclass +
        # JSON tuple/list ergonomics — _from_dict converts list to
        # tuple, but direct construction must use tuple).
        with pytest.raises(ValueError, match="heads"):
            _valid_run_config(heads=[_valid_head()])  # type: ignore[arg-type]

    def test_duplicate_head_names_rejected(self):
        # Cross-object invariant: head names must be unique within
        # a RunConfig (used as routing keys for compile loss / dict
        # outputs). Each individual HeadConfig is valid; only the
        # combination is rejected.
        with pytest.raises(ValueError, match="duplicate"):
            _valid_run_config(
                heads=(
                    _valid_head(name="cca", source_column="a"),
                    _valid_head(name="cca", source_column="b"),
                )
            )

    def test_multi_head_distinct_names_accepted(self):
        # Sanity check that the duplicate-name check doesn't reject
        # legitimate multi-head configurations.
        _valid_run_config(
            heads=(
                _valid_head(name="cca", source_column="cca_label"),
                _valid_head(name="immig", source_column="immig_label"),
            )
        )


# ---------------------------------------------------------------------------
# TestDerivedProperties
# ---------------------------------------------------------------------------


class TestDerivedProperties:
    def test_label_keys_single_head(self):
        cfg = _valid_run_config()
        assert cfg.label_keys == {"cca_targets": "cca_label"}

    def test_label_keys_multi_head(self):
        cfg = _valid_run_config(
            heads=(
                _valid_head(name="cca", source_column="cca_label"),
                _valid_head(name="immig", source_column="immig_label"),
            )
        )
        assert cfg.label_keys == {
            "cca_targets": "cca_label",
            "immig_targets": "immig_label",
        }

    def test_head_names(self):
        cfg = _valid_run_config(
            heads=(
                _valid_head(name="cca"),
                _valid_head(name="immig", source_column="immig_label"),
            )
        )
        assert cfg.head_names == ("cca", "immig")

    def test_expected_columns_includes_text_key_and_sources(self):
        cfg = _valid_run_config(
            text_key="headline_with_lead",
            heads=(
                _valid_head(name="cca", source_column="cca_label"),
                _valid_head(name="immig", source_column="immig_label"),
            ),
        )
        assert cfg.expected_columns == {
            "headline_with_lead",
            "cca_label",
            "immig_label",
        }


# ---------------------------------------------------------------------------
# TestJSONRoundTrip
# ---------------------------------------------------------------------------


class TestJSONRoundTrip:
    def test_default_config_round_trips(self, tmp_path):
        path = tmp_path / "default.config.json"
        DEFAULT_CCA_CONFIG.to_json(path)
        loaded = RunConfig.from_json(path)
        assert loaded == DEFAULT_CCA_CONFIG

    def test_variant_round_trips(self, tmp_path):
        # An experimental variant: different prior + epochs.
        original = _valid_run_config(
            epochs=10,
            heads=(
                _valid_head(prior=0.05),
            ),
        )
        path = tmp_path / "variant.config.json"
        original.to_json(path)
        loaded = RunConfig.from_json(path)
        assert loaded == original

    def test_multi_head_round_trips(self, tmp_path):
        original = _valid_run_config(
            heads=(
                _valid_head(name="cca", source_column="cca_label"),
                _valid_head(
                    name="immig", source_column="immig_label", prior=0.01
                ),
            )
        )
        path = tmp_path / "multi.config.json"
        original.to_json(path)
        loaded = RunConfig.from_json(path)
        assert loaded == original

    def test_json_is_human_readable(self, tmp_path):
        # The sidecar is a documentation artifact too — confirm it's
        # indented and the field names are recognizable.
        path = tmp_path / "default.config.json"
        DEFAULT_CCA_CONFIG.to_json(path)
        text = path.read_text()
        assert '"seq_length": 128' in text
        assert '"text_key": "headline_with_lead"' in text
        assert '"prior":' in text  # nested under heads[0].loss
        # Indented (default to_json uses indent=2)
        assert "\n  " in text


# ---------------------------------------------------------------------------
# TestJSONForwardCompat
# ---------------------------------------------------------------------------


class TestJSONForwardCompat:
    def test_unknown_top_level_field_warned_and_ignored(self, tmp_path):
        # Simulate a sidecar from a future schema with an extra
        # field this version doesn't know about.
        payload = dataclasses.asdict(_valid_run_config())
        payload["future_field"] = "future_value"
        path = tmp_path / "future.config.json"
        with open(path, "w") as f:
            json.dump(payload, f)

        with pytest.warns(UserWarning, match="future_field"):
            loaded = RunConfig.from_json(path)
        # Loaded config equals the version without the future
        # field — forward-compat: ignore unknown, don't fail.
        assert loaded == _valid_run_config()

    def test_unknown_nested_field_warned_and_ignored(self, tmp_path):
        payload = dataclasses.asdict(_valid_run_config())
        payload["heads"][0]["loss"]["future_loss_field"] = 42
        path = tmp_path / "future_nested.config.json"
        with open(path, "w") as f:
            json.dump(payload, f)

        with pytest.warns(UserWarning, match="future_loss_field"):
            loaded = RunConfig.from_json(path)
        assert loaded == _valid_run_config()

    def test_missing_required_field_fails_loud(self, tmp_path):
        payload = dataclasses.asdict(_valid_run_config())
        del payload["seq_length"]
        path = tmp_path / "missing.config.json"
        with open(path, "w") as f:
            json.dump(payload, f)

        # Failing loud on missing required fields is the right
        # behavior — silent defaults would hide schema regressions.
        with pytest.raises(ValueError, match="seq_length"):
            RunConfig.from_json(path)

    def test_missing_nested_required_field_fails_loud(self, tmp_path):
        payload = dataclasses.asdict(_valid_run_config())
        del payload["heads"][0]["loss"]["prior"]
        path = tmp_path / "missing_nested.config.json"
        with open(path, "w") as f:
            json.dump(payload, f)

        with pytest.raises(ValueError, match="prior"):
            RunConfig.from_json(path)

    def test_missing_sidecar_file_fails_loud(self, tmp_path):
        # The eval script's load path: if no sidecar exists at the
        # derived path, fail with a clear error (rather than
        # silently using DEFAULT_CCA_CONFIG, which would mask the
        # drift the sidecar exists to prevent).
        path = tmp_path / "nonexistent.config.json"
        with pytest.raises(FileNotFoundError, match="sidecar not found"):
            RunConfig.from_json(path)


# ---------------------------------------------------------------------------
# TestBackboneValidation
# ---------------------------------------------------------------------------


class TestBackboneValidation:
    def test_matching_hidden_dim_passes(self):
        cfg = _valid_run_config(
            heads=(_valid_head(hidden_dim=768),)
        )
        backbone = _FakeBackbone(hidden_dim=768)
        cfg.validate_against_backbone(backbone)  # no exception

    def test_mismatched_hidden_dim_raises(self):
        cfg = _valid_run_config(
            heads=(_valid_head(hidden_dim=768),)
        )
        backbone = _FakeBackbone(hidden_dim=1024)
        with pytest.raises(ValueError, match="hidden_dim"):
            cfg.validate_against_backbone(backbone)

    def test_multi_head_one_mismatch_raises(self):
        cfg = _valid_run_config(
            heads=(
                _valid_head(name="cca", source_column="cca_label", hidden_dim=768),
                _valid_head(
                    name="immig", source_column="immig_label", hidden_dim=512
                ),
            )
        )
        backbone = _FakeBackbone(hidden_dim=768)
        # First head matches, second doesn't — should still raise.
        with pytest.raises(ValueError, match="immig"):
            cfg.validate_against_backbone(backbone)

    def test_backbone_without_hidden_dim_attribute_raises(self):
        cfg = _valid_run_config()

        class BadBackbone:
            pass

        with pytest.raises(ValueError, match="hidden_dim"):
            cfg.validate_against_backbone(BadBackbone())


# ---------------------------------------------------------------------------
# TestPathHelper
# ---------------------------------------------------------------------------


class TestPathHelper:
    def test_weights_h5_path_substituted(self):
        result = config_path_for_weights("a/b/cca.weights.h5")
        assert result == Path("a/b/cca.config.json")

    def test_path_object_input_accepted(self):
        result = config_path_for_weights(Path("a/b/cca.weights.h5"))
        assert result == Path("a/b/cca.config.json")

    def test_unusual_path_appended(self):
        # Graceful fallback for paths that don't end in
        # `.weights.h5`: just append `.config.json`.
        result = config_path_for_weights("foo/bar.h5")
        assert result == Path("foo/bar.h5.config.json")

    def test_no_extension_appended(self):
        result = config_path_for_weights("foo/bar")
        assert result == Path("foo/bar.config.json")

    def test_preserves_directory(self):
        result = config_path_for_weights(
            "/abs/path/to/cca_classifier/cca.weights.h5"
        )
        assert result == Path(
            "/abs/path/to/cca_classifier/cca.config.json"
        )


# ---------------------------------------------------------------------------
# TestResolvedSteps (Tier 4 Piece 2 — I4)
# ---------------------------------------------------------------------------


class TestResolvedSteps:
    """ResolvedSteps captures the resolved LR schedule step counts
    that LRScheduleConfig factors are multiplied against at
    training time. See docs/notes/tier4-design.md Piece 2."""

    def test_construction_with_valid_positive_ints(self):
        rs = ResolvedSteps(
            warmup_steps=1250,
            decay_steps=15000,
            steps_per_epoch=5000,
        )
        assert rs.warmup_steps == 1250
        assert rs.decay_steps == 15000
        assert rs.steps_per_epoch == 5000

    def test_is_frozen(self):
        rs = ResolvedSteps(warmup_steps=1, decay_steps=1, steps_per_epoch=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            rs.warmup_steps = 2  # type: ignore[misc]

    @pytest.mark.parametrize("field,value", [
        ("warmup_steps", 0),
        ("warmup_steps", -1),
        ("decay_steps", 0),
        ("decay_steps", -1),
        ("steps_per_epoch", 0),
        ("steps_per_epoch", -1),
    ])
    def test_rejects_non_positive(self, field, value):
        kwargs = {"warmup_steps": 1, "decay_steps": 1, "steps_per_epoch": 1}
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            ResolvedSteps(**kwargs)

    @pytest.mark.parametrize("field,value", [
        ("warmup_steps", 1.5),
        ("warmup_steps", "1"),
        ("decay_steps", 1.5),
        ("steps_per_epoch", 1.5),
    ])
    def test_rejects_non_int(self, field, value):
        kwargs = {"warmup_steps": 1, "decay_steps": 1, "steps_per_epoch": 1}
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            ResolvedSteps(**kwargs)

    def test_from_dict_round_trips(self):
        original = ResolvedSteps(
            warmup_steps=1250, decay_steps=15000, steps_per_epoch=5000
        )
        payload = dataclasses.asdict(original)
        reconstructed = ResolvedSteps._from_dict(payload)
        assert reconstructed == original


# ---------------------------------------------------------------------------
# TestLRScheduleConfigResolvedField (Tier 4 Piece 2 — I4)
# ---------------------------------------------------------------------------


class TestLRScheduleConfigResolvedField:
    """LRScheduleConfig.resolved holds optional resolved step counts.
    See docs/notes/tier4-design.md Piece 2 for the rationale."""

    def test_default_is_none(self):
        cfg = LRScheduleConfig()
        assert cfg.resolved is None

    def test_accepts_resolved_steps_instance(self):
        rs = ResolvedSteps(
            warmup_steps=1, decay_steps=1, steps_per_epoch=1
        )
        cfg = LRScheduleConfig(resolved=rs)
        assert cfg.resolved is rs

    def test_rejects_non_resolved_steps_non_none(self):
        with pytest.raises(ValueError, match="resolved"):
            LRScheduleConfig(resolved={"warmup_steps": 1})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestLRScheduleConfigWithResolved (Tier 4 Piece 2 — I4)
# ---------------------------------------------------------------------------


class TestLRScheduleConfigWithResolved:
    """LRScheduleConfig.with_resolved populates the resolved field
    via math.floor(factor * steps_per_epoch). See
    docs/notes/tier4-design.md Piece 2."""

    def test_populates_resolved_from_factors(self):
        cfg = LRScheduleConfig(
            warmup_steps_factor=0.25, decay_steps_factor=3.0
        )
        resolved_cfg = cfg.with_resolved(steps_per_epoch=5000)

        assert resolved_cfg.resolved is not None
        assert resolved_cfg.resolved.warmup_steps == 1250  # floor(0.25 * 5000)
        assert resolved_cfg.resolved.decay_steps == 15000  # floor(3.0 * 5000)
        assert resolved_cfg.resolved.steps_per_epoch == 5000

    def test_uses_floor_for_non_integer_results(self):
        cfg = LRScheduleConfig(
            warmup_steps_factor=0.25, decay_steps_factor=3.0
        )
        # 2287 * 0.25 = 571.75 → 571
        resolved_cfg = cfg.with_resolved(steps_per_epoch=2287)

        assert resolved_cfg.resolved is not None
        assert resolved_cfg.resolved.warmup_steps == 571

    def test_returns_new_instance_not_mutation(self):
        cfg = LRScheduleConfig()
        assert cfg.resolved is None
        resolved_cfg = cfg.with_resolved(steps_per_epoch=100)

        # Original unchanged
        assert cfg.resolved is None
        # Returned is a different instance
        assert resolved_cfg is not cfg
        assert resolved_cfg.resolved is not None

    def test_preserves_other_fields(self):
        cfg = LRScheduleConfig(
            initial_lr=2e-4,
            warmup_target=2e-3,
            decay_alpha=5e-2,
            warmup_steps_factor=0.5,
            decay_steps_factor=4.0,
        )
        resolved_cfg = cfg.with_resolved(steps_per_epoch=1000)

        assert resolved_cfg.initial_lr == 2e-4
        assert resolved_cfg.warmup_target == 2e-3
        assert resolved_cfg.decay_alpha == 5e-2
        assert resolved_cfg.warmup_steps_factor == 0.5
        assert resolved_cfg.decay_steps_factor == 4.0

    def test_rejects_non_positive_steps_per_epoch(self):
        cfg = LRScheduleConfig()
        with pytest.raises(ValueError, match="steps_per_epoch"):
            cfg.with_resolved(steps_per_epoch=0)
        with pytest.raises(ValueError, match="steps_per_epoch"):
            cfg.with_resolved(steps_per_epoch=-1)


# ---------------------------------------------------------------------------
# TestLRScheduleConfigJSONRoundTripWithResolved (Tier 4 Piece 2 — I4)
# ---------------------------------------------------------------------------


class TestLRScheduleConfigJSONRoundTripWithResolved:
    """JSON round-trip preserves the resolved field when populated.
    See docs/notes/tier4-design.md Piece 2."""

    def test_round_trips_with_resolved_populated(self, tmp_path):
        # Construct a RunConfig with a resolved LRScheduleConfig
        original_run_config = _valid_run_config()
        original_run_config = dataclasses.replace(
            original_run_config,
            lr_schedule=original_run_config.lr_schedule.with_resolved(
                steps_per_epoch=5000
            ),
        )

        # Round-trip via JSON sidecar
        sidecar = tmp_path / "config.json"
        original_run_config.to_json(sidecar)
        reconstructed = RunConfig.from_json(sidecar)

        assert reconstructed.lr_schedule.resolved is not None
        assert reconstructed.lr_schedule.resolved.warmup_steps == 1250
        assert reconstructed.lr_schedule.resolved.decay_steps == 15000
        assert reconstructed.lr_schedule.resolved.steps_per_epoch == 5000
        # Equality check across the whole config
        assert reconstructed == original_run_config

    def test_round_trips_with_resolved_none(self, tmp_path):
        # Default (resolved=None) round-trip
        original_run_config = _valid_run_config()
        assert original_run_config.lr_schedule.resolved is None

        sidecar = tmp_path / "config.json"
        original_run_config.to_json(sidecar)
        reconstructed = RunConfig.from_json(sidecar)

        assert reconstructed.lr_schedule.resolved is None
        assert reconstructed == original_run_config

    def test_loads_pre_resolved_sidecar_without_key(self, tmp_path):
        """Backward compat: older sidecars written before Piece 2
        have no 'resolved' key in the lr_schedule dict."""
        # Construct an old-shape sidecar manually
        run_config = _valid_run_config()
        payload = dataclasses.asdict(run_config)
        del payload["lr_schedule"]["resolved"]

        sidecar = tmp_path / "old_config.json"
        with open(sidecar, "w") as f:
            json.dump(payload, f)

        reconstructed = RunConfig.from_json(sidecar)
        assert reconstructed.lr_schedule.resolved is None


class TestDiagnosticsConfigValidation:
    def test_default_constructs(self):
        from src.cca_config import DiagnosticsConfig

        c = DiagnosticsConfig()
        assert c.enable_gradient_norms is True
        assert c.enable_prediction_distribution is True
        assert c.gradient_norm_aggregations == ("max", "mean")
        assert c.prediction_summary_stats == ("mean", "std", "frac_above_0.5")

    def test_non_bool_enable_raises(self):
        from src.cca_config import DiagnosticsConfig

        with pytest.raises(ValueError, match="enable_gradient_norms"):
            DiagnosticsConfig(enable_gradient_norms=1)

    def test_empty_gradient_aggregations_raises(self):
        from src.cca_config import DiagnosticsConfig

        with pytest.raises(ValueError, match="gradient_norm_aggregations"):
            DiagnosticsConfig(gradient_norm_aggregations=())

    def test_invalid_gradient_aggregation_value_raises(self):
        from src.cca_config import DiagnosticsConfig

        with pytest.raises(ValueError, match="gradient_norm_aggregations"):
            DiagnosticsConfig(gradient_norm_aggregations=("median",))

    def test_non_tuple_gradient_aggregations_raises(self):
        from src.cca_config import DiagnosticsConfig

        with pytest.raises(ValueError, match="gradient_norm_aggregations"):
            DiagnosticsConfig(gradient_norm_aggregations=["max"])

    def test_invalid_summary_stat_raises(self):
        from src.cca_config import DiagnosticsConfig

        with pytest.raises(ValueError, match="prediction_summary_stats"):
            DiagnosticsConfig(prediction_summary_stats=("variance",))

    def test_non_tuple_summary_stats_raises(self):
        from src.cca_config import DiagnosticsConfig

        with pytest.raises(ValueError, match="prediction_summary_stats"):
            DiagnosticsConfig(prediction_summary_stats=["mean"])

    def test_enable_prediction_distribution_false_ok(self):
        from src.cca_config import DiagnosticsConfig

        assert DiagnosticsConfig(enable_prediction_distribution=False).enable_prediction_distribution is False


class TestDiagnosticsAggregationConstantSync:
    def test_valid_aggregations_in_sync_with_trackers(self):
        # Drift guard: cca_config deliberately duplicates the
        # ("max","mean") literal rather than importing from
        # diagnostics.trackers (preserves the "config does not import
        # the machinery it configures" invariant). This test recovers
        # drift-prevention at the test boundary.
        from src.cca_config import _VALID_GRADIENT_AGGREGATIONS
        from src.diagnostics.trackers import _VALID_AGGREGATIONS

        assert _VALID_GRADIENT_AGGREGATIONS == _VALID_AGGREGATIONS


class TestDiagnosticsConfigFromDict:
    def test_roundtrip_default(self):
        from src.cca_config import DiagnosticsConfig

        original = DiagnosticsConfig()
        payload = json.loads(json.dumps(dataclasses.asdict(original)))
        reconstructed = DiagnosticsConfig._from_dict(payload)
        assert reconstructed == original

    def test_roundtrip_non_default(self):
        from src.cca_config import DiagnosticsConfig

        original = DiagnosticsConfig(
            enable_overflow_proxy=False,
            enable_prediction_distribution=False,
            gradient_norm_aggregations=("mean",),
            prediction_summary_stats=("mean", "std"),
        )
        payload = json.loads(json.dumps(dataclasses.asdict(original)))
        reconstructed = DiagnosticsConfig._from_dict(payload)
        assert reconstructed == original

    def test_list_payload_coerced_to_tuple(self):
        from src.cca_config import DiagnosticsConfig

        # JSON arrays deserialize as lists; _from_dict must coerce.
        payload = {
            "gradient_norm_aggregations": ["max", "mean"],
            "prediction_summary_stats": ["mean", "std", "frac_above_0.5"],
        }
        c = DiagnosticsConfig._from_dict(payload)
        assert isinstance(c.gradient_norm_aggregations, tuple)
        assert isinstance(c.prediction_summary_stats, tuple)
        assert c.gradient_norm_aggregations == ("max", "mean")

    def test_unknown_key_warns_and_ignored(self):
        from src.cca_config import DiagnosticsConfig

        with pytest.warns(UserWarning, match="Unknown field"):
            c = DiagnosticsConfig._from_dict({"enable_gradient_norms": False, "bogus": 1})
        assert c.enable_gradient_norms is False

    def test_missing_keys_use_defaults(self):
        from src.cca_config import DiagnosticsConfig

        # All fields defaulted → empty payload reconstructs the default.
        assert DiagnosticsConfig._from_dict({}) == DiagnosticsConfig()

    def test_from_dict_does_not_bypass_validation_on_coerced_gradient_norms(self):
        from src.cca_config import DiagnosticsConfig

        # Coerced list payload with invalid entry must still fail validation.
        payload = {"gradient_norm_aggregations": ["median"]}
        with pytest.raises(ValueError, match="gradient_norm_aggregations"):
            DiagnosticsConfig._from_dict(payload)

    def test_from_dict_does_not_bypass_validation_on_coerced_prediction_stats(self):
        from src.cca_config import DiagnosticsConfig

        # Coerced list payload with invalid entry must still fail validation.
        payload = {"prediction_summary_stats": ["variance"]}
        with pytest.raises(ValueError, match="prediction_summary_stats"):
            DiagnosticsConfig._from_dict(payload)


class TestEscalationKnobs:
    """Encoder-unfreeze escalation knobs (freeze_encoder / unfreeze_top_n /
    layer_multipliers) on RunConfig — mirrors UsRunConfig's TestEscalationKnobs
    in tests/test_us_config.py."""

    def test_default_escalation_knobs(self):
        cfg = _valid_run_config()
        assert cfg.freeze_encoder is True
        assert cfg.unfreeze_top_n == 0
        assert cfg.layer_multipliers is None

    def test_freeze_encoder_false(self):
        cfg = _valid_run_config(freeze_encoder=False)
        assert cfg.freeze_encoder is False

    def test_unfreeze_top_n_positive(self):
        cfg = _valid_run_config(unfreeze_top_n=2)
        assert cfg.unfreeze_top_n == 2

    def test_unfreeze_top_n_at_upper_bound_accepted(self):
        cfg = _valid_run_config(unfreeze_top_n=12)
        assert cfg.unfreeze_top_n == 12

    def test_layer_multipliers_dict(self):
        multipliers = {"head": 1.0, "encoder_top": 0.1, "encoder_frozen": 0.0}
        cfg = _valid_run_config(layer_multipliers=multipliers)
        assert cfg.layer_multipliers == multipliers

    def test_unfreeze_top_n_negative_rejected(self):
        with pytest.raises(ValueError, match="unfreeze_top_n"):
            _valid_run_config(unfreeze_top_n=-1)

    def test_unfreeze_top_n_above_12_rejected(self):
        with pytest.raises(ValueError, match="unfreeze_top_n"):
            _valid_run_config(unfreeze_top_n=13)

    def test_layer_multipliers_non_dict_rejected(self):
        with pytest.raises(ValueError, match="layer_multipliers"):
            _valid_run_config(layer_multipliers="not a dict")

    def test_freeze_encoder_non_bool_rejected(self):
        with pytest.raises(ValueError, match="freeze_encoder"):
            _valid_run_config(freeze_encoder="yes")


class TestEscalationKnobsRoundTrip:
    """JSON round-trip for the escalation knobs, incl. back-compat for older
    sidecars written before these fields existed."""

    def test_round_trip_with_escalation_knobs(self, tmp_path):
        cfg = _valid_run_config(
            freeze_encoder=False,
            unfreeze_top_n=2,
            layer_multipliers={"head": 1.0, "encoder_top": 0.1},
        )
        path = tmp_path / "config.json"
        cfg.to_json(path)
        reloaded = RunConfig.from_json(path)
        assert reloaded == cfg
        assert reloaded.freeze_encoder is False
        assert reloaded.unfreeze_top_n == 2
        assert reloaded.layer_multipliers == {"head": 1.0, "encoder_top": 0.1}

    def test_back_compat_old_sidecar_missing_fields(self, tmp_path):
        """Old sidecars missing escalation knobs should load with the
        frozen-probe defaults (freeze_encoder=True, unfreeze_top_n=0,
        layer_multipliers=None)."""
        cfg = _valid_run_config()
        path = tmp_path / "config.json"
        cfg.to_json(path)

        payload = json.loads(path.read_text())
        del payload["freeze_encoder"]
        del payload["unfreeze_top_n"]
        del payload["layer_multipliers"]
        path.write_text(json.dumps(payload))

        reloaded = RunConfig.from_json(path)
        assert reloaded.freeze_encoder is True
        assert reloaded.unfreeze_top_n == 0
        assert reloaded.layer_multipliers is None
        assert reloaded.seq_length == cfg.seq_length
        assert reloaded.heads == cfg.heads

    def test_default_cca_config_has_frozen_probe_defaults(self):
        assert DEFAULT_CCA_CONFIG.freeze_encoder is True
        assert DEFAULT_CCA_CONFIG.unfreeze_top_n == 0
        assert DEFAULT_CCA_CONFIG.layer_multipliers is None


class TestHardFreezeKnob:
    """`hard_freeze` (Capability 1, docs/notes/branched-encoder-strategy.md):
    trainable=False sub-layers below the unfrozen top-N block, replacing the
    drift-prone zero-multiplier freeze. Mirrors TestEscalationKnobs above."""

    def test_default_is_false(self):
        cfg = _valid_run_config()
        assert cfg.hard_freeze is False

    def test_can_set_true(self):
        cfg = _valid_run_config(hard_freeze=True)
        assert cfg.hard_freeze is True

    def test_non_bool_rejected(self):
        with pytest.raises(ValueError, match="hard_freeze"):
            _valid_run_config(hard_freeze="yes")

    def test_default_cca_config_has_hard_freeze_false(self):
        # Back-compat: all historical artifacts were multiplier-frozen, not
        # trainable=False -- False is the historically-accurate default.
        assert DEFAULT_CCA_CONFIG.hard_freeze is False


class TestSeedKnob:
    """`seed` (training-time keras.utils.set_random_seed knob). Independent
    of and must not affect the seed=200 polars .sample() split seed in
    src/data_setup/data.py."""

    def test_default_is_200(self):
        cfg = _valid_run_config()
        assert cfg.seed == 200

    def test_can_set_custom_seed(self):
        cfg = _valid_run_config(seed=7)
        assert cfg.seed == 7

    def test_zero_seed_accepted(self):
        cfg = _valid_run_config(seed=0)
        assert cfg.seed == 0

    def test_non_int_rejected(self):
        with pytest.raises(ValueError, match="seed"):
            _valid_run_config(seed=1.5)

    def test_bool_rejected(self):
        # bool is a subclass of int in Python; must be explicitly excluded.
        with pytest.raises(ValueError, match="seed"):
            _valid_run_config(seed=True)

    def test_negative_seed_rejected(self):
        with pytest.raises(ValueError, match="seed"):
            _valid_run_config(seed=-1)

    def test_default_cca_config_has_seed_200(self):
        assert DEFAULT_CCA_CONFIG.seed == 200


class TestHardFreezeAndSeedRoundTrip:
    """JSON round-trip + back-compat for hard_freeze/seed, mirroring
    TestEscalationKnobsRoundTrip above."""

    def test_round_trip_with_hard_freeze_and_seed(self, tmp_path):
        cfg = _valid_run_config(hard_freeze=True, seed=7)
        path = tmp_path / "config.json"
        cfg.to_json(path)
        reloaded = RunConfig.from_json(path)
        assert reloaded == cfg
        assert reloaded.hard_freeze is True
        assert reloaded.seed == 7

    def test_back_compat_old_sidecar_missing_hard_freeze_and_seed(self, tmp_path):
        """Old sidecars missing hard_freeze/seed should load with the
        historically-accurate defaults (hard_freeze=False, seed=200) --
        mirrors test_back_compat_old_sidecar_missing_fields above."""
        cfg = _valid_run_config()
        path = tmp_path / "config.json"
        cfg.to_json(path)

        payload = json.loads(path.read_text())
        del payload["hard_freeze"]
        del payload["seed"]
        path.write_text(json.dumps(payload))

        reloaded = RunConfig.from_json(path)
        assert reloaded.hard_freeze is False
        assert reloaded.seed == 200
        assert reloaded.seq_length == cfg.seq_length
        assert reloaded.heads == cfg.heads


class TestHeadConfigLossWeightRoundTrip:
    """JSON round-trip + back-compat for HeadConfig.loss_weight, mirroring
    TestHardFreezeAndSeedRoundTrip above (nested-field variant: loss_weight
    lives on HeadConfig, reconstructed via HeadConfig._from_dict's generic
    _filter_known_fields default-fallback — no hand-written back-compat
    needed since the field carries a dataclass default)."""

    def test_round_trip_with_custom_loss_weight(self, tmp_path):
        cfg = _valid_run_config(heads=(_valid_head(loss_weight=0.3),))
        path = tmp_path / "config.json"
        cfg.to_json(path)
        reloaded = RunConfig.from_json(path)
        assert reloaded == cfg
        assert reloaded.heads[0].loss_weight == 0.3

    def test_back_compat_old_sidecar_missing_loss_weight(self, tmp_path):
        """A sidecar written before this knob existed lacks
        heads[*].loss_weight entirely; reloading must default to 1.0 (the
        historically-accurate value — every pre-knob run behaved as
        loss_weight=1.0)."""
        cfg = _valid_run_config()
        path = tmp_path / "config.json"
        cfg.to_json(path)

        payload = json.loads(path.read_text())
        del payload["heads"][0]["loss_weight"]
        path.write_text(json.dumps(payload))

        reloaded = RunConfig.from_json(path)
        assert reloaded.heads[0].loss_weight == 1.0
        assert reloaded.seq_length == cfg.seq_length

    def test_default_cca_config_has_loss_weight_one(self):
        assert DEFAULT_CCA_CONFIG.heads[0].loss_weight == 1.0


class TestRunConfigDiagnosticsIntegration:
    def test_default_factory_fires_when_constructed_without_diagnostics(self):
        from src.cca_config import DEFAULT_CCA_CONFIG, DiagnosticsConfig

        # DEFAULT_CCA_CONFIG does not pass diagnostics → default fires.
        assert DEFAULT_CCA_CONFIG.diagnostics == DiagnosticsConfig()

    def test_back_compat_sidecar_missing_diagnostics_key(self, tmp_path):
        from src.cca_config import DEFAULT_CCA_CONFIG, DiagnosticsConfig, RunConfig

        payload = dataclasses.asdict(DEFAULT_CCA_CONFIG)
        payload.pop("diagnostics", None)  # simulate pre-Tier-5 sidecar
        sidecar = tmp_path / "old.config.json"
        with open(sidecar, "w") as f:
            json.dump(payload, f)
        reconstructed = RunConfig.from_json(sidecar)
        assert reconstructed.diagnostics == DiagnosticsConfig()

    def test_present_diagnostics_key_roundtrips(self, tmp_path):
        from src.cca_config import DEFAULT_CCA_CONFIG, DiagnosticsConfig, RunConfig

        custom = dataclasses.replace(
            DEFAULT_CCA_CONFIG,
            diagnostics=DiagnosticsConfig(enable_overflow_proxy=False,
                                          gradient_norm_aggregations=("mean",)),
        )
        sidecar = tmp_path / "new.config.json"
        custom.to_json(sidecar)
        reconstructed = RunConfig.from_json(sidecar)
        assert reconstructed.diagnostics == custom.diagnostics

    def test_type_defense_rejects_non_diagnosticsconfig(self):
        from src.cca_config import DEFAULT_CCA_CONFIG

        with pytest.raises(ValueError, match="diagnostics"):
            dataclasses.replace(DEFAULT_CCA_CONFIG, diagnostics={"enable_gradient_norms": True})

    def test_null_diagnostics_in_payload_uses_default(self, tmp_path):
        from src.cca_config import DEFAULT_CCA_CONFIG, DiagnosticsConfig, RunConfig

        payload = dataclasses.asdict(DEFAULT_CCA_CONFIG)
        payload["diagnostics"] = None  # explicit null
        sidecar = tmp_path / "null.config.json"
        with open(sidecar, "w") as f:
            json.dump(payload, f)
        reconstructed = RunConfig.from_json(sidecar)
        assert reconstructed.diagnostics == DiagnosticsConfig()
