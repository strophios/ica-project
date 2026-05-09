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
import warnings
from pathlib import Path

import pytest

import src.cca_config as cca_config
from src.cca_config import (
    FLPULossConfig,
    HeadConfig,
    RatioBatchConfig,
    LRScheduleConfig,
    OptimizerConfig,
    RunConfig,
    DEFAULT_CCA_CONFIG,
    config_path_for_weights,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _valid_head(name="cca", source_column="cca_label", hidden_dim=768, prior=0.02):
    """Build a valid HeadConfig for use in tests that need one."""
    return HeadConfig(
        name=name,
        source_column=source_column,
        hidden_dim=hidden_dim,
        loss=FLPULossConfig(prior=prior),
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
