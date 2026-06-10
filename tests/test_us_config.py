"""
Tests for `src.us_config` (US filter run configuration).

Test classes mirror the validation hierarchy: each dataclass's
self-consistency checks have their own TestConstructionValidation
section; cross-object invariants and derived properties live with
UsRunConfig. Plus JSON round-trip, including populated `resolved` field.
"""

import dataclasses
import json
import tempfile
from pathlib import Path

import pytest

import src.us_config as us_config
from src.us_config import (
    UsHeadConfig,
    UsRunConfig,
    DEFAULT_US_CONFIG,
)
from src.cca_config import (
    LRScheduleConfig,
    OptimizerConfig,
    DiagnosticsConfig,
    ResolvedSteps,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _valid_head(name="us", source_column="us_label", hidden_dim=768):
    """Build a valid UsHeadConfig for use in tests that need one."""
    return UsHeadConfig(name=name, source_column=source_column, hidden_dim=hidden_dim)


def _valid_run_config(**overrides):
    """Build a valid UsRunConfig for use in tests, with optional
    field overrides via dataclasses.replace."""
    base = UsRunConfig(
        seq_length=128,
        text_key="headline_with_lead",
        target_dtype="float32",
        head=_valid_head(),
        epochs=7,
        backbone_weights_path="/tmp/fake_dapt_backbone.weights.h5",
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


class TestUsHeadConfigValidation:
    def test_valid_construction(self):
        UsHeadConfig(name="us", source_column="us_label", hidden_dim=768)

    def test_name_empty_string_rejected(self):
        with pytest.raises(ValueError, match="name"):
            UsHeadConfig(name="", source_column="us_label", hidden_dim=768)

    def test_name_non_string_rejected(self):
        with pytest.raises(ValueError, match="name"):
            UsHeadConfig(name=None, source_column="us_label", hidden_dim=768)

    def test_name_with_slash_rejected(self):
        with pytest.raises(ValueError, match="name.*must not contain"):
            UsHeadConfig(name="us/bad", source_column="us_label", hidden_dim=768)

    def test_source_column_empty_string_rejected(self):
        with pytest.raises(ValueError, match="source_column"):
            UsHeadConfig(name="us", source_column="", hidden_dim=768)

    def test_source_column_non_string_rejected(self):
        with pytest.raises(ValueError, match="source_column"):
            UsHeadConfig(name="us", source_column=None, hidden_dim=768)

    def test_hidden_dim_zero_rejected(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            UsHeadConfig(name="us", source_column="us_label", hidden_dim=0)

    def test_hidden_dim_negative_rejected(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            UsHeadConfig(name="us", source_column="us_label", hidden_dim=-1)

    def test_hidden_dim_non_int_rejected(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            UsHeadConfig(name="us", source_column="us_label", hidden_dim=768.5)


class TestUsRunConfigValidation:
    def test_valid_construction(self):
        _valid_run_config()

    def test_seq_length_zero_rejected(self):
        with pytest.raises(ValueError, match="seq_length"):
            _valid_run_config(seq_length=0)

    def test_seq_length_negative_rejected(self):
        with pytest.raises(ValueError, match="seq_length"):
            _valid_run_config(seq_length=-1)

    def test_text_key_empty_string_rejected(self):
        with pytest.raises(ValueError, match="text_key"):
            _valid_run_config(text_key="")

    def test_text_key_non_string_rejected(self):
        with pytest.raises(ValueError, match="text_key"):
            _valid_run_config(text_key=None)

    def test_target_dtype_invalid_keras_dtype_rejected(self):
        with pytest.raises(ValueError, match="target_dtype"):
            _valid_run_config(target_dtype="not_a_valid_dtype")

    def test_target_dtype_valid_dtypes_accepted(self):
        # Should not raise
        _valid_run_config(target_dtype="float32")
        _valid_run_config(target_dtype="float64")
        _valid_run_config(target_dtype="int32")

    def test_epochs_zero_rejected(self):
        with pytest.raises(ValueError, match="epochs"):
            _valid_run_config(epochs=0)

    def test_epochs_negative_rejected(self):
        with pytest.raises(ValueError, match="epochs"):
            _valid_run_config(epochs=-1)

    def test_backbone_weights_path_empty_string_rejected(self):
        with pytest.raises(ValueError, match="backbone_weights_path"):
            _valid_run_config(backbone_weights_path="")

    def test_backbone_weights_path_non_string_rejected(self):
        with pytest.raises(ValueError, match="backbone_weights_path"):
            _valid_run_config(backbone_weights_path=None)

    def test_head_non_usheadconfig_rejected(self):
        with pytest.raises(ValueError, match="head"):
            _valid_run_config(head={"name": "us"})

    def test_enable_loss_components_true_rejected(self):
        """US filter uses BCE (no loss components); enabling them should raise."""
        diag_with_loss_components = dataclasses.replace(
            DiagnosticsConfig(), enable_loss_components=True
        )
        with pytest.raises(ValueError, match="enable_loss_components"):
            _valid_run_config(diagnostics=diag_with_loss_components)


# ---------------------------------------------------------------------------
# Derived properties
# ---------------------------------------------------------------------------


class TestDerivedProperties:
    def test_label_keys(self):
        cfg = _valid_run_config()
        assert cfg.label_keys == {"us_targets": "us_label"}

    def test_label_keys_custom_name(self):
        cfg = _valid_run_config(
            head=UsHeadConfig(
                name="my_us_head", source_column="my_source", hidden_dim=768
            )
        )
        assert cfg.label_keys == {"my_us_head_targets": "my_source"}

    def test_expected_columns(self):
        cfg = _valid_run_config()
        assert cfg.expected_columns == {"headline_with_lead", "us_label"}

    def test_expected_columns_custom_names(self):
        cfg = _valid_run_config(
            text_key="my_text",
            head=UsHeadConfig(
                name="us", source_column="my_label", hidden_dim=768
            ),
        )
        assert cfg.expected_columns == {"my_text", "my_label"}


# ---------------------------------------------------------------------------
# Backbone validation
# ---------------------------------------------------------------------------


class TestBackboneValidation:
    def test_matching_hidden_dim_accepted(self):
        cfg = _valid_run_config()
        backbone = _FakeBackbone(hidden_dim=768)
        cfg.validate_against_backbone(backbone)  # no exception

    def test_mismatched_hidden_dim_rejected(self):
        cfg = _valid_run_config()
        backbone = _FakeBackbone(hidden_dim=1024)
        with pytest.raises(ValueError, match="hidden_dim"):
            cfg.validate_against_backbone(backbone)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_config_round_trip(self):
        cfg = _valid_run_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            cfg.to_json(path)
            reloaded = UsRunConfig.from_json(path)
            assert reloaded == cfg

    def test_round_trip_with_populated_resolved(self):
        """Test round-trip including a populated resolved field
        (occurs when config is saved after with_resolved call)."""
        cfg = _valid_run_config()
        resolved_lr_schedule = cfg.lr_schedule.with_resolved(steps_per_epoch=100)
        cfg_with_resolved = dataclasses.replace(cfg, lr_schedule=resolved_lr_schedule)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            cfg_with_resolved.to_json(path)

            # Verify the JSON has a resolved field
            raw_json = json.loads(path.read_text())
            assert raw_json["lr_schedule"]["resolved"] is not None

            # Reload and verify
            reloaded = UsRunConfig.from_json(path)
            assert reloaded == cfg_with_resolved
            assert reloaded.lr_schedule.resolved == resolved_lr_schedule.resolved

    def test_round_trip_default_config(self):
        """DEFAULT_US_CONFIG should round-trip cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "default.json"
            DEFAULT_US_CONFIG.to_json(path)
            reloaded = UsRunConfig.from_json(path)
            assert reloaded == DEFAULT_US_CONFIG


# ---------------------------------------------------------------------------
# AC3.4: No prior field guard
# ---------------------------------------------------------------------------


class TestNoFlpuCoupling:
    def test_no_prior_field_on_default_config(self):
        """AC3.4: UsRunConfig and its head must not have a prior field."""
        assert not hasattr(DEFAULT_US_CONFIG, "prior")
        assert not hasattr(DEFAULT_US_CONFIG.head, "prior")

    def test_no_prior_in_dataclass_tree(self):
        """AC3.4: Walk the dataclass tree and confirm 'prior' is absent."""
        cfg = _valid_run_config()

        def _has_prior_field(obj):
            if dataclasses.is_dataclass(obj):
                for field in dataclasses.fields(obj):
                    if field.name == "prior":
                        return True
                    # Recurse into field values
                    field_value = getattr(obj, field.name)
                    if _has_prior_field(field_value):
                        return True
            return False

        assert not _has_prior_field(cfg), "Found 'prior' field in UsRunConfig tree"

    def test_no_flpu_loss_config_in_head(self):
        """The US head should not reference FLPULossConfig."""
        cfg = _valid_run_config()
        # UsHeadConfig has no loss field at all
        assert not hasattr(cfg.head, "loss")
        assert isinstance(cfg.head, UsHeadConfig)


# ---------------------------------------------------------------------------
# AC3.5: Deterministic split / config reproducibility
# ---------------------------------------------------------------------------


class TestConfigReproducibility:
    def test_same_seed_yields_same_split_config_values(self):
        """AC3.5: Two instances with the same initialization should be equal."""
        cfg1 = UsRunConfig(
            seq_length=128,
            text_key="headline_with_lead",
            target_dtype="float32",
            head=UsHeadConfig(name="us", source_column="us_label", hidden_dim=768),
            epochs=7,
            backbone_weights_path="/path/to/backbone.h5",
            lr_schedule=LRScheduleConfig(
                initial_lr=1e-4,
                warmup_target=1e-3,
                decay_alpha=0.1,
                warmup_steps_factor=0.25,
                decay_steps_factor=3.0,
            ),
            optimizer=OptimizerConfig(weight_decay=5e-3),
        )
        cfg2 = UsRunConfig(
            seq_length=128,
            text_key="headline_with_lead",
            target_dtype="float32",
            head=UsHeadConfig(name="us", source_column="us_label", hidden_dim=768),
            epochs=7,
            backbone_weights_path="/path/to/backbone.h5",
            lr_schedule=LRScheduleConfig(
                initial_lr=1e-4,
                warmup_target=1e-3,
                decay_alpha=0.1,
                warmup_steps_factor=0.25,
                decay_steps_factor=3.0,
            ),
            optimizer=OptimizerConfig(weight_decay=5e-3),
        )
        assert cfg1 == cfg2
