# pattern: Imperative Shell
# Reason: file I/O and tempfile creation

"""Unit and integration tests for src/fusion/sidecar.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.fusion.combiner import FusionConfig
from src.fusion.sidecar import (
    fusion_path_for_weights,
    load_fusion,
    save_fusion,
)


class TestFusionPathForWeights:
    """Test fusion_path_for_weights function."""

    def test_weights_h5_to_fusion_json(self):
        """*.weights.h5 → *.fusion.json."""
        p = fusion_path_for_weights("model.weights.h5")
        assert p == Path("model.fusion.json")

    def test_weights_h5_with_directory(self):
        """Preserves directory path."""
        p = fusion_path_for_weights("/tmp/model.weights.h5")
        assert p == Path("/tmp/model.fusion.json")

    def test_weights_h5_with_stem(self):
        """Handles complex stems."""
        p = fusion_path_for_weights("dir/model_v2_best.weights.h5")
        assert p == Path("dir/model_v2_best.fusion.json")

    def test_non_h5_fallback(self):
        """Non-.weights.h5 files get .fusion.json appended."""
        p = fusion_path_for_weights("model.h5")
        assert p == Path("model.h5.fusion.json")

    def test_pathlib_input(self):
        """Accepts Path objects."""
        p = fusion_path_for_weights(Path("model.weights.h5"))
        assert isinstance(p, Path)
        assert p == Path("model.fusion.json")

    def test_string_input(self):
        """Accepts strings."""
        p = fusion_path_for_weights("model.weights.h5")
        assert isinstance(p, Path)


class TestSaveLoadRoundTripProduct:
    """Test save_fusion / load_fusion round-trip for product config."""

    def test_save_product_creates_json_file(self):
        """save_fusion creates a JSON file for product config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = FusionConfig(
                gate_threshold=0.5,
                combine="product",
                coefs=None,
                score_space="prob",
                includes_us=False,
            )
            path = Path(tmpdir) / "fusion.fusion.json"
            save_fusion(cfg, path)
            assert path.exists()
            assert path.suffix == ".json"

    def test_save_product_json_contains_all_fields(self):
        """Saved JSON contains all FusionConfig fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = FusionConfig(
                gate_threshold=0.5,
                combine="product",
                coefs=None,
                score_space="prob",
                includes_us=False,
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg, path)
            payload = json.loads(path.read_text())
            assert "gate_threshold" in payload
            assert "combine" in payload
            assert "coefs" in payload
            assert "score_space" in payload
            assert "includes_us" in payload

    def test_save_product_json_values(self):
        """Saved JSON values match input config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = FusionConfig(
                gate_threshold=0.3,
                combine="product",
                coefs=None,
                score_space="logit",
                includes_us=False,
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg, path)
            payload = json.loads(path.read_text())
            assert payload["gate_threshold"] == 0.3
            assert payload["combine"] == "product"
            assert payload["coefs"] is None
            assert payload["score_space"] == "logit"
            assert payload["includes_us"] is False

    def test_load_product_from_json(self):
        """load_fusion reconstructs product config from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_orig = FusionConfig(
                gate_threshold=0.5,
                combine="product",
                coefs=None,
                score_space="prob",
                includes_us=False,
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg_orig, path)
            cfg_loaded = load_fusion(path)
            assert cfg_loaded.gate_threshold == cfg_orig.gate_threshold
            assert cfg_loaded.combine == cfg_orig.combine
            assert cfg_loaded.coefs == cfg_orig.coefs
            assert cfg_loaded.score_space == cfg_orig.score_space
            assert cfg_loaded.includes_us == cfg_orig.includes_us

    def test_roundtrip_product_equality(self):
        """Product config round-trip: save → load → identical."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_orig = FusionConfig(
                gate_threshold=0.7,
                combine="product",
                coefs=None,
                score_space="prob",
                includes_us=False,
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg_orig, path)
            cfg_loaded = load_fusion(path)
            # Frozen dataclass, so == works for equality
            assert cfg_loaded == cfg_orig


class TestSaveLoadRoundTripLogReg:
    """Test save_fusion / load_fusion round-trip for logreg config."""

    def test_save_logreg_3coefs_no_us_creates_json(self):
        """save_fusion creates JSON file for logreg config (3 coefs: 2 slopes + intercept, no US)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = FusionConfig(
                gate_threshold=0.5,
                combine="logreg",
                coefs=[0.5, 0.3, -0.1],  # 2 slopes + intercept
                score_space="prob",
                includes_us=False,
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg, path)
            assert path.exists()

    def test_save_logreg_4coefs_with_us_creates_json(self):
        """save_fusion creates JSON file for logreg config (4 coefs: 3 slopes + intercept, with US)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = FusionConfig(
                gate_threshold=0.6,
                combine="logreg",
                coefs=[0.5, 0.3, 0.2, -0.1],  # 3 slopes + intercept
                score_space="prob",
                includes_us=True,
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg, path)
            assert path.exists()

    def test_save_logreg_json_values_3coefs_no_us(self):
        """Saved JSON preserves logreg coefs (3-param: slopes+intercept, no US)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = FusionConfig(
                gate_threshold=0.5,
                combine="logreg",
                coefs=[0.5, 0.3, -0.1],
                score_space="prob",
                includes_us=False,
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg, path)
            payload = json.loads(path.read_text())
            assert payload["combine"] == "logreg"
            assert payload["coefs"] == [0.5, 0.3, -0.1]
            assert payload["includes_us"] is False

    def test_save_logreg_json_values_4coefs_with_us(self):
        """Saved JSON preserves logreg coefs (4-param: slopes+intercept, with US)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = FusionConfig(
                gate_threshold=0.5,
                combine="logreg",
                coefs=[0.5, 0.3, 0.2, -0.1],
                score_space="prob",
                includes_us=True,
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg, path)
            payload = json.loads(path.read_text())
            assert payload["combine"] == "logreg"
            assert payload["coefs"] == [0.5, 0.3, 0.2, -0.1]
            assert payload["includes_us"] is True

    def test_load_logreg_from_json_3coefs_no_us(self):
        """load_fusion reconstructs logreg config (3 coefs, no US)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_orig = FusionConfig(
                gate_threshold=0.5,
                combine="logreg",
                coefs=[0.5, 0.3, -0.1],
                score_space="prob",
                includes_us=False,
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg_orig, path)
            cfg_loaded = load_fusion(path)
            assert cfg_loaded.coefs == [0.5, 0.3, -0.1]
            assert cfg_loaded.includes_us is False

    def test_load_logreg_from_json_4coefs_with_us(self):
        """load_fusion reconstructs logreg config (4 coefs, with US)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_orig = FusionConfig(
                gate_threshold=0.6,
                combine="logreg",
                coefs=[0.5, 0.3, 0.2, -0.1],
                score_space="prob",
                includes_us=True,
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg_orig, path)
            cfg_loaded = load_fusion(path)
            assert cfg_loaded.coefs == [0.5, 0.3, 0.2, -0.1]
            assert cfg_loaded.includes_us is True

    def test_roundtrip_logreg_3coefs_no_us_equality(self):
        """Logreg config round-trip (3 coefs, no US): save → load → identical."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_orig = FusionConfig(
                gate_threshold=0.5,
                combine="logreg",
                coefs=[0.5, 0.3, -0.1],
                score_space="prob",
                includes_us=False,
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg_orig, path)
            cfg_loaded = load_fusion(path)
            assert cfg_loaded == cfg_orig

    def test_roundtrip_logreg_4coefs_with_us_equality(self):
        """Logreg config round-trip (4 coefs, with US): save → load → identical."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_orig = FusionConfig(
                gate_threshold=0.6,
                combine="logreg",
                coefs=[0.5, 0.3, 0.2, -0.1],
                score_space="prob",
                includes_us=True,
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg_orig, path)
            cfg_loaded = load_fusion(path)
            assert cfg_loaded == cfg_orig


class TestLoadFusionValidation:
    """Test load_fusion error handling."""

    def test_missing_gate_threshold_raises(self):
        """Missing gate_threshold field raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fusion.json"
            payload = {
                # Missing gate_threshold
                "combine": "product",
                "coefs": None,
                "score_space": "prob",
                "includes_us": False,
            }
            path.write_text(json.dumps(payload))
            with pytest.raises(ValueError, match="missing required fields"):
                load_fusion(path)

    def test_missing_combine_raises(self):
        """Missing combine field raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fusion.json"
            payload = {
                "gate_threshold": 0.5,
                # Missing combine
                "coefs": None,
                "score_space": "prob",
                "includes_us": False,
            }
            path.write_text(json.dumps(payload))
            with pytest.raises(ValueError, match="missing required fields"):
                load_fusion(path)

    def test_missing_coefs_raises(self):
        """Missing coefs field raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fusion.json"
            payload = {
                "gate_threshold": 0.5,
                "combine": "product",
                # Missing coefs
                "score_space": "prob",
                "includes_us": False,
            }
            path.write_text(json.dumps(payload))
            with pytest.raises(ValueError, match="missing required fields"):
                load_fusion(path)

    def test_missing_score_space_raises(self):
        """Missing score_space field raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fusion.json"
            payload = {
                "gate_threshold": 0.5,
                "combine": "product",
                "coefs": None,
                # Missing score_space
                "includes_us": False,
            }
            path.write_text(json.dumps(payload))
            with pytest.raises(ValueError, match="missing required fields"):
                load_fusion(path)

    def test_missing_includes_us_raises(self):
        """Missing includes_us field raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fusion.json"
            payload = {
                "gate_threshold": 0.5,
                "combine": "product",
                "coefs": None,
                "score_space": "prob",
                # Missing includes_us
            }
            path.write_text(json.dumps(payload))
            with pytest.raises(ValueError, match="missing required fields"):
                load_fusion(path)

    def test_invalid_combine_value_raises(self):
        """Invalid combine value raises ValueError (caught by FusionConfig validation)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fusion.json"
            payload = {
                "gate_threshold": 0.5,
                "combine": "invalid",
                "coefs": None,
                "score_space": "prob",
                "includes_us": False,
            }
            path.write_text(json.dumps(payload))
            with pytest.raises(ValueError, match="combine"):
                load_fusion(path)

    def test_invalid_gate_threshold_raises(self):
        """Invalid gate_threshold value raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fusion.json"
            payload = {
                "gate_threshold": 1.5,  # Outside [0, 1]
                "combine": "product",
                "coefs": None,
                "score_space": "prob",
                "includes_us": False,
            }
            path.write_text(json.dumps(payload))
            with pytest.raises(ValueError, match="gate_threshold"):
                load_fusion(path)

    def test_invalid_json_raises(self):
        """Invalid JSON raises json.JSONDecodeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fusion.json"
            path.write_text("not valid json {")
            with pytest.raises(Exception):  # json.JSONDecodeError
                load_fusion(path)

    def test_logreg_without_coefs_raises(self):
        """logreg config without coefs raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fusion.json"
            payload = {
                "gate_threshold": 0.5,
                "combine": "logreg",
                "coefs": None,  # Invalid for logreg
                "score_space": "prob",
                "includes_us": False,
            }
            path.write_text(json.dumps(payload))
            with pytest.raises(ValueError, match="coefs.*required"):
                load_fusion(path)

    def test_product_with_coefs_raises(self):
        """product config with non-None coefs raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fusion.json"
            payload = {
                "gate_threshold": 0.5,
                "combine": "product",
                "coefs": [0.5, 0.3],  # Invalid for product
                "score_space": "prob",
                "includes_us": False,
            }
            path.write_text(json.dumps(payload))
            with pytest.raises(ValueError, match="coefs.*None"):
                load_fusion(path)


class TestAC42FusionJsonPersistence:
    """AC4.2: FusionConfig persists to .fusion.json and reloads to identical state."""

    def test_ac42_product_complete_roundtrip(self):
        """Product config: save → load → identical."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_orig = FusionConfig(
                gate_threshold=0.5,
                combine="product",
                coefs=None,
                score_space="prob",
                includes_us=False,
            )
            path = Path(tmpdir) / "weights.weights.h5.fusion.json"
            save_fusion(cfg_orig, path)
            cfg_loaded = load_fusion(path)
            assert cfg_loaded == cfg_orig

    def test_ac42_logreg_complete_roundtrip(self):
        """Logreg config: save → load → identical."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_orig = FusionConfig(
                gate_threshold=0.6,
                combine="logreg",
                coefs=[0.5, 0.3, 0.2, -0.1],  # 3 slopes + intercept for includes_us=True
                score_space="prob",
                includes_us=True,
            )
            path = Path(tmpdir) / "weights.weights.h5.fusion.json"
            save_fusion(cfg_orig, path)
            cfg_loaded = load_fusion(path)
            assert cfg_loaded == cfg_orig

    def test_fusion_path_for_weights_integration(self):
        """fusion_path_for_weights integrates correctly with save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = FusionConfig(
                gate_threshold=0.5,
                combine="product",
                coefs=None,
                score_space="prob",
                includes_us=False,
            )
            weights_path = Path(tmpdir) / "model.weights.h5"
            fusion_path = fusion_path_for_weights(weights_path)
            assert fusion_path == Path(tmpdir) / "model.fusion.json"

            save_fusion(cfg, fusion_path)
            cfg_loaded = load_fusion(fusion_path)
            assert cfg_loaded == cfg


class TestComposedPlattAndHeadCalibrators:
    """Test persistence of composed_platt and head_calibrators (AC3.3, AC4.2)."""

    def test_save_with_composed_platt_and_head_calibrators(self):
        """save_fusion preserves composed_platt and head_calibrators in JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = FusionConfig(
                gate_threshold=0.5,
                combine="product",
                coefs=None,
                score_space="prob",
                includes_us=False,
                composed_platt=[0.8, -0.5],
                head_calibrators={"cca": "cca_stem", "rel": "rel_stem", "us": "us_stem"},
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg, path)
            payload = json.loads(path.read_text())
            assert "composed_platt" in payload
            assert "head_calibrators" in payload
            assert payload["composed_platt"] == [0.8, -0.5]
            assert payload["head_calibrators"] == {"cca": "cca_stem", "rel": "rel_stem", "us": "us_stem"}

    def test_load_with_composed_platt_and_head_calibrators(self):
        """load_fusion reconstructs composed_platt and head_calibrators."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_orig = FusionConfig(
                gate_threshold=0.5,
                combine="product",
                coefs=None,
                score_space="prob",
                includes_us=False,
                composed_platt=[0.8, -0.5],
                head_calibrators={"cca": "cca_stem", "rel": "rel_stem", "us": "us_stem"},
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg_orig, path)
            cfg_loaded = load_fusion(path)
            assert cfg_loaded.composed_platt == [0.8, -0.5]
            assert cfg_loaded.head_calibrators == {"cca": "cca_stem", "rel": "rel_stem", "us": "us_stem"}

    def test_roundtrip_with_all_new_fields(self):
        """Roundtrip save/load preserves all fields including new ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_orig = FusionConfig(
                gate_threshold=0.6,
                combine="logreg",
                coefs=[0.5, 0.3, -0.1],  # 2 slopes + intercept for includes_us=False
                score_space="prob",
                includes_us=False,
                composed_platt=[0.9, -0.2],
                head_calibrators={"cca": "cca_cal", "rel": "rel_cal", "us": "us_cal"},
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg_orig, path)
            cfg_loaded = load_fusion(path)
            assert cfg_loaded == cfg_orig

    def test_backward_compat_missing_new_fields(self):
        """Old .fusion.json without new fields loads with None (backward compat)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate an old .fusion.json without composed_platt and head_calibrators
            path = Path(tmpdir) / "fusion.json"
            old_payload = {
                "gate_threshold": 0.5,
                "combine": "product",
                "coefs": None,
                "score_space": "prob",
                "includes_us": False,
                # composed_platt and head_calibrators are missing
            }
            path.write_text(json.dumps(old_payload))
            cfg_loaded = load_fusion(path)
            assert cfg_loaded.composed_platt is None
            assert cfg_loaded.head_calibrators is None
            assert cfg_loaded.gate_threshold == 0.5
            assert cfg_loaded.combine == "product"


class TestHeadFeatureSources:
    """head_feature_sources: additive optional field for branched-encoder
    apply (docs/design-plans/2026-08-18-stage4-joint-finetune.md)."""

    def test_save_preserves_head_feature_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = FusionConfig(
                gate_threshold=0.5,
                combine="product",
                coefs=None,
                score_space="prob",
                includes_us=False,
                head_feature_sources={"us": "base", "cca": "base", "rel": "rel_branch"},
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg, path)
            payload = json.loads(path.read_text())
            assert payload["head_feature_sources"] == {
                "us": "base", "cca": "base", "rel": "rel_branch"
            }

    def test_load_reconstructs_head_feature_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_orig = FusionConfig(
                gate_threshold=0.5,
                combine="product",
                coefs=None,
                score_space="prob",
                includes_us=False,
                head_feature_sources={"us": "base", "cca": "base", "rel": "rel_branch"},
            )
            path = Path(tmpdir) / "fusion.json"
            save_fusion(cfg_orig, path)
            cfg_loaded = load_fusion(path)
            assert cfg_loaded.head_feature_sources == {
                "us": "base", "cca": "base", "rel": "rel_branch"
            }
            assert cfg_loaded == cfg_orig

    def test_backward_compat_missing_head_feature_sources_loads_none(self):
        """Old .fusion.json (no head_feature_sources key) loads with None —
        the 'fusion without the field = no check' back-compat contract."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fusion.json"
            old_payload = {
                "gate_threshold": 0.5,
                "combine": "product",
                "coefs": None,
                "score_space": "prob",
                "includes_us": False,
            }
            path.write_text(json.dumps(old_payload))
            cfg_loaded = load_fusion(path)
            assert cfg_loaded.head_feature_sources is None

    def test_default_head_feature_sources_is_none(self):
        cfg = FusionConfig(
            gate_threshold=0.5,
            combine="product",
            coefs=None,
            score_space="prob",
            includes_us=False,
        )
        assert cfg.head_feature_sources is None

    def test_invalid_head_feature_sources_type_raises(self):
        with pytest.raises(ValueError, match="head_feature_sources"):
            FusionConfig(
                gate_threshold=0.5,
                combine="product",
                coefs=None,
                score_space="prob",
                includes_us=False,
                head_feature_sources=["not", "a", "dict"],
            )
