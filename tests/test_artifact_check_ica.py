"""Tests for reload_and_score_ica cross-process ICA artifact reload proof.

Verifies that:
  - reload_and_score_ica output matches IcaModel.predict_ica_from_features within tolerance
  - on a fresh construction from on-disk artifacts (weights + configs + calibrations + fusion)
  - proving the artifact triple is sufficient for cross-process reproduction (bitwise on
    frozen-feature path)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import keras  # noqa: F401  (initializes backend)

from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_feature_inference_model
from src.calibration.calibrator import PlattCalibrator, platt_fit
from src.calibration.sidecar import save_calibration, calibration_path_for_weights
from src.fusion.sidecar import save_fusion
from src.fusion.combiner import FusionConfig
from src.cca_config import (
    RunConfig,
    config_path_for_weights,
    LRScheduleConfig,
    OptimizerConfig,
    HeadConfig,
    FLPULossConfig,
    RatioBatchConfig,
)
from src.us_config import UsRunConfig, UsHeadConfig
from src.validation.artifact_check import reload_and_score_ica
from src.assemble_ica import IcaModel


HIDDEN_DIM = 768
BATCH = 20  # Larger batch for better statistical coverage


# =============================================================================
# Fixtures: Synthetic three-head artifact set with configs + calibrations + fusion
# =============================================================================


@pytest.fixture
def tiny_ica_artifact_set():
    """Create three tiny heads (us, cca, rel) with weights, configs, calibrations, fusion.

    Yields:
        dict with keys:
          - 'us_weights', 'cca_weights', 'rel_weights': path to weights files
          - 'fusion_path': path to fusion.json
          - 'tmpdir': Path to temp directory (for cleanup)
          - 'features': (BATCH, HIDDEN_DIM) test features (same batch for all heads)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create shared test features (same batch for all predictions)
        features = np.random.randn(BATCH, HIDDEN_DIM).astype(np.float32)

        # ====================================================================
        # Build US head artifact
        # ====================================================================
        us_head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
            name="us",
        )

        us_model = build_feature_inference_model({"us": us_head}, hidden_dim=HIDDEN_DIM)
        _ = us_model.predict({"features": features}, verbose=0)

        us_weights_path = tmpdir_path / "us.weights.h5"
        us_model.save_weights(str(us_weights_path))

        # Create US config sidecar
        us_config = UsRunConfig(
            seq_length=512,
            text_key="text",
            target_dtype="float32",
            backbone_weights_path="dummy.h5",
            head=UsHeadConfig(
                name="us",
                source_column="us_label",
                hidden_dim=HIDDEN_DIM,
            ),
            epochs=1,
            lr_schedule=LRScheduleConfig(
                initial_lr=2e-5,
                warmup_target=5e-5,
                decay_alpha=0.1,
                warmup_steps_factor=0.25,
                decay_steps_factor=3.0,
            ),
            optimizer=OptimizerConfig(weight_decay=0.01),
        )
        us_config_path = config_path_for_weights(us_weights_path)
        us_config.to_json(us_config_path)

        # Create US calibration
        us_logits = np.random.randn(BATCH).astype(np.float32)
        us_labels = np.random.randint(0, 2, BATCH)
        us_A, us_B = platt_fit(us_logits, us_labels)
        us_calibrator = PlattCalibrator(
            A=us_A, B=us_B, fit_population="test", n=BATCH, method="platt"
        )
        us_cal_path = calibration_path_for_weights(us_weights_path)
        save_calibration(us_calibrator, str(us_cal_path))

        # ====================================================================
        # Build CCA head artifact
        # ====================================================================
        cca_head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
            name="cca",
        )

        cca_model = build_feature_inference_model({"cca": cca_head}, hidden_dim=HIDDEN_DIM)
        _ = cca_model.predict({"features": features}, verbose=0)

        cca_weights_path = tmpdir_path / "cca.weights.h5"
        cca_model.save_weights(str(cca_weights_path))

        # Create CCA config sidecar
        cca_config = RunConfig(
            seq_length=512,
            text_key="text",
            target_dtype="float32",
            backbone_weights_path="dummy.h5",
            heads=(
                HeadConfig(
                    name="cca",
                    source_column="cca_label",
                    hidden_dim=HIDDEN_DIM,
                    loss=FLPULossConfig(prior=0.02),
                ),
            ),
            epochs=1,
            ratio_batch=RatioBatchConfig(train_pos=0.1, val_pos=0.5, test_pos=0.5),
            lr_schedule=LRScheduleConfig(
                initial_lr=2e-5,
                warmup_target=5e-5,
                decay_alpha=0.1,
                warmup_steps_factor=0.25,
                decay_steps_factor=3.0,
            ),
            optimizer=OptimizerConfig(weight_decay=0.01),
        )
        cca_config_path = config_path_for_weights(cca_weights_path)
        cca_config.to_json(cca_config_path)

        # Create CCA calibration
        cca_logits = np.random.randn(BATCH).astype(np.float32)
        cca_labels = np.random.randint(0, 2, BATCH)
        cca_A, cca_B = platt_fit(cca_logits, cca_labels)
        cca_calibrator = PlattCalibrator(
            A=cca_A, B=cca_B, fit_population="test", n=BATCH, method="platt"
        )
        cca_cal_path = calibration_path_for_weights(cca_weights_path)
        save_calibration(cca_calibrator, str(cca_cal_path))

        # ====================================================================
        # Build relevance head artifact
        # ====================================================================
        rel_head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
            name="rel",
        )

        rel_model = build_feature_inference_model({"rel": rel_head}, hidden_dim=HIDDEN_DIM)
        _ = rel_model.predict({"features": features}, verbose=0)

        rel_weights_path = tmpdir_path / "rel.weights.h5"
        rel_model.save_weights(str(rel_weights_path))

        # Create rel config sidecar
        rel_config = RunConfig(
            seq_length=512,
            text_key="text",
            target_dtype="float32",
            backbone_weights_path="dummy.h5",
            heads=(
                HeadConfig(
                    name="rel",
                    source_column="ica_event",
                    hidden_dim=HIDDEN_DIM,
                    loss=FLPULossConfig(prior=0.02),
                ),
            ),
            epochs=1,
            ratio_batch=RatioBatchConfig(train_pos=0.1, val_pos=0.5, test_pos=0.5),
            lr_schedule=LRScheduleConfig(
                initial_lr=2e-5,
                warmup_target=5e-5,
                decay_alpha=0.1,
                warmup_steps_factor=0.25,
                decay_steps_factor=3.0,
            ),
            optimizer=OptimizerConfig(weight_decay=0.01),
        )
        rel_config_path = config_path_for_weights(rel_weights_path)
        rel_config.to_json(rel_config_path)

        # Create rel calibration
        rel_logits = np.random.randn(BATCH).astype(np.float32)
        rel_labels = np.random.randint(0, 2, BATCH)
        rel_A, rel_B = platt_fit(rel_logits, rel_labels)
        rel_calibrator = PlattCalibrator(
            A=rel_A, B=rel_B, fit_population="test", n=BATCH, method="platt"
        )
        rel_cal_path = calibration_path_for_weights(rel_weights_path)
        save_calibration(rel_calibrator, str(rel_cal_path))

        # ====================================================================
        # Create fusion config
        # ====================================================================
        fusion_config = FusionConfig(
            gate_threshold=0.5,
            combine="product",
            coefs=None,  # product combiner doesn't use coefs
            score_space="prob",
            includes_us=True,
            composed_platt=None,  # No composed Platt for simplicity
            head_calibrators={"us": "test", "cca": "test", "rel": "test"},
        )

        fusion_path = tmpdir_path / "ica_fusion.fusion.json"
        save_fusion(fusion_config, str(fusion_path))

        yield {
            "us_weights": str(us_weights_path),
            "cca_weights": str(cca_weights_path),
            "rel_weights": str(rel_weights_path),
            "fusion_path": str(fusion_path),
            "tmpdir": tmpdir_path,
            "features": features,
        }


# =============================================================================
# Test: reload_and_score_ica matches IcaModel.predict_ica_from_features
# =============================================================================


def test_reload_and_score_ica_matches_ica_model(tiny_ica_artifact_set):
    """Verify that reload_and_score_ica output matches IcaModel.predict_ica_from_features
    within tolerance (bitwise on frozen-feature path).

    This test proves cross-process reproduction: construct the model fresh from disk,
    score the same features, and verify the outputs match bit-for-bit (or within float32
    tolerance).
    """
    artifact_set = tiny_ica_artifact_set

    # Construct IcaModel fresh from the fixture artifacts
    model = IcaModel(
        us_weights_path=artifact_set["us_weights"],
        cca_weights_path=artifact_set["cca_weights"],
        rel_weights_path=artifact_set["rel_weights"],
    )

    # Score via predict_ica_from_features (in-process)
    in_process_result = model.predict_ica_from_features(artifact_set["features"])

    # Score via reload_and_score_ica (cross-process simulation: fresh construction)
    cross_process_result = reload_and_score_ica(
        us_weights=artifact_set["us_weights"],
        cca_weights=artifact_set["cca_weights"],
        rel_weights=artifact_set["rel_weights"],
        fusion_path=artifact_set["fusion_path"],
        features=artifact_set["features"],
    )

    # Verify output structure
    assert set(in_process_result.keys()) == {"us", "cca", "rel", "ica_score"}
    assert set(cross_process_result.keys()) == {"us", "cca", "rel", "ica_score"}

    # Verify shapes
    n = len(artifact_set["features"])
    for key in ["us", "cca", "rel", "ica_score"]:
        assert in_process_result[key].shape == (n,), \
            f"in_process {key} shape mismatch: {in_process_result[key].shape}"
        assert cross_process_result[key].shape == (n,), \
            f"cross_process {key} shape mismatch: {cross_process_result[key].shape}"

    # Verify bitwise/near-bitwise equivalence (frozen-feature path, float32 precision)
    # Using allclose with tight tolerance to account for float32 rounding
    for key in ["us", "cca", "rel", "ica_score"]:
        np.testing.assert_allclose(
            in_process_result[key],
            cross_process_result[key],
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Mismatch in {key}",
        )

    print("✓ reload_and_score_ica matches IcaModel within tolerance:")
    print(f"  us:       min={cross_process_result['us'].min():.4f}, "
          f"max={cross_process_result['us'].max():.4f}")
    print(f"  cca:      min={cross_process_result['cca'].min():.4f}, "
          f"max={cross_process_result['cca'].max():.4f}")
    print(f"  rel:      min={cross_process_result['rel'].min():.4f}, "
          f"max={cross_process_result['rel'].max():.4f}")
    print(f"  ica_score: min={cross_process_result['ica_score'].min():.4f}, "
          f"max={cross_process_result['ica_score'].max():.4f}")


# =============================================================================
# Test: reload_and_score_ica requires all artifact files to exist
# =============================================================================


def test_reload_and_score_ica_missing_weights(tiny_ica_artifact_set):
    """Verify that missing weight files raise FileNotFoundError."""
    artifact_set = tiny_ica_artifact_set

    # Try with non-existent US weights
    with pytest.raises((FileNotFoundError, ValueError)):
        reload_and_score_ica(
            us_weights="/nonexistent/us.weights.h5",
            cca_weights=artifact_set["cca_weights"],
            rel_weights=artifact_set["rel_weights"],
            fusion_path=artifact_set["fusion_path"],
            features=artifact_set["features"],
        )


def test_reload_and_score_ica_missing_fusion(tiny_ica_artifact_set):
    """Verify that missing fusion config raises FileNotFoundError."""
    artifact_set = tiny_ica_artifact_set

    # Try with non-existent fusion.json
    with pytest.raises(FileNotFoundError):
        reload_and_score_ica(
            us_weights=artifact_set["us_weights"],
            cca_weights=artifact_set["cca_weights"],
            rel_weights=artifact_set["rel_weights"],
            fusion_path="/nonexistent/ica_fusion.fusion.json",
            features=artifact_set["features"],
        )


# =============================================================================
# Test: reload_and_score_ica output ranges are valid
# =============================================================================


def test_reload_and_score_ica_output_ranges(tiny_ica_artifact_set):
    """Verify that per-head probs and ica_score are in valid ranges [0, 1]."""
    artifact_set = tiny_ica_artifact_set

    result = reload_and_score_ica(
        us_weights=artifact_set["us_weights"],
        cca_weights=artifact_set["cca_weights"],
        rel_weights=artifact_set["rel_weights"],
        fusion_path=artifact_set["fusion_path"],
        features=artifact_set["features"],
    )

    # All per-head probs and ica_score should be in [0, 1]
    for key in ["us", "cca", "rel", "ica_score"]:
        assert np.all(result[key] >= 0.0), f"{key} contains negative values"
        assert np.all(result[key] <= 1.0), f"{key} contains values > 1"
        assert np.all(np.isfinite(result[key])), f"{key} contains non-finite values"


# =============================================================================
# Test: reload_and_score_ica input validation
# =============================================================================


def test_reload_and_score_ica_invalid_features_shape(tiny_ica_artifact_set):
    """Verify that invalid feature shape raises ValueError."""
    artifact_set = tiny_ica_artifact_set

    # Wrong feature shape (should be (n, 768))
    bad_features = np.random.randn(BATCH, 512).astype(np.float32)

    with pytest.raises(ValueError):
        reload_and_score_ica(
            us_weights=artifact_set["us_weights"],
            cca_weights=artifact_set["cca_weights"],
            rel_weights=artifact_set["rel_weights"],
            fusion_path=artifact_set["fusion_path"],
            features=bad_features,
        )


def test_reload_and_score_ica_wrong_feature_dims(tiny_ica_artifact_set):
    """Verify that 1D features raise ValueError."""
    artifact_set = tiny_ica_artifact_set

    # 1D array instead of 2D
    bad_features = np.random.randn(HIDDEN_DIM).astype(np.float32)

    with pytest.raises(ValueError):
        reload_and_score_ica(
            us_weights=artifact_set["us_weights"],
            cca_weights=artifact_set["cca_weights"],
            rel_weights=artifact_set["rel_weights"],
            fusion_path=artifact_set["fusion_path"],
            features=bad_features,
        )
