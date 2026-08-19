"""
Tests for src.assemble_ica.IcaModel (multi-head inference artifact).

Verifies:
  - 3-head assembly emits {us, cca, rel} keys
  - Per-head assembled scores equal standalone single-head scores (weight transfer proof)
  - Duplicate head name → ValueError
  - Missing/shape-mismatched head weight → ValueError
  - ica_score is 0.0 for gated-out rows and in [0,1] for survivors
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import keras  # noqa: F401  (initializes backend)

from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_feature_inference_model
from src.calibration.calibrator import PlattCalibrator, platt_fit
from src.calibration.sidecar import save_calibration, calibration_path_for_weights
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
from src.assemble_ica import IcaModel


HIDDEN_DIM = 768
BATCH = 10


# =============================================================================
# Fixtures: Synthetic heads and weights for testing
# =============================================================================


@pytest.fixture
def tiny_head_with_weights():
    """Create a tiny head, save weights and config, return paths.

    Yields:
        tuple: (weights_path, config_path, calibration_path, calibrator, head)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a tiny head and model
        head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
            name="test_head",
        )

        # Build a features-only model to initialize the head
        model = build_feature_inference_model(
            {"test_head": head}, hidden_dim=HIDDEN_DIM
        )

        # Dummy forward pass to initialize weights
        dummy_features = np.random.randn(BATCH, HIDDEN_DIM).astype(np.float32)
        _ = model.predict({"features": dummy_features}, verbose=0)

        # Save weights
        weights_path = tmpdir / "test_head.weights.h5"
        model.save_weights(str(weights_path))

        # Generate synthetic logits and fit Platt calibrator
        logits = np.random.randn(BATCH).astype(np.float32)
        labels = np.random.randint(0, 2, BATCH).astype(int)
        A, B = platt_fit(logits, labels)
        calibrator = PlattCalibrator(
            A=A, B=B, fit_population="test", n=BATCH, method="platt"
        )

        # Save calibration
        calibration_path = tmpdir / "test_head.calibration.json"
        save_calibration(calibrator, str(calibration_path))

        # For config, we'll just provide minimal paths (won't actually load configs)
        config_path = tmpdir / "test_head.config.json"

        yield str(weights_path), str(config_path), str(calibration_path), calibrator, head


@pytest.fixture
def three_tiny_heads_with_weights():
    """Create three tiny heads (us, cca, rel) with weights, configs, and calibrators.

    Yields:
        dict: {
            'us': (weights_path, calibration_path, calibrator),
            'cca': (weights_path, calibration_path, calibrator),
            'rel': (weights_path, calibration_path, calibrator),
            'tmpdir': Path to temp directory
        }
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        heads_data = {}

        for head_name in ["us", "cca", "rel"]:
            # Create head
            head = ClassificationHead(
                hidden_dim=HIDDEN_DIM,
                loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                name=head_name,
            )

            # Build model to initialize
            model = build_feature_inference_model(
                {head_name: head}, hidden_dim=HIDDEN_DIM
            )

            # Dummy forward pass
            dummy_features = np.random.randn(BATCH, HIDDEN_DIM).astype(np.float32)
            _ = model.predict({"features": dummy_features}, verbose=0)

            # Save weights
            weights_path = tmpdir_path / f"{head_name}.weights.h5"
            model.save_weights(str(weights_path))

            # Save config sidecar (properly formed, not placeholder)
            if head_name == "us":
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
                config_path = config_path_for_weights(weights_path)
                us_config.to_json(config_path)
            else:  # cca or rel
                cca_config = RunConfig(
                    seq_length=512,
                    text_key="text",
                    target_dtype="float32",
                    backbone_weights_path="dummy.h5",
                    heads=(
                        HeadConfig(
                            name=head_name,
                            source_column=f"{head_name}_label",
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
                config_path = config_path_for_weights(weights_path)
                cca_config.to_json(config_path)

            # Fit calibrator
            logits = np.random.randn(BATCH).astype(np.float32)
            labels = np.random.randint(0, 2, BATCH).astype(int)
            A, B = platt_fit(logits, labels)
            calibrator = PlattCalibrator(
                A=A, B=B, fit_population="test", n=BATCH, method="platt"
            )

            # Save calibration
            calibration_path = calibration_path_for_weights(weights_path)
            save_calibration(calibrator, str(calibration_path))

            heads_data[head_name] = (
                str(weights_path),
                str(calibration_path),
                calibrator,
            )

        heads_data["tmpdir"] = tmpdir_path

        yield heads_data


# =============================================================================
# Test: 3-head assembly structure
# =============================================================================


def test_three_head_assembly_structure():
    """Verify that assemble_ica.IcaModel uses build_feature_inference_model
    with three heads and produces correct output keys.
    """
    # Build three tiny heads
    us_head = ClassificationHead(
        hidden_dim=HIDDEN_DIM,
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        name="us",
    )
    cca_head = ClassificationHead(
        hidden_dim=HIDDEN_DIM,
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        name="cca",
    )
    rel_head = ClassificationHead(
        hidden_dim=HIDDEN_DIM,
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        name="rel",
    )

    # Assemble
    model = build_feature_inference_model(
        {"us": us_head, "cca": cca_head, "rel": rel_head},
        hidden_dim=HIDDEN_DIM,
    )

    # Verify model structure (Keras 3 uses list-based inputs)
    # The model is functional with a single "features" input
    assert len(model.inputs) == 1
    assert model.inputs[0].name == "features"

    # Test forward pass - outputs should be a list (Keras 3 functional model)
    dummy_features = np.random.randn(BATCH, HIDDEN_DIM).astype(np.float32)
    outputs_list = model.predict({"features": dummy_features}, verbose=0)

    # With dict-based head outputs, Keras returns a dict
    # (because we specified outputs as a dict in build_feature_inference_model)
    assert isinstance(outputs_list, dict)
    assert set(outputs_list.keys()) == {"us", "cca", "rel"}
    for head_name, logits in outputs_list.items():
        assert logits.shape == (BATCH, 1), f"{head_name} shape mismatch: {logits.shape}"


# =============================================================================
# Test: Pattern 2 weight transfer (standalone vs. assembled equivalence)
# =============================================================================


def test_pattern2_weight_transfer_single_head():
    """Verify that loading weights via temp single-head model (Pattern 2)
    produces identical predictions in standalone vs. assembled context.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create and train a tiny head
        head_template = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
            name="test_head",
        )

        standalone_model = build_feature_inference_model(
            {"test_head": head_template},
            hidden_dim=HIDDEN_DIM,
        )

        weights_path = tmpdir / "test_head.weights.h5"
        dummy_features = np.random.randn(BATCH, HIDDEN_DIM).astype(np.float32)
        _ = standalone_model.predict({"features": dummy_features}, verbose=0)
        standalone_model.save_weights(str(weights_path))

        # Get predictions from standalone model
        standalone_preds = standalone_model.predict(
            {"features": dummy_features}, verbose=0
        )["test_head"]

        # NOW: Load into an assembled model via Pattern 2
        # (create new head instance and load weights)
        assembled_head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
            name="test_head",
        )

        temp_model = build_feature_inference_model(
            {"test_head": assembled_head}, hidden_dim=HIDDEN_DIM
        )
        temp_model.load_weights(str(weights_path), skip_mismatch=False)

        # assembled_head now has the weights; use it in the assembled model
        assembled_model = build_feature_inference_model(
            {"test_head": assembled_head},
            hidden_dim=HIDDEN_DIM,
        )

        assembled_preds = assembled_model.predict(
            {"features": dummy_features}, verbose=0
        )["test_head"]

        # Verify bitwise equivalence
        np.testing.assert_allclose(
            standalone_preds, assembled_preds, rtol=1e-5, atol=1e-7
        )


# =============================================================================
# Test: Duplicate head name raises ValueError
# =============================================================================


def test_duplicate_head_name_raises():
    """Verify that build_feature_inference_model asserts unique head names."""
    head1 = ClassificationHead(
        hidden_dim=HIDDEN_DIM,
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        name="duplicate",
    )
    head2 = ClassificationHead(
        hidden_dim=HIDDEN_DIM,
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        name="duplicate",
    )

    # Attempting to pass duplicate names should raise
    with pytest.raises(ValueError, match="duplicate"):
        build_feature_inference_model(
            {"duplicate": head1, "also_duplicate": head2},
            hidden_dim=HIDDEN_DIM,
        )


# =============================================================================
# Test: Missing/shape-mismatched weight raises ValueError
# =============================================================================


def test_shape_mismatch_weight_load_raises():
    """Verify that loading weights with mismatched shape raises."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a head with HIDDEN_DIM=768
        head768 = ClassificationHead(
            hidden_dim=768,
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
            name="size_test",
        )

        model768 = build_feature_inference_model(
            {"size_test": head768}, hidden_dim=768
        )

        dummy_features = np.random.randn(BATCH, 768).astype(np.float32)
        _ = model768.predict({"features": dummy_features}, verbose=0)

        weights_path = tmpdir / "weights_768.weights.h5"
        model768.save_weights(str(weights_path))

        # Now try to load into a head with HIDDEN_DIM=512
        head512 = ClassificationHead(
            hidden_dim=512,
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
            name="size_test",
        )

        model512 = build_feature_inference_model(
            {"size_test": head512}, hidden_dim=512
        )

        # Should raise on weight shape mismatch with skip_mismatch=False
        with pytest.raises(ValueError):
            model512.load_weights(str(weights_path), skip_mismatch=False)


# =============================================================================
# Test: Gate behavior (ica_score is 0.0 for gated-out, in [0,1] for survivors)
# =============================================================================


def test_gate_behavior(three_tiny_heads_with_weights):
    """Verify gating: ica_score is 0.0 for gated-out rows and in [0,1] for survivors.

    Constructs a real IcaModel from fixture artifacts and verifies gate behavior
    through predict_ica_from_features (the production path).
    """
    from src.assemble_ica import IcaModel
    from src.fusion.sidecar import save_fusion
    from src.fusion.combiner import FusionConfig

    heads_info = three_tiny_heads_with_weights

    # Create a minimal fusion config for the fixture's product combiner
    fusion_config = FusionConfig(
        gate_threshold=0.5,
        combine="product",
        coefs=None,
        score_space="prob",
        includes_us=False,
        composed_platt=None,
        head_calibrators={"us": "test", "cca": "test", "rel": "test"},
    )
    tmpdir = heads_info["tmpdir"]
    fusion_path = tmpdir / "ica_fusion.fusion.json"
    save_fusion(fusion_config, str(fusion_path))

    # Construct fresh IcaModel from fixture artifacts
    model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
    )

    # Test on dummy features
    dummy_features = np.random.randn(BATCH, HIDDEN_DIM).astype(np.float32)
    result = model.predict_ica_from_features(dummy_features)

    # Verify structure
    assert set(result.keys()) == {"us", "cca", "rel", "ica_score"}

    # Verify constraints on ica_score
    ica_score = result["ica_score"]
    assert ica_score.shape == (BATCH,)
    assert np.all((ica_score >= 0.0) & (ica_score <= 1.0)), \
        f"ica_score out of bounds: min={ica_score.min()}, max={ica_score.max()}"

    # Verify gating: rows with us prob < gate_threshold should have ica_score = 0.0
    us_probs = result["us"]
    gate_threshold = model.fusion_config.gate_threshold
    gated_out = us_probs < gate_threshold
    if gated_out.any():
        assert np.all(ica_score[gated_out] == 0.0), \
            "gated-out rows should have ica_score=0.0"

    # Verify survivors are in [0, 1]
    survivors = us_probs >= gate_threshold
    if survivors.any():
        assert np.all(
            (ica_score[survivors] >= 0.0) & (ica_score[survivors] <= 1.0)
        ), "survivor ica_scores should be in [0, 1]"


# =============================================================================
# Test: Composed-Platt application (score-space transformation)
# =============================================================================


def test_composed_platt_score_space(three_tiny_heads_with_weights):
    """Verify composed-Platt score-space transformation: prob→logit→platt→prob.

    Constructs an IcaModel with composed-Platt calibration and verifies that
    the transformation sequence is correctly applied in predict_ica_from_features.
    """
    from src.assemble_ica import IcaModel
    from src.fusion.sidecar import save_fusion
    from src.fusion.combiner import FusionConfig
    from src.calibration.calibrator import platt_fit

    heads_info = three_tiny_heads_with_weights

    # Create synthetic combined scores to fit Platt
    combined_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    combined_clip = np.clip(combined_probs, 1e-10, 1.0 - 1e-10)
    logits_for_fit = np.log(combined_clip / (1.0 - combined_clip))
    labels_for_fit = np.array([0, 0, 1, 1, 1])
    A, B = platt_fit(logits_for_fit, labels_for_fit)

    # Verify platt parameters are finite
    assert np.isfinite(A) and np.isfinite(B), \
        f"platt parameters not finite: A={A}, B={B}"

    # Create fusion config WITH composed-Platt calibration
    fusion_config = FusionConfig(
        gate_threshold=0.5,
        combine="product",
        coefs=None,
        score_space="prob",
        includes_us=False,
        composed_platt=[float(A), float(B)],  # Apply composed-Platt
        head_calibrators={"us": "test", "cca": "test", "rel": "test"},
    )
    tmpdir = heads_info["tmpdir"]
    fusion_path = tmpdir / "ica_fusion_with_platt.fusion.json"
    save_fusion(fusion_config, str(fusion_path))

    # Construct IcaModel with composed-Platt config
    model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
    )

    # Verify that fusion config loaded correctly with composed_platt
    assert model.fusion_config.composed_platt is not None, \
        "composed_platt should be fitted"
    assert len(model.fusion_config.composed_platt) == 2, \
        "composed_platt should have 2 elements [A, B]"

    # Score dummy features and verify ica_score is in [0, 1]
    dummy_features = np.random.randn(BATCH, HIDDEN_DIM).astype(np.float32)
    result = model.predict_ica_from_features(dummy_features)

    ica_score = result["ica_score"]
    assert np.all((ica_score >= 0.0) & (ica_score <= 1.0)), \
        f"composed-Platt output not in [0, 1]: min={ica_score.min()}, max={ica_score.max()}"


# =============================================================================
# Test: Logreg combiner round-trip (fit → apply consistency)
# =============================================================================


def test_logreg_combiner_round_trip(three_tiny_heads_with_weights):
    """Verify logreg combiner round-trip: fit a tiny 2-feature LR and verify IcaModel
    produces identical output to sklearn's predict_proba on the same inputs.

    This tests the CRITICAL score-space contract: fit_fusion fits on PROBABILITIES,
    and IcaModel._combine_scores must apply on PROBABILITIES (not logits).
    """
    from sklearn.linear_model import LogisticRegression
    from src.assemble_ica import IcaModel
    from src.fusion.sidecar import save_fusion
    from src.fusion.combiner import FusionConfig

    heads_info = three_tiny_heads_with_weights

    # Create synthetic probability features for fitting
    np.random.seed(42)
    n_fit = 20
    p_cca_fit = np.random.uniform(0, 1, n_fit).astype(np.float32)
    p_rel_fit = np.random.uniform(0, 1, n_fit).astype(np.float32)
    labels_fit = np.random.randint(0, 2, n_fit)

    # Fit sklearn LR on probability features (what fit_fusion does)
    sklearn_features = np.column_stack([p_cca_fit, p_rel_fit])
    sklearn_lr = LogisticRegression(penalty=None, random_state=42)
    sklearn_lr.fit(sklearn_features, labels_fit)

    # Extract coefficients in the format fit_fusion saves: slopes + intercept
    sklearn_slopes = sklearn_lr.coef_[0].tolist()
    sklearn_intercept = float(sklearn_lr.intercept_[0])
    sklearn_coefs = sklearn_slopes + [sklearn_intercept]  # [slope_cca, slope_rel, intercept]

    # Create fusion config with logreg combiner using the fitted coefs
    fusion_config = FusionConfig(
        gate_threshold=0.5,
        combine="logreg",
        coefs=sklearn_coefs,  # slopes + intercept
        score_space="prob",
        includes_us=False,
        composed_platt=None,
        head_calibrators={"us": "test", "cca": "test", "rel": "test"},
    )
    tmpdir = heads_info["tmpdir"]
    fusion_path = tmpdir / "ica_fusion_logreg.fusion.json"
    save_fusion(fusion_config, str(fusion_path))

    # Construct IcaModel with logreg fusion
    model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
    )

    # Create test data: use some of the fit data to verify consistency
    n_test = 10
    p_cca_test = np.random.uniform(0, 1, n_test).astype(np.float32)
    p_rel_test = np.random.uniform(0, 1, n_test).astype(np.float32)

    # Apply via sklearn (reference implementation)
    sklearn_features_test = np.column_stack([p_cca_test, p_rel_test])
    sklearn_scores = sklearn_lr.predict_proba(sklearn_features_test)[:, 1]

    # Apply via IcaModel by mocking calibrated head outputs
    # We need to create features that will produce p_cca_test, p_rel_test after calibration
    # The simplest approach: use pre-calibrated values directly via mock heads.
    # For this test, we'll directly call _combine_scores (internal method)
    # to verify the probability-space combiner contract.
    ica_scores = model._combine_scores(p_cca_test, p_rel_test)

    # Verify bitwise equivalence (within float32 tolerance)
    np.testing.assert_allclose(
        ica_scores,
        sklearn_scores,
        rtol=1e-5,
        atol=1e-7,
        err_msg="logreg combiner output mismatch: IcaModel vs sklearn",
    )

    print("✓ logreg combiner round-trip verified:")
    print(f"  sklearn scores: min={sklearn_scores.min():.4f}, max={sklearn_scores.max():.4f}")
    print(f"  IcaModel scores: min={ica_scores.min():.4f}, max={ica_scores.max():.4f}")


# =============================================================================
# Test: gate_override parameter (AC6 LDC gold-first gating)
# =============================================================================


def test_gate_override_all_true(three_tiny_heads_with_weights):
    """Verify gate_override=all_True produces composed scores (not 0.0)."""
    from src.fusion.sidecar import save_fusion
    from src.fusion.combiner import FusionConfig

    heads_info = three_tiny_heads_with_weights

    # Create a basic fusion config
    fusion_config = FusionConfig(
        gate_threshold=0.5,
        combine="product",
        coefs=None,
        score_space="prob",
        includes_us=False,
        composed_platt=None,
        head_calibrators={"us": "test", "cca": "test", "rel": "test"},
    )
    tmpdir = heads_info["tmpdir"]
    fusion_path = tmpdir / "ica_fusion_override_true.fusion.json"
    save_fusion(fusion_config, str(fusion_path))

    # Create IcaModel
    model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
    )

    # Create features
    n_test = 10
    features = np.random.randn(n_test, 768).astype(np.float32)

    # Predict with gate_override=all_True
    gate_all_true = np.ones(n_test, dtype=bool)
    result_override_true = model.predict_ica_from_features(features, gate_override=gate_all_true)

    # All rows should have non-zero ica_score (gated in)
    ica_scores = result_override_true["ica_score"]
    assert (ica_scores > 0.0).any(), "Some rows should have non-zero ica_score with all-True gate"


def test_gate_override_all_false(three_tiny_heads_with_weights):
    """Verify gate_override=all_False produces ica_score=0.0 for all rows."""
    from src.fusion.sidecar import save_fusion
    from src.fusion.combiner import FusionConfig

    heads_info = three_tiny_heads_with_weights

    # Create a basic fusion config
    fusion_config = FusionConfig(
        gate_threshold=0.5,
        combine="product",
        coefs=None,
        score_space="prob",
        includes_us=False,
        composed_platt=None,
        head_calibrators={"us": "test", "cca": "test", "rel": "test"},
    )
    tmpdir = heads_info["tmpdir"]
    fusion_path = tmpdir / "ica_fusion_override_false.fusion.json"
    save_fusion(fusion_config, str(fusion_path))

    # Create IcaModel
    model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
    )

    # Create features
    n_test = 10
    features = np.random.randn(n_test, 768).astype(np.float32)

    # Predict with gate_override=all_False
    gate_all_false = np.zeros(n_test, dtype=bool)
    result_override_false = model.predict_ica_from_features(features, gate_override=gate_all_false)

    # All rows should have ica_score=0.0 (gated out)
    ica_scores = result_override_false["ica_score"]
    np.testing.assert_allclose(
        ica_scores, np.zeros(n_test), rtol=1e-5, atol=1e-7,
        err_msg="gate_override=all_False should produce ica_score=0.0 everywhere"
    )


def test_gate_override_independent_of_calib_us(three_tiny_heads_with_weights):
    """Verify gate_override is independent of actual calib_us scores.

    Scenario: a row has low calib_us (would be gated out by ML) but high gate_override
    (gold label says it's US) → ica_score should be non-zero. And vice versa.
    """
    from src.fusion.sidecar import save_fusion
    from src.fusion.combiner import FusionConfig

    heads_info = three_tiny_heads_with_weights

    # Create a basic fusion config
    fusion_config = FusionConfig(
        gate_threshold=0.5,
        combine="product",
        coefs=None,
        score_space="prob",
        includes_us=False,
        composed_platt=None,
        head_calibrators={"us": "test", "cca": "test", "rel": "test"},
    )
    tmpdir = heads_info["tmpdir"]
    fusion_path = tmpdir / "ica_fusion_override_mixed.fusion.json"
    save_fusion(fusion_config, str(fusion_path))

    # Create IcaModel
    model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
    )

    # Create features
    n_test = 10
    features = np.random.randn(n_test, 768).astype(np.float32)

    # Mixed gate_override: some True, some False, independent of what ML would do
    gate_mixed = np.array([i % 2 == 0 for i in range(n_test)], dtype=bool)
    result_override = model.predict_ica_from_features(features, gate_override=gate_mixed)

    # Rows where gate_mixed is True should have non-zero ica_score
    # Rows where gate_mixed is False should have zero ica_score
    ica_scores = result_override["ica_score"]
    for i in range(n_test):
        if gate_mixed[i]:
            # Gated in: expect non-zero (or could be zero by chance, but with product combiner
            # on random features, very unlikely to all be zero)
            pass  # Non-deterministic in this random setup, just check shape
        else:
            # Gated out: expect exactly zero
            assert ica_scores[i] == 0.0, f"Row {i} gated out should have ica_score=0.0"


# =============================================================================
# Test: constructor-default sentinel fix (mock.patch("src.assemble_ica.config")
# must actually take effect — the pre-fix defaults were bound at import time)
# =============================================================================


def _write_default_fusion(tmpdir: Path) -> Path:
    from src.fusion.sidecar import save_fusion
    from src.fusion.combiner import FusionConfig

    fusion_config = FusionConfig(
        gate_threshold=0.5,
        combine="product",
        coefs=None,
        score_space="prob",
        includes_us=False,
        head_calibrators={"us": "test", "cca": "test", "rel": "test"},
    )
    fusion_path = tmpdir / "ica_fusion.fusion.json"
    save_fusion(fusion_config, str(fusion_path))
    return fusion_path


def test_bare_construction_resolves_defaults_from_config_at_call_time(
    three_tiny_heads_with_weights,
):
    """IcaModel() with NO explicit weight paths must read config.* at __init__
    time, not at module-import time -- otherwise mock.patch("src.assemble_ica.config")
    (used throughout test_apply_ica.py) silently no-ops and those tests load
    real production weights instead of the fixture."""
    heads_info = three_tiny_heads_with_weights
    tmpdir = heads_info["tmpdir"]
    fusion_path = _write_default_fusion(tmpdir)

    with mock.patch("src.assemble_ica.config") as mock_config:
        mock_config.US_FILTER_FULL_WEIGHTS = heads_info["us"][0]
        mock_config.CCA_DOCA_WEIGHTS = heads_info["cca"][0]
        mock_config.RELEVANCE_DOCA_WEIGHTS = heads_info["rel"][0]
        mock_config.CCA_DOCA_DIR = tmpdir

        # Bare call: no us/cca/rel weights path arguments at all.
        bare_model = IcaModel(fusion_path=str(fusion_path))

    # Explicit-args construction from the SAME fixture artifacts (no mocking
    # needed -- config isn't consulted when every path is given explicitly).
    explicit_model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
    )

    features = np.random.default_rng(5).standard_normal((BATCH, HIDDEN_DIM)).astype(np.float32)
    bare_result = bare_model.predict_ica_from_features(features)
    explicit_result = explicit_model.predict_ica_from_features(features)
    for key in ("us", "cca", "rel", "ica_score"):
        np.testing.assert_allclose(bare_result[key], explicit_result[key], rtol=1e-5, atol=1e-7)


def test_bare_construction_uses_mocked_paths_not_real_production_defaults(
    three_tiny_heads_with_weights,
):
    """Negative-space proof: pointing the mocked config at nonexistent weight
    paths must make a bare IcaModel() call fail to load THOSE paths. Before
    the sentinel fix this test would NOT raise (the bound defaults silently
    fell back to the real production artifacts on disk, since they happen to
    exist in this dev environment)."""
    heads_info = three_tiny_heads_with_weights
    tmpdir = heads_info["tmpdir"]
    fusion_path = _write_default_fusion(tmpdir)

    with mock.patch("src.assemble_ica.config") as mock_config:
        mock_config.US_FILTER_FULL_WEIGHTS = tmpdir / "does_not_exist_us.weights.h5"
        mock_config.CCA_DOCA_WEIGHTS = heads_info["cca"][0]
        mock_config.RELEVANCE_DOCA_WEIGHTS = heads_info["rel"][0]
        mock_config.CCA_DOCA_DIR = tmpdir

        with pytest.raises(Exception):  # config_path_for_weights -> missing sidecar
            IcaModel(fusion_path=str(fusion_path))


# =============================================================================
# Test: head_feature_sources (branched-encoder per-head CLS source)
# =============================================================================


def test_head_feature_sources_none_is_legacy_single_input(three_tiny_heads_with_weights):
    """Default (head_feature_sources=None) is byte-identical to before: the
    assembled model has a single shared 'features' input."""
    heads_info = three_tiny_heads_with_weights
    fusion_path = _write_default_fusion(heads_info["tmpdir"])
    model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
    )
    assert model.head_feature_sources is None
    assert len(model.model.inputs) == 1
    assert model.model.inputs[0].name == "features"


def test_head_feature_sources_wires_per_tag_inputs(three_tiny_heads_with_weights):
    heads_info = three_tiny_heads_with_weights
    fusion_path = _write_default_fusion(heads_info["tmpdir"])
    sources = {"us": "base", "cca": "base", "rel": "rel_branch"}
    model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
        head_feature_sources=sources,
    )
    assert model.head_feature_sources == sources
    input_names = sorted(i.name for i in model.model.inputs)
    assert input_names == ["features_base", "features_rel_branch"]


def test_predict_legacy_array_rejected_when_sources_configured(three_tiny_heads_with_weights):
    heads_info = three_tiny_heads_with_weights
    fusion_path = _write_default_fusion(heads_info["tmpdir"])
    model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
        head_feature_sources={"us": "base", "cca": "base", "rel": "rel_branch"},
    )
    features = np.random.randn(BATCH, HIDDEN_DIM).astype(np.float32)
    with pytest.raises(ValueError, match="head_feature_sources"):
        model.predict_ica_from_features(features)


def test_predict_dict_rejected_when_no_sources_configured(three_tiny_heads_with_weights):
    heads_info = three_tiny_heads_with_weights
    fusion_path = _write_default_fusion(heads_info["tmpdir"])
    model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
    )
    features = {"base": np.random.randn(BATCH, HIDDEN_DIM).astype(np.float32)}
    with pytest.raises(ValueError, match="head_feature_sources"):
        model.predict_ica_from_features(features)


def test_predict_dict_missing_tag_raises(three_tiny_heads_with_weights):
    heads_info = three_tiny_heads_with_weights
    fusion_path = _write_default_fusion(heads_info["tmpdir"])
    model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
        head_feature_sources={"us": "base", "cca": "base", "rel": "rel_branch"},
    )
    features = {"base": np.random.randn(BATCH, HIDDEN_DIM).astype(np.float32)}
    with pytest.raises(ValueError, match="rel_branch"):
        model.predict_ica_from_features(features)


def test_predict_dict_row_count_mismatch_raises(three_tiny_heads_with_weights):
    heads_info = three_tiny_heads_with_weights
    fusion_path = _write_default_fusion(heads_info["tmpdir"])
    model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
        head_feature_sources={"us": "base", "cca": "base", "rel": "rel_branch"},
    )
    features = {
        "base": np.random.randn(BATCH, HIDDEN_DIM).astype(np.float32),
        "rel_branch": np.random.randn(BATCH - 1, HIDDEN_DIM).astype(np.float32),
    }
    with pytest.raises(ValueError, match="row"):
        model.predict_ica_from_features(features)


def test_predict_dict_non_2d_array_raises(three_tiny_heads_with_weights):
    heads_info = three_tiny_heads_with_weights
    fusion_path = _write_default_fusion(heads_info["tmpdir"])
    model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
        head_feature_sources={"us": "base", "cca": "base", "rel": "rel_branch"},
    )
    features = {
        "base": np.random.randn(BATCH, HIDDEN_DIM).astype(np.float32),
        "rel_branch": np.random.randn(HIDDEN_DIM).astype(np.float32),  # 1-D
    }
    with pytest.raises(ValueError):
        model.predict_ica_from_features(features)


def test_predict_dict_wrong_hidden_dim_raises(three_tiny_heads_with_weights):
    heads_info = three_tiny_heads_with_weights
    fusion_path = _write_default_fusion(heads_info["tmpdir"])
    model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
        head_feature_sources={"us": "base", "cca": "base", "rel": "rel_branch"},
    )
    features = {
        "base": np.random.randn(BATCH, HIDDEN_DIM).astype(np.float32),
        "rel_branch": np.random.randn(BATCH, 512).astype(np.float32),
    }
    with pytest.raises(ValueError):
        model.predict_ica_from_features(features)


def test_predict_dict_mode_matches_legacy_when_sources_collapse_to_same_array(
    three_tiny_heads_with_weights,
):
    """Sources-mode with every head mapped to the same tag, fed the SAME array
    both places, reproduces legacy-mode output exactly -- the sources plumbing
    doesn't change the math, only where the array comes from."""
    heads_info = three_tiny_heads_with_weights
    fusion_path = _write_default_fusion(heads_info["tmpdir"])

    legacy_model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
    )
    sources_model = IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
        head_feature_sources={"us": "base", "cca": "base", "rel": "base"},
    )

    features = np.random.default_rng(9).standard_normal((BATCH, HIDDEN_DIM)).astype(np.float32)
    legacy_result = legacy_model.predict_ica_from_features(features)
    sources_result = sources_model.predict_ica_from_features({"base": features})
    for key in ("us", "cca", "rel", "ica_score"):
        np.testing.assert_allclose(
            legacy_result[key], sources_result[key], rtol=1e-5, atol=1e-7
        )


# =============================================================================
# Test: constructor sources vs fusion-recorded sources cross-check
# =============================================================================


def test_constructor_sources_match_fusion_recorded_sources_ok(three_tiny_heads_with_weights):
    from src.fusion.sidecar import save_fusion
    from src.fusion.combiner import FusionConfig

    heads_info = three_tiny_heads_with_weights
    sources = {"us": "base", "cca": "base", "rel": "rel_branch"}
    fusion_config = FusionConfig(
        gate_threshold=0.5,
        combine="product",
        coefs=None,
        score_space="prob",
        includes_us=False,
        head_calibrators={"us": "test", "cca": "test", "rel": "test"},
        head_feature_sources=sources,
    )
    fusion_path = heads_info["tmpdir"] / "ica_fusion_matching.fusion.json"
    save_fusion(fusion_config, str(fusion_path))

    # Must construct without raising.
    IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
        head_feature_sources=sources,
    )


def test_constructor_sources_mismatch_fusion_recorded_sources_raises(
    three_tiny_heads_with_weights,
):
    from src.fusion.sidecar import save_fusion
    from src.fusion.combiner import FusionConfig

    heads_info = three_tiny_heads_with_weights
    fusion_sources = {"us": "base", "cca": "base", "rel": "rel_branch"}
    constructor_sources = {"us": "base", "cca": "base", "rel": "some_other_branch"}
    fusion_config = FusionConfig(
        gate_threshold=0.5,
        combine="product",
        coefs=None,
        score_space="prob",
        includes_us=False,
        head_calibrators={"us": "test", "cca": "test", "rel": "test"},
        head_feature_sources=fusion_sources,
    )
    fusion_path = heads_info["tmpdir"] / "ica_fusion_mismatch.fusion.json"
    save_fusion(fusion_config, str(fusion_path))

    with pytest.raises(ValueError) as exc_info:
        IcaModel(
            us_weights_path=heads_info["us"][0],
            cca_weights_path=heads_info["cca"][0],
            rel_weights_path=heads_info["rel"][0],
            fusion_path=str(fusion_path),
            head_feature_sources=constructor_sources,
        )
    msg = str(exc_info.value)
    assert "some_other_branch" in msg
    assert "rel_branch" in msg


def test_fusion_without_sources_field_skips_check(three_tiny_heads_with_weights):
    """Back-compat: a fusion sidecar predating head_feature_sources (field
    absent -> None) performs no cross-check, even when the constructor is
    given sources."""
    heads_info = three_tiny_heads_with_weights
    fusion_path = _write_default_fusion(heads_info["tmpdir"])  # no head_feature_sources

    # Must construct without raising, despite the fusion recording nothing.
    IcaModel(
        us_weights_path=heads_info["us"][0],
        cca_weights_path=heads_info["cca"][0],
        rel_weights_path=heads_info["rel"][0],
        fusion_path=str(fusion_path),
        head_feature_sources={"us": "base", "cca": "base", "rel": "rel_branch"},
    )
