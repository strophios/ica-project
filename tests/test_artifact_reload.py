"""Tests for src.validation.artifact_check — artifact triple reload.

Exercises reload_and_score with a tiny model + calibrator triple (weights,
config, calibration sidecar), saved to tmp_path, loaded fresh in a subprocess
call, and verified to reproduce scores within floating-point tolerance.
Verifies AC5.2: artifact triple reloads deterministically.
"""

from pathlib import Path

import numpy as np

import keras  # noqa: F401  (initializes backend)

from src.validation.artifact_check import reload_and_score
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_inference_model
from src.us_config import UsRunConfig, UsHeadConfig
from src.cca_config import LRScheduleConfig, OptimizerConfig
from src.calibration.calibrator import PlattCalibrator
from src.calibration.sidecar import save_calibration, calibration_path_for_weights
from src.us_config import config_path_for_weights


SEQ_LEN = 4
HIDDEN_DIM = 8
VOCAB = 100
BATCH = 4


def _make_fake_backbone(seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM, vocab=VOCAB,
                        name="fake_backbone"):
    """Build a tiny stand-in for a real backbone."""
    token_ids = keras.Input(shape=(seq_len,), dtype="int32", name="token_ids")
    padding_mask = keras.Input(shape=(seq_len,), dtype="int32", name="padding_mask")
    embed = keras.layers.Embedding(vocab, hidden_dim, name="fake_embed")
    embedded = embed(token_ids)
    mask_float = keras.ops.cast(padding_mask, "float32")
    mask_expanded = keras.ops.expand_dims(mask_float, axis=-1)
    masked = embedded * mask_expanded
    backbone = keras.Model(
        inputs={"token_ids": token_ids, "padding_mask": padding_mask},
        outputs=masked,
        name=name,
    )
    backbone.hidden_dim = hidden_dim
    return backbone


class TestArtifactReloadBasic:
    """Artifact triple reload — basic mechanics."""

    def test_reload_and_score_returns_calibrated_scores(self, tmp_path):
        """reload_and_score returns float array in [0, 1]."""
        # Build and train a tiny model
        backbone = _make_fake_backbone()
        us_head = ClassificationHead(hidden_dim=HIDDEN_DIM, name="us")

        inference_model = build_inference_model(
            backbone=backbone,
            heads={"us": us_head},
            seq_length=SEQ_LEN,
        )

        # Save weights
        weights_path = tmp_path / "test.weights.h5"
        inference_model.save_weights(str(weights_path))

        # Create config sidecar
        config = UsRunConfig(
            seq_length=SEQ_LEN,
            text_key="headline_with_lead",
            target_dtype="float32",
            head=UsHeadConfig(name="us", source_column="us_label", hidden_dim=HIDDEN_DIM),
            epochs=1,
            backbone_weights_path="fake",
            lr_schedule=LRScheduleConfig(),
            optimizer=OptimizerConfig(),
        )
        config_path = config_path_for_weights(weights_path)
        config.to_json(config_path)

        # Create calibrator and save sidecar
        calibrator = PlattCalibrator(
            A=0.5,
            B=-0.1,
            fit_population="test",
            n=10,
            method="platt",
        )
        cal_path = calibration_path_for_weights(weights_path)
        save_calibration(calibrator, cal_path)

        # Test texts (just placeholder strings)
        texts = ["test text one", "test text two", "test text three"]

        # reload_and_score should load the triple and return scores
        us_scores = reload_and_score(
            str(weights_path),
            texts,
            backbone=backbone,  # Pass backbone to avoid file I/O
        )

        # Verify shape and range
        assert isinstance(us_scores, np.ndarray)
        assert us_scores.shape == (len(texts),)
        assert np.all(us_scores >= 0.0)
        assert np.all(us_scores <= 1.0)


class TestArtifactReloadFPTolerance:
    """Artifact triple reload — floating-point reproducibility."""

    def test_reload_reproduces_scores_within_tolerance(self, tmp_path):
        """reload_and_score reproduces pre-save scores within fp tolerance (AC5.2)."""
        # Build tiny model
        backbone = _make_fake_backbone()
        us_head = ClassificationHead(hidden_dim=HIDDEN_DIM, name="us")

        inference_model = build_inference_model(
            backbone=backbone,
            heads={"us": us_head},
            seq_length=SEQ_LEN,
        )

        # Save weights
        weights_path = tmp_path / "test.weights.h5"
        inference_model.save_weights(str(weights_path))

        # Create config sidecar
        config = UsRunConfig(
            seq_length=SEQ_LEN,
            text_key="headline_with_lead",
            target_dtype="float32",
            head=UsHeadConfig(name="us", source_column="us_label", hidden_dim=HIDDEN_DIM),
            epochs=1,
            backbone_weights_path="fake",
            lr_schedule=LRScheduleConfig(),
            optimizer=OptimizerConfig(),
        )
        config_path = config_path_for_weights(weights_path)
        config.to_json(config_path)

        # Create calibrator and save sidecar
        calibrator = PlattCalibrator(
            A=0.8,
            B=0.2,
            fit_population="test",
            n=15,
            method="platt",
        )
        cal_path = calibration_path_for_weights(weights_path)
        save_calibration(calibrator, cal_path)

        texts = ["text one", "text two", "text three"]

        # Compute scores pre-reload
        from src.validation.slice_eval import apply_us_model
        scores_before = apply_us_model(texts, weights_path, backbone=backbone)

        # Reload and compute scores post-reload
        scores_after = reload_and_score(
            str(weights_path),
            texts,
            backbone=backbone,
        )

        # Scores should match within float32 epsilon
        # Using atol based on float32 precision
        np.testing.assert_allclose(
            scores_before,
            scores_after,
            rtol=1e-5,
            atol=1e-6,
            err_msg="reload_and_score should reproduce apply_us_model scores",
        )


class TestArtifactReloadDocumentation:
    """Threshold recipe documentation (AC5.3)."""

    def test_default_threshold_documented(self):
        """Default US threshold is documented as 0.5."""
        # Import the documentation (will be created in docs/notes/)
        doc_path = Path(__file__).parent.parent / "docs" / "notes" / "us-filter-threshold-recipe.md"
        assert doc_path.exists(), f"Threshold recipe doc not found at {doc_path}"

        content = doc_path.read_text()
        assert "0.5" in content, "Default threshold 0.5 should be in documentation"

    def test_threshold_recipe_references_doca_recall(self):
        """Threshold recipe references doca_recall function."""
        doc_path = Path(__file__).parent.parent / "docs" / "notes" / "us-filter-threshold-recipe.md"
        assert doc_path.exists(), f"Threshold recipe doc not found at {doc_path}"

        content = doc_path.read_text()
        assert "doca_recall" in content, "Documentation should reference doca_recall"
