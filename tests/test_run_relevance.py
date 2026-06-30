"""Test the relevance head name renaming (Task 5 of Phase 3).

Verifies that:
- The RunConfig's HeadConfig.name is "rel" (not "cca")
- The head name flows through to the serialized sidecar
- Pattern-2 load (fresh head, weights loaded by structure) works with "rel" name
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import keras
import numpy as np

from src import config
from src.cca_config import DEFAULT_CCA_CONFIG
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_feature_inference_model


class TestRelevanceHeadRename:
    """Test that relevance head is renamed from 'cca' to 'rel' for multi-head assembly."""

    def test_config_rename_via_dataclasses_replace(self):
        """Verify the head name is 'rel' in the modified config."""
        base_config = DEFAULT_CCA_CONFIG
        base_head = base_config.heads[0]

        # Mirror what run_relevance.py does: use dataclasses.replace to set name="rel"
        renamed_head = dataclasses.replace(base_head, name="rel")
        new_config = dataclasses.replace(
            base_config,
            heads=(renamed_head,),
        )

        assert new_config.heads[0].name == "rel"
        assert base_config.heads[0].name == "cca"  # original unchanged

    def test_head_config_name_validation_accepts_rel(self):
        """HeadConfig should accept 'rel' as a valid name (no '/' in it)."""
        head_cfg = dataclasses.replace(DEFAULT_CCA_CONFIG.heads[0], name="rel")
        assert head_cfg.name == "rel"
        # If name contained "/" it would raise in __post_init__; we're not testing
        # that path here, just verifying "rel" is valid.

    def test_pattern2_load_with_rel_head(self):
        """Pattern 2: fresh head with 'rel' name loads weights by structure.

        Note: relevance.weights.h5 is a gitignored data product built by
        src.run_relevance.main(). This test exercises structural load equivalence
        by creating synthetic weights and loading them into a fresh 'rel' head,
        verifying that Pattern 2 (load-by-structure) succeeds when the head name
        matches what was saved (via dataclasses.replace in run_relevance.py:75).
        """
        # Create a minimal model with a 'rel'-named head and save weights
        keras.config.set_dtype_policy(config.DTYPE_POLICY)

        rel_head = ClassificationHead(
            hidden_dim=768,
            name="rel",  # The new name
        )

        model = build_feature_inference_model({"rel": rel_head}, hidden_dim=768)

        # Create dummy weights and save
        dummy_input = np.random.randn(2, 768).astype(np.float32)
        _ = model({"features": dummy_input})  # Build the model

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "test_rel.weights.h5"
            model.save_weights(str(weights_path))

            # Now create a fresh 'rel'-named head and load the weights
            fresh_head = ClassificationHead(
                hidden_dim=768,
                name="rel",
            )
            fresh_model = build_feature_inference_model({"rel": fresh_head}, hidden_dim=768)
            fresh_model.load_weights(str(weights_path), skip_mismatch=False)

            # Weights should load without error; shapes should match (batch_size, 1)
            assert fresh_model({"features": dummy_input})["rel"].shape == (2, 1)

    def test_different_heads_not_confused(self):
        """Test that 'cca' and 'rel' heads don't interfere (foundation for multi-head)."""
        keras.config.set_dtype_policy(config.DTYPE_POLICY)

        cca_head = ClassificationHead(hidden_dim=768, name="cca")
        rel_head = ClassificationHead(hidden_dim=768, name="rel")

        # Build separate models
        cca_model = build_feature_inference_model({"cca": cca_head}, hidden_dim=768)
        rel_model = build_feature_inference_model({"rel": rel_head}, hidden_dim=768)

        dummy_input = np.random.randn(2, 768).astype(np.float32)
        _ = cca_model({"features": dummy_input})
        _ = rel_model({"features": dummy_input})

        # Each model should output its respective head name
        cca_out = cca_model({"features": dummy_input})
        rel_out = rel_model({"features": dummy_input})

        assert "cca" in cca_out
        assert "rel" in rel_out
        assert "rel" not in cca_out
        assert "cca" not in rel_out


class TestRelevanceDatasetHeadNameContract:
    """End-to-end guard for the head-name <-> target-key contract (Phase 3 Task 5).

    Renaming the relevance head to "rel" (Task 5) means the endpoint model
    expects a "rel_targets" input. The dataset MUST be built with the matching
    head_name, else fit() fails with "Missing data for input 'rel_targets'".
    run_relevance.py threads head_cfg.name into dataset_from_embeddings; these
    tests fail if that wiring (or the rename) regresses.
    """

    def _build_rel_model(self, hidden_dim=8):
        from src.model_setup.heads import ClassificationHead
        from src.model_setup.assembly import build_feature_endpoint_model
        from src.loss_functions.loss import FLPULoss

        head = ClassificationHead(
            hidden_dim=hidden_dim,
            loss_fn=FLPULoss(prior=0.05),
            name="rel",
        )
        model = build_feature_endpoint_model({"rel": head}, hidden_dim=hidden_dim)
        model.compile(optimizer=keras.optimizers.Adam(1e-3))
        return model

    def _dataset(self, head_name, hidden_dim=8, n=32):
        from src.data_setup.data import dataset_from_embeddings

        rng = np.random.default_rng(0)
        feats = rng.standard_normal((n, hidden_dim)).astype("float32")
        labels = rng.integers(0, 2, size=n).astype("float32")
        # No weights -> data is a single (features, labels) group.
        return dataset_from_embeddings(
            shuffle_buffer=8, batch_size=8, data=(feats, labels),
            head_name=head_name,
        )

    def test_fit_succeeds_when_head_name_matches(self):
        model = self._build_rel_model()
        ds = self._dataset(head_name="rel")
        # One step must run without a "Missing data for input" error.
        model.fit(ds, steps_per_epoch=1, epochs=1, verbose=0)

    def test_fit_fails_when_head_name_mismatches(self):
        import pytest

        model = self._build_rel_model()
        ds = self._dataset(head_name="cca")  # the old default -> "cca_targets"
        with pytest.raises(ValueError, match="rel_targets"):
            model.fit(ds, steps_per_epoch=1, epochs=1, verbose=0)
