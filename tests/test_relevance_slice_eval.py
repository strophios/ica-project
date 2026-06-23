# pattern: Functional Core (inference helper contract test)
"""Tests for src.validation.relevance_slice_eval — relevance-head features-mode inference."""

from __future__ import annotations

import numpy as np


def test_apply_relevance_model_output_shape_and_finiteness():
    """
    Verify apply_relevance_model(features, weights_path) returns shape (n,) logits.

    This is a contract/shape test. Since the real relevance weights may not be
    available locally (this is a pre-Phase-3 helper), we construct a synthetic
    head + model to verify the pattern works. The actual integration (with real
    weights) is tested when real artifacts are available.
    """
    # Avoid import at module level to prevent test collection failure if
    # weights or config don't exist locally.
    import tempfile
    from pathlib import Path


    from src.model_setup.heads import ClassificationHead
    from src.model_setup.assembly import build_feature_inference_model
    from src.validation.relevance_slice_eval import apply_relevance_model
    import src.cca_config as cca_config

    # Build a synthetic minimal model for testing
    hidden_dim = 768
    head = ClassificationHead(hidden_dim=hidden_dim, name="relevance")
    model = build_feature_inference_model({"relevance": head}, hidden_dim=hidden_dim)

    # Save to a temp file so we can use apply_relevance_model's standard load path
    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = Path(tmpdir) / "test_weights.weights.h5"
        model.save_weights(str(weights_path))

        # Create and save a synthetic config sidecar (uses CCA config structure)
        config_path = cca_config.config_path_for_weights(weights_path)
        default_config = cca_config.DEFAULT_CCA_CONFIG
        default_config.to_json(config_path)

        # Create dummy features batch
        n_samples = 5
        features = np.random.randn(n_samples, hidden_dim).astype(np.float32)

        # Call the helper
        logits = apply_relevance_model(features, weights_path)

        # Verify shape
        assert logits.shape == (n_samples,), f"Expected shape ({n_samples},), got {logits.shape}"

        # Verify finiteness
        assert np.isfinite(logits).all(), "Non-finite logits produced"

        # Verify it's a numpy array (or compatible)
        assert isinstance(logits, np.ndarray)


def test_apply_relevance_model_batch_size_preserved():
    """Verify logits length matches input feature count."""
    import tempfile
    from pathlib import Path


    from src.model_setup.heads import ClassificationHead
    from src.model_setup.assembly import build_feature_inference_model
    from src.validation.relevance_slice_eval import apply_relevance_model
    import src.cca_config as cca_config

    hidden_dim = 768
    head = ClassificationHead(hidden_dim=hidden_dim, name="relevance")
    model = build_feature_inference_model({"relevance": head}, hidden_dim=hidden_dim)

    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = Path(tmpdir) / "test_weights.weights.h5"
        model.save_weights(str(weights_path))

        # Create and save a synthetic config sidecar
        config_path = cca_config.config_path_for_weights(weights_path)
        default_config = cca_config.DEFAULT_CCA_CONFIG
        default_config.to_json(config_path)

        # Test with various batch sizes
        for batch_size in [1, 10, 100]:
            features = np.random.randn(batch_size, hidden_dim).astype(np.float32)
            logits = apply_relevance_model(features, weights_path)
            assert len(logits) == batch_size
            assert logits.ndim == 1
