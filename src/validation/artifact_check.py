# pattern: Imperative Shell (delegates artifact-triple reload + scoring to apply_us_model and IcaModel)
"""Artifact triple reload check — proves reproducibility cross-process.

Demonstrates that the durable artifact (weights + config + calibration sidecars)
is sufficient to reproduce calibrated scores. This is the proof that the triple
alone suffices for downstream use without needing training scripts.

For US model: delegates to apply_us_model.
For ICA model: constructs a fresh IcaModel from disk to prove cross-process reproduction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.validation.slice_eval import apply_us_model


def reload_and_score(
    weights_path: Path | str,
    texts: list[str],
    backbone=None,
) -> np.ndarray:
    """Load artifact triple and return calibrated scores for texts.

    Demonstrates Pattern-2 cross-process load: loads config sidecar → fresh head
    + inference model → load_weights(skip_mismatch=False) → calibrator → scores.
    Delegates to apply_us_model for the actual scoring logic; this function
    is isolated for artifact verification testing.

    Args:
        weights_path: Path to trained US model weights (.weights.h5)
        texts: List of article texts (headline + "</s>" + lead_paragraph)
        backbone: Optional pre-built backbone (for testing; if None, loads from config)

    Returns:
        Calibrated probability scores [0, 1], shape (len(texts),)
    """
    return apply_us_model(texts, weights_path=weights_path, backbone=backbone)


def reload_and_score_ica(
    us_weights: Path | str,
    cca_weights: Path | str,
    rel_weights: Path | str,
    fusion_path: Path | str,
    features: np.ndarray,
) -> dict:
    """Load ICA artifact triple (3 heads + calibrators + fusion) and score features cross-process.

    Demonstrates Pattern-2 cross-process load for multi-head assembly: fresh construction
    of IcaModel (no shared instances with prior models), load all artifact sidecars (configs,
    calibrations, fusion), assemble heads, and return composed ICA score + per-head probs.

    The point is to prove that the on-disk artifact set reproduces scores identically in a
    fresh process/construction, validating Pattern-2 reproducibility.

    Args:
        us_weights: Path to US head weights (.weights.h5)
        cca_weights: Path to CCA head weights (.weights.h5)
        rel_weights: Path to relevance head weights (.weights.h5)
        fusion_path: Path to fusion config (.fusion.json)
        features: shape (n, 768) float32 array of CLS embeddings

    Returns:
        dict with keys:
          - "us": (n,) calibrated US probabilities in [0, 1]
          - "cca": (n,) calibrated CCA probabilities in [0, 1]
          - "rel": (n,) calibrated relevance probabilities in [0, 1]
          - "ica_score": (n,) composed ICA score in [0, 1]

    Raises:
        ValueError: if any head config doesn't load, weight transfer fails, or validation fails
        FileNotFoundError: if weights, configs, calibrations, or fusion sidecar don't exist
    """
    from src.assemble_ica import IcaModel

    # Fresh construction: no shared instances. IcaModel loads all artifacts from disk
    # by path and assembles them internally.
    model = IcaModel(
        us_weights_path=us_weights,
        cca_weights_path=cca_weights,
        rel_weights_path=rel_weights,
    )

    # Validate that the fusion config path is accessible (defensive check)
    fusion_path = Path(fusion_path)
    if not fusion_path.exists():
        raise FileNotFoundError(f"fusion config does not exist: {fusion_path}")

    # Use the fresh model's predict_ica_from_features to score
    return model.predict_ica_from_features(features)
