# pattern: Imperative Shell (delegates artifact-triple reload + scoring to apply_us_model)
"""Artifact triple reload check — proves reproducibility cross-process.

Demonstrates that the durable artifact (weights + config + calibration sidecars)
is sufficient to reproduce calibrated scores. This is the proof that the triple
alone suffices for downstream use without needing training scripts.
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
