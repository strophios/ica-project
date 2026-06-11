# pattern: Imperative Shell (loads artifact triple and scores texts)
"""Artifact triple reload check — proves reproducibility cross-process.

Demonstrates that the durable artifact (weights + config + calibration sidecars)
is sufficient to reproduce calibrated scores. This is the proof that the triple
alone suffices for downstream use without needing training scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import src.config as config
from src.calibration.sidecar import (
    calibration_path_for_weights,
    load_calibration,
)
from src.preproc.preprocessor import ClassifierPreprocessor
from src.model_setup.backbone import load_dapt_backbone
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_inference_model
from src.us_config import UsRunConfig, config_path_for_weights


def reload_and_score(
    weights_path: Path | str,
    texts: list[str],
    backbone=None,
) -> np.ndarray:
    """Load artifact triple and return calibrated scores for texts.

    Demonstrates Pattern-2 cross-process load: loads config sidecar → fresh head
    + inference model → load_weights(skip_mismatch=False) → calibrator → scores.
    This mirrors apply_us_model's structure but is isolated for artifact
    verification testing.

    Args:
        weights_path: Path to trained US model weights (.weights.h5)
        texts: List of article texts (headline + "</s>" + lead_paragraph)
        backbone: Optional pre-built backbone (for testing; if None, loads from config)

    Returns:
        Calibrated probability scores [0, 1], shape (len(texts),)
    """
    weights_path = Path(weights_path)

    # Load run config sidecar
    config_sidecar_path = config_path_for_weights(weights_path)
    run_config = UsRunConfig.from_json(config_sidecar_path)

    # Build fresh head and inference model
    us_head = ClassificationHead(
        hidden_dim=run_config.head.hidden_dim,
        name=run_config.head.name,
    )

    if backbone is None:
        backbone = load_dapt_backbone(run_config.backbone_weights_path)
    run_config.validate_against_backbone(backbone)

    inference_model = build_inference_model(
        backbone=backbone,
        heads={run_config.head.name: us_head},
        seq_length=run_config.seq_length,
    )

    # Load weights
    inference_model.load_weights(str(weights_path), skip_mismatch=False)

    # Build preprocessor (inference mode)
    preproc = ClassifierPreprocessor(
        SEQ_LENGTH=run_config.seq_length,
        text_key=run_config.text_key,
        label_keys={},
        endpoint_model=True,
        target_dtype=run_config.target_dtype,
    )

    # Create input dict and preprocess
    input_dict = {run_config.text_key: texts}
    batch = preproc(input_dict)

    # Predict
    logits = inference_model.predict(batch, batch_size=256, verbose=0)

    # Handle dict or tensor output
    if isinstance(logits, dict):
        logits = logits[run_config.head.name]

    logits = logits.squeeze()

    # Assert finite
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite logits produced by model")

    # Load calibrator and transform to calibrated probabilities
    calibrator = load_calibration(calibration_path_for_weights(weights_path))
    us_scores = calibrator.transform(logits)

    return us_scores
