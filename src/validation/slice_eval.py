# pattern: Imperative Shell (model application + evaluation)
"""Transfer eval + proxy gap for pre-1986 slice validation.

Applies the trained US model to a text corpus, evaluates performance
against hand-labeled gold set, and computes the dateline-vs-event-location
proxy gap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

import keras
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


def apply_us_model(
    texts: list[str],
    weights_path: Path | str = config.US_FILTER_CLASSIFIER_WEIGHTS,
) -> np.ndarray:
    """Apply calibrated US model to texts (Pattern 2: fresh head, loaded weights).

    Follows the Pattern-2 shape from eval_cca_classifier.py:
    - Load UsRunConfig sidecar
    - Fresh ClassificationHead
    - build_inference_model
    - load_weights(skip_mismatch=False)
    - ClassifierPreprocessor in inference mode
    - predict + assert finite
    - load PlattCalibrator, transform logits -> calibrated us_score

    Args:
        texts: List of article texts (headline + lead_paragraph)
        weights_path: Path to trained US model weights

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


def evaluate_slice(
    gold_df: pl.DataFrame,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Evaluate model performance on gold-set slice.

    Computes precision, recall, F1 against us_event labels using
    us_score >= threshold for prediction.

    Args:
        gold_df: Gold-set dataframe with 'us_score' and 'us_event' columns
        threshold: Decision threshold for positive prediction

    Returns:
        Dict with keys: precision, recall, f1, n_pos, n_neg
    """
    # Assume us_score is present (from apply_us_model)
    predictions = (gold_df["us_score"] >= threshold).cast(pl.Boolean)
    labels = gold_df["us_event"]

    # Count positives and negatives
    n_pos = (labels == True).sum()
    n_neg = (labels == False).sum()

    # Confusion matrix
    tp = ((predictions == True) & (labels == True)).sum()
    fp = ((predictions == True) & (labels == False)).sum()
    fn = ((predictions == False) & (labels == True)).sum()
    tn = ((predictions == False) & (labels == False)).sum()

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0 and recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
    }


def proxy_gap(gold_df: pl.DataFrame) -> dict[str, float | int]:
    """Compute dateline-vs-event-location agreement proxy gap.

    For rows where both event_location is available, checks agreement
    between the us_event label and the event_location code.

    Args:
        gold_df: Gold-set dataframe with 'us_event' and 'event_location' columns

    Returns:
        Dict with keys: dateline_event_agreement (0-1), n (count of rows with both)
    """
    # Filter to rows with event_location present
    rows_with_loc = gold_df.filter(pl.col("event_location").is_not_null())

    if rows_with_loc.shape[0] == 0:
        return {"dateline_event_agreement": 0.0, "n": 0}

    # Check agreement: us_event=True (US event) should match event_location="US"
    event_location = rows_with_loc["event_location"]
    us_event = rows_with_loc["us_event"]

    # us_event=True should match event_location="US"
    is_us_location = event_location == "US"
    agreement = (us_event == is_us_location).sum() / rows_with_loc.shape[0]

    return {
        "dateline_event_agreement": float(agreement),
        "n": int(rows_with_loc.shape[0]),
    }
