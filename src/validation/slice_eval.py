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
    backbone=None,
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

    # Load weights. Use skip_mismatch=True to allow loading head-only weights
    # (the backbone is separately loaded and frozen, so we only need head weights).
    inference_model.load_weights(str(weights_path), skip_mismatch=True)

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
    us_score >= threshold for prediction. Rows with null us_event
    are silently excluded from metrics computation.

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
    n_pos = labels.sum()
    n_neg = (~labels).sum()

    # Confusion matrix
    tp = (predictions & labels).sum()
    fp = (predictions & ~labels).sum()
    fn = (~predictions & labels).sum()

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


def proxy_gap(
    gold_df: pl.DataFrame,
    dateline_labels_df: pl.DataFrame | None = None,
) -> dict[str, float | int]:
    """Compute dateline-vs-event-location agreement proxy gap (AC6.3).

    Measures agreement between DATELINE labels (looked up via alt_corpus_id from
    the LDC labeled parquet) and hand-coded event_location. This is a proxy for
    assessing whether the DATELINE label is a reliable stand-in for hand-coding
    when a gold set doesn't exist.

    If dateline_labels_df is not provided, reads from config.US_FILTER_LABELED_PARQUET
    and filters to label_source == "dateline". Expects both dataframes to have
    id columns cast to str for consistent joining.

    Agreement is computed as: mean(dateline_us_label == event_location_us_coding)
    where event_location_us_coding is derived by checking if event_location == "US".

    Args:
        gold_df: Gold-set dataframe with columns:
            - alt_corpus_id (str, nullable): LDC id for joining to dateline labels
            - event_location (str, nullable): hand-coded location ("US" or foreign)
            - us_event (bool, nullable): hand-coded US-ness (included for context)
        dateline_labels_df: Optional DataFrame with columns:
            - ldc_id (str): LDC article id
            - us_label (bool): dateline-derived label
            If None, read from config.US_FILTER_LABELED_PARQUET, cast id to str,
            filter to label_source == "dateline".

    Returns:
        Dict with keys:
        - dateline_event_agreement (float in [0, 1]): fraction of rows where
          dateline label matches hand-coded event_location
        - n (int): count of gold-set rows successfully joined with dateline labels
    """
    # Load dateline labels if not provided
    if dateline_labels_df is None:
        ldc_labeled = pl.read_parquet(config.US_FILTER_LABELED_PARQUET)
        # Filter to dateline-sourced labels only (avoid circular comparison)
        dateline_labels_df = ldc_labeled.filter(pl.col("label_source") == "dateline").select(
            pl.col("id").cast(pl.Utf8).alias("ldc_id"),
            pl.col("us_label")
        )

    # Filter gold_df to rows with both alt_corpus_id and event_location present
    joinable = gold_df.filter(
        (pl.col("alt_corpus_id").is_not_null())
        & (pl.col("event_location").is_not_null())
    )

    if joinable.shape[0] == 0:
        return {"dateline_event_agreement": 0.0, "n": 0}

    # Join on alt_corpus_id (API gold) to ldc_id (dateline labels)
    joined = joinable.join(
        dateline_labels_df,
        left_on="alt_corpus_id",
        right_on="ldc_id",
        how="inner",
    )

    if joined.shape[0] == 0:
        return {"dateline_event_agreement": 0.0, "n": 0}

    # Compute event_location-derived US coding: event_location == "US" → True
    event_location_us = joined["event_location"] == "US"

    # Agreement: fraction where dateline us_label matches event_location US coding
    agreement = (joined["us_label"] == event_location_us).sum() / joined.shape[0]

    return {
        "dateline_event_agreement": float(agreement),
        "n": int(joined.shape[0]),
    }
