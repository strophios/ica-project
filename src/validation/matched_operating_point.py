# pattern: Functional Core
"""Matched-operating-point comparison across models on different score scales.

Built for the US-head validate-before-swap comparison (scripts/eval_us_retrain.py):
the current production head emits Platt-calibrated probabilities, the retrain
candidate emits raw (uncalibrated -- calibration is deferred to the swap phase)
logits. Comparing "recall at threshold 0.5" across the two is meaningless since
0.5 means different things on each scale. `recall_at_matched_precision` instead
asks "at the point on THIS model's own precision-recall curve closest to a given
precision, what recall does it get" -- valid for any monotonic score, calibrated
or not.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score


def recall_at_matched_precision(
    scores: np.ndarray, labels: np.ndarray, target_precision: float
) -> dict:
    """Threshold/precision/recall at the operating point closest to `target_precision`.

    Scans `sklearn.metrics.precision_recall_curve` (thresholds ascending) and
    picks the point whose achieved precision is closest to `target_precision`.
    `target_precision` is not always exactly achievable (e.g. above the curve's
    max precision) -- the returned `precision` shows how close the match actually
    landed, so callers can judge whether the comparison is meaningful.

    `scores` need only be monotonically related to model confidence WITHIN this
    one array; no assumption of a shared scale across different models' scores.

    Raises ValueError on mismatched/empty inputs or an out-of-range target.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).astype(bool).reshape(-1)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(
            f"scores/labels length mismatch: {scores.shape[0]} vs {labels.shape[0]}"
        )
    if scores.shape[0] == 0:
        raise ValueError("scores/labels must be non-empty")
    if not (0.0 <= target_precision <= 1.0):
        raise ValueError(f"target_precision must be in [0,1]; got {target_precision}")

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # precision_recall_curve appends a final (precision=1, recall=0) sentinel point
    # with NO corresponding threshold (len(thresholds) == len(precision) - 1).
    # Drop it so every remaining candidate has a real, usable threshold.
    precision, recall = precision[:-1], recall[:-1]
    if precision.size == 0:
        raise ValueError("no thresholded operating points available (degenerate input)")

    idx = int(np.argmin(np.abs(precision - target_precision)))
    return {
        "target_precision": float(target_precision),
        "threshold": float(thresholds[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
    }


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """`roc_auc_score` with the bool cast callers otherwise repeat everywhere."""
    return float(
        roc_auc_score(np.asarray(labels).astype(bool), np.asarray(scores, dtype=np.float64))
    )
