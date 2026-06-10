# pattern: Functional Core
"""Canonical metrics for the US binary classifier (PN task).

Mirrors the CCA metric set, using logits-space thresholds and PR-AUC
for class imbalance robustness.
"""

import keras


def make_us_metrics() -> list[keras.metrics.Metric]:
    """Canonical binary-classification metrics for the US head (logits space).

    Same set as the CCA head: thresholds at 0.0 because outputs are logits.
    F1 is computed post-hoc from precision/recall (Keras F1 needs prob thresholds).
    """
    return [
        keras.metrics.BinaryAccuracy(threshold=0.0),
        keras.metrics.Precision(thresholds=0.0, name="precision"),
        keras.metrics.Recall(thresholds=0.0, name="recall"),
        keras.metrics.AUC(curve="PR", from_logits=True, name="pr_auc"),
    ]
