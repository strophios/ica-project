"""
Shared metric configuration for the CCA classifier.

Tier 3 closeout — interim fix for I8 (the adversarial review's
finding that the metrics list was duplicated by hand between
`run_cca_classification.py` and `eval_cca_classifier.py`, exactly
the shape of train/eval coupling Tier 3 set out to fix).

The proper fix is to put metric specifications in `RunConfig` so
each training run records its own metric choices in the JSON
sidecar. That requires either serializing keras Metric instances
(some ceremony) or encoding metric specs as a small DSL. Neither
is warranted yet — we use the same metric set across runs, and
JSON-serializing keras Metrics is the kind of API surface we
should reach for only when we have an actual reason.

This module is the interim version: a single shared function that
returns the canonical metrics list. Both production scripts call
it. Drift is impossible because there's only one source.

When metric specs eventually land in RunConfig, this module can
either be retired (config drives metric construction directly) or
become the implementation of "default metrics for a binary head."

The module is also handy for external tooling (notebooks, scripts)
that want the same metrics — `from src.cca_metrics import
make_cca_metrics` and use the result.

See `docs/notes/tier3-design.md` Piece 3 closeout subsection for
the design framing, and `docs/notes/pinned-questions.md` for the
fuller question of metric configurations as a research dimension.
"""

from __future__ import annotations

import keras


def make_cca_metrics() -> list[keras.metrics.Metric]:
    """Construct the canonical metric set for a binary CCA-style
    classification head.

    A new list of fresh metric instances is returned on each call
    — Keras metrics are stateful, and head construction requires
    fresh instances per head (the head clones them with a head-name
    prefix; see `ClassificationHead.__init__`). Callers should not
    share the returned list across heads.

    Rationale for these specific metrics:
      - With a ~2% class prior and 50/50-weighted validation
        batches, BinaryAccuracy alone is misleading (a cautious
        model scores well by predicting negative).
      - Precision and Recall capture the classification behavior
        under imbalance.
      - PR-AUC (`AUC(curve="PR")`) summarizes the precision/recall
        trade-off across thresholds — particularly informative
        with rare positives.
      - All thresholds are 0.0 because the head outputs logits
        (sigmoid(0) = 0.5).
      - F1 is omitted because `keras.metrics.F1Score` requires a
        threshold in (0, 1] (probability-space output);
        compute it post-hoc from precision and recall.
    """
    return [
        keras.metrics.BinaryAccuracy(threshold=0.0),
        keras.metrics.Precision(thresholds=0.0, name="precision"),
        keras.metrics.Recall(thresholds=0.0, name="recall"),
        keras.metrics.AUC(curve="PR", from_logits=True, name="pr_auc"),
    ]
