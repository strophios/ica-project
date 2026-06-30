# pattern: Functional Core (metrics) + Imperative Shell (reporting)
"""DoCA-matched recall diagnostic (AC6.4).

Computes the recall of the US filter over DoCA-matched articles.

Module-level contract: The input `scored_df` must carry `doca_id` + `us_score` columns,
assembled by the operator via one of two paths:

1. **Direct LDC scoring (default, simplest):** Filter LDC labeled parquet to
   doca_id non-null, apply US model to those rows, attach doca_id + us_score.

2. **Via gold-set alt_corpus_id join (pre-1986 extension, operator-gated on
   gold set + trained model):** Apply US model to API-scored rows, join via
   gold_df's alt_corpus_id (API id ↔ LDC id) to attach LDC doca_id.

Both paths cast LDC id (Int64) to str at the join boundary to match the API
id convention (pinned in Task 1).

Caveat: DoCA = US protest events; the recall diagnostic is biased by
topic-skew (DoCA is non-representative). The operator-gated real run confirms
this is the right tradeoff for the use case (Tier 5 empirical handoff).
"""

from __future__ import annotations

import polars as pl


# Module-level caveat string (AC6.4: topic-skew caveat)
DOCA_TOPIC_SKEW_CAVEAT = (
    "⚠️  TOPIC-SKEW CAVEAT: DoCA is a curated collection of US protest events, "
    "not a representative random sample. High recall over DoCA-matched articles "
    "does not imply high recall over the full LDC corpus. This diagnostic is "
    "useful as a ceiling estimate and validation of model focus, but is biased "
    "toward the specific topics DoCA covers (civil disobedience, strikes, etc.)."
)


def doca_recall(scored_df: pl.DataFrame, threshold: float = 0.5) -> dict[str, float | int]:
    """Fraction of DoCA-matched articles the calibrated filter scores US.

    Args:
        scored_df: DataFrame with columns doca_id (str, nullable) and us_score
                   (float [0, 1]). Rows with doca_id=null are ignored.
        threshold: Decision threshold for positive prediction (default 0.5).

    Returns:
        Dict with keys:
        - recall: fraction of DoCA rows with us_score >= threshold
        - n: number of rows with doca_id non-null
    """
    # Filter to rows with doca_id non-null
    doca_rows = scored_df.filter(pl.col("doca_id").is_not_null())

    n_doca = doca_rows.shape[0]
    if n_doca == 0:
        return {"recall": 0.0, "n": 0}

    # Count rows scoring above threshold
    n_recalled = (doca_rows["us_score"] >= threshold).sum()
    recall = n_recalled / n_doca

    return {
        "recall": float(recall),
        "n": int(n_doca),
    }


class ThresholdPickResult:
    """Result of picking a threshold from the recall recipe.

    Attributes:
        threshold: The selected threshold (largest meeting target_recall, or
                   lowest if none qualify).
        qualified: Whether the selected threshold meets or exceeds target_recall.
    """

    def __init__(self, threshold: float, qualified: bool):
        self.threshold = threshold
        self.qualified = qualified

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ThresholdPickResult):
            return NotImplemented
        return self.threshold == other.threshold and self.qualified == other.qualified

    def __repr__(self) -> str:
        return f"ThresholdPickResult(threshold={self.threshold}, qualified={self.qualified})"


def pick_us_threshold(
    scored_df: pl.DataFrame,
    target_recall: float,
    thresholds: list[float],
) -> ThresholdPickResult:
    """Pick the largest threshold whose DoCA recall >= target_recall.

    Implements the CCA-consumer recall recipe from docs/notes/us-filter-threshold-recipe.md:
    evaluate doca_recall over a threshold grid, return the largest threshold
    whose recall meets the target. If none qualify, return the lowest threshold
    with qualified=False.

    Args:
        scored_df: DataFrame with columns doca_id (str, nullable) and us_score
                   (float [0, 1]). Rows with doca_id=null are ignored.
        target_recall: Target recall level (e.g., 0.98 to preserve 98% of
                       DoCA-matched articles).
        thresholds: List of thresholds to evaluate. Must be non-empty.
                    Typically sorted ascending for clarity, but order doesn't
                    affect the result.

    Returns:
        ThresholdPickResult with:
        - threshold: Largest threshold meeting target_recall, or lowest
                     threshold if none qualify.
        - qualified: True if the selected threshold achieves ≥ target_recall,
                     False otherwise.

    Raises:
        ValueError: If thresholds is empty.
    """
    if not thresholds:
        raise ValueError("thresholds must be non-empty")

    # Evaluate recall at each threshold
    recalls = {}
    for t in thresholds:
        result = doca_recall(scored_df, threshold=t)
        recalls[t] = result["recall"]

    # Find the largest threshold whose recall >= target_recall
    qualifying = [t for t in thresholds if recalls[t] >= target_recall]

    if qualifying:
        # Return the largest qualifying threshold
        selected_threshold = max(qualifying)
        return ThresholdPickResult(threshold=selected_threshold, qualified=True)
    else:
        # None qualify: return the lowest threshold with qualified=False
        selected_threshold = min(thresholds)
        return ThresholdPickResult(threshold=selected_threshold, qualified=False)
