# pattern: Imperative Shell (apply_cca_model loads + runs a model; evaluate_cca_slice is pure)
"""Apply the trained CCA classifier to cached embeddings + slice evaluation.

Features-mode counterpart of `slice_eval.py` (which serves the US filter on raw
text). `apply_cca_model` scores cached CLS vectors with the DoCA-trained CCA head;
`evaluate_cca_slice` computes precision/recall/F1 of `cca_event` labels against a
score threshold. Parallels `slice_eval.evaluate_slice`.

Note on metrics from a score-stratified gold set: precision at a threshold is
unbiased (it is conditional on score >= T, and we sample within score strata).
Recall computed here is biased when the gold set oversamples high scores -- use
`doca_recall` (recall over known DoCA positives) as the trustworthy recall proxy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

import src.config as config
import src.cca_config as cca_config
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_feature_inference_model


def apply_cca_model(
    features: np.ndarray,
    weights_path: Path | str = config.CCA_DOCA_WEIGHTS,
) -> np.ndarray:
    """Score cached CLS vectors with the trained CCA head (Pattern 2 reload).

    Loads the RunConfig sidecar, builds a fresh features-mode inference model,
    loads weights by structure, and returns raw logits (shape (n,)). The head
    outputs logits; threshold at 0.0 for prob 0.5, or sigmoid for [0,1] scores.
    """
    weights_path = Path(weights_path)
    run_config = cca_config.RunConfig.from_json(
        cca_config.config_path_for_weights(weights_path)
    )
    head_cfg = run_config.heads[0]
    head = ClassificationHead(hidden_dim=head_cfg.hidden_dim, name=head_cfg.name)
    model = build_feature_inference_model(
        {head_cfg.name: head}, hidden_dim=head_cfg.hidden_dim
    )
    model.load_weights(str(weights_path), skip_mismatch=False)

    logits = model.predict({"features": features}, batch_size=512, verbose=0)
    if isinstance(logits, dict):
        logits = logits[head_cfg.name]
    logits = np.asarray(logits).reshape(-1)
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite CCA logits produced by model")
    return logits


def evaluate_cca_slice(
    gold_df: pl.DataFrame,
    threshold: float = 0.0,
    score_col: str = "cca_logit",
) -> dict[str, float | int]:
    """Precision/recall/F1 of `cca_event` vs `score_col >= threshold`.

    Rows with null `cca_event` are excluded. Mirrors `slice_eval.evaluate_slice`.
    Precision is the trustworthy number from a score-stratified gold set; treat
    recall as biased there (see module docstring).
    """
    coded = gold_df.filter(pl.col("cca_event").is_not_null())
    predictions = (coded[score_col] >= threshold).cast(pl.Boolean)
    labels = coded["cca_event"]

    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    tp = int((predictions & labels).sum())
    fp = int((predictions & ~labels).sum())
    fn = int((~predictions & labels).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if (precision == 0 and recall == 0) else (
        2 * precision * recall / (precision + recall)
    )
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "n_pos": n_pos, "n_neg": n_neg, "threshold": float(threshold),
    }


def recall_at_thresholds(
    scores: np.ndarray, thresholds: tuple[float, ...]
) -> list[dict[str, float | int]]:
    """Recall of a POSITIVE-ONLY score array at each threshold (fraction >= t).

    Used for the held-out DoCA-positive recall proxy: every score is a known
    positive, so recall = fraction at or above the threshold. Leakage-honest only
    if `scores` are test-split positives (never trained on).
    """
    scores = np.asarray(scores).reshape(-1)
    n = int(scores.size)
    return [
        {"threshold": float(t),
         "recall": float((scores >= t).mean()) if n else 0.0, "n": n}
        for t in thresholds
    ]


def band_ipw_weights(
    gold_band_counts: dict[str, int],
    corpus_band_counts: dict[str, int],
) -> dict[str, float]:
    """Inverse-probability weight per score band: `corpus_n / gold_n`.

    The gold set is sampled within score-band strata at unequal rates (the high
    band is oversampled so precision is estimable). To recover a corpus
    operating-point estimate, each coded row is weighted by the inverse of its
    band's sampling probability -- equivalently `corpus_band_size /
    gold_band_size` (a Horvitz-Thompson / IPW weight). Bands cancel: a uniform
    sampling rate yields equal weights, and the reweighted metrics then equal the
    raw ones.

    Raises ValueError if a coded band has non-positive gold count, or is absent /
    empty in the corpus counts (cannot have sampled from an empty stratum).
    """
    weights: dict[str, float] = {}
    for band, gold_n in gold_band_counts.items():
        if gold_n <= 0:
            raise ValueError(f"band {band!r} has non-positive gold count {gold_n}")
        corpus_n = corpus_band_counts.get(band, 0)
        if corpus_n <= 0:
            raise ValueError(
                f"band {band!r} coded but absent/empty in corpus counts "
                f"(corpus_n={corpus_n})"
            )
        weights[band] = corpus_n / gold_n
    return weights


def evaluate_cca_slice_weighted(
    gold_df: pl.DataFrame,
    threshold: float = 0.0,
    score_col: str = "cca_logit",
    weight_col: str = "ipw",
) -> dict[str, float | int]:
    """IPW-reweighted precision/recall/F1 -- the corpus operating-point estimate.

    Same shape as `evaluate_cca_slice`, but each row contributes its `weight_col`
    mass: precision = Σw·tp / Σw·(tp+fp), recall = Σw·tp / Σw·(tp+fn). Unlike the
    raw version, recall is unbiased on a score-stratified gold set. The UNWEIGHTED
    tp/fp/fn supports are returned too: a reweighted estimate over few coded rows
    in a heavily-weighted band is high-variance, so callers must show the support
    to flag unreliable cells (e.g. a low band weighted ~1000x with ~5 positives).
    """
    coded = gold_df.filter(pl.col("cca_event").is_not_null())
    pred = coded[score_col] >= threshold
    y = coded["cca_event"]
    w = coded[weight_col].cast(pl.Float64)

    wtp = float((w * (pred & y).cast(pl.Float64)).sum())
    wfp = float((w * (pred & ~y).cast(pl.Float64)).sum())
    wfn = float((w * (~pred & y).cast(pl.Float64)).sum())

    precision = wtp / (wtp + wfp) if (wtp + wfp) > 0 else 0.0
    recall = wtp / (wtp + wfn) if (wtp + wfn) > 0 else 0.0
    f1 = 0.0 if (precision == 0 and recall == 0) else (
        2 * precision * recall / (precision + recall)
    )
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "threshold": float(threshold),
        "support_tp": int((pred & y).sum()),
        "support_fp": int((pred & ~y).sum()),
        "support_fn": int((~pred & y).sum()),
    }
