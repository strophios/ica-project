# pattern: Imperative Shell
"""Fit + save Platt calibration for the relevance head.

Applies the trained relevance head to its natural-balance validation split
(drawn from the cached embeddings the same way training split them), fits
Platt on (val_logit, rel_label), reports ECE/Brier before vs after, and writes
the `*.calibration.json` sidecar — completing the artifact triple.

The fit is on the natural-balance val split (never rebalanced), so `B` maps to
the real prior rather than a rebalanced one; `fit_population` records the
distribution the scores are calibrated to. A large `|B|` is a diagnostic, not a
routine outcome — with train and val sharing a base rate it should sit near zero
(see docs/notes/calibration-notes.md).

Run from project root (after run_relevance):
    uv run python -m src.calibrate_relevance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import keras
import polars as pl

import src.config as config
from src.data_setup.data import create_relevance_data
from src.embed_corpus import load_cache
from src.validation.relevance_slice_eval import apply_relevance_model
from src.calibration.calibrator import PlattCalibrator
from src.calibration.report import calibration_report
from src.calibration.sidecar import save_calibration, calibration_path_for_weights

keras.config.set_dtype_policy(config.DTYPE_POLICY)
keras.utils.set_random_seed(200)


def main(suffix="relevance_train", weights_path=None,
         fit_population="relevance_train_val_natural_balance"):
    weights_path = (
        config.RELEVANCE_DOCA_WEIGHTS
        if weights_path is None
        else Path(weights_path)
    )

    meta, cls = load_cache(config.CCA_EMBED_CACHE_DIR / suffix)

    # Add labels and US restrictions to the metadata (same as run_relevance).
    # Load positives from the relevance candidates.
    RELEVANCE_DIR = config.PROJECT_ROOT / "relevance"
    positives = pl.read_parquet(RELEVANCE_DIR / "candidates.parquet")["id"].to_list()
    reliable_neg_ids = pl.read_parquet(RELEVANCE_DIR / "reliable_negatives.parquet")["id"].to_list()

    table = meta.with_columns(
        pl.col("id").is_in(positives).cast(pl.Int8).alias("cca_label"),
        (pl.col("us_logit") >= 0.5).alias("us"),
        pl.col("id").is_in(reliable_neg_ids).alias("reliable_neg"),
    )

    # Same split as training (seed 200), so "val" here is the val the model didn't fit.
    val_splits = create_relevance_data(table)["val"]
    # The val_splits is a dict with keys "pos", "neg", "unl". Concatenate them.
    val = pl.concat([val_splits["pos"], val_splits["neg"], val_splits["unl"]])
    val_feats = cls[val["emb_row"].to_numpy()]
    val_y = val["cca_label"].cast(pl.Int8).to_numpy()

    # Apply the trained head (features-mode) to the val embeddings -> logits.
    val_logits = apply_relevance_model(val_feats, weights_path=weights_path)

    cal = PlattCalibrator.fit(val_logits, val_y, fit_population=fit_population)

    # Report calibration before (raw sigmoid of the logit) vs after (Platt).
    raw_probs = 1.0 / (1.0 + np.exp(-val_logits))
    before = calibration_report(raw_probs, val_y)
    after = calibration_report(cal.transform(val_logits), val_y)

    save_calibration(cal, calibration_path_for_weights(weights_path))

    print(f"Platt fit on {cal.n} val rows (natural balance, rel_frac={float(val_y.mean()):.3f})")  # LOG
    print(f"  A={cal.A:.4f}  B={cal.B:.4f}  ({'LARGE |B| — investigate' if abs(cal.B) > 1.0 else 'B small, as expected'})")  # LOG
    print(f"  ECE   {before['ece']:.4f} -> {after['ece']:.4f}")  # LOG
    print(f"  Brier {before['brier']:.4f} -> {after['brier']:.4f}")  # LOG
    print(f"Saved {calibration_path_for_weights(weights_path)}")  # LOG
    return cal, before, after


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fit + save Platt calibration for the relevance head.")
    ap.add_argument("--suffix", default="relevance_train", help="embedding cache subdir")
    ap.add_argument("--out", default=None,
                    help="relevance weights path (default: relevance/relevance.weights.h5)")
    args = ap.parse_args()
    main(suffix=args.suffix, weights_path=args.out)
