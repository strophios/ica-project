# pattern: Imperative Shell
"""Fit + save Platt calibration for the features-mode US filter.

Applies the trained US head to its natural-balance validation split (drawn from the
cached embeddings the same way training split them), fits Platt on
(val_logit, us_label), reports ECE/Brier before vs after, and writes the
`*.calibration.json` sidecar — the third leg of the artifact triple.

The fit is on the natural-balance val split (never rebalanced), so `B` maps to the
real LDC-labeled prior rather than a rebalanced one; `fit_population` records the
distribution the scores are calibrated to. A large `|B|` is a diagnostic, not a
routine outcome — with train and val sharing a base rate it should sit near zero
(see docs/notes/calibration-notes.md).

Run from project root (after run_us_features):
    uv run python -m src.calibrate_us_filter
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import keras
import polars as pl

import src.config as config
import src.us_config as us_config
from src.us_config import config_path_for_weights
from src.data_setup.data import create_us_filter_data
from src.embed_corpus import load_cache
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_feature_inference_model
from src.calibration.calibrator import PlattCalibrator
from src.calibration.report import calibration_report
from src.calibration.sidecar import save_calibration, calibration_path_for_weights

keras.config.set_dtype_policy(config.DTYPE_POLICY)
keras.utils.set_random_seed(200)


def main(suffix="us_train_ldc", weights_path=None,
         fit_population="ldc_labeled_val_natural_balance"):
    weights_path = (
        config.US_FILTER_DIR / "us_classifier_full.weights.h5"
        if weights_path is None
        else Path(weights_path)
    )
    run_config = us_config.UsRunConfig.from_json(config_path_for_weights(weights_path))
    head_cfg = run_config.head

    meta, cls = load_cache(config.CCA_EMBED_CACHE_DIR / suffix)
    # Same split as training (seed 200), so "val" here is the val the model didn't fit.
    val = create_us_filter_data(meta)["val"]
    val_feats = cls[val["emb_row"].to_numpy()]
    val_y = val["us_label"].cast(pl.Int8).to_numpy()

    # Apply the trained head (features-mode) to the val embeddings -> logits.
    us_head = ClassificationHead(hidden_dim=head_cfg.hidden_dim, name=head_cfg.name)
    inference = build_feature_inference_model(
        {head_cfg.name: us_head}, hidden_dim=cls.shape[1]
    )
    inference.load_weights(str(weights_path), skip_mismatch=False)
    val_logits = inference.predict(
        {"features": val_feats}, verbose=0
    )[head_cfg.name].reshape(-1)

    cal = PlattCalibrator.fit(val_logits, val_y, fit_population=fit_population)

    # Report calibration before (raw sigmoid of the logit) vs after (Platt).
    raw_probs = 1.0 / (1.0 + np.exp(-val_logits))
    before = calibration_report(raw_probs, val_y)
    after = calibration_report(cal.transform(val_logits), val_y)

    save_calibration(cal, calibration_path_for_weights(weights_path))

    print(f"Platt fit on {cal.n} val rows (natural balance, us_frac={float(val_y.mean()):.3f})")  # LOG
    print(f"  A={cal.A:.4f}  B={cal.B:.4f}  ({'LARGE |B| — investigate' if abs(cal.B) > 1.0 else 'B small, as expected'})")  # LOG
    print(f"  ECE   {before['ece']:.4f} -> {after['ece']:.4f}")  # LOG
    print(f"  Brier {before['brier']:.4f} -> {after['brier']:.4f}")  # LOG
    print(f"Saved {calibration_path_for_weights(weights_path)}")  # LOG
    return cal, before, after


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fit + save Platt calibration for the US filter.")
    ap.add_argument("--suffix", default="us_train_ldc", help="embedding cache subdir")
    ap.add_argument("--out", default=None,
                    help="US filter weights path (default: us_filter/us_classifier_full.weights.h5)")
    args = ap.parse_args()
    main(suffix=args.suffix, weights_path=args.out)
