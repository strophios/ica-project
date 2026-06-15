# pattern: Imperative Shell
"""
Re-estimate the FLPU class prior for the DoCA/API/US-restricted population (DEDPUL).

Trains a tiny labeled/unlabeled classifier on the CACHED CLS embeddings (labeled =
DoCA positives, unlabeled = US-restricted background), feeds its P(unlabeled)
scores into DEDPUL's density-ratio EM, and reports pi_pos = 1 - alpha (alpha =
DEDPUL's estimated fraction of negatives in the unlabeled pool). Mirrors
run_prior_estimate.py but features-mode.

Deviation from the original L/U classifier: it rebalanced (ratio-batch + class
weights) because an end-to-end backbone needed it; here the backbone is frozen and
features are cached, so we train on the NATURAL mixture, giving a properly
calibrated P(unlabeled) -- which is what DEDPUL's density-ratio assumes.

Run from project root:
    uv run python -m src.run_cca_doca_prior --suffix train250k --threshold 0.0
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import keras
import polars as pl

import src.config as config
from src.embed_corpus import load_cache
from src.build_cca_doca_table import label_and_restrict
import src.prior_estimation.dedpul_em as dedpul_em

keras.config.set_dtype_policy(config.DTYPE_POLICY)
keras.utils.set_random_seed(200)


def train_lu_classifier(pos_feats, unl_feats, epochs=5, batch_size=256):
    """Tiny L/U classifier on cached features. Label 1 = unlabeled, 0 = positive,
    so sigmoid(logit) = P(unlabeled) directly (estimate_diff's `preds`/`target`
    convention). Trained on the natural mixture (no rebalancing)."""
    X = np.concatenate([pos_feats, unl_feats], axis=0).astype("float32")
    y = np.concatenate(
        [np.zeros(len(pos_feats)), np.ones(len(unl_feats))]
    ).astype("float32")
    model = keras.Sequential([
        keras.layers.Input(shape=(X.shape[1],)),
        keras.layers.Dense(1, activation=None),
    ])
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2)
    return model, X, y


def main(suffix="train250k", threshold=0.0, epochs=5):
    meta, cls = load_cache(config.CCA_EMBED_CACHE_DIR / suffix)
    positives = pl.read_parquet(config.CCA_DOCA_POSITIVES)["id"].to_list()
    table = label_and_restrict(meta, positives, threshold)

    pos_rows = table.filter(pl.col("cca_label") == 1)["emb_row"].to_numpy()
    unl_rows = table.filter(
        (pl.col("cca_label") == 0) & pl.col("us")
    )["emb_row"].to_numpy()
    pos_feats, unl_feats = cls[pos_rows], cls[unl_rows]
    print(f"L/U: positives={len(pos_feats)} unlabeled-US={len(unl_feats)}")  # LOG

    model, X, y = train_lu_classifier(pos_feats, unl_feats, epochs=epochs)
    logits = model.predict(X, batch_size=512, verbose=0).reshape(-1)
    preds = 1.0 / (1.0 + np.exp(-logits))  # P(unlabeled)
    target = y  # 0 = positive, 1 = unlabeled

    diffs = dedpul_em.estimate_diff(preds, target, tune=True, kde_mode="prob")
    alpha, _ = dedpul_em.estimate_poster_em(diffs, preds, target)
    pi_pos = 1.0 - alpha
    naive = len(pos_feats) / (len(pos_feats) + len(unl_feats))

    print(f"DEDPUL alpha (neg frac in U) = {alpha:.4f}  ->  pi_pos = {pi_pos:.4f}")  # LOG
    print(f"naive labeled rate = {naive:.4f}  (sanity floor; true pi >= this)")  # LOG
    if not (0.0 < pi_pos < 1.0) or pi_pos < naive * 0.5:
        print("  WARNING: pi_pos is implausible vs the naive rate — investigate "
              "before using (L/U separability, KDE bandwidth, label leakage).")  # LOG

    out = {
        "pi_pos": float(pi_pos), "alpha": float(alpha),
        "naive_labeled_rate": float(naive),
        "n_pos": int(len(pos_feats)), "n_unl": int(len(unl_feats)),
        "suffix": suffix, "threshold": threshold,
    }
    config.CCA_DOCA_DIR.mkdir(parents=True, exist_ok=True)
    (config.CCA_DOCA_DIR / "prior_estimate.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {config.CCA_DOCA_DIR / 'prior_estimate.json'}: {out}")  # LOG


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DEDPUL prior re-estimation (features-mode).")
    ap.add_argument("--suffix", default="train250k", help="embedding cache subdir")
    ap.add_argument("--threshold", type=float, default=0.0, help="US logit threshold")
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()
    main(suffix=args.suffix, threshold=args.threshold, epochs=args.epochs)
