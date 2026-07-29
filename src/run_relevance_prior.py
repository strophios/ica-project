# pattern: Imperative Shell
"""
Re-estimate the FLPU class prior for the immigrant-relevance population (DEDPUL).

The relevance analog of `run_cca_doca_prior.py`: trains a tiny labeled/unlabeled
classifier on cached CLS embeddings (labeled = US-restricted relevance positives,
unlabeled = the US-restricted background), feeds its P(unlabeled) scores into
DEDPUL's density-ratio EM, and reports pi_pos = 1 - alpha.

Differences from the CCA version:
  * Cache is `relevance_train` (266,018 rows; see `src.build_relevance_table`),
    not `train250k`.
  * Positives are `relevance/candidates.parquet` ids (immigrant-relevance
    descriptor candidates + hand-verified ICA anchors), not DoCA positives.
  * US restriction uses the FUSED gate (ML `us_logit` pass AND not-clearly-
    foreign), not the raw dateline/ML threshold alone -- reproduces
    `run_relevance.py`'s / `build_relevance_text_table.py`'s population
    derivation exactly (same fused-gate call, same "positives must also be
    US" restriction -- relevance descriptor-positives include genuinely
    foreign articles that the US gate correctly rejects, unlike DoCA events).
  * The ICA-eval holdout ids (`config.ICA_HOLDOUT_IDS`) are excluded from the
    population before splitting into labeled/unlabeled, mirroring
    `build_relevance_text_table.py`'s belt-and-suspenders holdout drop (the
    CCA prior script has no such holdout -- the relevance population needs it
    because the same hand-coded eval set backs the rel-vs-ICA comparisons in
    `scripts/eval_heads_own_terms.py` / `scripts/eval_rel_text_artifact.py`).

Run from project root:
    uv run python -m src.run_relevance_prior --suffix relevance_train --threshold 0.5
"""

from __future__ import annotations

import argparse
import json

import keras
import numpy as np
import polars as pl

import src.config as config
import src.prior_estimation.dedpul_em as dedpul_em
from src.build_cca_doca_table import label_and_restrict
from src.embed_corpus import load_cache
from src.preproc.us_location import apply_fused_us_gate, load_location_signals
from src.run_cca_doca_prior import train_lu_classifier

RELEVANCE_DIR = config.PROJECT_ROOT / "relevance"

keras.config.set_dtype_policy(config.DTYPE_POLICY)
keras.utils.set_random_seed(200)


def main(suffix="relevance_train", threshold=0.5, epochs=5, max_unl=40000):
    meta, cls = load_cache(config.CCA_EMBED_CACHE_DIR / suffix)
    positives = pl.read_parquet(RELEVANCE_DIR / "candidates.parquet")["id"].to_list()
    table = label_and_restrict(meta, positives, threshold)

    # Fused US gate (identical to run_relevance.py / build_relevance_text_table.py):
    # ML filter passes AND not clearly foreign.
    signals = load_location_signals(table["id"].to_list())
    table = table.join(signals, on="id", how="left").with_columns(
        pl.col("any_us").fill_null(False), pl.col("any_not_us").fill_null(False)
    )
    n_us_ml = int(table["us"].sum())
    table = apply_fused_us_gate(table)
    print(f"fused US gate: {int(table['us'].sum())}/{n_us_ml} of ML-gated kept "
          f"(clearly-foreign removed)")  # LOG

    # Relevance-specific: positives must ALSO be US (see run_relevance.py's
    # module docstring for the CCA-vs-relevance distinction here).
    n_pos_all = int((table["cca_label"] == 1).sum())
    table = table.with_columns(
        ((pl.col("cca_label") == 1) & pl.col("us")).cast(pl.Int8).alias("cca_label")
    )
    n_pos_us = int((table["cca_label"] == 1).sum())
    print(f"positives US-restricted: {n_pos_us}/{n_pos_all} kept")  # LOG

    # ICA-eval holdout exclusion (belt-and-suspenders, mirroring
    # build_relevance_text_table.py -- NOT present in run_cca_doca_prior.py,
    # added here because this population backs rel-vs-ICA eval comparisons).
    holdout_ids = pl.read_parquet(config.ICA_HOLDOUT_IDS)["id"].to_list()
    before = table.height
    table = table.filter(pl.col("id").is_in(holdout_ids).not_())
    n_holdout_dropped = before - table.height
    print(f"holdout dropped = {n_holdout_dropped} (pre-holdout population = {before})")  # LOG

    pos_rows = table.filter(pl.col("cca_label") == 1)["emb_row"].to_numpy()
    unl_rows = table.filter(
        (pl.col("cca_label") == 0) & pl.col("us")
    )["emb_row"].to_numpy()
    pos_feats = cls[pos_rows]
    unl_feats = cls[unl_rows]
    n_unl_full = len(unl_feats)
    # DEDPUL's density-ratio KDE is O(n^2) and stabilizes at tens of thousands of
    # points, so subsample the unlabeled (uniform -> preserves the positive
    # fraction DEDPUL estimates). All positives are kept.
    if max_unl is not None and n_unl_full > max_unl:
        rng = np.random.default_rng(200)
        unl_feats = unl_feats[rng.choice(n_unl_full, size=max_unl, replace=False)]
        print(f"subsampled unlabeled {n_unl_full} -> {max_unl} for DEDPUL KDE")  # LOG
    print(f"L/U: positives={len(pos_feats)} unlabeled(DEDPUL)={len(unl_feats)} "
          f"(full US pool={n_unl_full})")  # LOG

    model, X, y = train_lu_classifier(pos_feats, unl_feats, epochs=epochs)
    logits = model.predict(X, batch_size=512, verbose=0).reshape(-1)
    preds = 1.0 / (1.0 + np.exp(-logits))  # P(unlabeled)
    target = y  # 0 = positive, 1 = unlabeled

    diffs = dedpul_em.estimate_diff(preds, target, tune=True, kde_mode="prob")
    alpha, _ = dedpul_em.estimate_poster_em(diffs, preds, target)
    pi_pos = 1.0 - alpha
    naive = len(pos_feats) / (len(pos_feats) + n_unl_full)

    print(f"DEDPUL alpha (neg frac in U) = {alpha:.4f}  ->  pi_pos = {pi_pos:.4f}")  # LOG
    # The nnPU prior is the positive rate in the UNLABELED background (DEDPUL's
    # 1 - alpha), NOT bounded below by the labeled rate. The labeled rate here is
    # a sampling artifact (we force-include ALL positives + a fixed background
    # sample), so it is reported for context but is NOT a floor for pi.
    print(f"labeled rate n_pos/(n_pos+n_unl) = {naive:.4f}  "
          "(sampling artifact; NOT a floor for the nnPU prior)")  # LOG
    if not (0.0 < pi_pos < 1.0):
        print("  WARNING: pi_pos out of (0,1) — degenerate DEDPUL output; investigate.")  # LOG
    elif pi_pos < 0.002 or pi_pos > 0.5:
        print(f"  NOTE: pi_pos={pi_pos:.4f} is unusual — confirm L/U separability "
              "and robustness (re-run with a different seed / kde_mode).")  # LOG

    out = {
        "pi_pos": float(pi_pos), "alpha": float(alpha),
        "naive_labeled_rate": float(naive),
        "n_pos": int(len(pos_feats)), "n_unl": int(n_unl_full),
        "n_unl_dedpul": int(len(unl_feats)),
        "n_pos_all_candidates": n_pos_all,
        "n_holdout_dropped": n_holdout_dropped,
        "suffix": suffix, "threshold": threshold,
    }
    RELEVANCE_DIR.mkdir(parents=True, exist_ok=True)
    (RELEVANCE_DIR / "prior_estimate.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {RELEVANCE_DIR / 'prior_estimate.json'}: {out}")  # LOG


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DEDPUL prior re-estimation for the relevance population (features-mode).")
    ap.add_argument("--suffix", default="relevance_train", help="embedding cache subdir")
    ap.add_argument("--threshold", type=float, default=0.5, help="calibrated US probability threshold")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-unl", type=int, default=40000,
                    help="subsample unlabeled to this many for DEDPUL's O(n^2) KDE")
    args = ap.parse_args()
    main(suffix=args.suffix, threshold=args.threshold, epochs=args.epochs,
         max_unl=args.max_unl)
