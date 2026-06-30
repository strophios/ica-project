# pattern: Imperative Shell
"""Fit + save IPW-weighted Platt calibration for a CCA model.

The CCA gold set is score-stratified (the high band is oversampled), so an unweighted
Platt fit would calibrate to the gold set's ~34% positive rate, not the corpus ~2%. We
fit Platt with the SAME inverse-probability weights the precision reweighting uses
(corpus_band / gold_band), so the calibrated probability maps to the corpus base rate.
`fit_population` records that the calibration is corpus-proportioned via IPW.

Read the calibrated value as a *ranking* probability: CCA scores are PU/nnPU logits, so
the number is "P(collective action) relative to the DoCA-labeled positives, in corpus
proportion," not a clean physical event probability. This is a multi-head-future
investment — the memo operates on PR-curve thresholds, and the consumer that needs
calibrated probabilities is the eventual probabilistic combination of head scores.

Run from project root (after the gold set is coded):
    uv run python -m src.calibrate_cca
    uv run python -m src.calibrate_cca --weights ../cca_doca/cca_doca_street.weights.h5 --form-filter any_street
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import keras
import polars as pl

import src.config as config
from src.embed_corpus import load_cache
from src.validation.cca_slice_eval import band_ipw_weights, restrict_label_to_form
from src.validation.run_cca_eval import (
    _FORM_EVENT_TYPE,
    _attach_gold_scores,
    _corpus_band_counts,
    _gold_band_counts,
    _parse_cca_event,
)
from src.calibration.calibrator import PlattCalibrator
from src.calibration.sidecar import calibration_path_for_weights, save_calibration

keras.config.set_dtype_policy(config.DTYPE_POLICY)
keras.utils.set_random_seed(200)


def main(weights_path=None, suffix="train250k", coded_path=None, form_filter=None):
    weights_path = config.CCA_DOCA_WEIGHTS if weights_path is None else Path(weights_path)
    coded_path = (
        config.VALIDATION_DIR / "cca_coding_first500_coded.csv"
        if coded_path is None else Path(coded_path)
    )
    fit_population = (
        f"cca_gold_ipw_corpus_weighted[{_FORM_EVENT_TYPE[form_filter]}]"
        if form_filter else "cca_gold_ipw_corpus_weighted[all_forms]"
    )

    meta, cls = load_cache(config.CCA_EMBED_CACHE_DIR / suffix)
    coded = _parse_cca_event(pl.read_csv(coded_path))
    coded = _attach_gold_scores(coded, meta, cls, weights_path)
    if form_filter:
        coded = restrict_label_to_form(coded, _FORM_EVENT_TYPE[form_filter])

    # IPW weights (corpus_band / gold_band) — same as the precision reweighting.
    weights = band_ipw_weights(_gold_band_counts(coded), _corpus_band_counts())
    coded = coded.with_columns(
        pl.col("sample_stratum").replace_strict(weights).alias("ipw")
    ).filter(pl.col("cca_event").is_not_null())

    logits = coded["cca_logit"].to_numpy()
    y = coded["cca_event"].cast(pl.Int8).to_numpy()
    w = coded["ipw"].to_numpy()
    cal = PlattCalibrator.fit(logits, y, fit_population=fit_population, sample_weight=w)
    save_calibration(cal, calibration_path_for_weights(weights_path))

    print(f"CCA Platt (IPW-weighted) for {weights_path.name}: "
          f"A={cal.A:.4f}  B={cal.B:.4f}  n={cal.n}")  # LOG
    print(f"  fit_population={fit_population}")  # LOG
    # Calibration check: the IPW-weighted mean calibrated prob should match the
    # IPW-weighted positive rate (the corpus base rate) if calibrated in aggregate.
    wy = float((w * y).sum() / w.sum())
    wp = float((w * cal.transform(logits)).sum() / w.sum())
    print(f"  IPW base rate: observed={wy:.4f}  calibrated-mean={wp:.4f}  "
          f"(close => aggregate-calibrated)  |  P(CCA|logit=0)={1 / (1 + np.exp(-cal.B)):.3f}")  # LOG
    grid = np.array([0.0, 1.0, 1.5, 2.0])
    probs = cal.transform(grid)
    print("  calibrated P:  " + "  ".join(
        f"logit{t:+.1f}->{p:.3f}" for t, p in zip(grid, probs)))  # LOG
    print(f"Saved {calibration_path_for_weights(weights_path)}")  # LOG


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fit IPW-weighted Platt calibration for a CCA model.")
    ap.add_argument("--weights", default=None, help="CCA weights (default: CCA_DOCA_WEIGHTS)")
    ap.add_argument("--suffix", default="train250k", help="embedding cache subdir")
    ap.add_argument("--coded", default=None, help="coded gold CSV (default: first500)")
    ap.add_argument("--form-filter", default=None,
                    choices=["any_street", "any_boycott", "any_conventional", "any_lawsuit"],
                    help="match the model's training form restriction")
    args = ap.parse_args()
    main(weights_path=args.weights, suffix=args.suffix, coded_path=args.coded,
         form_filter=args.form_filter)
