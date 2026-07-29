# pattern: Imperative Shell
"""Per-head "own terms" evaluation on the hand-coded ICA eval set.

Each component head is scored against its OWN hand-coded dimension
(us_event, cca_event, immig_relevant) rather than the composed ICA label:

- US head   (calibrated) vs us_event
- CCA head  (calibrated) vs cca_event
- rel head  (calibrated) vs immig_relevant

Scoring recipe mirrors src/fit_fusion.py step 1-2 exactly (same cache, same
helpers, same calibrators), so numbers are comparable to the fusion/memo run.
Also reports ROC-AUC vs ica_event per head as a bridge to the memo's
decomposed table (ml_memo/ica_model_state_2026-06.md).

Caveat carried into the output: the eval set is boundary-enriched FOR ICA
(strata on cca x relevance score bands + DoCA anchors), so per-dimension base
rates are population-specific, not corpus rates.

Run from project root: uv run python -m scripts.eval_heads_own_terms
(module form required — `python scripts/...` puts scripts/ on sys.path,
breaking the src.* imports)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import src.config as config
from src.calibration.report import calibration_report
from src.calibration.sidecar import calibration_path_for_weights, load_calibration
from src.embed_corpus import load_cache
from src.validation.cca_slice_eval import apply_cca_model
from src.validation.relevance_slice_eval import apply_relevance_model
from src.validation.slice_eval import apply_us_model

EVAL_CSV = config.PROJECT_ROOT / "validation" / "ica_coding_template_coded.csv"
THRESHOLDS = [0.3, 0.5, 0.7]
# The cache this eval joins the eval-set ids against by default -- matches
# fit_fusion.py's Step 1 (see module docstring). A tuned comparison run passes
# a matching `*_tuned` cache via --cache-suffix so CCA/rel features (and the
# US head's headline/lead text, which is cache-independent) line up with the
# tuned weights under evaluation.
DEFAULT_CACHE_SUFFIX = "relevance_train"
DEFAULT_OUT = config.CCA_DOCA_DIR / "experiments" / "eval_heads_own_terms.json"


def head_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    cal = calibration_report(p, y, n_bins=15)
    operating_points = []
    for t in THRESHOLDS:
        pred = p >= t
        operating_points.append(
            {
                "threshold": t,
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "flagged_frac": float(pred.mean()),
            }
        )
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ece": float(cal["ece"]),
        "brier": float(cal["brier"]),
        "operating_points": operating_points,
    }


def main(
    cca_weights: Path | str | None = None,
    rel_weights: Path | str | None = None,
    us_weights: Path | str | None = None,
    cache_suffix: str = DEFAULT_CACHE_SUFFIX,
    out_path: Path | str | None = None,
) -> dict:
    """Score all three heads' own-terms ROC/PR-AUC/calibration on the eval set.

    Defaults reproduce the production own-terms eval byte-for-byte. Pass
    `cca_weights`/`rel_weights`/`us_weights` (each falls back independently to
    its production default) plus a matching `cache_suffix` to run the SAME
    eval against a tuned artifact for a negative-transfer comparison; pass
    `out_path` so the tuned run doesn't overwrite the production JSON.
    """
    cca_weights = Path(cca_weights) if cca_weights is not None else config.CCA_DOCA_WEIGHTS
    rel_weights = Path(rel_weights) if rel_weights is not None else config.RELEVANCE_DOCA_WEIGHTS
    us_weights = Path(us_weights) if us_weights is not None else config.US_FILTER_FULL_WEIGHTS
    out_path = Path(out_path) if out_path is not None else DEFAULT_OUT

    eval_df = pl.read_csv(EVAL_CSV)
    n_raw = eval_df.shape[0]

    # Join to the embed cache used by fit_fusion (drives CCA/rel scoring)
    meta_full, cls = load_cache(config.CCA_EMBED_CACHE_DIR / cache_suffix)
    meta_by_id = meta_full.with_columns(
        pl.col("id").cast(pl.Utf8).alias("id_str")
    ).select(["id_str", "emb_row"])
    eval_df = eval_df.with_columns(pl.col("id").cast(pl.Utf8).alias("id_str")).join(
        meta_by_id, on="id_str", how="left"
    )
    n_missing_emb = eval_df.filter(pl.col("emb_row").is_null()).shape[0]
    eval_df = eval_df.filter(pl.col("emb_row").is_not_null())
    print(
        f"rows: {n_raw} coded, {n_missing_emb} missing embeddings, {eval_df.shape[0]} scored"
    )

    features = cls[eval_df["emb_row"].to_numpy().astype(int)]

    # --- score all three calibrated heads (same recipe as fit_fusion.py) ---
    cca_logits = apply_cca_model(features, weights_path=cca_weights)
    cca_cal = load_calibration(calibration_path_for_weights(cca_weights))
    p_cca = cca_cal.transform(cca_logits)

    rel_logits = apply_relevance_model(features, weights_path=rel_weights)
    rel_cal = load_calibration(calibration_path_for_weights(rel_weights))
    p_rel = rel_cal.transform(rel_logits)

    texts = [
        f"{h if h else ''}</s>{lead if lead else ''}"
        for h, lead in zip(
            eval_df["headline"].to_list(), eval_df["lead_paragraph"].to_list()
        )
    ]
    # apply_us_model rebuilds a token-mode backbone from the US sidecar's own
    # backbone_weights_path -- a tuned us_weights sidecar (recorded via
    # run_us_features.py --backbone-weights) re-embeds through the matching
    # tuned backbone automatically; no extra knob needed here.
    p_us = apply_us_model(texts, weights_path=us_weights, skip_mismatch=True)

    for name, p in [("us", p_us), ("cca", p_cca), ("rel", p_rel)]:
        assert np.isfinite(p).all(), f"non-finite {name} probabilities"

    heads = {
        "us": (p_us, "us_event"),
        "cca": (p_cca, "cca_event"),
        "rel": (p_rel, "immig_relevant"),
    }

    ica = eval_df["ica_event"].to_numpy()
    results: dict = {
        "eval_csv": str(EVAL_CSV),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_coded_rows": n_raw,
        "n_missing_embeddings": n_missing_emb,
        "n_scored": int(eval_df.shape[0]),
        "cache_suffix": cache_suffix,
        "weights": {
            "cca": str(cca_weights),
            "rel": str(rel_weights),
            "us": str(us_weights),
        },
        "note": (
            "Eval set is boundary-enriched for ICA (cca x relevance score-band strata "
            "+ DoCA anchors), so per-dimension base rates and precision are "
            "population-specific, not corpus rates. Scoring recipe identical to "
            "fit_fusion.py (relevance_train cache; calibrated heads; US text-mode)."
        ),
        "heads": {},
    }

    for name, (p, label_col) in heads.items():
        y_raw = eval_df[label_col].to_numpy()
        mask = ~pl.Series(y_raw).is_null().to_numpy()
        y = y_raw[mask].astype(bool)
        m = head_metrics(p[mask], y)
        m["label"] = label_col

        # bridge to the memo's decomposed vs-ICA table
        ica_mask = mask & ~pl.Series(ica).is_null().to_numpy()
        m["roc_auc_vs_ica"] = float(
            roc_auc_score(ica[ica_mask].astype(bool), p[ica_mask])
        )
        results["heads"][name] = m

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}\n")

    hdr = f"{'head':<5} {'label':<15} {'n':>5} {'base':>6} {'ROC':>6} {'PR-AUC':>7} {'ECE':>6} {'Brier':>6} {'vsICA':>6}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in results["heads"].items():
        print(
            f"{name:<5} {m['label']:<15} {m['n']:>5} {m['base_rate']:>6.2f} "
            f"{m['roc_auc']:>6.3f} {m['pr_auc']:>7.3f} {m['ece']:>6.3f} "
            f"{m['brier']:>6.3f} {m['roc_auc_vs_ica']:>6.3f}"
        )
    print()
    for name, m in results["heads"].items():
        for op in m["operating_points"]:
            print(
                f"{name} @ {op['threshold']:.1f}: P={op['precision']:.3f} "
                f"R={op['recall']:.3f} F1={op['f1']:.3f} flagged={op['flagged_frac']:.2f}"
            )
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Per-head own-terms eval on the hand-coded ICA eval set."
    )
    ap.add_argument("--cca-weights", default=None, help="default: CCA_DOCA_WEIGHTS")
    ap.add_argument("--rel-weights", default=None, help="default: RELEVANCE_DOCA_WEIGHTS")
    ap.add_argument("--us-weights", default=None, help="default: US_FILTER_FULL_WEIGHTS")
    ap.add_argument(
        "--cache-suffix", default=DEFAULT_CACHE_SUFFIX,
        help="embed cache subdir the eval-set ids are joined against "
             "(default: relevance_train)",
    )
    ap.add_argument(
        "--out", default=None,
        help="output JSON path (default: cca_doca/experiments/eval_heads_own_terms.json "
             "-- pass a distinct path for a tuned-artifact run so it doesn't overwrite "
             "the production comparison)",
    )
    args = ap.parse_args()
    main(
        cca_weights=args.cca_weights,
        rel_weights=args.rel_weights,
        us_weights=args.us_weights,
        cache_suffix=args.cache_suffix,
        out_path=args.out,
    )
