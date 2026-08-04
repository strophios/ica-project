# pattern: Imperative Shell (one pure helper; joins + table assembly in main)
"""Precision-recall operating-point tables for the August memo (2026-08-04).

All scores come from the CLUSTER-applied candidates files (true-math scoring),
so precision and recall columns share one scoring basis:

  1. CCA level (1960-1995): IPW-reweighted corpus precision (500-row gold set,
     corpus_band/gold_band weights — the June machinery) vs DoCA-anchor recall,
     across calibrated cca_score thresholds.
     Caveat: DoCA anchors include the CCA head's training positives, so recall
     is an operational retrieve-the-known-events number, not de-novo recall.
  2. ICA level (1960-1995): eval-set precision (boundary-enriched — UPPER BOUND
     on corpus precision) vs recall of the 552-event ICA-subset-of-DoCA
     (event_type4 domains), across ica_score thresholds. Anchor recall is split
     held-out (196 events; excluded from all training) vs in-training (356).
  3. Forward yield (1996-2025): candidate volumes at the same ica_score
     operating points, per era (2025 = abstract-register channel, flagged).

Run: uv run python -m scripts.memo_pr_tables
Output: cca_doca/experiments/memo_pr_tables.json + printed tables.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

import src.config as config
from src.validation.cca_slice_eval import band_ipw_weights
from src.validation.run_cca_eval import (
    _corpus_band_counts,
    _gold_band_counts,
    _parse_cca_event,
)

CCA_THRESHOLDS = [0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
ICA_THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 0.7]


def recall_at_threshold(anchor_scores: np.ndarray, n_anchors: int, t: float) -> float:
    """Fraction of ALL anchors whose (found) score is >= t; missing anchors
    count as misses via the denominator."""
    if n_anchors == 0:
        return 0.0
    return float((anchor_scores >= t).sum() / n_anchors)


def main() -> dict:
    cand_dir = config.ICA_CANDIDATES_DIR
    exp_dir = config.CCA_DOCA_DIR / "experiments"
    hist = pl.read_parquet(cand_dir / "api_1960_1995.parquet")
    fwd = pl.read_parquet(cand_dir / "api_1996_2025.parquet").with_columns(
        pl.col("year").cast(pl.Int64)
    )
    results: dict = {}

    # --- 1. CCA level: IPW precision vs DoCA recall ------------------------
    coded = _parse_cca_event(
        pl.read_csv(config.PROJECT_ROOT / "validation" / "cca_coding_first500_coded.csv",
                    infer_schema_length=600)
    ).filter(pl.col("cca_event").is_not_null())
    weights = band_ipw_weights(_gold_band_counts(coded), _corpus_band_counts())
    coded = coded.with_columns(
        pl.col("sample_stratum").replace_strict(weights).alias("ipw")
    )
    # True-math calibrated cca_score joined from the cluster-applied candidates.
    coded = coded.drop("cca_score").join(
        hist.select(["id", "cca_score"]), on="id", how="inner"
    )
    print(f"gold rows joined to candidates: {coded.height} (of 500 drawn)")

    doca_ids = pl.read_parquet(
        config.CCA_DOCA_DIR / "cca_doca_positives.parquet"
    )["id"].cast(pl.Utf8).unique().to_list()
    doca_scores = hist.filter(pl.col("id").is_in(doca_ids))["cca_score"].to_numpy()

    y = coded["cca_event"].cast(pl.Int8).to_numpy()
    s = coded["cca_score"].to_numpy()
    w = coded["ipw"].to_numpy()
    cca_rows = []
    for t in CCA_THRESHOLDS:
        flag = s >= t
        prec = float((w * y * flag).sum() / (w * flag).sum()) if flag.any() else None
        cca_rows.append({
            "cca_score_ge": t,
            "ipw_precision": prec,
            "raw_precision": float(y[flag].mean()) if flag.any() else None,
            "doca_recall": recall_at_threshold(doca_scores, len(doca_ids), t),
            "corpus_flagged_frac": float((hist["cca_score"] >= t).mean()),
            "corpus_flagged_n": int((hist["cca_score"] >= t).sum()),
        })
    results["cca_level_1960_1995"] = {
        "rows": cca_rows,
        "n_gold": coded.height,
        "n_doca_anchors": len(doca_ids),
        "note": "ipw_precision = corpus-reweighted gold-set precision (all-forms "
                "cca_event); DoCA anchors include CCA training positives "
                "(operational retrieval number, not de-novo recall)",
    }

    # --- 2. ICA level: eval precision vs 552-event anchor recall -----------
    eval_df = pl.read_csv(
        config.PROJECT_ROOT / "validation" / "ica_coding_template_coded.csv"
    ).filter(pl.col("ica_event").is_not_null())
    eval_df = eval_df.join(hist.select(["id", "ica_score"]), on="id", how="inner")
    print(f"eval rows joined to candidates: {eval_df.height}")
    ye = eval_df["ica_event"].cast(pl.Int8).to_numpy()
    se = eval_df["ica_score"].to_numpy()

    anchors = pl.read_parquet(config.PROJECT_ROOT / "relevance" / "ica_anchors.parquet")
    holdout = set(
        pl.read_parquet(config.PROJECT_ROOT / "validation" / "ica_holdout_ids.parquet")
        ["id"].cast(pl.Utf8).to_list()
    )
    anchors = anchors.with_columns(
        pl.col("article_id").cast(pl.Utf8).alias("id"),
        pl.col("article_id").cast(pl.Utf8).is_in(list(holdout)).alias("held_out"),
    ).join(hist.select(["id", "ica_score"]), on="id", how="left")
    n_missing = anchors.filter(pl.col("ica_score").is_null()).height
    print(f"anchor events: {anchors.height}, missing from candidates: {n_missing}")

    def _event_recall(frame: pl.DataFrame, t: float) -> float:
        scores = frame["ica_score"].fill_null(-1.0).to_numpy()
        return recall_at_threshold(scores, frame.height, t)

    ica_rows = []
    for t in ICA_THRESHOLDS:
        flag = se >= t
        ica_rows.append({
            "ica_score_ge": t,
            "eval_precision_enriched": float(ye[flag].mean()) if flag.any() else None,
            "anchor_recall_all_552": _event_recall(anchors, t),
            "anchor_recall_heldout_196": _event_recall(anchors.filter(pl.col("held_out")), t),
            "anchor_recall_intraining_356": _event_recall(anchors.filter(~pl.col("held_out")), t),
            "corpus_flagged_frac": float((hist["ica_score"] >= t).mean()),
            "corpus_flagged_n": int((hist["ica_score"] >= t).sum()),
        })
    by_domain = {}
    for dom in anchors["event_type4"].unique().sort().to_list():
        sub = anchors.filter(pl.col("event_type4") == dom)
        sub_h = sub.filter(pl.col("held_out"))
        by_domain[dom] = {
            "n_events": sub.height,
            "n_heldout": sub_h.height,
            "recall_by_threshold": {
                str(t): {"all": _event_recall(sub, t),
                         "heldout": _event_recall(sub_h, t) if sub_h.height else None}
                for t in ICA_THRESHOLDS
            },
        }
    results["ica_level_1960_1995"] = {
        "rows": ica_rows,
        "by_domain": by_domain,
        "n_eval": eval_df.height,
        "note": "eval precision is boundary-enriched — an UPPER BOUND on corpus "
                "precision; anchor recall is event-level (552 events / 466 "
                "articles), held-out = excluded from all training",
    }

    # --- 3. forward yield at the same operating points ---------------------
    eras = {"1996_2011": (1996, 2011), "2012_2024": (2012, 2024), "2025": (2025, 2025)}
    fwd_rows = []
    for t in ICA_THRESHOLDS:
        row: dict = {"ica_score_ge": t, "total": int((fwd["ica_score"] >= t).sum())}
        for era, (lo, hi) in eras.items():
            sub = fwd.filter(pl.col("year").is_between(lo, hi))
            n = int((sub["ica_score"] >= t).sum())
            row[era] = n
            row[f"{era}_per_year"] = round(n / (hi - lo + 1), 1)
        fwd_rows.append(row)
    results["forward_yield_1996_2025"] = {
        "rows": fwd_rows,
        "note": "2025 rides the abstract-register channel (coalesce policy) — "
                "volumes not directly comparable pending the paired-channel check",
    }

    out = exp_dir / "memo_pr_tables.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}\n")

    print("CCA level (1960-1995): threshold | IPW precision | DoCA recall | corpus flagged")
    for r in cca_rows:
        print(f"  cca>={r['cca_score_ge']:.1f} | P={r['ipw_precision']:.3f} | "
              f"R={r['doca_recall']:.3f} | {r['corpus_flagged_n']:>7} ({r['corpus_flagged_frac']:.4f})")
    print("\nICA level (1960-1995): threshold | eval P (enriched) | anchor recall all/heldout | corpus flagged")
    for r in ica_rows:
        print(f"  ica>={r['ica_score_ge']:.1f} | P={r['eval_precision_enriched']:.3f} | "
              f"R={r['anchor_recall_all_552']:.3f}/{r['anchor_recall_heldout_196']:.3f} | "
              f"{r['corpus_flagged_n']:>7} ({r['corpus_flagged_frac']:.4f})")
    print("\nICA anchor recall by domain (all / held-out), at ica>=0.3:")
    for dom, d in by_domain.items():
        r3 = d["recall_by_threshold"]["0.3"]
        ho = f"{r3['heldout']:.3f}" if r3["heldout"] is not None else "n/a"
        print(f"  {dom:<14} n={d['n_events']:>3} (ho {d['n_heldout']:>3}): {r3['all']:.3f} / {ho}")
    print("\nforward yield: threshold | total | 1996-2011/yr | 2012-2024/yr | 2025")
    for r in fwd_rows:
        print(f"  ica>={r['ica_score_ge']:.1f} | {r['total']:>6} | "
              f"{r['1996_2011_per_year']:>7} | {r['2012_2024_per_year']:>7} | {r['2025']:>5}")
    return results


if __name__ == "__main__":
    main()
