# pattern: Imperative Shell (pure rank/recall helpers at top; I/O in main)
"""Topline eval of the cluster-applied ICA candidates (2026-08, pre-meeting).

Inputs (produced on the cluster with true-math scoring, synced down):
  cca_doca/ica_candidates/api_1960_1995.parquet   (full in-DoCA-period range)
  cca_doca/ica_candidates/api_1996_2025.parquet   (forward corpus, coalesce channel)

Reports:
  1. DoCA-anchor recall on 1960-1995 — the ~15.6k DoCA-matched article ids
     (cca_doca/cca_doca_positives.parquet) against the ica_score ranking, at
     review budgets (top 1/5/10% and absolute top-Ks). The recall comparison
     the 2026-07-10 meeting asked for.
  2. Hand-coded ICA-positive ranks — the eval set's ica_event rows that carry
     API ids, located in the ranked lists (era-matched file).
  3. Per-year candidate rates 1996-2025 at score thresholds, 2025 reported
     separately (abstract-register channel).
  4. Face-validity CSVs: top-K per file with headlines joined from api_corpus.

Outputs: cca_doca/experiments/topline_candidates_eval.json + face_validity CSVs
in cca_doca/experiments/. Run: uv run python -m scripts.topline_candidates_eval
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

import src.config as config


# ---------------------------------------------------------------------------
# Functional core
# ---------------------------------------------------------------------------
def anchor_ranks(
    candidates: pl.DataFrame, anchor_ids: list[str]
) -> tuple[int, np.ndarray]:
    """Locate anchors in a ranked candidates frame.

    `candidates` must be sorted by ica_score descending (the candidates-file
    contract). Returns (n_found, 1-based ranks of the found anchors). Anchors
    absent from the frame are counted by the caller as misses via the recall
    denominator.
    """
    ranked = candidates.with_row_index("rank", offset=1)
    hits = ranked.filter(pl.col("id").is_in(list(anchor_ids)))
    return hits.height, hits["rank"].to_numpy().astype(np.int64)


def recall_at_top(ranks: np.ndarray, n_anchors: int, k: int) -> float:
    """Fraction of ALL anchors ranked within the top k (inclusive).

    Missing anchors (not in `ranks`) count against recall — an anchor the
    ranking never surfaces is a miss, not an exclusion.
    """
    if n_anchors == 0:
        return 0.0
    return float((ranks <= k).sum() / n_anchors)


def budget_table(
    ranks: np.ndarray, n_anchors: int, n_candidates: int
) -> list[dict]:
    """Recall at fractional and absolute review budgets."""
    rows = []
    for frac in [0.001, 0.01, 0.05, 0.10]:
        k = max(1, int(n_candidates * frac))
        rows.append({"budget": f"top {frac:.1%}", "k": k,
                     "recall": recall_at_top(ranks, n_anchors, k)})
    for k in [1_000, 10_000, 50_000]:
        if k <= n_candidates:
            rows.append({"budget": f"top {k}", "k": k,
                         "recall": recall_at_top(ranks, n_anchors, k)})
    return rows


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------
def main(top_k_csv: int = 100) -> dict:
    cand_dir = config.ICA_CANDIDATES_DIR
    exp_dir = config.CCA_DOCA_DIR / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"files": {}}

    frames = {
        name: pl.read_parquet(cand_dir / f"{name}.parquet")
        for name in ["api_1960_1995", "api_1996_2025"]
    }
    for name, df in frames.items():
        assert df["ica_score"].is_sorted(descending=True), f"{name} not rank-sorted"
        results["files"][name] = {"n": df.height,
                                  "gated_in_frac": float(df["gated"].mean())}

    # --- 1. DoCA-anchor recall (1960-1995) --------------------------------
    doca_ids = pl.read_parquet(
        config.CCA_DOCA_DIR / "cca_doca_positives.parquet"
    )["id"].cast(pl.Utf8).unique().to_list()
    df = frames["api_1960_1995"]
    results["doca_recall_1960_1995"] = {}
    # cca_score is the dimension DoCA anchors actually label (collective
    # action of ANY kind); ica_score is reported alongside for context —
    # most DoCA events are not immigration-relevant, so low ica-ranked
    # recall is expected, not a defect.
    for score_col in ["cca_score", "ica_score"]:
        ranked = df.sort(score_col, descending=True)
        n_found, ranks = anchor_ranks(ranked, doca_ids)
        results["doca_recall_1960_1995"][score_col] = {
            "n_anchors": len(doca_ids),
            "n_found_in_candidates": n_found,
            "median_rank_of_found": float(np.median(ranks)) if ranks.size else None,
            "budgets": budget_table(ranks, len(doca_ids), df.height),
        }
    results["doca_recall_1960_1995"]["note"] = (
        "anchors = unique DoCA-matched article ids; recall counts "
        "missing-from-candidates as misses; cca_score ranking is the "
        "dimension-appropriate recall check"
    )

    # --- 2. hand-coded ICA positives --------------------------------------
    eval_df = pl.read_csv(config.PROJECT_ROOT / "validation" / "ica_coding_template_coded.csv")
    pos = eval_df.filter(pl.col("ica_event") == True)  # noqa: E712  (polars expr)
    pos_ids = pos["id"].cast(pl.Utf8).to_list()
    ica_block: dict = {"n_hand_coded_positives": len(pos_ids), "by_file": {}}
    for name, df in frames.items():
        n_found, ranks = anchor_ranks(df, pos_ids)
        ica_block["by_file"][name] = {
            "n_found": n_found,
            "median_rank_of_found": float(np.median(ranks)) if ranks.size else None,
            "budgets": budget_table(ranks, n_found, df.height) if n_found else [],
            "note": "recall over the positives PRESENT in this file (the eval "
                    "set spans corpora; era determines the file)",
        }
    results["hand_coded_ica_positives"] = ica_block

    # --- 3. per-year rates 1996-2025 --------------------------------------
    fwd = frames["api_1996_2025"].with_columns(pl.col("year").cast(pl.Int64))
    per_year = (
        fwd.group_by("year")
        .agg(
            pl.len().alias("n_articles"),
            (pl.col("ica_score") >= 0.5).sum().alias("n_ge_050"),
            (pl.col("ica_score") >= 0.8).sum().alias("n_ge_080"),
        )
        .sort("year")
        .with_columns(
            (pl.col("n_ge_050") / pl.col("n_articles")).alias("rate_ge_050"),
            (pl.col("year") >= 2025).alias("abstract_register_era"),
        )
    )
    results["per_year_1996_2025"] = per_year.to_dicts()

    # --- 4. face-validity CSVs --------------------------------------------
    headlines = (
        pl.scan_parquet(str(config.API_CORPUS_DIR / "*.parquet"))
        .select(["id", "headline", "lead_paragraph", "abstrct"])
        .collect()
        .unique(subset="id", keep="first")
    )
    for name, df in frames.items():
        top = df.head(top_k_csv).join(headlines, on="id", how="left")
        out_csv = exp_dir / f"face_validity_{name}_top{top_k_csv}.csv"
        top.write_csv(out_csv)
        print(f"wrote {out_csv}")

    out_json = exp_dir / "topline_candidates_eval.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_json}\n")

    # --- console summary ---------------------------------------------------
    for score_col in ["cca_score", "ica_score"]:
        d = results["doca_recall_1960_1995"][score_col]
        print(f"DoCA anchors ranked by {score_col}: {d['n_anchors']} anchors, "
              f"{d['n_found_in_candidates']} found, median rank {d['median_rank_of_found']:.0f}")
        for b in d["budgets"]:
            print(f"  recall @ {b['budget']:<10} (k={b['k']:>7}): {b['recall']:.3f}")
    for name, blk in ica_block["by_file"].items():
        print(f"ICA positives in {name}: {blk['n_found']}/{len(pos_ids)}"
              + (f" | median rank {blk['median_rank_of_found']:.0f}" if blk["n_found"] else ""))
        for b in blk["budgets"]:
            print(f"  recall @ {b['budget']:<10} (k={b['k']:>7}): {b['recall']:.3f}")
    print("\nper-year (1996-2025), n >= 0.5 threshold:")
    for r in results["per_year_1996_2025"]:
        flag = "  <- abstract-register era" if r["abstract_register_era"] else ""
        print(f"  {r['year']}: {r['n_ge_050']:>5} / {r['n_articles']:>6} ({r['rate_ge_050']:.4f}){flag}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Topline eval of the ICA candidates files.")
    ap.add_argument("--top-k-csv", type=int, default=100,
                    help="rows per face-validity CSV (default 100)")
    args = ap.parse_args()
    main(top_k_csv=args.top_k_csv)
