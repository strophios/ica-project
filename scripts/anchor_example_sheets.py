# pattern: Imperative Shell
"""Best/medium/worst anchor examples + same-score non-anchor comparators.

For each of (DoCA anchors x cca_score) and (ICA-552 anchors x ica_score) on
the 1960-1995 candidates: the 10 best-scored, 10 median-scored, and 10
worst-scored anchor articles (what the model finds easy / middling / hard),
plus ~10 NON-anchor articles drawn from the same score band as each group
(what else lives at that score). Headlines joined from the corpus.

Output: cca_doca/experiments/anchor_examples_{cca_doca,ica552}.csv
Run: uv run python -m scripts.anchor_example_sheets
"""

from __future__ import annotations

import polars as pl

import src.config as config

N_PER_BAND = 10


def main() -> None:
    hist = pl.read_parquet(config.ICA_CANDIDATES_DIR / "api_1960_1995.parquet")
    headlines = (
        pl.scan_parquet(str(config.API_CORPUS_DIR / "*.parquet"))
        .select(["id", "headline", "lead_paragraph"])
        .collect()
        .unique(subset="id", keep="first")
    )
    exp_dir = config.CCA_DOCA_DIR / "experiments"

    anchor_sets = {
        "cca_doca": (
            pl.read_parquet(config.CCA_DOCA_DIR / "cca_doca_positives.parquet")
            .select(pl.col("id").cast(pl.Utf8)).unique(),
            "cca_score",
        ),
        "ica552": (
            pl.read_parquet(config.PROJECT_ROOT / "relevance" / "ica_anchors.parquet")
            .select(pl.col("article_id").cast(pl.Utf8).alias("id"), "event_type4")
            .unique(subset="id"),
            "ica_score",
        ),
    }

    for name, (anchors, score_col) in anchor_sets.items():
        scored = anchors.join(
            hist.select(["id", "year", score_col]), on="id", how="inner"
        ).sort(score_col, descending=True)
        n = scored.height
        mid_lo = max(0, n // 2 - N_PER_BAND // 2)
        bands = {
            "best": scored.head(N_PER_BAND),
            "median": scored.slice(mid_lo, N_PER_BAND),
            "worst": scored.tail(N_PER_BAND),
        }
        rows = []
        for band, frame in bands.items():
            frame = frame.with_columns(
                pl.lit(band).alias("band"), pl.lit("anchor").alias("kind")
            )
            rows.append(frame)
            # Non-anchor comparators from the same score range: closest-scored
            # corpus articles not in the anchor set (what ELSE gets this score).
            lo, hi = frame[score_col].min(), frame[score_col].max()
            in_band = hist.filter(
                pl.col(score_col).is_between(lo, hi)
                & ~pl.col("id").is_in(scored["id"].to_list())
            )
            comparators = (
                in_band.sample(n=min(N_PER_BAND, in_band.height), seed=200)
                .select(["id", "year", score_col])
                .with_columns(
                    pl.lit(band).alias("band"), pl.lit("comparator").alias("kind")
                )
            )
            rows.append(comparators)
        out = (
            pl.concat(rows, how="diagonal")
            .join(headlines, on="id", how="left")
            .select(
                ["band", "kind", "year", score_col, "headline", "lead_paragraph", "id"]
                + (["event_type4"] if "event_type4" in rows[0].columns else [])
            )
        )
        out_csv = exp_dir / f"anchor_examples_{name}.csv"
        out.write_csv(out_csv)
        print(f"wrote {out_csv} ({out.height} rows)")
        for band in ["best", "median", "worst"]:
            sub = out.filter((pl.col("band") == band) & (pl.col("kind") == "anchor"))
            print(f"  {name} {band} anchors ({score_col} "
                  f"{sub[score_col].min():.3f}-{sub[score_col].max():.3f}):")
            for r in sub.head(3).to_dicts():
                h = (r["headline"] or "")[:70]
                print(f"    {r['year']} {r[score_col]:.3f} | {h}")


if __name__ == "__main__":
    main()
