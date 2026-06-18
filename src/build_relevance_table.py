# pattern: Imperative Shell (merge_caches is the pure core)
"""Build the merged immigrant-relevance training cache from two embedding caches.

The relevance head's positives and background live in DIFFERENT caches:
  * POSITIVES   -> `relevance_pos`  (the 17,396 candidate articles, embedded fresh)
  * BACKGROUND  -> `train250k`      (the 235k stratified unlabeled sample reused
                                     from the CCA run, plus its CCA positives,
                                     which -- being protests, ~0.2% immigration --
                                     act as informative not-immigrant hard negatives)

This step merges them (dedup by id, positives win on overlap), re-scores `us_logit`
with the NEW hardened US head against the cached CLS (the existing
`_rescore_us_restriction`; valid because every cache shares the frozen DAPT
backbone), and writes a single `relevance_train` cache that `run_relevance`
consumes like any other. It also reports the US-gated label counts and a
face-validity sample -- in particular the noisier ethnic-only tier -- so the
keep/tier/drop call on those positives is made on US-gated text.

Run from project root:
    uv run python -m src.build_relevance_table --threshold 0.5
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

import src.config as config
from src.embed_corpus import load_cache, write_shard
from src.run_cca_doca import _rescore_us_restriction

US_FULL_WEIGHTS = config.US_FILTER_DIR / "us_classifier_full.weights.h5"
RELEVANCE_DIR = config.PROJECT_ROOT / "relevance"
RELEVANCE_CACHE = config.CCA_EMBED_CACHE_DIR / "relevance_train"


def merge_caches(
    meta_pos: pl.DataFrame, cls_pos: np.ndarray,
    meta_bg: pl.DataFrame, cls_bg: np.ndarray,
) -> tuple[pl.DataFrame, np.ndarray]:
    """Pure: stack positive-cache rows then background-cache rows whose id is not
    already present, re-aligning CLS and assigning a contiguous `emb_row`.

    Positives win on id overlap (their CLS is identical anyway -- same backbone).
    Returns (meta[id, year, us_logit], cls) row-aligned, emb_row 0..N-1.
    """
    pos_ids = set(meta_pos["id"].to_list())
    keep_bg = meta_bg.filter(~pl.col("id").is_in(list(pos_ids)))
    bg_rows = keep_bg["emb_row"].to_numpy()
    cls = np.concatenate([cls_pos, cls_bg[bg_rows]], axis=0)
    cols = ["id", "year", "us_logit"]
    meta = pl.concat([meta_pos.select(cols), keep_bg.select(cols)], how="vertical")
    meta = meta.with_columns(pl.arange(0, meta.height).alias("emb_row"))
    return meta, cls


def _report(table: pl.DataFrame, candidates: pl.DataFrame, threshold: float) -> None:
    n = table.height
    n_pos = int(table.filter(pl.col("cca_label") == 1).height)
    n_us = int(table.filter(pl.col("us")).height)
    pos_us = int(table.filter((pl.col("cca_label") == 1) & pl.col("us")).height)
    unl_us = int(table.filter((pl.col("cca_label") == 0) & pl.col("us")).height)
    print(f"merged rows={n}  relevance-positives={n_pos}")  # LOG
    print(f"US(calib>={threshold}): total={n_us}  "
          f"positives-US={pos_us}/{n_pos} ({100 * pos_us / max(n_pos,1):.1f}%)  "
          f"unlabeled-US background={unl_us}")  # LOG

    # Face validity on US-gated positives, split by tier (ethnic-only is noisier).
    anchors = set(
        pl.read_parquet(RELEVANCE_DIR / "ica_anchors.parquet")["article_id"].to_list()
    )
    cand = candidates.with_columns(
        pl.col("matched").list.eval(
            pl.element().str.to_lowercase().str.contains("-american")
        ).list.all().fill_null(False).alias("ethnic_only")
    )
    us_pos_ids = set(
        table.filter((pl.col("cca_label") == 1) & pl.col("us"))["id"].to_list()
    )
    for tier, mask in [
        ("descriptor (content)", ~cand["ethnic_only"] & ~cand["id"].is_in(list(anchors))),
        ("ethnic-only", cand["ethnic_only"]),
    ]:
        sub = cand.filter(mask & pl.col("id").is_in(list(us_pos_ids)))
        print(f"\n--- US-gated face validity: {tier} (n={sub.height}) ---")  # LOG
        for r in sub.sample(min(10, sub.height), seed=2).iter_rows(named=True):
            print(f"   [{r['year']}] {str(r['headline'])[:74]}")  # LOG


def main(threshold: float = 0.5) -> None:
    candidates = pl.read_parquet(RELEVANCE_DIR / "candidates.parquet")
    cand_ids = candidates["id"].to_list()

    meta_pos, cls_pos = load_cache(config.CCA_EMBED_CACHE_DIR / "relevance_pos")
    meta_bg, cls_bg = load_cache(config.CCA_EMBED_CACHE_DIR / "train250k")
    print(f"relevance_pos={meta_pos.height}  train250k={meta_bg.height}")  # LOG

    meta, cls = merge_caches(meta_pos, cls_pos, meta_bg, cls_bg)
    meta = _rescore_us_restriction(meta, cls, US_FULL_WEIGHTS)
    print(f"merged={meta.height}, us_logit re-scored via {US_FULL_WEIGHTS.name}")  # LOG

    # Write the merged cache for run_relevance (shards carry id, year, us_logit).
    RELEVANCE_CACHE.mkdir(parents=True, exist_ok=True)
    shard_size = 250_000
    for i, start in enumerate(range(0, meta.height, shard_size)):
        chunk = meta.slice(start, shard_size)
        write_shard(RELEVANCE_CACHE, i, cls[start:start + chunk.height],
                    chunk.select("id", "year", "us_logit"))
    print(f"wrote {RELEVANCE_CACHE} ({meta.height} rows)")  # LOG

    table = meta.with_columns(
        pl.col("id").is_in(cand_ids).cast(pl.Int8).alias("cca_label"),
        (pl.col("us_logit") >= threshold).alias("us"),
    )
    _report(table, candidates, threshold)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge + label the relevance training cache.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="calibrated US probability gate (0.5 default; 0.25 = higher recall)")
    args = ap.parse_args()
    main(threshold=args.threshold)
