# pattern: Imperative Shell (label_and_restrict is the pure core)
"""
Build the DoCA-labeled, US-restricted training table from the embedding cache.

Joins the cached metadata (id, year, us_logit, emb_row) against the DoCA positive
ids and applies the US logit threshold, producing the table that
`create_cca_doca_data` splits for training. Also reports the us_logit distribution
and label/US counts — the informal-calibration view used to pick the threshold.

Run from project root:
    uv run python -m src.build_cca_doca_table --suffix train250k --threshold 0.0
"""

from __future__ import annotations

import argparse

import polars as pl

import src.config as config
from src.embed_corpus import load_cache_meta


def label_and_restrict(
    meta: pl.DataFrame, positive_ids, threshold: float = 0.0
) -> pl.DataFrame:
    """Pure: attach `cca_label` (id in DoCA positives) and `us` (us_logit >= threshold).

    Returns the input columns plus `cca_label` (Int8 0/1) and `us` (bool).
    """
    pos_list = list(dict.fromkeys(positive_ids))
    return meta.with_columns(
        pl.col("id").is_in(pos_list).cast(pl.Int8).alias("cca_label"),
        (pl.col("us_logit") >= threshold).alias("us"),
    )


def _report(table: pl.DataFrame, threshold: float) -> None:
    n = table.height
    n_pos = int(table.filter(pl.col("cca_label") == 1).height)
    n_us = int(table.filter(pl.col("us")).height)
    n_unl_us = int(table.filter((pl.col("cca_label") == 0) & pl.col("us")).height)
    pos_us = int(table.filter((pl.col("cca_label") == 1) & pl.col("us")).height)
    print(f"rows={n}  positives={n_pos}  "
          f"US(thr={threshold})={n_us} ({100 * n_us / max(n, 1):.1f}%)")  # LOG
    print(f"  unlabeled-US (training background) = {n_unl_us}")  # LOG
    print(f"  positives scored US = {pos_us}/{n_pos} "
          f"({100 * pos_us / max(n_pos, 1):.1f}%) — positives kept regardless")  # LOG
    print("  us_logit quantiles:")  # LOG
    for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
        print(f"    {q:>4}: {table['us_logit'].quantile(q):.3f}")  # LOG


def main(cache_suffix: str = "train250k", threshold: float = 0.0) -> None:
    meta = load_cache_meta(config.CCA_EMBED_CACHE_DIR / cache_suffix)
    positive_ids = pl.read_parquet(config.CCA_DOCA_POSITIVES)["id"].to_list()
    table = label_and_restrict(meta, positive_ids, threshold)
    config.CCA_DOCA_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(config.CCA_DOCA_TABLE)
    _report(table, threshold)
    print(f"Wrote {config.CCA_DOCA_TABLE}")  # LOG


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the CCA/DoCA training table.")
    ap.add_argument("--suffix", default="train250k", help="embedding cache subdir")
    ap.add_argument("--threshold", type=float, default=0.0, help="US logit threshold")
    args = ap.parse_args()
    main(cache_suffix=args.suffix, threshold=args.threshold)
