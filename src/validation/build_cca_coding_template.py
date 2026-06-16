# pattern: Imperative Shell (build_cca_template is the pure core)
"""
Score-stratified CCA coding template for the gold set.

Samples the UNLABELED background (cca_label==0 -- the deployment population whose
cca_event is unknown) stratified by CCA-score band, OVERSAMPLING the high band so
the coded set has enough predicted-positives to estimate precision (the positive
class is ~2%, so a uniform sample would have almost none). Rows are ordered so any
prefix is approximately band-proportional -- the MVP floor (first 500) is itself a
valid stratified mini-gold-set, and coding more later just continues down the list
(draw once). The coder fills cca_event (and may also fill us_event/event_location
on the same rows, since CCA "discovered" events include foreign protests).

Run from project root (after src.score_cca_doca writes scored_candidates.parquet):
    uv run python -m src.validation.build_cca_coding_template --target 3000
"""

from __future__ import annotations

import argparse

import polars as pl

import src.config as config
from src.validation.schema import validate_gold_set

_HIGH, _LOW = 1.0, -1.0  # cca_logit band boundaries
_DEFAULT_ALLOC = {"cca_score_high": 1200, "cca_score_mid": 900, "cca_score_low": 900}

_SCHEMA_COLS = [
    "id", "corpus", "year", "news_desk", "section_name", "headline",
    "lead_paragraph", "sample_stratum", "us_event", "event_location",
    "cca_event", "event_type", "immig_relevant", "ica_event", "alt_corpus_id",
    "cca_logit", "cca_score",
]


def build_cca_template(
    scored: pl.DataFrame, alloc: dict[str, int] | None = None, seed: int = 200
) -> pl.DataFrame:
    """Build a schema-conformant, prefix-stratified CCA coding template.

    `scored` must carry: id, year, news_desk, section_name, headline,
    lead_paragraph, cca_label, cca_logit. Samples cca_label==0 by score band
    (alloc per band, min(alloc, available)); orders rows by within-band fractional
    rank so any prefix is band-proportional. Label columns are emitted null.
    """
    alloc = alloc or _DEFAULT_ALLOC
    pool = (
        scored.filter(
            (pl.col("cca_label") == 0)
            & pl.col("year").is_not_null()
            & pl.col("id").is_not_null()
        )
        .with_columns(
            pl.when(pl.col("cca_logit") >= _HIGH).then(pl.lit("cca_score_high"))
            .when(pl.col("cca_logit") < _LOW).then(pl.lit("cca_score_low"))
            .otherwise(pl.lit("cca_score_mid"))
            .alias("sample_stratum")
        )
    )

    parts = []
    for band, n in alloc.items():
        g = pool.filter(pl.col("sample_stratum") == band)
        take = min(n, g.height)
        if take == 0:
            continue
        s = g.sample(n=take, seed=seed, with_replacement=False)
        # fractional within-band rank -> interleaves bands when sorted (prefix-strat)
        s = (
            s.with_row_index("_bi")
            .with_columns(((pl.col("_bi") + 0.5) / take).alias("_frac"))
            .drop("_bi")
        )
        parts.append(s)

    template = pl.concat(parts).sort("_frac").drop("_frac")
    template = template.with_columns(
        pl.lit("api").alias("corpus"),
        pl.col("year").cast(pl.Int64),
        pl.col("news_desk").fill_null(""),
        pl.col("section_name").fill_null(""),
        pl.col("headline").fill_null(""),
        pl.col("lead_paragraph").fill_null(""),
        pl.col("cca_logit").cast(pl.Float64),
        (1.0 / (1.0 + (-pl.col("cca_logit").cast(pl.Float64)).exp())).alias("cca_score"),
        pl.lit(None, dtype=pl.Utf8).alias("alt_corpus_id"),
        pl.lit(None, dtype=pl.Boolean).alias("us_event"),
        pl.lit(None, dtype=pl.Utf8).alias("event_location"),
        pl.lit(None, dtype=pl.Boolean).alias("cca_event"),
        pl.lit(None, dtype=pl.Utf8).alias("event_type"),
        pl.lit(None, dtype=pl.Boolean).alias("immig_relevant"),
        pl.lit(None, dtype=pl.Boolean).alias("ica_event"),
    ).select(_SCHEMA_COLS)

    validate_gold_set(template)
    return template


def main(target: int = 3000, seed: int = 200) -> None:
    scored = pl.read_parquet(config.CCA_DOCA_DIR / "scored_candidates.parquet")
    # Scale the default 4:3:3 allocation to the requested target.
    scale = target / sum(_DEFAULT_ALLOC.values())
    alloc = {k: int(round(v * scale)) for k, v in _DEFAULT_ALLOC.items()}
    template = build_cca_template(scored, alloc=alloc, seed=seed)

    out = config.VALIDATION_DIR / "cca_coding_template.parquet"
    config.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    template.write_parquet(out)

    bands = template["sample_stratum"].value_counts().sort("sample_stratum")
    print(f"Wrote {template.height}-row CCA coding template -> {out}")  # LOG
    print(f"  band counts:\n{bands}")  # LOG
    first500 = template.head(500)["sample_stratum"].value_counts().sort("sample_stratum")
    print(f"  first-500 band counts (prefix-stratification check):\n{first500}")  # LOG
    print("  MVP floor: hand-code cca_event for the first 500 rows; "
          "evaluate with evaluate_cca_slice.")  # LOG


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the score-stratified CCA coding template.")
    ap.add_argument("--target", type=int, default=3000, help="total template size")
    ap.add_argument("--seed", type=int, default=200)
    args = ap.parse_args()
    main(target=args.target, seed=args.seed)
