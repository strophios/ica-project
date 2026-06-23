# pattern: Functional Core
"""
Composed-score ICA boundary sampler (Phase 2, Task 3).

Builds a schema-conformant coding template by stratifying candidates over
bins of CCA strength × relevance band. Deliberately includes high-CCA /
low-marginal-relevance cells (where contextual ICA hides), not just
high-relevance rows.

The sampler excludes anchor positives and previously-coded rows, emits
rows with null labels for holistic hand-coding, and tags each row with
its stratum (e.g., "cca_high_relev_low") for downstream analysis.
"""

from __future__ import annotations

import polars as pl

from src.validation.schema import validate_gold_set

# Score band boundaries
_CCA_HIGH, _CCA_LOW = 1.0, -1.0
_RELEV_HIGH, _RELEV_LOW = 0.5, -0.5

# Default allocation across 6 strata
_DEFAULT_ALLOC = {
    "cca_high_relev_high": 200,
    "cca_high_relev_low": 200,
    "cca_mid_relev_high": 150,
    "cca_mid_relev_low": 150,
    "cca_low_relev_high": 100,
    "cca_low_relev_low": 100,
}

# Column order for output (schema-conformant)
_SCHEMA_COLS = [
    "id", "corpus", "year", "news_desk", "section_name", "headline",
    "lead_paragraph", "sample_stratum", "us_event", "event_location",
    "cca_event", "event_type", "immig_relevant", "ica_event", "alt_corpus_id",
    "cca_logit", "cca_score", "relevance_logit", "relevance_score",
]


def _assign_cca_band(logit: pl.Expr) -> pl.Expr:
    """Map CCA logit to band label: high/mid/low."""
    return (
        pl.when(logit >= _CCA_HIGH).then(pl.lit("cca_high"))
        .when(logit < _CCA_LOW).then(pl.lit("cca_low"))
        .otherwise(pl.lit("cca_mid"))
    )


def _assign_relev_band(logit: pl.Expr) -> pl.Expr:
    """Map relevance logit to band label: high/low."""
    return (
        pl.when(logit >= _RELEV_HIGH).then(pl.lit("relev_high"))
        .otherwise(pl.lit("relev_low"))
    )


def _compose_stratum(cca_band: pl.Expr, relev_band: pl.Expr) -> pl.Expr:
    """Compose a stratum label from CCA and relevance bands."""
    return pl.concat_str([cca_band, relev_band], separator="_")


def build_ica_template(
    scored: pl.DataFrame,
    anchor_ids: list[str] | None = None,
    coded500_ids: list[str] | None = None,
    alloc: dict[str, int] | None = None,
    seed: int = 200,
) -> pl.DataFrame:
    """Build a schema-conformant, composed-score ICA coding template.

    Stratifies candidate rows over 6 strata: CCA (high/mid/low) ×
    Relevance (high/low), deliberately including high-CCA / low-relevance
    cells where contextual ICA hides. Excludes anchors and previously-coded
    rows. Orders rows by within-stratum fractional rank so any prefix is
    approximately stratum-proportional.

    Args:
        scored: DataFrame with columns id, year, news_desk, section_name,
                headline, lead_paragraph, cca_logit, relevance_logit.
        anchor_ids: list of ids to exclude (anchor holdout set).
        coded500_ids: list of ids to exclude (previously coded rows).
        alloc: dict[stratum_name] → count, defaults to _DEFAULT_ALLOC.
        seed: random seed for deterministic sampling.

    Returns:
        Schema-conformant DataFrame with null labels and sample_stratum tagged.

    Raises:
        ValueError: if scored lacks required columns.
    """
    anchor_ids = anchor_ids or []
    coded500_ids = coded500_ids or []
    alloc = alloc or _DEFAULT_ALLOC
    exclude_ids = set(anchor_ids) | set(coded500_ids)

    # Filter to valid candidates (not excluded, complete metadata)
    pool = (
        scored.filter(
            ~pl.col("id").is_in(list(exclude_ids))
            & pl.col("year").is_not_null()
            & pl.col("id").is_not_null()
            & pl.col("cca_logit").is_not_null()
            & pl.col("relevance_logit").is_not_null()
        )
        .with_columns(
            _assign_cca_band(pl.col("cca_logit")).alias("_cca_band"),
            _assign_relev_band(pl.col("relevance_logit")).alias("_relev_band"),
        )
        .with_columns(
            _compose_stratum(
                pl.col("_cca_band"), pl.col("_relev_band")
            ).alias("sample_stratum")
        )
        .drop("_cca_band", "_relev_band")
    )

    # Sample from each stratum
    parts = []
    for stratum, n in alloc.items():
        g = pool.filter(pl.col("sample_stratum") == stratum)
        take = min(n, g.height)
        if take == 0:
            continue

        s = g.sample(n=take, seed=seed, with_replacement=False)
        # Fractional within-stratum rank for prefix-stratification
        s = (
            s.with_row_index("_si")
            .with_columns(((pl.col("_si") + 0.5) / take).alias("_frac"))
            .drop("_si")
        )
        parts.append(s)

    if not parts:
        # No data available; return empty but schema-conformant frame
        template = pl.DataFrame({
            col: pl.Series([], dtype=pl.Utf8 if col == "id" else pl.Float64)
            for col in _SCHEMA_COLS
        })
        return template

    template = pl.concat(parts).sort("_frac").drop("_frac")

    # Add label columns (all null for hand-coding) and compute sigmoid scores
    template = template.with_columns(
        pl.lit("api").alias("corpus"),
        pl.col("year").cast(pl.Int64),
        pl.col("news_desk").fill_null(""),
        pl.col("section_name").fill_null(""),
        pl.col("headline").fill_null(""),
        pl.col("lead_paragraph").fill_null(""),
        pl.col("cca_logit").cast(pl.Float64),
        (1.0 / (1.0 + (-pl.col("cca_logit").cast(pl.Float64)).exp())).alias("cca_score"),
        pl.col("relevance_logit").cast(pl.Float64),
        (1.0 / (1.0 + (-pl.col("relevance_logit").cast(pl.Float64)).exp())).alias(
            "relevance_score"
        ),
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
