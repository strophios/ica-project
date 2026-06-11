# pattern: Imperative Shell
"""Generate stratified sampling candidate for hand-coding.

Samples the API pre-1986 corpus (years < 1987) stratified by era-bucket ×
news_desk. Optionally merges a DoCA-matched positive set. Writes output
conforming to the gold-set schema with label columns null and corpus='api'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl

from src.validation.schema import validate_gold_set


def stratified_sample(
    df: pl.DataFrame,
    n_per_cell: int,
    era_boundaries: Optional[list[int]] = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Stratified sample by era-bucket and news_desk.

    Args:
        df: Dataframe with 'year' and 'news_desk' columns
        n_per_cell: Number of samples per era-desk cell
        era_boundaries: Era boundaries (e.g., [1960, 1970, 1980, 1987])
            Default: [1960, 1970, 1980, 1987]
        seed: Random seed for reproducibility

    Returns:
        Sampled dataframe, stratified by era × desk
    """
    if era_boundaries is None:
        era_boundaries = [1960, 1970, 1980, 1987]

    # Add era column using cut; get string representation of the bins
    df_with_era = df.with_columns(
        era=pl.col("year").cut(era_boundaries).cast(pl.Utf8)
    )

    # Sample up to n_per_cell from each era/desk combination
    # (handles sparse cells gracefully)
    def sample_group(g):
        """Sample min(n_per_cell, group_size) from group."""
        actual_n = min(n_per_cell, g.shape[0])
        return g.sample(n=actual_n, seed=seed, with_replacement=False)

    sampled = (
        df_with_era
        .with_row_index("row_num")
        .group_by("era", "news_desk")
        .map_groups(sample_group)
        .drop(["era", "row_num"])
    )

    return sampled


def build_and_write_template(
    api_df: pl.DataFrame,
    output_path: Path | str,
    n_per_cell: int = 5,
    doca_matched_df: Optional[pl.DataFrame] = None,
    seed: int = 42,
) -> None:
    """Build and write a coding template conforming to the gold-set schema.

    Samples the API corpus stratified by era-bucket and news_desk. Optionally
    merges DoCA-matched rows. Writes output with:
    - corpus="api"
    - alt_corpus_id=None
    - label columns (us_event, event_location, etc.) null
    - sample_stratum set appropriately ("random_pre1986" or "doca_matched")

    Args:
        api_df: API corpus dataframe (must have id, year, news_desk, headline, lead_paragraph)
        output_path: Path to write the template parquet
        n_per_cell: Rows per era-desk cell
        doca_matched_df: Optional DataFrame with 'id' column of DoCA-matched articles
        seed: Random seed for reproducibility
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Track which ids are doca_matched (if provided)
    doca_ids = set()
    if doca_matched_df is not None:
        doca_ids = set(doca_matched_df["id"].to_list())

    # Stratified sample
    sampled = stratified_sample(api_df, n_per_cell=n_per_cell, seed=seed)

    # Add schema columns
    template = sampled.with_columns([
        pl.lit("api").alias("corpus"),
        pl.lit(None, dtype=pl.Utf8).alias("alt_corpus_id"),
        pl.lit(None, dtype=pl.Boolean).alias("us_event"),
        pl.lit(None, dtype=pl.Utf8).alias("event_location"),
        pl.lit(None, dtype=pl.Boolean).alias("cca_event"),
        pl.lit(None, dtype=pl.Boolean).alias("immig_relevant"),
        pl.lit(None, dtype=pl.Boolean).alias("ica_event"),
        pl.when(pl.col("id").is_in(doca_ids)).then(
            pl.lit("doca_matched")
        ).otherwise(
            pl.lit("random_pre1986")
        ).alias("sample_stratum"),
    ])

    # Ensure only required columns are present
    required_cols = [
        "id", "corpus", "year", "news_desk", "section_name",
        "headline", "lead_paragraph", "sample_stratum",
        "us_event", "event_location", "cca_event", "immig_relevant",
        "ica_event", "alt_corpus_id",
    ]
    template = template.select([col for col in required_cols if col in template.columns])

    # Validate against schema
    validate_gold_set(template)

    # Write
    template.write_parquet(output_path)


if __name__ == "__main__":
    import argparse
    import src.config as config

    parser = argparse.ArgumentParser(
        description="Generate hand-coding candidate template for gold-set validation."
    )
    parser.add_argument(
        "--n-per-cell",
        type=int,
        default=5,
        help="Number of articles per era-desk cell (default: 5, matching the 546-row template)",
    )
    parser.add_argument(
        "--doca-matched",
        type=str,
        default=None,
        help="Optional path to parquet with DoCA-matched ids; rows will be marked as 'doca_matched'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.VALIDATION_DIR / "coding_template.parquet"),
        help=f"Output path for the template parquet (default: {config.VALIDATION_DIR / 'coding_template.parquet'})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Load API corpus (years < 1987)
    api_df = pl.read_parquet(config.API_CORPUS_DIR)
    api_df = api_df.filter(pl.col("year") < 1987)

    # Optionally load DoCA-matched ids
    doca_matched_df = None
    if args.doca_matched:
        doca_matched_df = pl.read_parquet(args.doca_matched)

    # Generate and write template
    build_and_write_template(
        api_df=api_df,
        output_path=args.output,
        n_per_cell=args.n_per_cell,
        doca_matched_df=doca_matched_df,
        seed=args.seed,
    )

    print(f"Coding template written to: {args.output}")
    print(f"Stratified sampling: {args.n_per_cell} rows per era-desk cell, seed={args.seed}")
    if args.doca_matched:
        print(f"DoCA-matched set merged from: {args.doca_matched}")
