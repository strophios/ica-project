# pattern: Imperative Shell
"""Build stripped-text LDC 1996-2007 source parquet for cluster embed.

Joins ldc_corpus[1996-2007] to ldc_labeled[id, stripped_text, us_label]
and writes a single source parquet suitable for embed_corpus.py with
--lead-column stripped_text.

Output columns: id, stripped_text, us_label (nullable), year, and any
additional columns from the LDC corpus (headline, etc.) needed by embed.
"""

from __future__ import annotations

import argparse

import polars as pl

import src.config as config


def main(limit: int | None = None) -> None:
    """Build stripped-LDC source by joining corpus to labeled parquet.

    Reads the ldc_corpus (1996-2007, hive-partitioned by publication_year),
    joins to ldc_labeled.parquet (id, us_label nullable, label_source, stripped_text),
    and writes output to config.US_FILTER_DIR / "ldc_9607_stripped_source.parquet".

    Args:
        limit: Optional row limit for smoke-testing. If provided, only this many
               rows from the full joined result are written.
    """
    # Read ldc_labeled (has stripped_text and us_label)
    print("Reading ldc_labeled.parquet...")
    labeled = pl.read_parquet(config.US_FILTER_LABELED_PARQUET)
    print(f"  Loaded {labeled.shape[0]} labeled rows")

    # Read ldc_corpus 1996-2007 via hive-partitioned scan
    print("Reading ldc_corpus[1996-2007] (hive-partitioned by publication_year)...")
    corpus = pl.scan_parquet(
        config.LDC_CORPUS / "*.parquet",
        hive_partitioning=True
    ).filter(
        (pl.col("publication_year") >= 1996) & (pl.col("publication_year") <= 2007)
    ).collect()
    print(f"  Loaded {corpus.shape[0]} corpus rows from years 1996-2007")

    # Join corpus to labeled on id
    print("Joining corpus to labeled...")
    joined = corpus.join(labeled.select(["id", "stripped_text", "us_label"]),
                         on="id", how="inner")
    print(f"  Joined result: {joined.shape[0]} rows")

    # Apply limit if requested (for smoke-testing)
    if limit is not None:
        joined = joined.head(limit)
        print(f"  Limited to {limit} rows for smoke test")

    # Ensure output directory exists
    output_dir = config.US_FILTER_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write to output path
    output_path = output_dir / "ldc_9607_stripped_source.parquet"
    print(f"Writing to {output_path}...")
    joined.write_parquet(output_path)
    print(f"  Complete: {joined.shape[0]} rows written")

    # Log the schema for verification
    print("\nOutput schema:")
    for col, dtype in zip(joined.columns, joined.dtypes):
        print(f"  {col}: {dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build stripped-LDC source parquet for cluster embed"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit output to this many rows (for smoke-testing)",
    )
    args = parser.parse_args()

    main(limit=args.limit)
