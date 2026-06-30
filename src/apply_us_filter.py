# pattern: Imperative Shell (applies model, reads/writes, computes statistics)
"""Batch apply calibrated US filter to API corpus and write per-year outputs.

Reads the full 1960-1995 API corpus via data_from_parquet, applies the
trained US model with calibration via apply_us_model, computes a boolean
us column by thresholding calibrated scores, and writes per-year parquet
files to US_FILTER_SCORES_DIR with id, us_score, us columns.
"""

from __future__ import annotations

import numpy as np
import polars as pl

import src.config as config
from src.data_setup.data import data_from_parquet
from src.validation.slice_eval import apply_us_model


def main(threshold: float = 0.5) -> None:
    """Apply calibrated US model to full API corpus, write per-year outputs.

    Reads API corpus via data_from_parquet, applies calibrated US model to
    headline+lead text, thresholds calibrated scores to produce boolean us
    column, and writes per-year parquets to US_FILTER_SCORES_DIR with columns
    id (str), us_score (float in [0,1]), us (bool).

    Args:
        threshold: Decision threshold for us_score -> us conversion (default 0.5).
    """
    # Read the whole API corpus once
    df = data_from_parquet(
        config.PROJECT_ROOT,
        "api_corpus",
        addl_columns=["year"],
        lead_column="lead_paragraph"
    )

    # Apply model to texts, get calibrated scores
    texts = df["headline_with_lead"].to_list()
    us_scores = apply_us_model(texts)

    # Verify finite and same length as corpus
    assert np.isfinite(us_scores).all(), "Non-finite scores produced by model"
    assert len(us_scores) == df.shape[0], (
        f"Score count {len(us_scores)} != corpus row count {df.shape[0]}"
    )

    # Compute boolean us column
    us_boolean = us_scores >= threshold

    # Attach scores and boolean to dataframe
    df = df.with_columns(
        pl.Series("us_score", us_scores),
        pl.Series("us", us_boolean),
    )

    # Ensure id is string (no cast needed since API id is already string)
    df = df.with_columns(
        pl.col("id").cast(pl.Utf8)
    )

    # Write per-year outputs
    output_dir = config.US_FILTER_SCORES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for year in sorted(df["year"].unique()):
        year_df = df.filter(pl.col("year") == year).select([
            "id", "us_score", "us"
        ])

        output_path = output_dir / f"{year}.parquet"
        year_df.write_parquet(output_path)

        # Print per-year stats
        n_rows = year_df.shape[0]
        n_positive = year_df["us"].sum()
        frac_positive = n_positive / n_rows if n_rows > 0 else 0.0

        print(
            f"Year {year}: {n_rows} rows, "
            f"{n_positive} US-positive ({frac_positive:.1%})"
        )


if __name__ == "__main__":
    main()
