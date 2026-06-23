# pattern: Functional Core
"""ICA evaluation label logic and anchor reservation helpers.

Provides pure functions for:
- Deriving ICA negative labels based on US and CCA scope gates
- Reconciling legacy immig (0/1) into immig_relevant (bool)
- Reserving anchor articles from a holdout set
- Assembling holdout ids across multiple sources
"""

from __future__ import annotations

import polars as pl


def derive_ica_negatives(df: pl.DataFrame) -> pl.DataFrame:
    """Set ica_event=False where us_event==False OR cca_event==False.

    Leaves ica_event null for every us_event==True ∧ cca_event==True row
    (the holistic-coding region). Never reads immig_relevant.

    Args:
        df: DataFrame with us_event and cca_event columns (can be bool or nullable bool)

    Returns:
        DataFrame with new or updated ica_event column (nullable bool)
    """
    # Create or update ica_event column:
    # - False where us_event==False OR cca_event==False
    # - null where us_event==True AND cca_event==True
    result = df.with_columns(
        pl.when((~pl.col("us_event")) | (~pl.col("cca_event")))
        .then(False)
        .otherwise(None)
        .alias("ica_event")
    )
    return result


def reconcile_immig_column(df: pl.DataFrame) -> pl.DataFrame:
    """Map legacy immig (0/1) → immig_relevant (bool) without overwriting hand-coded values.

    If immig column is present:
    - 1 → immig_relevant=True
    - 0 → immig_relevant=False
    - Only applies if immig_relevant is not already present (hand-coded)
    - Preserves immig as immig_advisory with immig_source="legacy"

    Args:
        df: DataFrame that may contain immig (0/1) and/or immig_relevant (bool)

    Returns:
        DataFrame with immig_relevant populated (if not already present) and
        immig renamed to immig_advisory, plus immig_source="legacy" annotation
    """
    result = df.clone()

    # Only reconcile if immig column exists and immig_relevant is not already present
    if "immig" in result.columns and "immig_relevant" not in result.columns:
        result = result.with_columns(
            pl.col("immig")
            .cast(pl.Boolean)
            .alias("immig_relevant")
        )

    # Preserve immig as advisory if it exists
    if "immig" in result.columns:
        result = result.rename({"immig": "immig_advisory"})
        # Add source annotation if immig_advisory was just created
        if "immig_source" not in result.columns:
            result = result.with_columns(
                pl.lit("legacy").alias("immig_source")
            )

    return result


def reserve_anchor_holdout(
    anchor_df: pl.DataFrame,
    frac: float = 0.30,
    seed: int = 200
) -> tuple[list[str], list[str]]:
    """Dedupe anchors by article_id and deterministically split into holdout.

    Args:
        anchor_df: DataFrame with article_id column (may have duplicates)
        frac: Fraction to reserve as holdout (default 0.30 = 30%)
        seed: Random seed for deterministic split (default 200)

    Returns:
        Tuple of (holdout_ids, train_eligible_ids), both sorted lists of str
    """
    # Dedupe by article_id and sort for consistent ordering
    deduped = anchor_df.select("article_id").unique().sort("article_id")

    # Deterministic split: shuffle then partition
    shuffled = deduped.sample(fraction=1.0, seed=seed)
    n_total = shuffled.height
    n_holdout = max(1, int(n_total * frac))  # At least 1 if any rows

    holdout = sorted(shuffled.head(n_holdout)["article_id"].to_list())
    train_eligible = sorted(shuffled.tail(n_total - n_holdout)["article_id"].to_list())

    return (holdout, train_eligible)


def assemble_holdout_ids(*id_sets: list[str]) -> list[str]:
    """Union multiple id lists and dedupe.

    Args:
        *id_sets: Variable number of lists of str ids

    Returns:
        Deduplicated sorted list of all unique ids
    """
    all_ids = set()
    for id_set in id_sets:
        all_ids.update(id_set)
    return sorted(list(all_ids))
