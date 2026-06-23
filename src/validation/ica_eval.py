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


def assemble_eval_frame(
    anchor_rows: pl.DataFrame,
    coded_survivor_rows: pl.DataFrame,
    boundary_rows: pl.DataFrame,
) -> pl.DataFrame:
    """Assemble a clean joint-ICA evaluation frame from three sources.

    Merges anchor positives (marked ica_event=True), coded survivors (with
    us_event/cca_event preserved, ica_event=null for hand-coding), and boundary
    draw (stratified candidate pool). Ensures schema conformance via null-filling
    missing columns, applies ICA negative derivation, and returns the merged frame.

    Args:
        anchor_rows: DataFrame with anchors marked ica_event=True, sample_stratum="anchor"
        coded_survivor_rows: DataFrame with us_event/cca_event=True, ica_event=null, sample_stratum="coded_reuse"
        boundary_rows: DataFrame with mixed scope gates, stratified logits, ica_event=null, sample_stratum="cca_*_relev_*"

    Returns:
        Merged DataFrame with schema-conformant columns (null-filled as needed),
        ICA negatives derived, ready for validation and hand-coding.

    Note: Does NOT validate the output — caller is responsible for calling
    validate_gold_set() after assembly. Does NOT include the holdout ids;
    caller is responsible for collecting those separately.
    """
    # Define schema column expectations
    required_cols = set([
        "id", "corpus", "year", "news_desk", "section_name", "headline",
        "lead_paragraph", "sample_stratum", "us_event", "event_location",
        "cca_event", "event_type", "immig_relevant", "ica_event", "alt_corpus_id",
        "cca_logit", "cca_score", "relevance_logit", "relevance_score",
    ])

    def pad_frame(df: pl.DataFrame) -> pl.DataFrame:
        """Add any missing columns as null, with proper type casting."""
        # Define expected types for schema columns (match schema.py)
        bool_cols = {"us_event", "cca_event", "ica_event", "immig_relevant"}
        float_cols = {"cca_logit", "cca_score", "relevance_logit", "relevance_score"}
        int_cols = {"year"}
        str_cols = {
            "id", "corpus", "news_desk", "section_name", "headline",
            "lead_paragraph", "sample_stratum", "event_location", "event_type",
            "alt_corpus_id"
        }

        # Add missing columns with correct types
        for col in required_cols:
            if col not in df.columns:
                if col in bool_cols:
                    df = df.with_columns(pl.lit(None, dtype=pl.Boolean).alias(col))
                elif col in float_cols:
                    df = df.with_columns(pl.lit(None, dtype=pl.Float32).alias(col))
                elif col in int_cols:
                    df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias(col))
                else:  # str_cols
                    if col == "corpus":
                        df = df.with_columns(pl.lit("api", dtype=pl.Utf8).alias(col))
                    else:
                        df = df.with_columns(pl.lit("", dtype=pl.Utf8).alias(col))

        # Fill nulls in corpus with "api" (for rows from scored candidates)
        if "corpus" in df.columns:
            df = df.with_columns(pl.col("corpus").fill_null("api"))

        # Fill nulls in lead_paragraph with empty string (required but often missing)
        if "lead_paragraph" in df.columns:
            df = df.with_columns(pl.col("lead_paragraph").fill_null(""))

        # Standardize types for concat compatibility
        for col in float_cols:
            if col in df.columns and df[col].dtype != pl.Float32:
                df = df.with_columns(pl.col(col).cast(pl.Float32))

        # Ensure year is Int64
        if "year" in df.columns:
            df = df.with_columns(pl.col("year").cast(pl.Int64))

        # Ensure all string columns are actually strings
        for col in str_cols:
            if col in df.columns and df[col].dtype != pl.Utf8:
                df = df.with_columns(pl.col(col).cast(pl.Utf8))

        return df.select(sorted(required_cols))

    # Pad all source frames to match schema
    anchor_rows = pad_frame(anchor_rows)
    coded_survivor_rows = pad_frame(coded_survivor_rows)
    boundary_rows = pad_frame(boundary_rows)

    # Concatenate all sources
    full_template = pl.concat([anchor_rows, coded_survivor_rows, boundary_rows])

    # Apply ICA label logic: set ica_event=False where us_event==False OR cca_event==False.
    # We inline this logic (rather than reusing derive_ica_negatives) because derive_ica_negatives
    # unconditionally overwrites all ica_event values, and we must preserve the anchor positives'
    # ica_event=True values set above (rows that are NOT in the holistic coding region).
    # The conditional `when(is_null & scope_gates_fail)` preserves anchors and marks scope
    # gate violations only. This prevents a future refactor from accidentally reverting anchors
    # to null by reusing the unrestricted helper.
    full_template = full_template.with_columns(
        pl.when(
            (pl.col("ica_event").is_null()) &
            ((~pl.col("us_event")) | (~pl.col("cca_event")))
        ).then(False)
        .otherwise(pl.col("ica_event"))
        .alias("ica_event")
    )

    return full_template
