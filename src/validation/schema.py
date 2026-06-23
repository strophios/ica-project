# pattern: Functional Core
"""Gold-set validation schema.

Defines the column set, dtypes, and enumeration values for the
hand-labeled gold set used for pre-1986 slice evaluation and
eventual full-model validation. The schema is durable across model
iterations (US filter, then CCA/immig/ICA heads).
"""

from __future__ import annotations

import polars as pl


# Required columns that must be present and non-null
REQUIRED_COLUMNS = {
    "id": pl.Utf8,
    "corpus": pl.Utf8,
    "year": pl.Int64,
    "news_desk": pl.Utf8,
    "section_name": pl.Utf8,
    "headline": pl.Utf8,
    "lead_paragraph": pl.Utf8,
    "sample_stratum": pl.Utf8,
}

# Label columns that may be present but can be null
LABEL_COLUMNS = {
    "us_event": pl.Boolean,
    "event_location": pl.Utf8,
    "cca_event": pl.Boolean,
    # Hand-coded collective-action form (coder's judgment, free vocab; typical
    # values: street, strike, boycott, conventional, lawsuit, other). Lets us
    # measure detection by event type and test whether non-prototypical forms
    # drive label noise.
    "event_type": pl.Utf8,
    "immig_relevant": pl.Boolean,
    "ica_event": pl.Boolean,
}

# Optional columns that may or may not be present
OPTIONAL_COLUMNS = {
    "alt_corpus_id": pl.Utf8,
    # Model scores attached to a score-stratified template (CCA gold set). The
    # raw logit drives thresholding (evaluate_cca_slice); cca_score is its sigmoid
    # for human readability.
    "cca_logit": pl.Float64,
    "cca_score": pl.Float64,
    # Relevance-head model scores (for ICA boundary sampling). Like CCA scores,
    # the logit is raw and score is sigmoid.
    "relevance_logit": pl.Float64,
    "relevance_score": pl.Float64,
}

# Enumeration values for categorical columns
VALID_CORPUS = {"api", "ldc"}
# US-filter sampling modes ("doca_matched", "random_pre1986", "ambiguous") plus
# CCA score-band strata for the score-stratified CCA gold set, plus
# ICA composed strata (CCA strength × relevance band) for the ICA boundary sampler, plus
# ICA eval set strata ("anchor" for held-out positives, "coded_reuse" for re-coded rows).
VALID_SAMPLE_STRATUM = {
    "doca_matched", "random_pre1986", "ambiguous",
    "cca_score_high", "cca_score_mid", "cca_score_low",
    "cca_high_relev_high", "cca_high_relev_low",
    "cca_mid_relev_high", "cca_mid_relev_low",
    "cca_low_relev_high", "cca_low_relev_low",
    "anchor", "coded_reuse",
}


def validate_gold_set(df: pl.DataFrame) -> None:
    """Validate a gold-set dataframe against the schema.

    Raises ValueError enumerating all validation failures:
    - Missing required columns
    - Columns with wrong dtype
    - Invalid enumeration values in corpus, sample_stratum
    - Null values in required columns

    Args:
        df: The dataframe to validate

    Raises:
        ValueError: With all validation issues enumerated in the message
    """
    errors = []

    # Check for missing required columns
    missing_required = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]
    if missing_required:
        errors.append(f"Missing required columns: {missing_required}")

    # Check for null values in required columns
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            null_count = df[col].is_null().sum()
            if null_count > 0:
                errors.append(
                    f"Column '{col}' has {null_count} null values, but is required to be non-null"
                )

    # Check dtypes for present columns
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            expected_dtype = REQUIRED_COLUMNS[col]
            actual_dtype = df.schema[col]
            if actual_dtype != expected_dtype:
                errors.append(
                    f"Column '{col}' has dtype {actual_dtype}, "
                    f"expected {expected_dtype}"
                )

    # Check dtypes for label columns (if present)
    # Label columns can be nullable, so we compare base types
    for col in LABEL_COLUMNS:
        if col in df.columns:
            expected_dtype = LABEL_COLUMNS[col]
            actual_dtype = df.schema[col]
            # Handle nullable types: get base type by removing nullability
            # In Polars, df[col].dtype gives the full dtype; we need to extract base
            actual_base = actual_dtype.base_type() if hasattr(actual_dtype, 'base_type') else actual_dtype
            expected_base = expected_dtype.base_type() if hasattr(expected_dtype, 'base_type') else expected_dtype

            # Also allow Null dtype (when column is all None)
            if actual_dtype != pl.Null and actual_base != expected_base:
                errors.append(
                    f"Column '{col}' has dtype {actual_dtype}, "
                    f"expected {expected_dtype}"
                )

    # Check dtypes for optional columns (if present)
    # Optional columns can be nullable
    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            expected_dtype = OPTIONAL_COLUMNS[col]
            actual_dtype = df.schema[col]
            # Handle nullable types
            actual_base = actual_dtype.base_type() if hasattr(actual_dtype, 'base_type') else actual_dtype
            expected_base = expected_dtype.base_type() if hasattr(expected_dtype, 'base_type') else expected_dtype

            # Allow Null dtype (all None)
            # For float columns, accept both Float32 and Float64 (interchange via casting)
            is_float_type = expected_base in (pl.Float32, pl.Float64)
            actual_is_float = actual_base in (pl.Float32, pl.Float64)

            if actual_dtype != pl.Null and not (is_float_type and actual_is_float) and actual_base != expected_base:
                errors.append(
                    f"Column '{col}' has dtype {actual_dtype}, "
                    f"expected {expected_dtype}"
                )

    # Check enumeration values
    if "corpus" in df.columns:
        invalid_corpus = set(df["corpus"].unique()) - VALID_CORPUS
        if invalid_corpus:
            errors.append(
                f"Column 'corpus' contains invalid values: {invalid_corpus}. "
                f"Expected: {VALID_CORPUS}"
            )

    if "sample_stratum" in df.columns:
        invalid_stratum = set(df["sample_stratum"].unique()) - VALID_SAMPLE_STRATUM
        if invalid_stratum:
            errors.append(
                f"Column 'sample_stratum' contains invalid values: {invalid_stratum}. "
                f"Expected: {VALID_SAMPLE_STRATUM}"
            )

    if errors:
        raise ValueError("\n".join(errors))
