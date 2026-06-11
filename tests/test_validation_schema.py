# pattern: Functional Core (pure data validation)
"""Tests for src.validation.schema — gold-set validation."""

import pytest
import polars as pl

from src.validation.schema import validate_gold_set


class TestValidateGoldSet:
    """Validation of gold-set dataframes against the schema."""

    def test_valid_frame_passes(self):
        """A frame with all required columns and correct dtypes passes."""
        df = pl.DataFrame({
            "id": ["1", "2"],
            "corpus": ["api", "ldc"],
            "year": [1990, 1995],
            "news_desk": ["National", "World"],
            "section_name": ["US", "International"],
            "headline": ["Test", "Article"],
            "lead_paragraph": ["Lead 1", "Lead 2"],
            "sample_stratum": ["random_pre1986", "doca_matched"],
        })
        # Should not raise
        validate_gold_set(df)

    def test_missing_required_column_id_raises(self):
        """Missing 'id' raises ValueError enumerating the missing columns."""
        df = pl.DataFrame({
            "corpus": ["api"],
            "year": [1990],
            "news_desk": ["National"],
            "section_name": ["US"],
            "headline": ["Test"],
            "lead_paragraph": ["Lead"],
            "sample_stratum": ["random_pre1986"],
        })
        with pytest.raises(ValueError, match="id"):
            validate_gold_set(df)

    def test_missing_multiple_required_columns_enumerates_all(self):
        """Missing multiple required columns enumerates all in error message."""
        df = pl.DataFrame({
            "corpus": ["api"],
            "headline": ["Test"],
        })
        with pytest.raises(ValueError) as exc_info:
            validate_gold_set(df)
        msg = str(exc_info.value)
        # Check that multiple missing columns are enumerated
        assert "id" in msg
        assert "year" in msg or "news_desk" in msg

    def test_invalid_corpus_enum_value_raises(self):
        """corpus with invalid enum value raises ValueError."""
        df = pl.DataFrame({
            "id": ["1"],
            "corpus": ["invalid"],
            "year": [1990],
            "news_desk": ["National"],
            "section_name": ["US"],
            "headline": ["Test"],
            "lead_paragraph": ["Lead"],
            "sample_stratum": ["random_pre1986"],
        })
        with pytest.raises(ValueError, match="corpus"):
            validate_gold_set(df)

    def test_invalid_sample_stratum_enum_value_raises(self):
        """sample_stratum with invalid enum value raises ValueError."""
        df = pl.DataFrame({
            "id": ["1"],
            "corpus": ["api"],
            "year": [1990],
            "news_desk": ["National"],
            "section_name": ["US"],
            "headline": ["Test"],
            "lead_paragraph": ["Lead"],
            "sample_stratum": ["invalid_stratum"],
        })
        with pytest.raises(ValueError, match="sample_stratum"):
            validate_gold_set(df)

    def test_null_in_required_column_raises(self):
        """A null value in a required column raises (required = non-null)."""
        df = pl.DataFrame({
            "id": ["1", "2"],
            "corpus": ["api", "ldc"],
            "year": [1990, 1995],
            "news_desk": ["National", None],
            "section_name": ["US", "International"],
            "headline": ["Test", "Article"],
            "lead_paragraph": ["Lead 1", "Lead 2"],
            "sample_stratum": ["random_pre1986", "doca_matched"],
        })
        with pytest.raises(ValueError, match="null values"):
            validate_gold_set(df)

    def test_label_columns_present_but_null_tolerated(self):
        """Label columns can be present and null."""
        df = pl.DataFrame({
            "id": ["1"],
            "corpus": ["api"],
            "year": [1990],
            "news_desk": ["National"],
            "section_name": ["US"],
            "headline": ["Test"],
            "lead_paragraph": ["Lead"],
            "sample_stratum": ["random_pre1986"],
            "us_event": [None],
            "event_location": [None],
            "cca_event": [None],
            "immig_relevant": [None],
            "ica_event": [None],
        })
        # Should not raise
        validate_gold_set(df)

    def test_alt_corpus_id_optional_nullable(self):
        """alt_corpus_id is optional and can be null."""
        df = pl.DataFrame({
            "id": ["1"],
            "corpus": ["api"],
            "year": [1990],
            "news_desk": ["National"],
            "section_name": ["US"],
            "headline": ["Test"],
            "lead_paragraph": ["Lead"],
            "sample_stratum": ["random_pre1986"],
            "alt_corpus_id": [None],
        })
        # Should not raise
        validate_gold_set(df)

    def test_alt_corpus_id_missing_is_ok(self):
        """alt_corpus_id missing entirely is ok."""
        df = pl.DataFrame({
            "id": ["1"],
            "corpus": ["api"],
            "year": [1990],
            "news_desk": ["National"],
            "section_name": ["US"],
            "headline": ["Test"],
            "lead_paragraph": ["Lead"],
            "sample_stratum": ["random_pre1986"],
        })
        # Should not raise
        validate_gold_set(df)

    def test_wrong_dtype_raises(self):
        """Column with wrong dtype raises ValueError."""
        df = pl.DataFrame({
            "id": [1],  # Wrong: should be str
            "corpus": ["api"],
            "year": [1990],
            "news_desk": ["National"],
            "section_name": ["US"],
            "headline": ["Test"],
            "lead_paragraph": ["Lead"],
            "sample_stratum": ["random_pre1986"],
        })
        with pytest.raises(ValueError, match="id|dtype"):
            validate_gold_set(df)

    def test_valid_corpus_api(self):
        """corpus='api' is valid."""
        df = pl.DataFrame({
            "id": ["1"],
            "corpus": ["api"],
            "year": [1990],
            "news_desk": ["National"],
            "section_name": ["US"],
            "headline": ["Test"],
            "lead_paragraph": ["Lead"],
            "sample_stratum": ["random_pre1986"],
        })
        validate_gold_set(df)

    def test_valid_corpus_ldc(self):
        """corpus='ldc' is valid."""
        df = pl.DataFrame({
            "id": ["1"],
            "corpus": ["ldc"],
            "year": [1990],
            "news_desk": ["National"],
            "section_name": ["US"],
            "headline": ["Test"],
            "lead_paragraph": ["Lead"],
            "sample_stratum": ["random_pre1986"],
        })
        validate_gold_set(df)

    def test_valid_sample_strata(self):
        """All valid sample_stratum values are accepted."""
        for stratum in ["doca_matched", "random_pre1986", "ambiguous"]:
            df = pl.DataFrame({
                "id": ["1"],
                "corpus": ["api"],
                "year": [1990],
                "news_desk": ["National"],
                "section_name": ["US"],
                "headline": ["Test"],
                "lead_paragraph": ["Lead"],
                "sample_stratum": [stratum],
            })
            validate_gold_set(df)

    def test_empty_dataframe_raises(self):
        """Empty dataframe raises (missing required columns)."""
        df = pl.DataFrame()
        with pytest.raises(ValueError):
            validate_gold_set(df)
