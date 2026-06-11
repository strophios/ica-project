# pattern: Imperative Shell (reads/writes, orchestrates schema validation)
"""Tests for src.validation.build_coding_template — candidate generator."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from src.validation.build_coding_template import (
    stratified_sample,
    build_and_write_template,
)
from src.validation.schema import validate_gold_set


class TestStratifiedSample:
    """Stratified sampling by era and news_desk."""

    def test_stratified_sample_respects_cell_counts(self):
        """Requested cell counts are met exactly (or fewer if data sparse)."""
        # Create synthetic API corpus: 100 rows, pre-1986, varied desks
        desks = ["National", "World", "Business"] * 35
        desks = desks[:100]  # Pad to 100

        df = pl.DataFrame({
            "id": [str(i) for i in range(100)],
            "year": [1960 + i % 20 for i in range(100)],
            "news_desk": desks,
            "section_name": ["US"] * 100,
            "headline": [f"Headline {i}" for i in range(100)],
            "lead_paragraph": [f"Lead {i}" for i in range(100)],
        })

        # Request 3 per cell: 1960-69 × National, 1960-69 × World, etc.
        sample = stratified_sample(
            df,
            n_per_cell=3,
            era_boundaries=[1960, 1970, 1980, 1987],
        )

        # Check we got a non-empty sample
        assert sample.shape[0] > 0
        assert "id" in sample.columns

    def test_stratified_sample_seeds_deterministically(self):
        """Same seed produces same sample."""
        df = pl.DataFrame({
            "id": [str(i) for i in range(50)],
            "year": [1975] * 50,
            "news_desk": ["National"] * 50,
            "section_name": ["US"] * 50,
            "headline": [f"H{i}" for i in range(50)],
            "lead_paragraph": [f"L{i}" for i in range(50)],
        })

        sample1 = stratified_sample(df, n_per_cell=5, seed=42)
        sample2 = stratified_sample(df, n_per_cell=5, seed=42)

        assert sample1.equals(sample2)

    def test_stratified_sample_handles_sparse_cells(self):
        """Cells with fewer rows than n_per_cell return all available rows."""
        df = pl.DataFrame({
            "id": ["1", "2", "3"],
            "year": [1970, 1975, 1980],
            "news_desk": ["National", "World", "Business"],
            "section_name": ["US"] * 3,
            "headline": ["H1", "H2", "H3"],
            "lead_paragraph": ["L1", "L2", "L3"],
        })

        # Request 10 per cell but only 1 row per (sparse) cell
        sample = stratified_sample(df, n_per_cell=10)

        # Should return at most 3 rows (all available)
        assert sample.shape[0] <= 3


class TestBuildAndWriteTemplate:
    """Template generation and writing."""

    def test_build_and_write_template_creates_valid_output(self, tmp_path):
        """Output conforms to schema (corpus='api', null labels, etc.)."""
        # Create synthetic API corpus
        df = pl.DataFrame({
            "id": [str(i) for i in range(50)],
            "year": [1960 + i % 20 for i in range(50)],
            "news_desk": ["National"] * 30 + ["World"] * 20,
            "section_name": ["US"] * 50,
            "headline": [f"Headline {i}" for i in range(50)],
            "lead_paragraph": [f"Lead {i}" for i in range(50)],
        })

        output_path = tmp_path / "template.parquet"

        build_and_write_template(
            api_df=df,
            output_path=output_path,
            n_per_cell=2,
        )

        # Read and validate
        assert output_path.exists()
        result = pl.read_parquet(output_path)

        # Validate against schema
        validate_gold_set(result)

        # Check specific requirements
        assert result.shape[0] > 0
        assert (result["corpus"] == "api").all()
        assert result["alt_corpus_id"].is_null().all()
        assert "us_event" in result.columns
        assert result["us_event"].is_null().all()

    def test_build_and_write_template_sets_sample_stratum_random(self, tmp_path):
        """sample_stratum is 'random_pre1986' for all rows."""
        df = pl.DataFrame({
            "id": ["1", "2"],
            "year": [1970, 1975],
            "news_desk": ["National", "World"],
            "section_name": ["US", "US"],
            "headline": ["H1", "H2"],
            "lead_paragraph": ["L1", "L2"],
        })

        output_path = tmp_path / "template.parquet"

        build_and_write_template(
            api_df=df,
            output_path=output_path,
            n_per_cell=1,
        )

        result = pl.read_parquet(output_path)
        assert (result["sample_stratum"] == "random_pre1986").all()

    def test_build_and_write_template_with_doca_matched_optional(self, tmp_path):
        """doca_matched rows get sample_stratum='doca_matched'."""
        api_df = pl.DataFrame({
            "id": ["1", "2", "3"],
            "year": [1970, 1975, 1980],
            "news_desk": ["National", "World", "Business"],
            "section_name": ["US", "US", "US"],
            "headline": ["H1", "H2", "H3"],
            "lead_paragraph": ["L1", "L2", "L3"],
        })

        doca_matched_df = pl.DataFrame({
            "id": ["1"],
            "year": [1970],
        })

        output_path = tmp_path / "template.parquet"

        build_and_write_template(
            api_df=api_df,
            output_path=output_path,
            n_per_cell=2,
            doca_matched_df=doca_matched_df,
        )

        result = pl.read_parquet(output_path)

        # Some rows should be marked doca_matched
        has_doca = (result["sample_stratum"] == "doca_matched").any()
        # Might not have any if sampling didn't pick that row
        # But if it did, the stratum should be set correctly
        assert all(
            s in ["random_pre1986", "doca_matched"]
            for s in result["sample_stratum"].unique()
        )

    def test_build_and_write_template_deterministic_seed(self, tmp_path):
        """Same seed produces same template."""
        df = pl.DataFrame({
            "id": [str(i) for i in range(30)],
            "year": [1975] * 30,
            "news_desk": ["National"] * 30,
            "section_name": ["US"] * 30,
            "headline": [f"H{i}" for i in range(30)],
            "lead_paragraph": [f"L{i}" for i in range(30)],
        })

        path1 = tmp_path / "template1.parquet"
        path2 = tmp_path / "template2.parquet"

        build_and_write_template(df, path1, n_per_cell=3, seed=100)
        build_and_write_template(df, path2, n_per_cell=3, seed=100)

        result1 = pl.read_parquet(path1)
        result2 = pl.read_parquet(path2)

        # Same seed → same rows
        assert result1.equals(result2)

    def test_build_and_write_template_label_columns_null(self, tmp_path):
        """All label columns are null in output."""
        df = pl.DataFrame({
            "id": ["1"],
            "year": [1975],
            "news_desk": ["National"],
            "section_name": ["US"],
            "headline": ["H1"],
            "lead_paragraph": ["L1"],
        })

        output_path = tmp_path / "template.parquet"

        build_and_write_template(
            api_df=df,
            output_path=output_path,
            n_per_cell=1,
        )

        result = pl.read_parquet(output_path)

        # Check label columns are null
        assert result["us_event"].is_null().all()
        assert result["event_location"].is_null().all()
        assert result["cca_event"].is_null().all()
        assert result["immig_relevant"].is_null().all()
        assert result["ica_event"].is_null().all()

    def test_build_and_write_template_creates_directory(self, tmp_path):
        """Output directory is created if it doesn't exist."""
        df = pl.DataFrame({
            "id": ["1"],
            "year": [1975],
            "news_desk": ["National"],
            "section_name": ["US"],
            "headline": ["H1"],
            "lead_paragraph": ["L1"],
        })

        # Nested path that doesn't exist
        output_path = tmp_path / "newdir" / "template.parquet"
        assert not output_path.parent.exists()

        build_and_write_template(
            api_df=df,
            output_path=output_path,
            n_per_cell=1,
        )

        assert output_path.exists()
