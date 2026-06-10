"""
Tests for missing-value handling and string concatenation in
`data_from_parquet` (`src/data_setup/data.py`).

The function reads parquet files from `{project_root}/{db_folder}/**/*.parquet`
with `hive_partitioning=True`, replaces missing values in `headline` and
`lead_paragraph` (both true polars nulls and the literal string "NA" become
""), then builds `headline_with_lead = headline + "</s>" + lead_paragraph`.

Invariants tested:
  1. True polars nulls in `headline` → "" in output.
  2. True polars nulls in `lead_paragraph` → "" in output.
  3. Literal "NA" strings in `headline` → "" in output.
  4. Literal "NA" strings in `lead_paragraph` → "" in output.
  5. Mixed: null headline + "NA" lead → both become "".
  6. `headline_with_lead` concatenation is correct across all null/NA/normal
     combinations (separates with "</s>").
  7. Normal (non-null, non-"NA") values pass through unchanged.

Implementation note: tests write a small parquet file into pytest's `tmp_path`
fixture and call `data_from_parquet(tmp_path, db_folder=..., addl_columns=None)`
so that only `id`, `headline`, `lead_paragraph` are needed in the fixture data.
"""

import polars as pl
import pytest

from src.data_setup.data import data_from_parquet


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _write_parquet(
    tmp_path,
    rows: list[dict],
    db_folder: str = "ldc_corpus",
    lead_column: str = "lead_paragraph",
) -> None:
    """Write a list of row-dicts to a parquet file under
    `tmp_path / db_folder / test_data.parquet`.

    Each dict must have keys: id, headline, and the specified lead_column name.
    Use None for a true polars null.
    """
    corpus_dir = tmp_path / db_folder
    corpus_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "id": [r["id"] for r in rows],
            "headline": pl.Series(
                "headline", [r["headline"] for r in rows], dtype=pl.Utf8
            ),
            lead_column: pl.Series(
                lead_column,
                [r[lead_column] for r in rows],
                dtype=pl.Utf8,
            ),
        }
    )
    df.write_parquet(corpus_dir / "test_data.parquet")


def _row_by_id(df: pl.DataFrame, row_id: str) -> pl.DataFrame:
    return df.filter(pl.col("id") == row_id)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

class TestParquetMissingValueHandling:
    """Tests for null and "NA" replacement in `data_from_parquet`."""

    def test_null_headline_becomes_empty_string(self, tmp_path):
        """True polars null in `headline` → "" in the returned dataframe."""
        _write_parquet(
            tmp_path,
            [{"id": "r1", "headline": None, "lead_paragraph": "some lead"}],
        )
        result = data_from_parquet(tmp_path, db_folder="ldc_corpus")
        row = _row_by_id(result, "r1")
        assert row["headline"][0] == ""

    def test_null_lead_becomes_empty_string(self, tmp_path):
        """True polars null in `lead_paragraph` → "" in the returned dataframe."""
        _write_parquet(
            tmp_path,
            [{"id": "r1", "headline": "some headline", "lead_paragraph": None}],
        )
        result = data_from_parquet(tmp_path, db_folder="ldc_corpus")
        row = _row_by_id(result, "r1")
        assert row["lead_paragraph"][0] == ""

    def test_na_string_headline_becomes_empty_string(self, tmp_path):
        """Literal string "NA" in `headline` → "" in the returned dataframe."""
        _write_parquet(
            tmp_path,
            [{"id": "r1", "headline": "NA", "lead_paragraph": "some lead"}],
        )
        result = data_from_parquet(tmp_path, db_folder="ldc_corpus")
        row = _row_by_id(result, "r1")
        assert row["headline"][0] == ""

    def test_na_string_lead_becomes_empty_string(self, tmp_path):
        """Literal string "NA" in `lead_paragraph` → "" in the returned dataframe."""
        _write_parquet(
            tmp_path,
            [{"id": "r1", "headline": "some headline", "lead_paragraph": "NA"}],
        )
        result = data_from_parquet(tmp_path, db_folder="ldc_corpus")
        row = _row_by_id(result, "r1")
        assert row["lead_paragraph"][0] == ""

    def test_mixed_null_headline_and_na_lead(self, tmp_path):
        """Row with null headline and "NA" lead → both fields become ""."""
        _write_parquet(
            tmp_path,
            [{"id": "r1", "headline": None, "lead_paragraph": "NA"}],
        )
        result = data_from_parquet(tmp_path, db_folder="ldc_corpus")
        row = _row_by_id(result, "r1")
        assert row["headline"][0] == ""
        assert row["lead_paragraph"][0] == ""

    def test_normal_values_pass_through_unchanged(self, tmp_path):
        """Non-null, non-"NA" values are not modified."""
        _write_parquet(
            tmp_path,
            [{"id": "r1", "headline": "Workers Strike", "lead_paragraph": "Hundreds walked out."}],
        )
        result = data_from_parquet(tmp_path, db_folder="ldc_corpus")
        row = _row_by_id(result, "r1")
        assert row["headline"][0] == "Workers Strike"
        assert row["lead_paragraph"][0] == "Hundreds walked out."


class TestHeadlineWithLeadConcatenation:
    """Tests for `headline_with_lead` concatenation in `data_from_parquet`.

    The column is built as: headline + "</s>" + lead_paragraph, applied after
    null/"NA" replacement, so:
      - empty headline + empty lead → "</s>"
      - normal headline + empty lead → "<headline></s>"
      - empty headline + normal lead → "</s><lead>"
      - normal headline + normal lead → "<headline></s><lead>"
    """

    def test_both_empty_produces_separator_only(self, tmp_path):
        """headline="" and lead="" → headline_with_lead == "</s>"."""
        _write_parquet(
            tmp_path,
            [{"id": "r1", "headline": None, "lead_paragraph": None}],
        )
        result = data_from_parquet(tmp_path, db_folder="ldc_corpus")
        row = _row_by_id(result, "r1")
        assert row["headline_with_lead"][0] == "</s>"

    def test_empty_headline_with_normal_lead(self, tmp_path):
        """headline="" and non-empty lead → "</s>" + lead."""
        _write_parquet(
            tmp_path,
            [{"id": "r1", "headline": "NA", "lead_paragraph": "Some lead text."}],
        )
        result = data_from_parquet(tmp_path, db_folder="ldc_corpus")
        row = _row_by_id(result, "r1")
        assert row["headline_with_lead"][0] == "</s>Some lead text."

    def test_normal_headline_with_empty_lead(self, tmp_path):
        """Non-empty headline and lead="" → headline + "</s>"."""
        _write_parquet(
            tmp_path,
            [{"id": "r1", "headline": "Strike Hits City", "lead_paragraph": None}],
        )
        result = data_from_parquet(tmp_path, db_folder="ldc_corpus")
        row = _row_by_id(result, "r1")
        assert row["headline_with_lead"][0] == "Strike Hits City</s>"

    def test_normal_headline_and_lead(self, tmp_path):
        """Both fields normal → headline + "</s>" + lead."""
        _write_parquet(
            tmp_path,
            [{"id": "r1", "headline": "Workers Walk Out", "lead_paragraph": "Hundreds joined."}],
        )
        result = data_from_parquet(tmp_path, db_folder="ldc_corpus")
        row = _row_by_id(result, "r1")
        assert row["headline_with_lead"][0] == "Workers Walk Out</s>Hundreds joined."

    def test_multiple_rows_concatenated_independently(self, tmp_path):
        """Each row's concatenation is independent; multiple rows all produce
        correct headline_with_lead values."""
        _write_parquet(
            tmp_path,
            [
                {"id": "r1", "headline": "Title A", "lead_paragraph": "Lead A."},
                {"id": "r2", "headline": None, "lead_paragraph": "Lead B."},
                {"id": "r3", "headline": "Title C", "lead_paragraph": "NA"},
            ],
        )
        result = data_from_parquet(tmp_path, db_folder="ldc_corpus")
        assert _row_by_id(result, "r1")["headline_with_lead"][0] == "Title A</s>Lead A."
        assert _row_by_id(result, "r2")["headline_with_lead"][0] == "</s>Lead B."
        assert _row_by_id(result, "r3")["headline_with_lead"][0] == "Title C</s>"


class TestLeadColumnParameterization:
    """Tests for the lead_column parameter in `data_from_parquet`.

    The function now accepts a `lead_column` parameter (keyword, default 'lead_paragraph')
    to enable assembly from alternate lead sources like `stripped_text`.
    """

    def test_default_lead_column_is_lead_paragraph(self, tmp_path):
        """Calling without lead_column parameter uses "lead_paragraph" by default
        (regression test to ensure backward compatibility)."""
        _write_parquet(
            tmp_path,
            [{"id": "r1", "headline": "Title", "lead_paragraph": "Lead text."}],
        )
        result = data_from_parquet(tmp_path, db_folder="ldc_corpus")
        row = _row_by_id(result, "r1")
        assert row["headline_with_lead"][0] == "Title</s>Lead text."
        # Verify the column name is still there
        assert "lead_paragraph" in result.columns

    def test_stripped_text_column_parameter(self, tmp_path):
        """Calling with lead_column="stripped_text" assembles from that column."""
        _write_parquet(
            tmp_path,
            [{"id": "r1", "headline": "Title", "stripped_text": "Stripped lead."}],
            db_folder="us_filter",
            lead_column="stripped_text",
        )
        result = data_from_parquet(
            tmp_path, db_folder="us_filter", lead_column="stripped_text"
        )
        row = _row_by_id(result, "r1")
        assert row["headline_with_lead"][0] == "Title</s>Stripped lead."
        # Verify the stripped_text column is present
        assert "stripped_text" in result.columns

    def test_stripped_text_with_null_handling(self, tmp_path):
        """Null and "NA" in stripped_text are handled the same as lead_paragraph."""
        _write_parquet(
            tmp_path,
            [
                {"id": "r1", "headline": "Title A", "stripped_text": None},
                {"id": "r2", "headline": "Title B", "stripped_text": "NA"},
                {"id": "r3", "headline": "Title C", "stripped_text": "Real text."},
            ],
            db_folder="us_filter",
            lead_column="stripped_text",
        )
        result = data_from_parquet(
            tmp_path, db_folder="us_filter", lead_column="stripped_text"
        )
        assert _row_by_id(result, "r1")["headline_with_lead"][0] == "Title A</s>"
        assert _row_by_id(result, "r2")["headline_with_lead"][0] == "Title B</s>"
        assert _row_by_id(result, "r3")["headline_with_lead"][0] == "Title C</s>Real text."

