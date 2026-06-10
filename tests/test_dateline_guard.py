"""
Tests for the no-residue dateline guard (`src/preproc/dateline_guard.py`).

This guard is a Python port of the *detection* half of the R extractor
`extract_dateline_block` in `r/dateline/resolve_dateline.R`. The guard
detects datelines that WOULD be stripped by the R extractor (conditional
on date fields, recognized qualifiers, or bare AP-list cities).

Acceptance criteria:
  - AC2.1: Guard passes on correctly-stripped input (no residue)
  - AC2.2: Guard fails loudly on seeded leaks (dateline in input)
  - AC2.3: raw_text and stripped_text relationship (R-side correctness)
"""

import polars as pl
import pytest

from src.data_setup.data import data_from_parquet
from src.preproc.dateline_guard import has_dateline_prefix, assert_no_dateline_residue


class TestHasDatelinePrefix:
    """Unit tests for `has_dateline_prefix` detection function."""

    def test_empty_string_returns_false(self):
        """Empty string has no dateline."""
        assert has_dateline_prefix("") is False

    def test_none_returns_false(self):
        """None (if somehow passed) should not crash; return False."""
        assert has_dateline_prefix(None) is False

    def test_simple_dateline_with_em_dash(self):
        """WASHINGTON, July 30 — text → True (date field triggers strip)."""
        assert has_dateline_prefix("WASHINGTON, July 30 — text") is True

    def test_dateline_with_country_qualifier(self):
        """LISBON, Portugal — Officials said. → True (recognized country)."""
        assert has_dateline_prefix("LISBON, Portugal — Officials said.") is True

    def test_dateline_with_spaced_hyphen(self):
        """WASHINGTON, March 2 - A girl... → True (spaced hyphen is valid)."""
        assert has_dateline_prefix("WASHINGTON, March 2 - A girl from Westmont") is True

    def test_special_credit_line_prefix(self):
        """Special to The New York Times CHICAGO — text → True."""
        assert (
            has_dateline_prefix("Special to The New York Times CHICAGO — text") is True
        )

    def test_special_credit_case_insensitive(self):
        """Case-insensitive credit line stripping."""
        assert (
            has_dateline_prefix(
                "special to the new york times WASHINGTON — text"
            )
            is True
        )

    def test_regular_prose_returns_false(self):
        """The workers met to discuss terms. → False (not a dateline)."""
        assert has_dateline_prefix("The workers met to discuss terms.") is False

    def test_emphasis_caps_lede_not_stripped(self):
        """PILOBOLUS - that dance troupe specializing... → False (no date/qualifier/AP city)."""
        assert (
            has_dateline_prefix(
                "PILOBOLUS - that dance troupe specializing in mad scrambles"
            )
            is False
        )

    def test_memory_emphasis_caps_not_stripped(self):
        """MEMORY, memory - is there ever enough of it? → False (unrecognized qualifier)."""
        assert (
            has_dateline_prefix("MEMORY, memory - is there ever enough of it?")
            is False
        )

    def test_dateline_with_em_dash_variant(self):
        """Handles em-dash correctly."""
        assert has_dateline_prefix("BOSTON, Jan. 15 — Text here.") is True

    def test_dateline_with_double_dash(self):
        """Handles -- (double hyphen) as delimiter."""
        assert has_dateline_prefix("CHICAGO, Sept. 3 -- Text here.") is True

    def test_bare_recognized_us_city(self):
        """Bare AP-30 US city (no qualifier) → True (AP-list lookup)."""
        # NEW YORK is in ap_us_cities, so it should be stripped
        assert has_dateline_prefix("NEW YORK — Text") is True

    def test_bare_recognized_foreign_city(self):
        """Bare AP-46 foreign city (no qualifier) → True (AP-list lookup)."""
        # BERLIN is in ap_foreign_cities, so it should be stripped
        assert has_dateline_prefix("BERLIN — Text") is True

    def test_unrecognized_bare_caps_block_not_stripped(self):
        """UNKNOWNCITY — Text → False (not in AP lists, no qualifier)."""
        assert has_dateline_prefix("UNKNOWNCITY — Text") is False


class TestAssertNoDatelineResidue:
    """Unit tests for the assertion function."""

    def test_clean_texts_do_not_raise(self):
        """Texts with no datelines pass without raising."""
        texts = [
            "The workers met to discuss.",
            "A protest was held yesterday.",
            "",
        ]
        # Should not raise
        assert_no_dateline_residue(texts)

    def test_none_and_empty_in_iterable_do_not_raise(self):
        """None and empty strings in the iterable do not raise (Phase-4 runtime safety)."""
        texts = [
            None,
            "",
            "Clean text without dateline",
            "",
            None,
        ]
        # Should not raise—None and empty strings are safe
        assert_no_dateline_residue(texts)

    def test_one_dateline_raises(self):
        """One text with a dateline raises ValueError."""
        texts = ["Clean text", "WASHINGTON — leaked dateline", "Another clean"]
        with pytest.raises(ValueError, match="Dateline residue detected"):
            assert_no_dateline_residue(texts)

    def test_multiple_datelines_reported(self):
        """Multiple datelines are all reported in the error."""
        texts = ["BOSTON — bad", "CHICAGO — bad", "DENVER — bad"]
        with pytest.raises(ValueError) as exc_info:
            assert_no_dateline_residue(texts)
        error_msg = str(exc_info.value)
        assert "3" in error_msg  # Count of offenders


class TestAC21_NoResidueOnCleanData:
    """AC2.1: Guard passes on correctly-stripped input."""

    def test_fixture_parquet_clean_input(self, tmp_path):
        """Parquet with clean stripped_text passes the guard."""
        # Write a parquet with datelined raw_text and correctly-stripped stripped_text
        corpus_dir = tmp_path / "us_filter"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame(
            {
                "id": ["a1", "a2", "a3"],
                "headline": ["Title A", "Title B", "Title C"],
                "raw_text": [
                    "WASHINGTON, July 30 — Text about the protest.",
                    "Clean lead paragraph.",
                    "BOSTON — Another article.",
                ],
                "stripped_text": [
                    "Text about the protest.",
                    "Clean lead paragraph.",
                    "Another article.",
                ],
                "label_source": ["test", "test", "test"],
            }
        )
        df.write_parquet(corpus_dir / "test_data.parquet")

        # Load and check
        result = data_from_parquet(
            tmp_path,
            db_folder="us_filter",
            lead_column="stripped_text",
            addl_columns=["raw_text", "label_source"],
        )

        # Guard should not raise on clean stripped_text
        assert_no_dateline_residue(result["stripped_text"].to_list())


class TestAC22_GuardDetectsLeaks:
    """AC2.2: Guard fails on seeded leaks."""

    def test_seeded_leak_in_stripped_text_raises(self, tmp_path):
        """If stripped_text still contains a dateline, guard raises."""
        corpus_dir = tmp_path / "us_filter"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame(
            {
                "id": ["leaked1"],
                "headline": ["Title"],
                "raw_text": ["WASHINGTON — Original"],
                "stripped_text": ["WASHINGTON — Oops still there"],
                "label_source": ["test"],
            }
        )
        df.write_parquet(corpus_dir / "test_data.parquet")

        result = data_from_parquet(
            tmp_path,
            db_folder="us_filter",
            lead_column="stripped_text",
            addl_columns=["raw_text", "label_source"],
        )

        # Guard should raise because stripped_text contains a dateline
        with pytest.raises(ValueError, match="Dateline residue detected"):
            assert_no_dateline_residue(result["stripped_text"].to_list())

    def test_has_dateline_prefix_returns_true_on_leak(self):
        """has_dateline_prefix detects the leaked dateline."""
        leaked_text = "WASHINGTON — Oops still there"
        assert has_dateline_prefix(leaked_text) is True


class TestAC23_RawVsStripped:
    """AC2.3: raw_text differs from stripped_text by the dateline span."""

    def test_datelined_raw_text_vs_clean_stripped(self, tmp_path):
        """raw_text with dateline, stripped_text without it."""
        corpus_dir = tmp_path / "us_filter"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame(
            {
                "id": ["datelined1"],
                "headline": ["Title"],
                "raw_text": ["WASHINGTON, July 30 — The workers assembled."],
                "stripped_text": ["The workers assembled."],
                "label_source": ["test"],
            }
        )
        df.write_parquet(corpus_dir / "test_data.parquet")

        result = data_from_parquet(
            tmp_path,
            db_folder="us_filter",
            lead_column="stripped_text",
            addl_columns=["raw_text", "label_source"],
        )

        row = result.filter(pl.col("id") == "datelined1")
        raw = row["raw_text"][0]
        stripped = row["stripped_text"][0]

        # raw should have the dateline prefix
        assert has_dateline_prefix(raw) is True

        # stripped should NOT have it
        assert has_dateline_prefix(stripped) is False

        # The stripped text should appear at the end of raw (modulo whitespace)
        # (Exact position depends on R extractor's match_len, but at minimum
        # the dateline was removed)
        assert stripped in raw or raw.strip().endswith(stripped.strip())
