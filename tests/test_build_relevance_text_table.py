"""Tests for src.build_relevance_text_table (text-bearing relevance population).

Covers the pure text-assembly helper (assemble_headline_with_lead), the
targeted api_corpus text reader (read_api_text), and an end-to-end
integration test of main() against synthetic fixtures verifying label
derivation + fused US gate + holdout exclusion + text attach together.
"""

from __future__ import annotations

import polars as pl
import pytest

import src.config as config
import src.build_relevance_text_table as brtt
from src.build_relevance_text_table import assemble_headline_with_lead, read_api_text


# ---------------------------------------------------------------------------
# assemble_headline_with_lead (pure)
# ---------------------------------------------------------------------------
class TestAssembleHeadlineWithLead:
    def test_basic_concatenation(self):
        df = pl.DataFrame({"headline": ["A"], "lead_paragraph": ["lede"]})
        out = assemble_headline_with_lead(df)
        assert out["headline_with_lead"].to_list() == ["A</s>lede"]

    def test_null_headline_becomes_empty_string(self):
        df = pl.DataFrame({"headline": [None], "lead_paragraph": ["lede"]})
        out = assemble_headline_with_lead(df)
        assert out["headline_with_lead"].to_list() == ["</s>lede"]

    def test_null_lead_becomes_empty_string(self):
        df = pl.DataFrame({"headline": ["A"], "lead_paragraph": [None]})
        out = assemble_headline_with_lead(df)
        assert out["headline_with_lead"].to_list() == ["A</s>"]

    def test_literal_na_string_becomes_empty(self):
        """Legacy upstream export convention: literal 'NA' string, not a
        true polars null. Must be treated the same as null."""
        df = pl.DataFrame({"headline": ["NA"], "lead_paragraph": ["NA"]})
        out = assemble_headline_with_lead(df)
        assert out["headline_with_lead"].to_list() == ["</s>"]

    def test_custom_lead_column(self):
        df = pl.DataFrame({"headline": ["A"], "stripped_text": ["s"]})
        out = assemble_headline_with_lead(df, lead_column="stripped_text")
        assert out["headline_with_lead"].to_list() == ["A</s>s"]

    def test_preserves_other_columns(self):
        df = pl.DataFrame({"id": ["x1"], "headline": ["A"], "lead_paragraph": ["b"]})
        out = assemble_headline_with_lead(df)
        assert "id" in out.columns
        assert out["id"].to_list() == ["x1"]


# ---------------------------------------------------------------------------
# read_api_text (targeted shell read)
# ---------------------------------------------------------------------------
class TestReadApiText:
    def _write_corpus(self, tmp_path, rows):
        d = tmp_path / "api_corpus"
        d.mkdir()
        pl.DataFrame(rows).write_parquet(d / "1990.parquet")
        return d

    def test_filters_to_wanted_ids(self, tmp_path):
        corpus_dir = self._write_corpus(tmp_path, {
            "id": ["a", "b", "c"],
            "headline": ["HA", "HB", "HC"],
            "lead_paragraph": ["la", "lb", "lc"],
        })
        out = read_api_text(["a", "c"], corpus_dir=corpus_dir)
        assert sorted(out["id"].to_list()) == ["a", "c"]

    def test_multiple_files_concatenated(self, tmp_path):
        d = tmp_path / "api_corpus"
        d.mkdir()
        pl.DataFrame({"id": ["a"], "headline": ["HA"], "lead_paragraph": ["la"]}).write_parquet(
            d / "1990.parquet"
        )
        pl.DataFrame({"id": ["b"], "headline": ["HB"], "lead_paragraph": ["lb"]}).write_parquet(
            d / "1991.parquet"
        )
        out = read_api_text(["a", "b"], corpus_dir=d)
        assert sorted(out["id"].to_list()) == ["a", "b"]

    def test_no_matching_ids_returns_empty_with_schema(self, tmp_path):
        corpus_dir = self._write_corpus(tmp_path, {
            "id": ["a"], "headline": ["HA"], "lead_paragraph": ["la"],
        })
        out = read_api_text(["nonexistent"], corpus_dir=corpus_dir)
        assert out.height == 0
        assert set(out.columns) == {"id", "headline", "lead_paragraph"}

    def test_empty_corpus_dir_returns_empty(self, tmp_path):
        d = tmp_path / "empty_corpus"
        d.mkdir()
        out = read_api_text(["a"], corpus_dir=d)
        assert out.height == 0


# ---------------------------------------------------------------------------
# main() integration: label derivation + fused US gate + holdout exclusion +
# text attach, all together, against a tiny synthetic population.
# ---------------------------------------------------------------------------
class TestMainIntegration:
    """Fixture population (5 ids):
    - pos1:            candidate, us_logit=0.9, glocation "United States" -> US-restricted positive
    - pos2_foreignonly: candidate, us_logit=0.9, glocation "Cuba" only -> clearly foreign -> gated out
    - neg1:            reliable-negative, us_logit=0.1 (fails ML gate), no location signal
    - unl1:            unlabeled background, us_logit=0.9, no location signal -> US background
    - hold1:           candidate, us_logit=0.9, glocation "United States" -- WOULD be a valid
                        US-restricted positive, but is in the ICA-eval holdout -> must be dropped
    """

    IDS = ["pos1", "pos2_foreignonly", "neg1", "unl1", "hold1"]

    def _keywords(self, glocation):
        if glocation is None:
            return []
        return [{"type": "glocations", "value": glocation, "rank": 1, "major": "N"}]

    def _setup(self, tmp_path, monkeypatch):
        # --- embed cache meta (id, year, us_logit) ---
        cache_dir = tmp_path / "embed_cache" / "relevance_train"
        cache_dir.mkdir(parents=True)
        pl.DataFrame({
            "id": self.IDS,
            "year": ["1990"] * 5,
            "us_logit": [0.9, 0.9, 0.1, 0.9, 0.9],
        }).write_parquet(cache_dir / "shard_000_meta.parquet")
        monkeypatch.setattr(config, "CCA_EMBED_CACHE_DIR", tmp_path / "embed_cache")

        # --- relevance/ data products ---
        rel_dir = tmp_path / "relevance"
        rel_dir.mkdir()
        pl.DataFrame({"id": ["pos1", "pos2_foreignonly", "hold1"]}).write_parquet(
            rel_dir / "candidates.parquet"
        )
        pl.DataFrame({"id": ["neg1"]}).write_parquet(rel_dir / "reliable_negatives.parquet")
        monkeypatch.setattr(brtt, "RELEVANCE_DIR", rel_dir)

        # --- holdout ---
        validation_dir = tmp_path / "validation"
        validation_dir.mkdir()
        pl.DataFrame({"id": ["hold1"]}).write_parquet(validation_dir / "ica_holdout_ids.parquet")
        monkeypatch.setattr(config, "ICA_HOLDOUT_IDS", validation_dir / "ica_holdout_ids.parquet")

        # --- api_corpus (text + location signals) ---
        corpus_dir = tmp_path / "api_corpus"
        corpus_dir.mkdir()
        glocations = {
            "pos1": "United States",
            "pos2_foreignonly": "Cuba",
            "neg1": None,
            "unl1": None,
            "hold1": "United States",
        }
        pl.DataFrame(
            {
                "id": self.IDS,
                "headline": [f"H_{i}" for i in self.IDS],
                "lead_paragraph": [f"lede_{i}" for i in self.IDS],
                "keywords": [self._keywords(glocations[i]) for i in self.IDS],
                "news_desk": [None] * 5,
                "section_name": [None] * 5,
            },
            schema_overrides={"news_desk": pl.Utf8, "section_name": pl.Utf8},
        ).write_parquet(corpus_dir / "1990.parquet")
        monkeypatch.setattr(config, "API_CORPUS_DIR", corpus_dir)

        # --- output path ---
        out_path = tmp_path / "relevance_text_table.parquet"
        monkeypatch.setattr(config, "RELEVANCE_TEXT_TABLE", out_path)
        return out_path

    def test_holdout_excluded_from_output(self, tmp_path, monkeypatch):
        out_path = self._setup(tmp_path, monkeypatch)
        brtt.main()
        table = pl.read_parquet(out_path)
        assert "hold1" not in table["id"].to_list()

    def test_row_count_matches_population_minus_holdout(self, tmp_path, monkeypatch):
        out_path = self._setup(tmp_path, monkeypatch)
        brtt.main()
        table = pl.read_parquet(out_path)
        # 5 in the cache, 1 (hold1) dropped as holdout -> 4 remain
        assert table.height == 4

    def test_clearly_foreign_positive_is_gated_out(self, tmp_path, monkeypatch):
        out_path = self._setup(tmp_path, monkeypatch)
        brtt.main()
        table = pl.read_parquet(out_path)
        row = table.filter(pl.col("id") == "pos2_foreignonly")
        assert row.height == 1
        assert row["cca_label"].item() == 0  # US-restricted: foreign candidate is not a positive
        assert row["us"].item() is False

    def test_us_positive_is_kept(self, tmp_path, monkeypatch):
        out_path = self._setup(tmp_path, monkeypatch)
        brtt.main()
        table = pl.read_parquet(out_path)
        row = table.filter(pl.col("id") == "pos1")
        assert row["cca_label"].item() == 1
        assert row["us"].item() is True

    def test_reliable_negative_flagged(self, tmp_path, monkeypatch):
        out_path = self._setup(tmp_path, monkeypatch)
        brtt.main()
        table = pl.read_parquet(out_path)
        row = table.filter(pl.col("id") == "neg1")
        assert row["reliable_neg"].item() is True

    def test_text_attached(self, tmp_path, monkeypatch):
        out_path = self._setup(tmp_path, monkeypatch)
        brtt.main()
        table = pl.read_parquet(out_path)
        row = table.filter(pl.col("id") == "pos1")
        assert row["headline"].item() == "H_pos1"
        assert row["headline_with_lead"].item() == "H_pos1</s>lede_pos1"

    def test_missing_text_raises(self, tmp_path, monkeypatch):
        """If an id in the population has no matching api_corpus row, main()
        must raise rather than silently dropping it."""
        self._setup(tmp_path, monkeypatch)
        # Remove one id's row from the api_corpus fixture entirely.
        corpus_dir = config.API_CORPUS_DIR
        df = pl.read_parquet(corpus_dir / "1990.parquet")
        df.filter(pl.col("id") != "unl1").write_parquet(corpus_dir / "1990.parquet")

        with pytest.raises(ValueError, match="not found in api_corpus text"):
            brtt.main()
