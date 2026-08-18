"""Tests for src.build_joint_text_table (the union CCA+rel text-bearing table,
docs/design-plans/2026-08-18-stage4-joint-finetune.md item 2).

Covers the pure population-union and rel-label-derivation helpers, plus an
end-to-end integration test of main() against synthetic fixtures verifying:
union-by-id (rel channel wins on `us_logit`/`year` conflict), cca_label kept
regardless of the US gate, rel_label US-restricted post-fused-gate, reliable
negatives, and holdout exclusion -- all together, mirroring
test_build_relevance_text_table.py's TestMainIntegration.
"""

from __future__ import annotations

import polars as pl
import pytest

import src.config as config
import src.build_joint_text_table as bjtt
from src.build_joint_text_table import apply_rel_label, union_population


# ---------------------------------------------------------------------------
# union_population (pure)
# ---------------------------------------------------------------------------
class TestUnionPopulation:
    def test_disjoint_ids_both_kept(self):
        cca = pl.DataFrame({"id": ["a"], "us_logit": [3.0], "year": ["1990"]})
        rel = pl.DataFrame({"id": ["b"], "us_logit": [0.9], "year": ["1991"]})
        out = union_population(cca, rel).sort("id")
        assert out["id"].to_list() == ["a", "b"]
        assert out["in_cca_pop"].to_list() == [True, False]
        assert out["in_rel_pop"].to_list() == [False, True]

    def test_overlap_id_marked_in_both_pops(self):
        cca = pl.DataFrame({"id": ["x"], "us_logit": [3.0], "year": ["1990"]})
        rel = pl.DataFrame({"id": ["x"], "us_logit": [0.9], "year": ["1990"]})
        out = union_population(cca, rel)
        assert out.height == 1
        assert out["in_cca_pop"].item() is True
        assert out["in_rel_pop"].item() is True

    def test_overlap_prefers_rel_us_logit(self):
        # cca-side raw logit is a large negative number; rel-side calibrated
        # probability is high. The union must carry the REL value.
        cca = pl.DataFrame({"id": ["x"], "us_logit": [-5.0], "year": ["1990"]})
        rel = pl.DataFrame({"id": ["x"], "us_logit": [0.9], "year": ["1990"]})
        out = union_population(cca, rel)
        assert out["us_logit"].item() == pytest.approx(0.9)

    def test_overlap_prefers_rel_year(self):
        cca = pl.DataFrame({"id": ["x"], "us_logit": [3.0], "year": ["1990"]})
        rel = pl.DataFrame({"id": ["x"], "us_logit": [0.9], "year": ["1995"]})
        out = union_population(cca, rel)
        assert out["year"].item() == "1995"

    def test_cca_only_row_keeps_cca_us_logit(self):
        cca = pl.DataFrame({"id": ["a"], "us_logit": [3.0], "year": ["1990"]})
        rel = pl.DataFrame({"id": ["b"], "us_logit": [0.9], "year": ["1990"]})
        out = union_population(cca, rel)
        row = out.filter(pl.col("id") == "a")
        assert row["us_logit"].item() == pytest.approx(3.0)

    def test_dedupes_duplicate_ids_within_a_single_meta(self):
        cca = pl.DataFrame({"id": ["a", "a"], "us_logit": [3.0, 4.0], "year": ["1990", "1991"]})
        rel = pl.DataFrame({"id": ["b"], "us_logit": [0.9], "year": ["1990"]})
        out = union_population(cca, rel)
        assert out.filter(pl.col("id") == "a").height == 1

    def test_row_count_is_union_size(self):
        cca = pl.DataFrame({"id": ["a", "x"], "us_logit": [3.0, -5.0], "year": ["1990", "1990"]})
        rel = pl.DataFrame({"id": ["b", "x"], "us_logit": [0.9, 0.9], "year": ["1990", "1990"]})
        out = union_population(cca, rel)
        assert out.height == 3  # a, b, x (x is the overlap)


# ---------------------------------------------------------------------------
# apply_rel_label (pure)
# ---------------------------------------------------------------------------
class TestApplyRelLabel:
    def test_candidate_and_us_true_is_positive(self):
        table = pl.DataFrame({"id": ["a"], "us": [True]})
        out = apply_rel_label(table, ["a"])
        assert out["rel_label"].to_list() == [1]
        assert out["rel_label"].dtype == pl.Int8

    def test_candidate_but_us_false_is_negative(self):
        """The US-restriction: a candidate that fails the (already fused) US
        gate is NOT a rel positive."""
        table = pl.DataFrame({"id": ["a"], "us": [False]})
        out = apply_rel_label(table, ["a"])
        assert out["rel_label"].to_list() == [0]

    def test_non_candidate_is_negative_regardless_of_us(self):
        table = pl.DataFrame({"id": ["a"], "us": [True]})
        out = apply_rel_label(table, ["other"])
        assert out["rel_label"].to_list() == [0]

    def test_mixed_batch(self):
        table = pl.DataFrame({
            "id": ["cand_us", "cand_notus", "noncand_us"],
            "us": [True, False, True],
        })
        out = apply_rel_label(table, ["cand_us", "cand_notus"])
        assert out["rel_label"].to_list() == [1, 0, 0]


# ---------------------------------------------------------------------------
# main() integration: union + label derivation + fused US gate + holdout +
# text attach, all together, against a tiny synthetic population.
# ---------------------------------------------------------------------------
class TestMainIntegration:
    """Fixture population.

    cca cache (train250k): cca_pos_foreign, overlap_pos, cca_unl, neg1, hold1
    rel cache (relevance_train): overlap_pos, rel_pos_domestic, rel_pos_foreign,
        rel_unl, hold1

    Union = {cca_pos_foreign, overlap_pos, cca_unl, neg1, hold1,
             rel_pos_domestic, rel_pos_foreign, rel_unl}  (8 ids, 2 overlap)

    - cca_pos_foreign: DoCA positive, cca-only, us_logit=-5.0 (raw, fails the
      0.5 threshold) -> cca_label=1 but us=False: proves "kept regardless".
    - overlap_pos: DoCA positive AND rel candidate, in both caches; cca-side
      us_logit=-5.0 (would fail), rel-side us_logit=0.9 (passes) -> proves the
      union prefers the rel channel (rel_label=1 requires the rel value to win).
      Location "United States" (domestic).
    - cca_unl: cca-only, unlabeled background, us_logit=3.0 (passes), no
      location signal.
    - neg1: cca-only, reliable-negative, us_logit=-8.0 (fails), no location.
    - hold1: DoCA positive AND rel candidate, in both caches, good US location
      -- but in the ICA-eval holdout, so must be dropped entirely.
    - rel_pos_domestic: rel-only candidate, us_logit=0.9, "United States" ->
      rel_label=1.
    - rel_pos_foreign: rel-only candidate, us_logit=0.9 (ML passes) but
      glocation "Cuba" only -> clearly foreign -> fused gate rejects -> proves
      the US-restriction drops it (rel_label=0 despite being a candidate).
    - rel_unl: rel-only, unlabeled background, us_logit=0.9, no location signal.
    """

    CCA_IDS = ["cca_pos_foreign", "overlap_pos", "cca_unl", "neg1", "hold1"]
    CCA_US_LOGIT = [-5.0, -5.0, 3.0, -8.0, 3.0]
    REL_IDS = ["overlap_pos", "rel_pos_domestic", "rel_pos_foreign", "rel_unl", "hold1"]
    REL_US_LOGIT = [0.9, 0.9, 0.9, 0.9, 0.9]

    GLOCATIONS = {
        "cca_pos_foreign": "Uganda",
        "overlap_pos": "United States",
        "cca_unl": None,
        "neg1": None,
        "hold1": "United States",
        "rel_pos_domestic": "United States",
        "rel_pos_foreign": "Cuba",
        "rel_unl": None,
    }
    ALL_IDS = list(dict.fromkeys(CCA_IDS + REL_IDS))

    def _keywords(self, glocation):
        if glocation is None:
            return []
        return [{"type": "glocations", "value": glocation, "rank": 1, "major": "N"}]

    def _setup(self, tmp_path, monkeypatch):
        # --- embed cache metas (id, year, us_logit) ---
        cache_root = tmp_path / "embed_cache"
        cca_dir = cache_root / "train250k"
        rel_dir_cache = cache_root / "relevance_train"
        cca_dir.mkdir(parents=True)
        rel_dir_cache.mkdir(parents=True)
        pl.DataFrame({
            "id": self.CCA_IDS,
            "year": ["1990"] * len(self.CCA_IDS),
            "us_logit": self.CCA_US_LOGIT,
        }).write_parquet(cca_dir / "shard_000_meta.parquet")
        pl.DataFrame({
            "id": self.REL_IDS,
            "year": ["1990"] * len(self.REL_IDS),
            "us_logit": self.REL_US_LOGIT,
        }).write_parquet(rel_dir_cache / "shard_000_meta.parquet")
        monkeypatch.setattr(config, "CCA_EMBED_CACHE_DIR", cache_root)

        # --- relevance/ data products ---
        rel_dir = tmp_path / "relevance"
        rel_dir.mkdir()
        pl.DataFrame({"id": ["overlap_pos", "rel_pos_domestic", "rel_pos_foreign", "hold1"]}).write_parquet(
            rel_dir / "candidates.parquet"
        )
        pl.DataFrame({"id": ["neg1"]}).write_parquet(rel_dir / "reliable_negatives.parquet")
        monkeypatch.setattr(bjtt, "RELEVANCE_DIR", rel_dir)

        # --- CCA/DoCA positives ---
        cca_doca_dir = tmp_path / "cca_doca"
        cca_doca_dir.mkdir()
        pl.DataFrame({"id": ["cca_pos_foreign", "overlap_pos", "hold1"]}).write_parquet(
            cca_doca_dir / "cca_doca_positives.parquet"
        )
        monkeypatch.setattr(config, "CCA_DOCA_POSITIVES", cca_doca_dir / "cca_doca_positives.parquet")

        # --- holdout ---
        validation_dir = tmp_path / "validation"
        validation_dir.mkdir()
        pl.DataFrame({"id": ["hold1"]}).write_parquet(validation_dir / "ica_holdout_ids.parquet")
        monkeypatch.setattr(config, "ICA_HOLDOUT_IDS", validation_dir / "ica_holdout_ids.parquet")

        # --- api_corpus (text + location signals) ---
        corpus_dir = tmp_path / "api_corpus"
        corpus_dir.mkdir()
        pl.DataFrame(
            {
                "id": self.ALL_IDS,
                "headline": [f"H_{i}" for i in self.ALL_IDS],
                "lead_paragraph": [f"lede_{i}" for i in self.ALL_IDS],
                "keywords": [self._keywords(self.GLOCATIONS[i]) for i in self.ALL_IDS],
                "news_desk": [None] * len(self.ALL_IDS),
                "section_name": [None] * len(self.ALL_IDS),
            },
            schema_overrides={"news_desk": pl.Utf8, "section_name": pl.Utf8},
        ).write_parquet(corpus_dir / "1990.parquet")
        monkeypatch.setattr(config, "API_CORPUS_DIR", corpus_dir)

        # --- output path ---
        out_path = tmp_path / "joint_text_table.parquet"
        monkeypatch.setattr(config, "JOINT_TEXT_TABLE", out_path)
        return out_path

    def test_holdout_excluded_from_output(self, tmp_path, monkeypatch):
        out_path = self._setup(tmp_path, monkeypatch)
        bjtt.main()
        table = pl.read_parquet(out_path)
        assert "hold1" not in table["id"].to_list()

    def test_row_count_is_union_minus_holdout(self, tmp_path, monkeypatch):
        out_path = self._setup(tmp_path, monkeypatch)
        bjtt.main()
        table = pl.read_parquet(out_path)
        # 8 unique ids in the union, 1 (hold1) dropped as holdout -> 7 remain
        assert table.height == 7

    def test_cca_positive_kept_regardless_of_us_gate(self, tmp_path, monkeypatch):
        out_path = self._setup(tmp_path, monkeypatch)
        bjtt.main()
        table = pl.read_parquet(out_path)
        row = table.filter(pl.col("id") == "cca_pos_foreign")
        assert row["cca_label"].item() == 1
        assert row["us"].item() is False  # failed US gate but kept as cca positive

    def test_overlap_uses_rel_channel_for_us_gate(self, tmp_path, monkeypatch):
        """overlap_pos's cca-side us_logit (-5.0) would fail the 0.5 threshold;
        its rel-side us_logit (0.9) passes. rel_label=1 proves the rel value won."""
        out_path = self._setup(tmp_path, monkeypatch)
        bjtt.main()
        table = pl.read_parquet(out_path)
        row = table.filter(pl.col("id") == "overlap_pos")
        assert row["cca_label"].item() == 1
        assert row["rel_label"].item() == 1
        assert row["us"].item() is True
        assert row["in_cca_pop"].item() is True
        assert row["in_rel_pop"].item() is True

    def test_clearly_foreign_rel_candidate_gated_out(self, tmp_path, monkeypatch):
        out_path = self._setup(tmp_path, monkeypatch)
        bjtt.main()
        table = pl.read_parquet(out_path)
        row = table.filter(pl.col("id") == "rel_pos_foreign")
        assert row["us"].item() is False
        assert row["rel_label"].item() == 0  # candidate, but US-restricted out

    def test_rel_domestic_positive_kept(self, tmp_path, monkeypatch):
        out_path = self._setup(tmp_path, monkeypatch)
        bjtt.main()
        table = pl.read_parquet(out_path)
        row = table.filter(pl.col("id") == "rel_pos_domestic")
        assert row["rel_label"].item() == 1
        assert row["cca_label"].item() == 0  # not a DoCA positive

    def test_reliable_negative_flagged(self, tmp_path, monkeypatch):
        out_path = self._setup(tmp_path, monkeypatch)
        bjtt.main()
        table = pl.read_parquet(out_path)
        row = table.filter(pl.col("id") == "neg1")
        assert row["reliable_neg"].item() is True
        assert row["cca_label"].item() == 0
        assert row["rel_label"].item() == 0

    def test_unlabeled_background_rows_present(self, tmp_path, monkeypatch):
        out_path = self._setup(tmp_path, monkeypatch)
        bjtt.main()
        table = pl.read_parquet(out_path)
        for uid in ("cca_unl", "rel_unl"):
            row = table.filter(pl.col("id") == uid)
            assert row.height == 1
            assert row["cca_label"].item() == 0
            assert row["rel_label"].item() == 0

    def test_text_attached(self, tmp_path, monkeypatch):
        out_path = self._setup(tmp_path, monkeypatch)
        bjtt.main()
        table = pl.read_parquet(out_path)
        row = table.filter(pl.col("id") == "overlap_pos")
        assert row["headline"].item() == "H_overlap_pos"
        assert row["headline_with_lead"].item() == "H_overlap_pos</s>lede_overlap_pos"

    def test_missing_text_raises(self, tmp_path, monkeypatch):
        self._setup(tmp_path, monkeypatch)
        corpus_dir = config.API_CORPUS_DIR
        df = pl.read_parquet(corpus_dir / "1990.parquet")
        df.filter(pl.col("id") != "cca_unl").write_parquet(corpus_dir / "1990.parquet")

        with pytest.raises(ValueError, match="not found in api_corpus text"):
            bjtt.main()

    def test_duplicate_api_corpus_id_does_not_fan_out_the_table(self, tmp_path, monkeypatch):
        """api_corpus carries a known duplicate-id family (same id, two physical
        rows, e.g. across two files) -- `read_api_text` concatenates both with
        no dedupe. main() must not let that fan out the final table (mirrors
        the fix already applied to `load_location_signals` for the same
        underlying data-quality flag; see docs/notes/metal-execution-findings.md)."""
        out_path = self._setup(tmp_path, monkeypatch)
        # Add a SECOND physical row for an existing id in a different file.
        corpus_dir = config.API_CORPUS_DIR
        dup_row = pl.DataFrame(
            {
                "id": ["cca_unl"],
                "headline": ["H_cca_unl_DUP"],
                "lead_paragraph": ["lede_cca_unl_DUP"],
                "keywords": [self._keywords(None)],
                "news_desk": [None],
                "section_name": [None],
            },
            schema_overrides={"news_desk": pl.Utf8, "section_name": pl.Utf8},
        )
        dup_row.write_parquet(corpus_dir / "1991.parquet")

        bjtt.main()
        table = pl.read_parquet(out_path)
        assert table.height == table["id"].n_unique()
        assert table.filter(pl.col("id") == "cca_unl").height == 1

    def test_custom_threshold_arg(self, tmp_path, monkeypatch):
        """main(threshold=...) is respected -- a very high threshold gates
        out even the rel-side 0.9 rows."""
        out_path = self._setup(tmp_path, monkeypatch)
        bjtt.main(threshold=0.95)
        table = pl.read_parquet(out_path)
        row = table.filter(pl.col("id") == "rel_pos_domestic")
        assert row["us"].item() is False
        assert row["rel_label"].item() == 0
