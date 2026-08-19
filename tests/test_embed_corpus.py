"""Tests for the embedding extractor's pure helpers + cache round-trip.

Covers the Functional Core (article selection, stratified sampling) and the
thin cache I/O. Does NOT exercise the keras/backbone forward pass.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.embed_corpus import (
    build_arg_parser,
    stratified_sample_by_year,
    select_articles,
    write_shard,
    load_cache,
    provenance_record,
    parse_branch_spec,
    parse_branch_specs,
    _resolve_branch_groups,
    _count_existing_shards,
)


def _corpus(n_per_year=100, years=(1965, 1975, 1985)):
    rows = []
    for y in years:
        for i in range(n_per_year):
            rows.append({"id": f"nyt://article/{y}-{i}", "year": str(y),
                         "headline_with_lead": f"hl {y} {i}"})
    return pl.DataFrame(rows)


def test_stratified_sample_is_deterministic_and_sized():
    df = _corpus()
    a = stratified_sample_by_year(df, 90, seed=200)
    b = stratified_sample_by_year(df, 90, seed=200)
    assert a.equals(b)  # deterministic under seed
    assert 60 <= a.height <= 120  # ~90, allowing per-year rounding
    # every sampled row comes from the input
    assert set(a["id"]).issubset(set(df["id"]))
    # proportional: all three years represented
    assert a["year"].n_unique() == 3


def test_stratified_sample_edge_cases():
    df = _corpus()
    assert stratified_sample_by_year(df, 0).height == 0
    assert stratified_sample_by_year(df, -5).height == 0
    assert stratified_sample_by_year(df, 10_000).height == df.height


def test_select_articles_forces_includes_and_dedupes():
    df = _corpus()
    include = [df["id"][0], df["id"][150], df["id"][250]]
    out = select_articles(df, include_ids=include, sample_n=30, full=False)
    # all forced includes present
    assert set(include).issubset(set(out["id"]))
    # unique ids (includes + disjoint sample)
    assert out["id"].n_unique() == out.height
    # sample drawn from the remainder, so total ~ len(include) + ~30
    assert out.height >= len(include)


def test_select_articles_full_returns_all():
    df = _corpus()
    out = select_articles(df, include_ids=None, sample_n=0, full=True)
    assert out.height == df.height


def test_cache_round_trip(tmp_path):
    cls0 = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    meta0 = pl.DataFrame({"id": ["a", "b"], "year": ["1965", "1965"],
                          "us_logit": [0.5, -1.0]})
    cls1 = np.arange(3 * 4, dtype=np.float32).reshape(3, 4) + 100
    meta1 = pl.DataFrame({"id": ["c", "d", "e"], "year": ["1975", "1975", "1985"],
                          "us_logit": [2.0, -0.3, 1.1]})
    write_shard(tmp_path, 0, cls0, meta0)
    write_shard(tmp_path, 1, cls1, meta1)

    meta, cls = load_cache(tmp_path)
    assert meta.height == 5
    assert cls.shape == (5, 4)
    # order preserved across shards
    assert meta["id"].to_list() == ["a", "b", "c", "d", "e"]
    # emb_row indexes into cls, and the row matches the source shard row
    assert meta["emb_row"].to_list() == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(cls[2], cls1[0])
    assert meta["us_logit"][2] == pytest.approx(2.0)


def test_write_shard_rejects_misaligned():
    cls = np.zeros((3, 4), dtype=np.float32)
    meta = pl.DataFrame({"id": ["a"], "year": ["1965"], "us_logit": [0.0]})
    with pytest.raises(ValueError, match="cls rows"):
        write_shard(__import__("pathlib").Path("/tmp"), 0, cls, meta)


# ---------------------------------------------------------------------------
# provenance_record: backbone_weights_override (additive field)
# ---------------------------------------------------------------------------
def test_provenance_record_override_absent_is_null(tmp_path):
    backbone = tmp_path / "dapt_backbone.weights.h5"
    backbone.write_bytes(b"x")
    us_weights = tmp_path / "us_classifier.weights.h5"
    us_weights.write_bytes(b"y")

    prov = provenance_record(
        backbone_weights=backbone, us_weights=us_weights, seq_length=128,
        text_channel="headline_with_lead", stamp="20260729",
        n_rows=10, n_included=2, sample_n=8, full=False,
    )
    assert prov["backbone_weights_override"] is None
    # existing field's meaning is preserved: stat of the backbone actually used.
    assert prov["backbone_weights"]["path"] == str(backbone)
    assert prov["backbone_weights"]["exists"] is True


def test_provenance_record_override_present_is_recorded(tmp_path):
    default_backbone = tmp_path / "dapt_backbone.weights.h5"
    default_backbone.write_bytes(b"x")
    tuned_backbone = tmp_path / "tuned_backbone.job8823087.weights.h5"
    tuned_backbone.write_bytes(b"tuned")
    us_weights = tmp_path / "us_classifier.weights.h5"
    us_weights.write_bytes(b"y")

    prov = provenance_record(
        backbone_weights=tuned_backbone,  # the ACTUAL backbone used
        us_weights=us_weights, seq_length=128, text_channel="headline_with_lead",
        stamp="20260729", n_rows=10, n_included=2, sample_n=8, full=False,
        backbone_weights_override=tuned_backbone,
    )
    assert prov["backbone_weights"]["path"] == str(tuned_backbone)
    assert prov["backbone_weights_override"]["path"] == str(tuned_backbone)
    assert prov["backbone_weights_override"]["exists"] is True
    assert prov["backbone_weights_override"]["size"] == len(b"tuned")


def test_provenance_record_override_missing_file_still_recorded(tmp_path):
    # A typo'd override path shouldn't crash provenance -- it should surface
    # as exists=False so the run is auditable even if something else fails.
    missing = tmp_path / "does_not_exist.weights.h5"
    us_weights = tmp_path / "us_classifier.weights.h5"
    us_weights.write_bytes(b"y")

    prov = provenance_record(
        backbone_weights=missing, us_weights=us_weights, seq_length=128,
        text_channel="headline_with_lead", stamp="20260729",
        n_rows=10, n_included=2, sample_n=8, full=False,
        backbone_weights_override=missing,
    )
    assert prov["backbone_weights_override"]["exists"] is False
    assert prov["backbone_weights_override"]["size"] is None


# ---------------------------------------------------------------------------
# CLI: --backbone-weights default
# ---------------------------------------------------------------------------
def test_backbone_weights_flag_defaults_to_none():
    args = build_arg_parser().parse_args(["--stamp", "20260729", "--out-suffix", "test"])
    assert args.backbone_weights is None


def test_backbone_weights_flag_accepts_a_path():
    args = build_arg_parser().parse_args([
        "--stamp", "20260729", "--out-suffix", "test",
        "--backbone-weights", "relevance/tuned_backbone.job8823087.weights.h5",
    ])
    assert args.backbone_weights == "relevance/tuned_backbone.job8823087.weights.h5"


# ---------------------------------------------------------------------------
# CLI + main() wiring: --lead-fallback-column (the post-1995 coalesce channel)
# ---------------------------------------------------------------------------
def test_lead_fallback_flag_defaults_to_none():
    """Default invocation must not request a fallback column (prior behavior)."""
    args = build_arg_parser().parse_args(["--stamp", "20260731", "--out-suffix", "test"])
    assert args.lead_fallback_column is None


def test_lead_fallback_flag_accepts_a_column():
    args = build_arg_parser().parse_args([
        "--stamp", "20260731", "--out-suffix", "test",
        "--lead-fallback-column", "abstrct",
    ])
    assert args.lead_fallback_column == "abstrct"


class _LoaderMarker(Exception):
    """Raised by the monkeypatched loader to stop main() after the call we test."""


def test_main_passes_lead_fallback_to_loader(monkeypatch, tmp_path):
    """main(lead_fallback_column=...) must hand the column through to
    data_from_parquet (the coalesce lives in the loader, not the embed loop)."""
    import src.embed_corpus as embed_corpus

    captured = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        raise _LoaderMarker()

    monkeypatch.setattr(embed_corpus, "data_from_parquet", _capture)
    monkeypatch.setattr(embed_corpus.config, "CCA_EMBED_CACHE_DIR", tmp_path)
    with pytest.raises(_LoaderMarker):
        embed_corpus.main(
            full=True, stamp="20260731", out_suffix="test",
            lead_fallback_column="abstrct",
        )
    assert captured.get("lead_fallback_column") == "abstrct"


def test_main_default_omits_lead_fallback(monkeypatch, tmp_path):
    """Without the knob, the loader is called with lead_fallback_column=None
    (back-compat: the source parquet needs no fallback column)."""
    import src.embed_corpus as embed_corpus

    captured = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        raise _LoaderMarker()

    monkeypatch.setattr(embed_corpus, "data_from_parquet", _capture)
    monkeypatch.setattr(embed_corpus.config, "CCA_EMBED_CACHE_DIR", tmp_path)
    with pytest.raises(_LoaderMarker):
        embed_corpus.main(full=True, stamp="20260731", out_suffix="test")
    assert captured.get("lead_fallback_column") is None


# ---------------------------------------------------------------------------
# dedupe_by_id: deterministic duplicate-id resolution (post-1995 API pull
# overlaps: 911 dup ids, 698 spanning two year files). Preference order:
# non-empty effective lead first (headline_with_lead not ending in "</s>"),
# then earliest year, then text — fully deterministic under permutation.
# ---------------------------------------------------------------------------
def _dup_frame(rows):
    return pl.DataFrame(
        [{"id": r[0], "year": r[1], "headline_with_lead": r[2]} for r in rows]
    )


def test_dedupe_by_id_unique_input_unchanged():
    from src.embed_corpus import dedupe_by_id
    df = _dup_frame([("a", "2020", "A</s>lead a"), ("b", "2021", "B</s>lead b")])
    out = dedupe_by_id(df)
    assert out.sort("id").equals(df.sort("id"))


def test_dedupe_by_id_prefers_nonempty_lead():
    from src.embed_corpus import dedupe_by_id
    df = _dup_frame([
        ("a", "2020", "Title</s>"),          # lead-empty copy, earlier year
        ("a", "2021", "Title</s>real lead"),  # lead-bearing copy, later year
    ])
    out = dedupe_by_id(df)
    assert out.height == 1
    assert out["headline_with_lead"][0] == "Title</s>real lead"


def test_dedupe_by_id_ties_break_by_earliest_year():
    from src.embed_corpus import dedupe_by_id
    df = _dup_frame([
        ("a", "2021", "Title</s>lead x"),
        ("a", "2020", "Title</s>lead x"),
    ])
    out = dedupe_by_id(df)
    assert out.height == 1
    assert out["year"][0] == "2020"


def test_dedupe_by_id_deterministic_under_permutation():
    from src.embed_corpus import dedupe_by_id
    rows = [
        ("a", "2021", "Title</s>lead x"),
        ("a", "2020", "Title</s>"),
        ("b", "2020", "B</s>b lead"),
        ("c", "2022", "C</s>"),
        ("c", "2022", "C</s>c lead"),
    ]
    out1 = dedupe_by_id(_dup_frame(rows)).sort("id")
    out2 = dedupe_by_id(_dup_frame(list(reversed(rows)))).sort("id")
    assert out1.equals(out2)
    assert out1["id"].to_list() == ["a", "b", "c"]


def test_dedupe_by_id_without_year_column():
    """The LDC-style sources have no year column; dedupe still works
    (preference: non-empty lead, then text)."""
    from src.embed_corpus import dedupe_by_id
    df = pl.DataFrame([
        {"id": "a", "headline_with_lead": "T</s>"},
        {"id": "a", "headline_with_lead": "T</s>lead"},
    ])
    out = dedupe_by_id(df)
    assert out.height == 1
    assert out["headline_with_lead"][0] == "T</s>lead"


class _ModelMarker(Exception):
    """Raised by the monkeypatched model builder: control flow passed the guard."""


def _main_with_corpus(monkeypatch, tmp_path, corpus_df, **main_kwargs):
    """Drive embed_corpus.main() with a fake in-memory corpus, stopping at the
    model builder. Returns nothing; raises whatever main() raises."""
    import src.embed_corpus as embed_corpus

    def _fake_loader(*args, **kwargs):
        return corpus_df

    def _raise_model_marker(*args, **kwargs):
        raise _ModelMarker()

    monkeypatch.setattr(embed_corpus, "data_from_parquet", _fake_loader)
    monkeypatch.setattr(embed_corpus, "_build_embed_model", _raise_model_marker)
    monkeypatch.setattr(embed_corpus.config, "CCA_EMBED_CACHE_DIR", tmp_path)
    embed_corpus.main(full=True, stamp="20260731", out_suffix="test", **main_kwargs)


def test_main_dup_ids_without_flag_still_raises(monkeypatch, tmp_path):
    """Back-compat: duplicate ids with dedupe off must hit the existing
    defense-in-depth guard (loud failure, no silent dedupe)."""
    dup = _dup_frame([("a", "2020", "T</s>x"), ("a", "2021", "T</s>x")])
    with pytest.raises(ValueError, match="duplicate ids"):
        _main_with_corpus(monkeypatch, tmp_path, dup)


def test_main_dedupe_flag_resolves_dups_and_proceeds(monkeypatch, tmp_path):
    """With dedupe_ids=True the dup frame passes the guard and reaches the
    model builder."""
    dup = _dup_frame([("a", "2020", "T</s>x"), ("a", "2021", "T</s>x"),
                      ("b", "2020", "B</s>y")])
    with pytest.raises(_ModelMarker):
        _main_with_corpus(monkeypatch, tmp_path, dup, dedupe_ids=True)


def test_dedupe_flag_defaults_to_off():
    args = build_arg_parser().parse_args(["--stamp", "20260731", "--out-suffix", "test"])
    assert args.dedupe_ids is False


def test_dedupe_flag_parses_on():
    args = build_arg_parser().parse_args([
        "--stamp", "20260731", "--out-suffix", "test", "--dedupe-ids",
    ])
    assert args.dedupe_ids is True



# ---------------------------------------------------------------------------
# write_shard / load_cache: variant support (stage-4 branched embed model,
# docs/design-plans/2026-08-18-stage4-joint-finetune.md "Branched
# productionization -- implementation contract"). Cache layout: an extra
# per-shard array named `shard_{idx:03d}_cls.{variant}.npy` (DOTTED --
# distinct from the base `shard_{idx:03d}_cls.npy`).
# ---------------------------------------------------------------------------
def _meta(ids, years, us_logits):
    return pl.DataFrame({"id": ids, "year": years, "us_logit": us_logits})


def test_write_shard_with_variants_creates_dotted_files(tmp_path):
    cls = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    variant = np.arange(2 * 4, dtype=np.float32).reshape(2, 4) + 50
    meta = _meta(["a", "b"], ["1965", "1965"], [0.1, 0.2])
    write_shard(tmp_path, 0, cls, meta, variants={"rel_branch": variant})
    loaded = np.load(tmp_path / "shard_000_cls.rel_branch.npy")
    np.testing.assert_array_equal(loaded, variant)
    # base cls file is unaffected by the extra variant.
    np.testing.assert_array_equal(np.load(tmp_path / "shard_000_cls.npy"), cls)


def test_write_shard_variants_default_none_is_byte_identical_to_prior_behavior(tmp_path):
    cls = np.zeros((1, 4), dtype=np.float32)
    meta = _meta(["a"], ["1965"], [0.1])
    write_shard(tmp_path, 0, cls, meta)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "shard_000_cls.npy", "shard_000_meta.parquet",
    ]


def test_write_shard_variant_row_mismatch_raises(tmp_path):
    cls = np.zeros((2, 4), dtype=np.float32)
    variant = np.zeros((3, 4), dtype=np.float32)  # wrong row count
    meta = _meta(["a", "b"], ["1965", "1965"], [0.1, 0.2])
    with pytest.raises(ValueError, match="rel_branch"):
        write_shard(tmp_path, 0, cls, meta, variants={"rel_branch": variant})


def test_variant_shard_filename_not_matched_by_bare_cls_glob(tmp_path):
    """Regression pin (brief-identified glob-safety requirement): the
    append-mode `shard_*_cls.npy` glob must NOT match the dotted variant
    filename `shard_000_cls.rel_branch.npy`."""
    cls = np.zeros((2, 4), dtype=np.float32)
    variant = np.ones((2, 4), dtype=np.float32)
    meta = _meta(["a", "b"], ["1965", "1965"], [0.1, 0.2])
    write_shard(tmp_path, 0, cls, meta, variants={"rel_branch": variant})
    assert (tmp_path / "shard_000_cls.rel_branch.npy").exists()
    bare_matches = list(tmp_path.glob("shard_*_cls.npy"))
    assert [p.name for p in bare_matches] == ["shard_000_cls.npy"]


def test_load_cache_with_variant_returns_variant_array(tmp_path):
    cls0 = np.zeros((2, 4), dtype=np.float32)
    var0 = np.full((2, 4), 1.0, dtype=np.float32)
    meta0 = _meta(["a", "b"], ["1965", "1965"], [0.1, 0.2])
    cls1 = np.zeros((1, 4), dtype=np.float32)
    var1 = np.full((1, 4), 2.0, dtype=np.float32)
    meta1 = _meta(["c"], ["1975"], [0.3])
    write_shard(tmp_path, 0, cls0, meta0, variants={"rel_branch": var0})
    write_shard(tmp_path, 1, cls1, meta1, variants={"rel_branch": var1})

    meta, cls_variant = load_cache(tmp_path, variant="rel_branch")
    assert meta.height == 3
    assert cls_variant.shape == (3, 4)
    np.testing.assert_array_equal(cls_variant[:2], var0)
    np.testing.assert_array_equal(cls_variant[2:], var1)
    assert meta["id"].to_list() == ["a", "b", "c"]


def test_load_cache_variant_none_still_returns_base_cls_even_with_variants_present(tmp_path):
    cls0 = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    var0 = cls0 + 100
    meta0 = _meta(["a", "b"], ["1965", "1965"], [0.1, 0.2])
    write_shard(tmp_path, 0, cls0, meta0, variants={"rel_branch": var0})
    meta, cls = load_cache(tmp_path)
    np.testing.assert_array_equal(cls, cls0)


def test_load_cache_missing_variant_file_raises_naming_the_shard(tmp_path):
    cls0 = np.zeros((2, 4), dtype=np.float32)
    meta0 = _meta(["a", "b"], ["1965", "1965"], [0.1, 0.2])
    write_shard(tmp_path, 0, cls0, meta0)  # no variant written
    with pytest.raises(FileNotFoundError, match="shard 0"):
        load_cache(tmp_path, variant="rel_branch")


def test_load_cache_missing_variant_on_second_shard_names_that_shard(tmp_path):
    cls0 = np.zeros((2, 4), dtype=np.float32)
    var0 = np.ones((2, 4), dtype=np.float32)
    meta0 = _meta(["a", "b"], ["1965", "1965"], [0.1, 0.2])
    cls1 = np.zeros((1, 4), dtype=np.float32)
    meta1 = _meta(["c"], ["1975"], [0.3])
    write_shard(tmp_path, 0, cls0, meta0, variants={"rel_branch": var0})
    write_shard(tmp_path, 1, cls1, meta1)  # shard 1 has no variant array
    with pytest.raises(FileNotFoundError, match="shard 1"):
        load_cache(tmp_path, variant="rel_branch")


def test_load_cache_variant_misaligned_raises(tmp_path):
    cls0 = np.zeros((2, 4), dtype=np.float32)
    var0 = np.ones((2, 4), dtype=np.float32)
    meta0 = _meta(["a", "b"], ["1965", "1965"], [0.1, 0.2])
    write_shard(tmp_path, 0, cls0, meta0, variants={"rel_branch": var0})
    # Corrupt the variant file directly: wrong row count vs meta.
    np.save(tmp_path / "shard_000_cls.rel_branch.npy", np.ones((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="misaligned"):
        load_cache(tmp_path, variant="rel_branch")


# ---------------------------------------------------------------------------
# _count_existing_shards: append-mode shard-offset source of truth, hardened
# to count meta parquets rather than the `_cls.npy` glob (brief-identified
# fragility; behavior-identical for well-formed caches).
# ---------------------------------------------------------------------------
def test_count_existing_shards_counts_meta_parquets(tmp_path):
    cls = np.zeros((2, 4), dtype=np.float32)
    meta = _meta(["a", "b"], ["1965", "1965"], [0.1, 0.2])
    write_shard(tmp_path, 0, cls, meta)
    write_shard(tmp_path, 1, cls, meta)
    assert _count_existing_shards(tmp_path) == 2


def test_count_existing_shards_empty_dir_is_zero(tmp_path):
    assert _count_existing_shards(tmp_path) == 0


def test_count_existing_shards_unaffected_by_variant_files(tmp_path):
    cls = np.zeros((2, 4), dtype=np.float32)
    variant = np.ones((2, 4), dtype=np.float32)
    meta = _meta(["a", "b"], ["1965", "1965"], [0.1, 0.2])
    write_shard(tmp_path, 0, cls, meta, variants={"rel_branch": variant})
    assert _count_existing_shards(tmp_path) == 1


# ---------------------------------------------------------------------------
# parse_branch_spec / parse_branch_specs: `--branch <variant>=<donor>[:<top_n>]`
# CLI syntax (stage-4 branched embed model).
# ---------------------------------------------------------------------------
class TestParseBranchSpec:
    def test_parses_variant_donor_and_top_n(self):
        assert parse_branch_spec(
            "rel_branch=../relevance/tuned_backbone.job8823087.weights.h5:1"
        ) == ("rel_branch", "../relevance/tuned_backbone.job8823087.weights.h5", 1)

    def test_top_n_defaults_to_one(self):
        assert parse_branch_spec("rel_branch=donor.weights.h5") == (
            "rel_branch", "donor.weights.h5", 1,
        )

    def test_rejects_missing_equals(self):
        with pytest.raises(ValueError, match="variant"):
            parse_branch_spec("donor.weights.h5")

    def test_rejects_empty_variant(self):
        with pytest.raises(ValueError, match="variant"):
            parse_branch_spec("=donor.weights.h5")

    def test_rejects_empty_donor_path(self):
        with pytest.raises(ValueError, match="donor"):
            parse_branch_spec("rel_branch=")

    def test_rejects_non_integer_top_n(self):
        with pytest.raises(ValueError, match="top_n"):
            parse_branch_spec("rel_branch=donor.weights.h5:abc")


class TestParseBranchSpecs:
    def test_empty_or_none_returns_empty_dict(self):
        assert parse_branch_specs(None) == {}
        assert parse_branch_specs([]) == {}

    def test_multiple_specs(self):
        specs = parse_branch_specs([
            "rel_branch=donor1.weights.h5:1",
            "cca_branch=donor2.weights.h5:2",
        ])
        assert specs == {
            "rel_branch": ("donor1.weights.h5", 1),
            "cca_branch": ("donor2.weights.h5", 2),
        }

    def test_rejects_duplicate_variant(self):
        with pytest.raises(ValueError, match="duplicate"):
            parse_branch_specs(["v=donor1.weights.h5", "v=donor2.weights.h5"])

    def test_rejects_variant_colliding_with_base_output_keys(self):
        with pytest.raises(ValueError, match="collides"):
            parse_branch_specs(["cls=donor.weights.h5"])
        with pytest.raises(ValueError, match="collides"):
            parse_branch_specs(["us=donor.weights.h5"])


# ---------------------------------------------------------------------------
# --branch CLI flag
# ---------------------------------------------------------------------------
def test_branch_flag_defaults_to_none():
    args = build_arg_parser().parse_args(["--stamp", "x", "--out-suffix", "test"])
    assert args.branch is None


def test_branch_flag_repeatable():
    args = build_arg_parser().parse_args([
        "--stamp", "x", "--out-suffix", "test",
        "--branch", "rel_branch=../relevance/tuned_backbone.job8823087.weights.h5:1",
        "--branch", "cca_branch=donor2.weights.h5",
    ])
    assert args.branch == [
        "rel_branch=../relevance/tuned_backbone.job8823087.weights.h5:1",
        "cca_branch=donor2.weights.h5",
    ]


# ---------------------------------------------------------------------------
# _resolve_branch_groups: explicit top_n vs. sidecar fallback.
# ---------------------------------------------------------------------------
class TestResolveBranchGroups:
    def test_explicit_top_n(self):
        groups, top_n = _resolve_branch_groups(__import__("pathlib").Path("donor.weights.h5"), 2)
        assert groups == {"transformer_layer_11", "transformer_layer_10"}
        assert top_n == 2

    def test_reads_sidecar_when_top_n_none(self, tmp_path):
        import dataclasses
        from src.cca_config import DEFAULT_CCA_CONFIG, config_path_for_weights

        donor = tmp_path / "donor.weights.h5"
        donor.write_bytes(b"x")
        cfg = dataclasses.replace(DEFAULT_CCA_CONFIG, unfreeze_top_n=3)
        cfg.to_json(config_path_for_weights(donor))

        groups, top_n = _resolve_branch_groups(donor, None)
        assert top_n == 3
        assert groups == {
            "transformer_layer_11", "transformer_layer_10", "transformer_layer_9",
        }

    def test_raises_when_no_sidecar_and_no_explicit_top_n(self, tmp_path):
        donor = tmp_path / "donor_no_sidecar.weights.h5"
        donor.write_bytes(b"x")
        with pytest.raises(ValueError, match="top_n"):
            _resolve_branch_groups(donor, None)


# ---------------------------------------------------------------------------
# main() threading: --branch specs must reach _build_embed_model unmodified,
# same pattern as the --lead-fallback-column / --dedupe-ids threading tests
# above (stub _build_embed_model via a marker exception; no real backbone).
# ---------------------------------------------------------------------------
def test_main_passes_branch_specs_to_build_embed_model(monkeypatch, tmp_path):
    import src.embed_corpus as embed_corpus

    captured = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        raise _ModelMarker()

    df = _corpus()
    monkeypatch.setattr(embed_corpus, "data_from_parquet", lambda *a, **k: df)
    monkeypatch.setattr(embed_corpus, "_build_embed_model", _capture)
    monkeypatch.setattr(embed_corpus.config, "CCA_EMBED_CACHE_DIR", tmp_path)
    specs = {"rel_branch": ("donor.weights.h5", 1)}
    with pytest.raises(_ModelMarker):
        embed_corpus.main(full=True, stamp="x", out_suffix="test", branch_specs=specs)
    assert captured.get("branch_specs") == specs


def test_main_default_omits_branch_specs(monkeypatch, tmp_path):
    import src.embed_corpus as embed_corpus

    captured = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        raise _ModelMarker()

    df = _corpus()
    monkeypatch.setattr(embed_corpus, "data_from_parquet", lambda *a, **k: df)
    monkeypatch.setattr(embed_corpus, "_build_embed_model", _capture)
    monkeypatch.setattr(embed_corpus.config, "CCA_EMBED_CACHE_DIR", tmp_path)
    with pytest.raises(_ModelMarker):
        embed_corpus.main(full=True, stamp="x", out_suffix="test")
    assert captured.get("branch_specs") is None


def test_dedupe_by_id_drops_empty_and_null_ids():
    """Rows with an empty or null id are junk (2025 transform artifacts: 13
    rows, no headline/lead/abstract) — dropped entirely, not collapsed to one
    untraceable row."""
    from src.embed_corpus import dedupe_by_id
    df = pl.DataFrame(
        {
            "id": ["", "", None, "a"],
            "year": ["2025", "2025", "2025", "2025"],
            "headline_with_lead": ["</s>", "</s>", "</s>", "T</s>lead"],
        }
    )
    out = dedupe_by_id(df)
    assert out["id"].to_list() == ["a"]
