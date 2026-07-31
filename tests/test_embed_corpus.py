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
