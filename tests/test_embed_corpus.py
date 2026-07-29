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
