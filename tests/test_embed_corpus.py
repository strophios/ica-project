"""Tests for the embedding extractor's pure helpers + cache round-trip.

Covers the Functional Core (article selection, stratified sampling) and the
thin cache I/O. Does NOT exercise the keras/backbone forward pass.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.embed_corpus import (
    stratified_sample_by_year,
    select_articles,
    write_shard,
    load_cache,
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
