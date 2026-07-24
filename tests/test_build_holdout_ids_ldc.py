"""US-head retrain v1: translating API-space holdout ids to their LDC twins."""

from __future__ import annotations

import polars as pl

from src.build_holdout_ids_ldc import translate_holdout_to_ldc


def _matched():
    return pl.DataFrame({
        "api_id": ["nyt://a", "nyt://b", "nyt://c"],
        "ldc_id": ["100", "200", "300"],
    })


def test_translates_only_matched_holdout_ids():
    out = translate_holdout_to_ldc(["nyt://a", "nyt://z"], _matched())
    assert out == ["100"]


def test_unmatched_holdout_ids_contribute_nothing():
    out = translate_holdout_to_ldc(["nyt://z", "nyt://y"], _matched())
    assert out == []


def test_dedupes_and_sorts():
    matched = pl.DataFrame({
        "api_id": ["nyt://a", "nyt://a-dup"],
        "ldc_id": ["300", "300"],
    })
    out = translate_holdout_to_ldc(["nyt://a", "nyt://a-dup"], matched)
    assert out == ["300"]


def test_non_string_ldc_id_dtype_is_cast():
    matched = pl.DataFrame({"api_id": ["nyt://a"], "ldc_id": [100]})  # Int64
    out = translate_holdout_to_ldc(["nyt://a"], matched)
    assert out == ["100"]


def test_empty_holdout_returns_empty():
    assert translate_holdout_to_ldc([], _matched()) == []
