"""US-head retrain v1: resolving LDC-format DoCA positive ids to stripped text."""

from __future__ import annotations

import polars as pl
import pytest

from src.build_us_pos_ldc345_source import join_ldc_positives_to_stripped_text


def _ldc_corpus():
    return pl.DataFrame({
        "id": [13450, 17408, 18907],
        "file_id": ["0013450.xml", "0017408.xml", "0018907.xml"],
    })


def _ldc_labeled():
    return pl.DataFrame({
        "id": [13450, 17408, 18907],
        "headline": ["A", "B", "C"],
        "stripped_text": ["text a", "text b", "text c"],
    })


def test_resolves_every_id_to_stripped_text():
    out = join_ldc_positives_to_stripped_text(
        ["0013450.xml", "0017408.xml"], _ldc_corpus(), _ldc_labeled()
    )
    assert out.sort("id")["id"].to_list() == [13450, 17408]
    assert out.sort("id")["stripped_text"].to_list() == ["text a", "text b"]


def test_dedupes_repeated_ids():
    out = join_ldc_positives_to_stripped_text(
        ["0013450.xml", "0013450.xml"], _ldc_corpus(), _ldc_labeled()
    )
    assert out.height == 1


def test_missing_file_id_raises():
    with pytest.raises(ValueError, match="ldc_corpus.file_id"):
        join_ldc_positives_to_stripped_text(
            ["0013450.xml", "9999999.xml"], _ldc_corpus(), _ldc_labeled()
        )


def test_missing_from_ldc_labeled_raises():
    labeled = _ldc_labeled().filter(pl.col("id") != 18907)
    with pytest.raises(ValueError, match="ldc_labeled"):
        join_ldc_positives_to_stripped_text(
            ["0013450.xml", "0018907.xml"], _ldc_corpus(), labeled
        )
