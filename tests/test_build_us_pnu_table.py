"""US-head retrain v1 P/N/U table builder.

- us-pnu.AC1: id-space unification (API ids pass through, LDC ids -> Utf8).
- us-pnu.AC2: P/N/U assignment rules, incl. heuristic negatives NOT becoming N.
- us-pnu.AC3: holdout exclusion (both id spaces), belt-and-suspenders.
- us-pnu.AC4: U sampling excludes P ids and is deterministic.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.build_us_pnu_table import (
    _as_splits_view,
    assemble_pnu_table,
    label_api_positives,
    label_dateline_negatives,
    label_ldc_positives,
    sample_unlabeled,
    validate_ids_present,
)
from src.data_setup.data import assert_holdout_excluded


# ---------------------------------------------------------------------------
# label_api_positives
# ---------------------------------------------------------------------------
def _train250k_meta():
    return pl.DataFrame({
        "emb_row": [0, 1, 2],
        "id": ["nyt://a", "nyt://b", "nyt://c"],
        "year": ["1965", "1970", "1980"],
        "us_logit": [3.0, -1.0, 0.5],
    })


def test_label_api_positives_tags_source_and_cache():
    out = label_api_positives(_train250k_meta(), ["nyt://a", "nyt://b"])
    assert out.sort("id")["id"].to_list() == ["nyt://a", "nyt://b"]
    assert set(out["cache"].to_list()) == {"train250k"}
    assert set(out["pnu_label"].to_list()) == {"pos"}
    assert set(out["source"].to_list()) == {"doca_api"}
    assert out["id"].dtype == pl.Utf8


def test_label_api_positives_excludes_holdout():
    out = label_api_positives(_train250k_meta(), ["nyt://a", "nyt://b"], holdout_ids=["nyt://a"])
    assert out["id"].to_list() == ["nyt://b"]


def test_label_api_positives_missing_id_raises():
    with pytest.raises(ValueError, match="not found in train250k"):
        label_api_positives(_train250k_meta(), ["nyt://a", "nyt://zzz"])


# ---------------------------------------------------------------------------
# label_ldc_positives
# ---------------------------------------------------------------------------
def _ldc345_meta():
    return pl.DataFrame({
        "emb_row": [0, 1, 2],
        "id": [13450, 17408, 18907],  # Int64, LDC space
        "us_logit": [5.1, 5.9, 5.6],
    })


def test_label_ldc_positives_unifies_id_to_string():
    out = label_ldc_positives(_ldc345_meta())
    assert out.height == 3
    assert out["id"].dtype == pl.Utf8
    assert sorted(out["id"].to_list()) == ["13450", "17408", "18907"]
    assert set(out["cache"].to_list()) == {"us_pos_ldc345"}
    assert set(out["source"].to_list()) == {"doca_ldc"}


def test_label_ldc_positives_excludes_holdout():
    out = label_ldc_positives(_ldc345_meta(), holdout_ids=["13450"])
    assert sorted(out["id"].to_list()) == ["17408", "18907"]


# ---------------------------------------------------------------------------
# label_dateline_negatives
# ---------------------------------------------------------------------------
def _ldc_labeled():
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "us_label": [False, True, False, False, None],
        "label_source": ["dateline", "dateline", "heuristic", "dateline", None],
    })


def test_label_dateline_negatives_only_dateline_foreign():
    out = label_dateline_negatives(_ldc_labeled())
    # id=1 (dateline, False) and id=4 (dateline, False) qualify.
    # id=2 is True (not foreign); id=3 is heuristic (excluded by design);
    # id=5 has null us_label/label_source.
    assert sorted(out["id"].to_list()) == ["1", "4"]
    assert set(out["source"].to_list()) == {"ldc_dateline_neg"}
    assert set(out["cache"].to_list()) == {"us_train_ldc"}


def test_label_dateline_negatives_heuristic_not_included():
    """Heuristic-sourced negatives must NOT become N, even though us_label=False."""
    labeled = pl.DataFrame({
        "id": [10],
        "us_label": [False],
        "label_source": ["heuristic"],
    })
    out = label_dateline_negatives(labeled)
    assert out.height == 0


def test_label_dateline_negatives_excludes_holdout():
    out = label_dateline_negatives(_ldc_labeled(), holdout_ids=["1"])
    assert out["id"].to_list() == ["4"]


# ---------------------------------------------------------------------------
# sample_unlabeled
# ---------------------------------------------------------------------------
def _full_meta(n=100):
    return pl.DataFrame({
        "emb_row": list(range(n)),
        "id": [f"nyt://u{i}" for i in range(n)],
        "year": ["1965"] * n,
        "us_logit": [1.0] * n,
    })


def test_sample_unlabeled_excludes_positive_ids():
    meta = _full_meta(50)
    out = sample_unlabeled(meta, exclude_ids=[f"nyt://u{i}" for i in range(10)], n=20, seed=200)
    assert out.height == 20
    assert not set(out["id"].to_list()) & {f"nyt://u{i}" for i in range(10)}
    assert set(out["cache"].to_list()) == {"full"}
    assert set(out["pnu_label"].to_list()) == {"unl"}
    assert set(out["source"].to_list()) == {"api_unlabeled"}


def test_sample_unlabeled_deterministic():
    meta = _full_meta(50)
    a = sample_unlabeled(meta, exclude_ids=[], n=10, seed=200)
    b = sample_unlabeled(meta, exclude_ids=[], n=10, seed=200)
    assert a["id"].to_list() == b["id"].to_list()


def test_sample_unlabeled_returns_all_when_n_exceeds_pool():
    meta = _full_meta(5)
    out = sample_unlabeled(meta, exclude_ids=[], n=1000, seed=200)
    assert out.height == 5


# ---------------------------------------------------------------------------
# assemble_pnu_table
# ---------------------------------------------------------------------------
def _tagged(ids, cache, label, source):
    return pl.DataFrame({
        "id": ids,
        "cache": [cache] * len(ids),
        "pnu_label": [label] * len(ids),
        "source": [source] * len(ids),
        "year": [None] * len(ids),
    }, schema={"id": pl.Utf8, "cache": pl.Utf8, "pnu_label": pl.Utf8, "source": pl.Utf8, "year": pl.Utf8})


def test_assemble_pnu_table_concatenates_all_sources():
    pos_api = _tagged(["a1"], "train250k", "pos", "doca_api")
    pos_ldc = _tagged(["100"], "us_pos_ldc345", "pos", "doca_ldc")
    neg = _tagged(["200"], "us_train_ldc", "neg", "ldc_dateline_neg")
    unl = _tagged(["u1", "u2"], "full", "unl", "api_unlabeled")
    table = assemble_pnu_table(pos_api, pos_ldc, neg, unl)
    assert table.height == 5
    assert set(table["pnu_label"].to_list()) == {"pos", "neg", "unl"}


def test_assemble_pnu_table_raises_on_id_collision():
    pos_api = _tagged(["dup"], "train250k", "pos", "doca_api")
    pos_ldc = _tagged(["100"], "us_pos_ldc345", "pos", "doca_ldc")
    neg = _tagged(["200"], "us_train_ldc", "neg", "ldc_dateline_neg")
    unl = _tagged(["dup"], "full", "unl", "api_unlabeled")  # collides with pos_api
    with pytest.raises(ValueError, match="more than one source"):
        assemble_pnu_table(pos_api, pos_ldc, neg, unl)


# ---------------------------------------------------------------------------
# validate_ids_present
# ---------------------------------------------------------------------------
def test_validate_ids_present_passes_when_subset():
    validate_ids_present(["1", "2"], ["1", "2", "3"])


def test_validate_ids_present_raises_on_missing():
    with pytest.raises(ValueError, match="not found in target cache"):
        validate_ids_present(["1", "99"], ["1", "2", "3"])


# ---------------------------------------------------------------------------
# Holdout exclusion, end to end (belt-and-suspenders)
# ---------------------------------------------------------------------------
def _full_table():
    pos_api = _tagged([f"nyt://p{i}" for i in range(5)], "train250k", "pos", "doca_api")
    pos_ldc = _tagged([str(100 + i) for i in range(3)], "us_pos_ldc345", "pos", "doca_ldc")
    neg = _tagged([str(200 + i) for i in range(4)], "us_train_ldc", "neg", "ldc_dateline_neg")
    unl = _tagged([f"nyt://u{i}" for i in range(10)], "full", "unl", "api_unlabeled")
    return assemble_pnu_table(pos_api, pos_ldc, neg, unl)


def test_clean_table_passes_holdout_assertion():
    table = _full_table()
    assert_holdout_excluded(_as_splits_view(table), {"nyt://not_here", "999"})


def test_planted_api_leak_raises():
    table = _full_table()
    leaked_id = table.filter(pl.col("source") == "doca_api")["id"][0]
    with pytest.raises(ValueError, match="leaked"):
        assert_holdout_excluded(_as_splits_view(table), {leaked_id})


def test_planted_ldc_leak_raises():
    table = _full_table()
    leaked_id = table.filter(pl.col("source") == "ldc_dateline_neg")["id"][0]
    with pytest.raises(ValueError, match="leaked"):
        assert_holdout_excluded(_as_splits_view(table), {leaked_id})


def test_planted_unlabeled_leak_raises():
    table = _full_table()
    leaked_id = table.filter(pl.col("source") == "api_unlabeled")["id"][0]
    with pytest.raises(ValueError, match="leaked"):
        assert_holdout_excluded(_as_splits_view(table), {leaked_id})
