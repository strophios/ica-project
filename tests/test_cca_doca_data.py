"""DoCA labeling + US-restricted PU split (Phase 1).

- cca-doca.AC1.1: id in DoCA positives -> positive; else unlabeled.
- cca-doca.AC1.2: positives kept regardless of US score; unlabeled pool US-only.
- cca-doca.AC1.3: disjoint train/val/test (unique id, no leakage), deterministic.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.build_cca_doca_table import filter_positives_by_form, label_and_restrict
from src.data_setup.data import create_cca_doca_data, create_relevance_data, assert_holdout_excluded


def test_filter_positives_by_form_partitions_ids():
    pos = pl.DataFrame({
        "id": ["a", "b", "c", "d"],
        "any_street": [1, 0, 1, None],  # int flags; null -> not-set
    })
    keep, drop = filter_positives_by_form(pos, "any_street")
    assert set(keep) == {"a", "c"}
    assert set(drop) == {"b", "d"}  # null treated as not-set


def test_filter_positives_by_form_unknown_column_raises():
    pos = pl.DataFrame({"id": ["a"], "any_street": [1]})
    with pytest.raises(ValueError):
        filter_positives_by_form(pos, "any_strike")  # no such form


def test_label_and_restrict_labels_and_thresholds():
    meta = pl.DataFrame({
        "emb_row": [0, 1, 2, 3],
        "id": ["a", "b", "c", "d"],
        "year": ["1965", "1970", "1980", "1990"],
        "us_logit": [3.0, -1.0, 0.5, -0.01],
    })
    out = label_and_restrict(meta, positive_ids=["a", "c"], threshold=0.0)
    assert out.sort("id")["cca_label"].to_list() == [1, 0, 1, 0]  # a,c positive
    # us = us_logit >= 0.0 -> a(T), b(F), c(T), d(F)
    assert out.sort("id")["us"].to_list() == [True, False, True, False]


def _synthetic_table():
    # 120 positives (alternating us), 1000 unlabeled (700 us True, 300 False).
    rows = []
    r = 0
    for i in range(120):
        rows.append({"emb_row": r, "id": f"p{i}", "cca_label": 1,
                     "us": (i % 2 == 0)})
        r += 1
    for i in range(1000):
        rows.append({"emb_row": r, "id": f"u{i}", "cca_label": 0,
                     "us": (i < 700)})
        r += 1
    return pl.DataFrame(rows)


def test_split_keeps_all_positives_regardless_of_us():
    table = _synthetic_table()
    out = create_cca_doca_data(table)
    pos_ids = set()
    for split in ("train", "val", "test"):
        pos_ids |= set(out[split]["pos"]["id"].to_list())
    # all 120 positives present, including the us=False ones
    assert pos_ids == {f"p{i}" for i in range(120)}


def test_split_unlabeled_pool_is_us_only():
    table = _synthetic_table()
    out = create_cca_doca_data(table)
    for split in ("train", "val", "test"):
        unl = out[split]["unl"]
        assert unl["us"].all()  # every unlabeled row is US
        # none of the us=False unlabeled (u700..u999) leaked in
        assert not any(int(x[1:]) >= 700 for x in unl["id"].to_list())


def test_holdout_ids_excluded_from_every_split():
    # cca-doca.AC1.4: gold-set ids dropped from the training pool (leakage guard).
    table = _synthetic_table()
    # mix unlabeled + one positive to document "drop from the whole table".
    holdout = {"u0", "u1", "u300", "p0"}
    out = create_cca_doca_data(table, holdout_ids=holdout)
    seen = set()
    for split in ("train", "val", "test"):
        for grp in ("pos", "unl"):
            seen |= set(out[split][grp]["id"].to_list())
    assert seen.isdisjoint(holdout)
    # everything NOT held out still present (120 pos - p0, US-unl 700 - u0,u1,u300)
    assert len(seen & {f"p{i}" for i in range(120)}) == 119
    assert "u0" not in seen and "u300" not in seen


def test_holdout_ids_none_is_a_noop():
    table = _synthetic_table()
    base = create_cca_doca_data(table)
    held = create_cca_doca_data(table, holdout_ids=None)
    assert base["train"]["unl"].equals(held["train"]["unl"])
    assert base["train"]["pos"].equals(held["train"]["pos"])


def test_holdout_ids_empty_is_a_noop():
    table = _synthetic_table()
    base = create_cca_doca_data(table)
    held = create_cca_doca_data(table, holdout_ids=[])
    assert base["train"]["unl"].equals(held["train"]["unl"])


def test_splits_are_disjoint_and_deterministic():
    table = _synthetic_table()
    out = create_cca_doca_data(table)
    # within-group split disjointness
    for grp in ("pos", "unl"):
        tr = set(out["train"][grp]["id"])
        va = set(out["val"][grp]["id"])
        te = set(out["test"][grp]["id"])
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    # pos vs unl disjoint
    all_pos = set().union(*(set(out[s]["pos"]["id"]) for s in ("train", "val", "test")))
    all_unl = set().union(*(set(out[s]["unl"]["id"]) for s in ("train", "val", "test")))
    assert all_pos.isdisjoint(all_unl)
    # emb_row carried through
    assert "emb_row" in out["train"]["pos"].columns
    # deterministic under seed
    out2 = create_cca_doca_data(table)
    assert out["train"]["pos"].equals(out2["train"]["pos"])
    assert out["train"]["unl"].equals(out2["train"]["unl"])


class TestAssertHoldoutExcludedOnRelevance:
    """Verify assert_holdout_excluded works correctly on relevance splits (pos/neg/unl)."""

    @staticmethod
    def _make_relevance_splits():
        """Build a realistic relevance-data table and split it."""
        rows = []
        r = 0
        # 50 positives
        for i in range(50):
            rows.append({
                "emb_row": r, "id": f"p{i}",
                "cca_label": 1, "reliable_neg": False, "us": True
            })
            r += 1
        # 30 reliable negatives
        for i in range(30):
            rows.append({
                "emb_row": r, "id": f"rn{i}",
                "cca_label": 0, "reliable_neg": True, "us": True
            })
            r += 1
        # 500 unlabeled US-only
        for i in range(500):
            rows.append({
                "emb_row": r, "id": f"u{i}",
                "cca_label": 0, "reliable_neg": False, "us": True
            })
            r += 1
        table = pl.DataFrame(rows)
        return create_relevance_data(table)

    def test_clean_relevance_split_passes(self):
        """Clean split with no holdout should pass."""
        splits = self._make_relevance_splits()
        assert_holdout_excluded(splits, None)
        assert_holdout_excluded(splits, set())

    def test_relevance_holdout_in_train_pos_raises(self):
        """Holdout id in train[pos] should raise."""
        splits = self._make_relevance_splits()
        # Get an actual positive id from train
        actual_pos_id = splits["train"]["pos"]["id"][0]
        with pytest.raises(ValueError, match="leaked"):
            assert_holdout_excluded(splits, [actual_pos_id])

    def test_relevance_holdout_in_train_neg_raises(self):
        """Holdout id in train[neg] should raise."""
        splits = self._make_relevance_splits()
        # Get an actual negative id from train
        actual_neg_id = splits["train"]["neg"]["id"][0]
        with pytest.raises(ValueError, match="leaked"):
            assert_holdout_excluded(splits, [actual_neg_id])

    def test_relevance_holdout_in_train_unl_raises(self):
        """Holdout id in train[unl] should raise."""
        splits = self._make_relevance_splits()
        # Get an actual unlabeled id from train
        actual_unl_id = splits["train"]["unl"]["id"][0]
        with pytest.raises(ValueError, match="leaked"):
            assert_holdout_excluded(splits, [actual_unl_id])

    def test_relevance_holdout_in_val_raises(self):
        """Holdout ids in val (any group) should raise."""
        splits = self._make_relevance_splits()
        actual_val_id = splits["val"]["pos"]["id"][0]
        with pytest.raises(ValueError, match="leaked"):
            assert_holdout_excluded(splits, [actual_val_id])

    def test_relevance_holdout_in_test_is_safe(self):
        """Holdout ids in test (evaluation only) should not raise."""
        splits = self._make_relevance_splits()
        test_ids = list(
            set(splits["test"]["pos"]["id"].to_list())
            | set(splits["test"]["neg"]["id"].to_list())
            | set(splits["test"]["unl"]["id"].to_list())
        )[:3]
        # Should not raise
        assert_holdout_excluded(splits, test_ids)
