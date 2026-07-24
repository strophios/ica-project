"""US-head PNU retrain split (Phase: cca-doca-retrain, us-head-v1).

- us-pnu-data.AC1: 90/5/5 split applied separately to pos/neg/unl groups.
- us-pnu-data.AC2: disjoint, deterministic under seed.
- us-pnu-data.AC3: holdout ids dropped from the whole table before splitting.
- us-pnu-data.AC4: assert_holdout_excluded wiring (train/val leak raises, test is safe).
"""

from __future__ import annotations

import polars as pl
import pytest

from src.data_setup.data import assert_holdout_excluded, create_us_pnu_data


def _synthetic_table():
    """150 pos (train250k/us_pos_ldc345 mixed), 400 neg (us_train_ldc), 1000 unl (full)."""
    rows = []
    r = 0
    for i in range(100):
        rows.append({"id": f"p{i}", "pnu_label": "pos", "cache": "train250k", "emb_row": r})
        r += 1
    r = 0
    for i in range(100, 150):
        rows.append({"id": f"p{i}", "pnu_label": "pos", "cache": "us_pos_ldc345", "emb_row": r})
        r += 1
    r = 0
    for i in range(400):
        rows.append({"id": f"n{i}", "pnu_label": "neg", "cache": "us_train_ldc", "emb_row": r})
        r += 1
    r = 0
    for i in range(1000):
        rows.append({"id": f"u{i}", "pnu_label": "unl", "cache": "full", "emb_row": r})
        r += 1
    return pl.DataFrame(rows)


def test_split_covers_every_row_across_groups():
    table = _synthetic_table()
    out = create_us_pnu_data(table)
    seen = set()
    for split in ("train", "val", "test"):
        for grp in ("pos", "neg", "unl"):
            seen |= set(out[split][grp]["id"].to_list())
    assert seen == set(table["id"].to_list())


def test_split_ratios_approximately_90_5_5():
    table = _synthetic_table()
    out = create_us_pnu_data(table)
    for grp, n in (("pos", 150), ("neg", 400), ("unl", 1000)):
        n_train = sum(out[s][grp].height for s in ("train",))
        n_val = out["val"][grp].height
        n_test = out["test"][grp].height
        assert n_train + n_val + n_test == n
        assert abs(n_train / n - 0.9) < 0.05
        assert abs(n_val / n - 0.05) < 0.05
        assert abs(n_test / n - 0.05) < 0.05


def test_splits_are_disjoint_and_deterministic():
    table = _synthetic_table()
    out = create_us_pnu_data(table)
    for grp in ("pos", "neg", "unl"):
        tr = set(out["train"][grp]["id"])
        va = set(out["val"][grp]["id"])
        te = set(out["test"][grp]["id"])
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    # cross-group disjointness
    all_pos = set().union(*(set(out[s]["pos"]["id"]) for s in ("train", "val", "test")))
    all_neg = set().union(*(set(out[s]["neg"]["id"]) for s in ("train", "val", "test")))
    all_unl = set().union(*(set(out[s]["unl"]["id"]) for s in ("train", "val", "test")))
    assert all_pos.isdisjoint(all_neg)
    assert all_pos.isdisjoint(all_unl)
    assert all_neg.isdisjoint(all_unl)
    # emb_row + cache carried through
    assert {"emb_row", "cache"} <= set(out["train"]["pos"].columns)
    # deterministic under seed
    out2 = create_us_pnu_data(table)
    assert out["train"]["pos"].equals(out2["train"]["pos"])
    assert out["train"]["neg"].equals(out2["train"]["neg"])
    assert out["train"]["unl"].equals(out2["train"]["unl"])


def test_holdout_ids_dropped_from_whole_table():
    table = _synthetic_table()
    holdout = {"p0", "n0", "u0", "u1"}
    out = create_us_pnu_data(table, holdout_ids=holdout)
    seen = set()
    for split in ("train", "val", "test"):
        for grp in ("pos", "neg", "unl"):
            seen |= set(out[split][grp]["id"].to_list())
    assert seen.isdisjoint(holdout)
    assert len(seen) == table.height - len(holdout)


def test_holdout_none_and_empty_are_noops():
    table = _synthetic_table()
    base = create_us_pnu_data(table)
    for holdout in (None, [], set()):
        held = create_us_pnu_data(table, holdout_ids=holdout)
        assert base["train"]["pos"].equals(held["train"]["pos"])
        assert base["train"]["neg"].equals(held["train"]["neg"])
        assert base["train"]["unl"].equals(held["train"]["unl"])


def test_non_unique_id_raises():
    table = pl.concat([_synthetic_table(), _synthetic_table().head(1)])
    with pytest.raises(AssertionError, match="not unique"):
        create_us_pnu_data(table)


class TestAssertHoldoutExcludedWiring:
    def test_clean_split_passes(self):
        out = create_us_pnu_data(_synthetic_table())
        assert_holdout_excluded(out, None)
        assert_holdout_excluded(out, set())

    def test_leak_in_train_pos_raises(self):
        out = create_us_pnu_data(_synthetic_table())
        leaked = out["train"]["pos"]["id"][0]
        with pytest.raises(ValueError, match="leaked"):
            assert_holdout_excluded(out, [leaked])

    def test_leak_in_train_neg_raises(self):
        out = create_us_pnu_data(_synthetic_table())
        leaked = out["train"]["neg"]["id"][0]
        with pytest.raises(ValueError, match="leaked"):
            assert_holdout_excluded(out, [leaked])

    def test_leak_in_val_unl_raises(self):
        out = create_us_pnu_data(_synthetic_table())
        leaked = out["val"]["unl"]["id"][0]
        with pytest.raises(ValueError, match="leaked"):
            assert_holdout_excluded(out, [leaked])

    def test_leak_in_test_is_safe(self):
        out = create_us_pnu_data(_synthetic_table())
        test_ids = list(
            set(out["test"]["pos"]["id"].to_list())
            | set(out["test"]["neg"]["id"].to_list())
            | set(out["test"]["unl"]["id"].to_list())
        )[:3]
        assert_holdout_excluded(out, test_ids)
