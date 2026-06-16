"""DoCA labeling + US-restricted PU split (Phase 1).

- cca-doca.AC1.1: id in DoCA positives -> positive; else unlabeled.
- cca-doca.AC1.2: positives kept regardless of US score; unlabeled pool US-only.
- cca-doca.AC1.3: disjoint train/val/test (unique id, no leakage), deterministic.
"""

from __future__ import annotations

import polars as pl

from src.build_cca_doca_table import label_and_restrict
from src.data_setup.data import create_cca_doca_data


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
