"""
Invariant tests for the train/val/test splitting logic in
`create_classifier_data`.

The split logic uses polars `sample(fraction=0.9, seed=200)` + `is_in(...).not_()`
to carve out the held-out halves. The invariants that matter for any
downstream model training:

  1. Given a fixed input and seed, the same rows end up in the same split
     on every call (determinism).
  2. Train / val / test are mutually exclusive (no id appears in more than
     one split).
  3. Every input row ends up in exactly one split (coverage).
  4. The split ratios are approximately 90 / 5 / 5, applied separately to
     labeled and unlabeled subsets.
  5. Non-unique `id` is loudly rejected (rather than silently producing
     leakage).
"""

import polars as pl
import pytest

from src.data_setup.data import create_classifier_data, assert_holdout_excluded


# -----------------------------------------------------------------------------
# Fixture builders
# -----------------------------------------------------------------------------

def _fake_dataframe(n_labeled: int, n_unlabeled: int) -> pl.DataFrame:
    """Build a minimal dataframe with the columns `create_classifier_data`
    reads: id, cca, cca_descriptor, immig, immig_descriptor."""
    total = n_labeled + n_unlabeled
    return pl.DataFrame(
        {
            "id": [f"doc_{i}" for i in range(total)],
            # First n_labeled rows are positive via cca_descriptor; the rest
            # are negatives for both sources.
            "cca": [False] * total,
            "cca_descriptor": [i < n_labeled for i in range(total)],
            "immig": [False] * total,
            "immig_descriptor": [False] * total,
        }
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

class TestDeterminism:
    def test_split_is_deterministic(self):
        """Same input + same seed → same split. This invariant is what we
        lean on to have reproducible training runs."""
        df = _fake_dataframe(n_labeled=200, n_unlabeled=2000)
        out_a = create_classifier_data(df, separate_labels=False)
        out_b = create_classifier_data(df, separate_labels=False)

        for split_name in ("train", "val", "test"):
            ids_a = sorted(out_a[split_name]["id"].to_list())
            ids_b = sorted(out_b[split_name]["id"].to_list())
            assert ids_a == ids_b, (
                f"Split '{split_name}' differed between runs. "
                f"Determinism broken."
            )


class TestMutualExclusion:
    def test_no_id_appears_in_multiple_splits(self):
        df = _fake_dataframe(n_labeled=200, n_unlabeled=2000)
        out = create_classifier_data(df, separate_labels=False)
        train_ids = set(out["train"]["id"].to_list())
        val_ids = set(out["val"]["id"].to_list())
        test_ids = set(out["test"]["id"].to_list())

        assert not (train_ids & val_ids), "train and val overlap"
        assert not (train_ids & test_ids), "train and test overlap"
        assert not (val_ids & test_ids), "val and test overlap"


class TestCoverage:
    def test_every_row_lands_in_exactly_one_split(self):
        df = _fake_dataframe(n_labeled=200, n_unlabeled=2000)
        out = create_classifier_data(df, separate_labels=False)
        total_rows = sum(out[s].shape[0] for s in ("train", "val", "test"))
        assert total_rows == df.shape[0], (
            f"Expected {df.shape[0]} rows total across splits, got {total_rows}"
        )

    def test_separate_labels_mode_covers_every_row(self):
        df = _fake_dataframe(n_labeled=200, n_unlabeled=2000)
        out = create_classifier_data(df, separate_labels=True)
        total_rows = sum(
            out[split][pu].shape[0]
            for split in ("train", "val", "test")
            for pu in ("pos", "unl")
        )
        assert total_rows == df.shape[0]


class TestRatios:
    def test_approximate_90_5_5_split(self):
        df = _fake_dataframe(n_labeled=1000, n_unlabeled=10000)
        out = create_classifier_data(df, separate_labels=False)
        total = df.shape[0]
        train_frac = out["train"].shape[0] / total
        val_frac = out["val"].shape[0] / total
        test_frac = out["test"].shape[0] / total

        # Allow some slack because the split is applied to labeled and
        # unlabeled separately, and polars `sample(fraction=...)` gives a
        # deterministic but not-exactly-that-fraction sample.
        assert 0.87 < train_frac < 0.93, f"train fraction {train_frac:.3f}"
        assert 0.03 < val_frac < 0.07, f"val fraction {val_frac:.3f}"
        assert 0.03 < test_frac < 0.07, f"test fraction {test_frac:.3f}"


class TestIdUniquenessAssertion:
    def test_duplicate_ids_are_rejected(self):
        """The split logic depends on unique `id` values — duplicate ids
        would silently cause data leakage between splits. Duplicate ids
        should raise loudly."""
        df = pl.DataFrame(
            {
                "id": ["a", "a", "b", "c", "d"],
                "cca": [False] * 5,
                "cca_descriptor": [True, False, True, False, False],
                "immig": [False] * 5,
                "immig_descriptor": [False] * 5,
            }
        )
        with pytest.raises(AssertionError, match="not unique"):
            create_classifier_data(df)


# -----------------------------------------------------------------------------
# Label-construction tests
# -----------------------------------------------------------------------------

def _labeled_dataframe(rows: list[dict]) -> pl.DataFrame:
    """Build a dataframe from explicit per-row dicts with the four boolean
    columns `create_classifier_data` reads to construct labels.

    Each dict must have keys: id, cca, cca_descriptor, immig, immig_descriptor.
    We need at least one row where cca=True or cca_descriptor=True (so that
    create_classifier_data's labeled split is non-empty and doesn't error).
    """
    return pl.DataFrame(
        {
            "id": [r["id"] for r in rows],
            "cca": [r["cca"] for r in rows],
            "cca_descriptor": [r["cca_descriptor"] for r in rows],
            "immig": [r["immig"] for r in rows],
            "immig_descriptor": [r["immig_descriptor"] for r in rows],
        }
    )


def _all_rows(out: dict) -> pl.DataFrame:
    """Concatenate the three splits back into a single dataframe for
    label inspection (we don't care which split each row lands in)."""
    return pl.concat([out["train"], out["val"], out["test"]])


class TestLabelConstruction:
    """Tests for the cca_label / immig_label construction logic in
    `create_classifier_data`.

    The function computes:
        cca_label   = 1 if (cca OR cca_descriptor) else 0
        immig_label = 1 if (immig OR immig_descriptor) else 0

    These tests verify all four boolean combinations for each label and that
    the output columns have integer dtype.
    """

    # We need enough rows with cca_label=1 for the labeled split to be
    # non-empty, and enough rows with cca_label=0 for the unlabeled split.
    # Strategy: include a block of "anchor" rows (cca=True, immig=False) so
    # that every test dataframe satisfies the non-empty-split precondition
    # regardless of what the test-specific rows contain.
    _N_ANCHOR = 20  # rows with cca_label=1, immig_label=0

    def _anchor_rows(self) -> list[dict]:
        """Return anchor rows that ensure labeled+unlabeled splits are
        non-empty for every test."""
        return [
            {
                "id": f"anchor_{i}",
                "cca": True,
                "cca_descriptor": False,
                "immig": False,
                "immig_descriptor": False,
            }
            for i in range(self._N_ANCHOR)
        ]

    def _unlabeled_rows(self, n: int, offset: int = 0) -> list[dict]:
        """Return rows with all four booleans False (cca_label=0, immig_label=0)."""
        return [
            {
                "id": f"unl_{offset + i}",
                "cca": False,
                "cca_descriptor": False,
                "immig": False,
                "immig_descriptor": False,
            }
            for i in range(n)
        ]

    # --- cca_label combinations ---

    def test_cca_false_false_gives_label_0(self):
        """(cca=False, cca_descriptor=False) → cca_label=0."""
        test_row = {
            "id": "test_row",
            "cca": False,
            "cca_descriptor": False,
            "immig": False,
            "immig_descriptor": False,
        }
        df = _labeled_dataframe(
            [test_row] + self._anchor_rows() + self._unlabeled_rows(20)
        )
        all_rows = _all_rows(create_classifier_data(df, separate_labels=False))
        row = all_rows.filter(pl.col("id") == "test_row")
        assert row["cca_label"][0] == 0

    def test_cca_true_false_gives_label_1(self):
        """(cca=True, cca_descriptor=False) → cca_label=1."""
        test_row = {
            "id": "test_row",
            "cca": True,
            "cca_descriptor": False,
            "immig": False,
            "immig_descriptor": False,
        }
        df = _labeled_dataframe(
            [test_row] + self._anchor_rows() + self._unlabeled_rows(20)
        )
        all_rows = _all_rows(create_classifier_data(df, separate_labels=False))
        row = all_rows.filter(pl.col("id") == "test_row")
        assert row["cca_label"][0] == 1

    def test_cca_false_true_gives_label_1(self):
        """(cca=False, cca_descriptor=True) → cca_label=1."""
        test_row = {
            "id": "test_row",
            "cca": False,
            "cca_descriptor": True,
            "immig": False,
            "immig_descriptor": False,
        }
        df = _labeled_dataframe(
            [test_row] + self._anchor_rows() + self._unlabeled_rows(20)
        )
        all_rows = _all_rows(create_classifier_data(df, separate_labels=False))
        row = all_rows.filter(pl.col("id") == "test_row")
        assert row["cca_label"][0] == 1

    def test_cca_true_true_gives_label_1(self):
        """(cca=True, cca_descriptor=True) → cca_label=1."""
        test_row = {
            "id": "test_row",
            "cca": True,
            "cca_descriptor": True,
            "immig": False,
            "immig_descriptor": False,
        }
        df = _labeled_dataframe(
            [test_row] + self._anchor_rows() + self._unlabeled_rows(20)
        )
        all_rows = _all_rows(create_classifier_data(df, separate_labels=False))
        row = all_rows.filter(pl.col("id") == "test_row")
        assert row["cca_label"][0] == 1

    # --- immig_label combinations ---

    def test_immig_false_false_gives_label_0(self):
        """(immig=False, immig_descriptor=False) → immig_label=0."""
        test_row = {
            "id": "test_row",
            "cca": False,
            "cca_descriptor": False,
            "immig": False,
            "immig_descriptor": False,
        }
        df = _labeled_dataframe(
            [test_row] + self._anchor_rows() + self._unlabeled_rows(20)
        )
        all_rows = _all_rows(create_classifier_data(df, separate_labels=False))
        row = all_rows.filter(pl.col("id") == "test_row")
        assert row["immig_label"][0] == 0

    def test_immig_true_false_gives_label_1(self):
        """(immig=True, immig_descriptor=False) → immig_label=1."""
        test_row = {
            "id": "test_row",
            "cca": False,
            "cca_descriptor": False,
            "immig": True,
            "immig_descriptor": False,
        }
        df = _labeled_dataframe(
            [test_row] + self._anchor_rows() + self._unlabeled_rows(20)
        )
        all_rows = _all_rows(create_classifier_data(df, separate_labels=False))
        row = all_rows.filter(pl.col("id") == "test_row")
        assert row["immig_label"][0] == 1

    def test_immig_false_true_gives_label_1(self):
        """(immig=False, immig_descriptor=True) → immig_label=1."""
        test_row = {
            "id": "test_row",
            "cca": False,
            "cca_descriptor": False,
            "immig": False,
            "immig_descriptor": True,
        }
        df = _labeled_dataframe(
            [test_row] + self._anchor_rows() + self._unlabeled_rows(20)
        )
        all_rows = _all_rows(create_classifier_data(df, separate_labels=False))
        row = all_rows.filter(pl.col("id") == "test_row")
        assert row["immig_label"][0] == 1

    def test_immig_true_true_gives_label_1(self):
        """(immig=True, immig_descriptor=True) → immig_label=1."""
        test_row = {
            "id": "test_row",
            "cca": True,
            "cca_descriptor": False,
            "immig": True,
            "immig_descriptor": True,
        }
        df = _labeled_dataframe(
            [test_row] + self._anchor_rows() + self._unlabeled_rows(20)
        )
        all_rows = _all_rows(create_classifier_data(df, separate_labels=False))
        row = all_rows.filter(pl.col("id") == "test_row")
        assert row["immig_label"][0] == 1

    # --- dtype and independence ---

    def test_cca_label_is_integer_dtype(self):
        """cca_label output column is integer (0/1 ints), not boolean."""
        df = _labeled_dataframe(self._anchor_rows() + self._unlabeled_rows(20))
        all_rows = _all_rows(create_classifier_data(df, separate_labels=False))
        assert all_rows["cca_label"].dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64), (
            f"Expected integer dtype, got {all_rows['cca_label'].dtype}"
        )

    def test_immig_label_is_integer_dtype(self):
        """immig_label output column is integer (0/1 ints), not boolean."""
        df = _labeled_dataframe(self._anchor_rows() + self._unlabeled_rows(20))
        all_rows = _all_rows(create_classifier_data(df, separate_labels=False))
        assert all_rows["immig_label"].dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64), (
            f"Expected integer dtype, got {all_rows['immig_label'].dtype}"
        )

    def test_labels_are_independent_per_row(self):
        """Each row's labels depend only on its own boolean columns, not
        on other rows. Verify by checking two rows with different combinations
        produce the expected labels independently."""
        rows = [
            # Row with cca=True, immig=False
            {
                "id": "cca_only",
                "cca": True,
                "cca_descriptor": False,
                "immig": False,
                "immig_descriptor": False,
            },
            # Row with cca=False, immig=True
            {
                "id": "immig_only",
                "cca": False,
                "cca_descriptor": False,
                "immig": True,
                "immig_descriptor": False,
            },
        ]
        df = _labeled_dataframe(rows + self._anchor_rows() + self._unlabeled_rows(20))
        all_rows = _all_rows(create_classifier_data(df, separate_labels=False))

        cca_row = all_rows.filter(pl.col("id") == "cca_only")
        immig_row = all_rows.filter(pl.col("id") == "immig_only")

        assert cca_row["cca_label"][0] == 1
        assert cca_row["immig_label"][0] == 0
        assert immig_row["cca_label"][0] == 0
        assert immig_row["immig_label"][0] == 1


class TestCreateRelevanceData:
    """The 3-way PNU split: positives / reliable-negatives / unlabeled."""

    @staticmethod
    def _table():
        # 8 positives (US), 6 reliable-neg, 20 unlabeled-US, 4 unlabeled-not-US.
        # (count, cca_label, us, reliable_neg)
        spec = [(8, 1, True, False), (6, 0, True, True), (20, 0, True, False), (4, 0, False, False)]
        cca, us, rneg = [], [], []
        for n, lab, is_us, rn in spec:
            cca += [lab] * n
            us += [is_us] * n
            rneg += [rn] * n
        return pl.DataFrame({
            "id": [f"d{i}" for i in range(len(cca))],
            "cca_label": cca, "us": us, "reliable_neg": rneg,
            "emb_row": list(range(len(cca))),
        })

    def _splits(self):
        from src.data_setup.data import create_relevance_data
        return create_relevance_data(self._table())

    def test_groups_are_disjoint_and_correctly_assigned(self):
        s = self._splits()
        all_pos = pl.concat([s[k]["pos"] for k in ("train", "val", "test")])
        all_neg = pl.concat([s[k]["neg"] for k in ("train", "val", "test")])
        all_unl = pl.concat([s[k]["unl"] for k in ("train", "val", "test")])
        pos_ids, neg_ids, unl_ids = (set(g["id"].to_list()) for g in (all_pos, all_neg, all_unl))
        # positives = cca_label==1; reliable negatives carved out; unlabeled is
        # US-restricted and excludes the reliable negatives.
        assert all(all_pos["cca_label"] == 1)
        assert all(all_neg["reliable_neg"])
        assert pos_ids.isdisjoint(neg_ids)
        assert neg_ids.isdisjoint(unl_ids)
        assert pos_ids.isdisjoint(unl_ids)
        # non-US unlabeled rows (4) are dropped from the unlabeled pool.
        assert len(unl_ids) == 20
        assert "emb_row" in all_unl.columns

    def test_holdout_drops_ids_from_all_groups(self):
        from src.data_setup.data import create_relevance_data
        held = ["d0", "d8", "d14"]  # one positive, one reliable-neg, one unlabeled
        s = create_relevance_data(self._table(), holdout_ids=held)
        seen = set()
        for k in ("train", "val", "test"):
            for grp in ("pos", "neg", "unl"):
                seen |= set(s[k][grp]["id"].to_list())
        assert seen.isdisjoint(set(held))


# =============================================================================
# Tests for assert_holdout_excluded
# =============================================================================

class TestAssertHoldoutExcluded:
    """Runtime leakage-guard assertion for train/val pools."""

    @staticmethod
    def _cca_splits(train_ids, val_ids, test_ids):
        """Build a CCA-shape splits dict: {"train":{"pos","unl"},...}."""
        return {
            "train": {
                "pos": pl.DataFrame({"id": train_ids[:2]}),
                "unl": pl.DataFrame({"id": train_ids[2:]}),
            },
            "val": {
                "pos": pl.DataFrame({"id": val_ids[:1]}),
                "unl": pl.DataFrame({"id": val_ids[1:]}),
            },
            "test": {
                "pos": pl.DataFrame({"id": test_ids[:1]}),
                "unl": pl.DataFrame({"id": test_ids[1:]}),
            },
        }

    @staticmethod
    def _relevance_splits(train_ids, val_ids, test_ids):
        """Build a relevance-shape splits dict: {"train":{"pos","neg","unl"},...}."""
        return {
            "train": {
                "pos": pl.DataFrame({"id": train_ids[:2]}),
                "neg": pl.DataFrame({"id": train_ids[2:3]}),
                "unl": pl.DataFrame({"id": train_ids[3:]}),
            },
            "val": {
                "pos": pl.DataFrame({"id": val_ids[:1]}),
                "neg": pl.DataFrame({"id": val_ids[1:2]}),
                "unl": pl.DataFrame({"id": val_ids[2:]}),
            },
            "test": {
                "pos": pl.DataFrame({"id": test_ids[:1]}),
                "neg": pl.DataFrame({"id": test_ids[1:2]}),
                "unl": pl.DataFrame({"id": test_ids[2:]}),
            },
        }

    def test_cca_clean_split_passes(self):
        """CCA split with no holdout ids should pass (no-op)."""
        splits = self._cca_splits(
            train_ids=["t1", "t2", "t3"],
            val_ids=["v1", "v2"],
            test_ids=["te1", "te2"],
        )
        # Should not raise
        assert_holdout_excluded(splits, None)
        assert_holdout_excluded(splits, set())
        assert_holdout_excluded(splits, [])

    def test_cca_clean_split_with_non_overlapping_holdout(self):
        """CCA split where holdout ids don't appear anywhere should pass."""
        splits = self._cca_splits(
            train_ids=["t1", "t2", "t3"],
            val_ids=["v1", "v2"],
            test_ids=["te1", "te2"],
        )
        # holdout has "other1", "other2" which don't appear in train/val
        assert_holdout_excluded(splits, ["other1", "other2"])

    def test_cca_leaks_in_train_pos(self):
        """CCA: holdout id in train[pos] should raise with id enumerated."""
        splits = self._cca_splits(
            train_ids=["t1", "t2", "t3"],
            val_ids=["v1", "v2"],
            test_ids=["te1", "te2"],
        )
        with pytest.raises(ValueError, match="leaked.*t2"):
            assert_holdout_excluded(splits, ["t2"])

    def test_cca_leaks_in_train_unl(self):
        """CCA: holdout id in train[unl] should raise with id enumerated."""
        splits = self._cca_splits(
            train_ids=["t1", "t2", "t3"],
            val_ids=["v1", "v2"],
            test_ids=["te1", "te2"],
        )
        with pytest.raises(ValueError, match="leaked.*t3"):
            assert_holdout_excluded(splits, ["t3"])

    def test_cca_leaks_in_val_pos(self):
        """CCA: holdout id in val[pos] should raise with id enumerated."""
        splits = self._cca_splits(
            train_ids=["t1", "t2", "t3"],
            val_ids=["v1", "v2"],
            test_ids=["te1", "te2"],
        )
        with pytest.raises(ValueError, match="leaked.*v1"):
            assert_holdout_excluded(splits, ["v1"])

    def test_cca_leaks_in_val_unl(self):
        """CCA: holdout id in val[unl] should raise with id enumerated."""
        splits = self._cca_splits(
            train_ids=["t1", "t2", "t3"],
            val_ids=["v1", "v2"],
            test_ids=["te1", "te2"],
        )
        with pytest.raises(ValueError, match="leaked.*v2"):
            assert_holdout_excluded(splits, ["v2"])

    def test_cca_safe_in_test(self):
        """CCA: holdout ids in test[] are OK (test is evaluation-only)."""
        splits = self._cca_splits(
            train_ids=["t1", "t2", "t3"],
            val_ids=["v1", "v2"],
            test_ids=["te1", "te2"],
        )
        # holdout ids in test should not raise
        assert_holdout_excluded(splits, ["te1", "te2"])

    def test_cca_multiple_leaked_ids_enumerated(self):
        """CCA: multiple leaked ids should be enumerated in error."""
        splits = self._cca_splits(
            train_ids=["t1", "t2", "t3"],
            val_ids=["v1", "v2"],
            test_ids=["te1", "te2"],
        )
        with pytest.raises(ValueError) as exc_info:
            assert_holdout_excluded(splits, ["t1", "t2", "v1"])
        error_msg = str(exc_info.value)
        assert "t1" in error_msg
        assert "t2" in error_msg
        assert "v1" in error_msg

    def test_relevance_clean_split_passes(self):
        """Relevance split with no holdout ids should pass."""
        splits = self._relevance_splits(
            train_ids=["t1", "t2", "t3", "t4"],
            val_ids=["v1", "v2", "v3"],
            test_ids=["te1", "te2", "te3"],
        )
        assert_holdout_excluded(splits, None)
        assert_holdout_excluded(splits, set())

    def test_relevance_clean_split_with_non_overlapping_holdout(self):
        """Relevance split where holdout ids don't appear should pass."""
        splits = self._relevance_splits(
            train_ids=["t1", "t2", "t3", "t4"],
            val_ids=["v1", "v2", "v3"],
            test_ids=["te1", "te2", "te3"],
        )
        assert_holdout_excluded(splits, ["other1", "other2"])

    def test_relevance_leaks_in_train_pos(self):
        """Relevance: holdout in train[pos] should raise."""
        splits = self._relevance_splits(
            train_ids=["t1", "t2", "t3", "t4"],
            val_ids=["v1", "v2", "v3"],
            test_ids=["te1", "te2", "te3"],
        )
        with pytest.raises(ValueError, match="leaked.*t1"):
            assert_holdout_excluded(splits, ["t1"])

    def test_relevance_leaks_in_train_neg(self):
        """Relevance: holdout in train[neg] should raise."""
        splits = self._relevance_splits(
            train_ids=["t1", "t2", "t3", "t4"],
            val_ids=["v1", "v2", "v3"],
            test_ids=["te1", "te2", "te3"],
        )
        with pytest.raises(ValueError, match="leaked.*t3"):
            assert_holdout_excluded(splits, ["t3"])

    def test_relevance_leaks_in_train_unl(self):
        """Relevance: holdout in train[unl] should raise."""
        splits = self._relevance_splits(
            train_ids=["t1", "t2", "t3", "t4"],
            val_ids=["v1", "v2", "v3"],
            test_ids=["te1", "te2", "te3"],
        )
        with pytest.raises(ValueError, match="leaked.*t4"):
            assert_holdout_excluded(splits, ["t4"])

    def test_relevance_leaks_in_val_pos(self):
        """Relevance: holdout in val[pos] should raise."""
        splits = self._relevance_splits(
            train_ids=["t1", "t2", "t3", "t4"],
            val_ids=["v1", "v2", "v3"],
            test_ids=["te1", "te2", "te3"],
        )
        with pytest.raises(ValueError, match="leaked.*v1"):
            assert_holdout_excluded(splits, ["v1"])

    def test_relevance_leaks_in_val_neg(self):
        """Relevance: holdout in val[neg] should raise."""
        splits = self._relevance_splits(
            train_ids=["t1", "t2", "t3", "t4"],
            val_ids=["v1", "v2", "v3"],
            test_ids=["te1", "te2", "te3"],
        )
        with pytest.raises(ValueError, match="leaked.*v2"):
            assert_holdout_excluded(splits, ["v2"])

    def test_relevance_leaks_in_val_unl(self):
        """Relevance: holdout in val[unl] should raise."""
        splits = self._relevance_splits(
            train_ids=["t1", "t2", "t3", "t4"],
            val_ids=["v1", "v2", "v3"],
            test_ids=["te1", "te2", "te3"],
        )
        with pytest.raises(ValueError, match="leaked.*v3"):
            assert_holdout_excluded(splits, ["v3"])

    def test_relevance_safe_in_test(self):
        """Relevance: holdout ids in test[] are OK."""
        splits = self._relevance_splits(
            train_ids=["t1", "t2", "t3", "t4"],
            val_ids=["v1", "v2", "v3"],
            test_ids=["te1", "te2", "te3"],
        )
        assert_holdout_excluded(splits, ["te1", "te2", "te3"])

    def test_missing_train_split_raises(self):
        """Missing 'train' key in splits dict should raise."""
        splits = {
            "val": {"pos": pl.DataFrame({"id": ["v1"]}), "unl": pl.DataFrame({"id": ["v2"]})},
            "test": {"pos": pl.DataFrame({"id": ["te1"]}), "unl": pl.DataFrame({"id": ["te2"]})},
        }
        with pytest.raises(ValueError, match="'train'.*not found"):
            assert_holdout_excluded(splits, ["holdout"])

    def test_missing_val_split_raises(self):
        """Missing 'val' key in splits dict should raise."""
        splits = {
            "train": {"pos": pl.DataFrame({"id": ["t1"]}), "unl": pl.DataFrame({"id": ["t2"]})},
            "test": {"pos": pl.DataFrame({"id": ["te1"]}), "unl": pl.DataFrame({"id": ["te2"]})},
        }
        with pytest.raises(ValueError, match="'val'.*not found"):
            assert_holdout_excluded(splits, ["holdout"])

    def test_missing_pos_in_train_raises(self):
        """Missing 'pos' group in train split should raise."""
        splits = {
            "train": {"unl": pl.DataFrame({"id": ["t1"]})},
            "val": {"pos": pl.DataFrame({"id": ["v1"]}), "unl": pl.DataFrame({"id": ["v2"]})},
            "test": {"pos": pl.DataFrame({"id": ["te1"]}), "unl": pl.DataFrame({"id": ["te2"]})},
        }
        with pytest.raises(ValueError, match="'pos'.*not found"):
            assert_holdout_excluded(splits, ["holdout"])

    def test_missing_unl_in_val_raises(self):
        """Missing 'unl' group in val split should raise."""
        splits = {
            "train": {"pos": pl.DataFrame({"id": ["t1"]}), "unl": pl.DataFrame({"id": ["t2"]})},
            "val": {"pos": pl.DataFrame({"id": ["v1"]})},
            "test": {"pos": pl.DataFrame({"id": ["te1"]}), "unl": pl.DataFrame({"id": ["te2"]})},
        }
        with pytest.raises(ValueError, match="'unl'.*not found"):
            assert_holdout_excluded(splits, ["holdout"])

    def test_non_dict_split_raises(self):
        """Non-dict value in splits should raise."""
        splits = {
            "train": pl.DataFrame({"id": ["t1"]}),  # Wrong: should be dict
            "val": {"pos": pl.DataFrame({"id": ["v1"]}), "unl": pl.DataFrame({"id": ["v2"]})},
            "test": {"pos": pl.DataFrame({"id": ["te1"]}), "unl": pl.DataFrame({"id": ["te2"]})},
        }
        with pytest.raises(ValueError, match="'train'.*not a dict"):
            assert_holdout_excluded(splits, ["holdout"])

    def test_non_dataframe_group_raises(self):
        """Non-DataFrame value in group should raise."""
        splits = {
            "train": {
                "pos": ["t1", "t2"],  # Wrong: should be DataFrame
                "unl": pl.DataFrame({"id": ["t3"]}),
            },
            "val": {"pos": pl.DataFrame({"id": ["v1"]}), "unl": pl.DataFrame({"id": ["v2"]})},
            "test": {"pos": pl.DataFrame({"id": ["te1"]}), "unl": pl.DataFrame({"id": ["te2"]})},
        }
        with pytest.raises(ValueError, match="'pos'.*not a DataFrame"):
            assert_holdout_excluded(splits, ["holdout"])
