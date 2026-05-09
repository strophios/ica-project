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

from src.data_setup.data import create_classifier_data


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
