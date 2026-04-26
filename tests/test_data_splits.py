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
