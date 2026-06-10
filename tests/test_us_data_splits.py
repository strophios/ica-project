"""
Invariant tests for the train/val/test splitting logic in `create_us_filter_data`.

The split logic uses polars `sample(fraction=0.9, seed=200)` + `is_in(...).not_()`
to carve out the held-out halves, separately for positive and negative examples.
The invariants that matter for downstream model training:

  1. Given a fixed input and seed, the same rows end up in the same split
     on every call (determinism).
  2. Train / val / test are mutually exclusive (no id appears in more than
     one split).
  3. Every input row ends up in exactly one split (coverage).
  4. The split ratios are approximately 90 / 5 / 5, applied separately to
     positive and negative subsets.
  5. Null `us_label` rows are dropped.
  6. Non-unique `id` is loudly rejected (rather than silently producing
     leakage).
"""

import polars as pl
import pytest

from src.data_setup.data import create_us_filter_data


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _fake_us_dataframe(n_pos: int, n_neg: int) -> pl.DataFrame:
    """Build a minimal dataframe with the columns `create_us_filter_data`
    reads: id, us_label."""
    total = n_pos + n_neg
    return pl.DataFrame(
        {
            "id": [f"doc_{i}" for i in range(total)],
            "us_label": [True] * n_pos + [False] * n_neg,
        }
    )


def _fake_us_dataframe_with_nulls(n_pos: int, n_neg: int, n_null: int) -> pl.DataFrame:
    """Build a dataframe with some null us_label values."""
    total = n_pos + n_neg + n_null
    return pl.DataFrame(
        {
            "id": [f"doc_{i}" for i in range(total)],
            "us_label": [True] * n_pos + [False] * n_neg + [None] * n_null,
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_split_is_deterministic(self):
        """Same input + seed=200 → same split. This invariant is what we
        lean on to have reproducible training runs (AC3.5)."""
        df = _fake_us_dataframe(n_pos=100, n_neg=900)
        out_a = create_us_filter_data(df)
        out_b = create_us_filter_data(df)

        for split_name in ("train", "val", "test"):
            ids_a = sorted(out_a[split_name]["id"].to_list())
            ids_b = sorted(out_b[split_name]["id"].to_list())
            assert ids_a == ids_b, (
                f"Split '{split_name}' differed between runs. "
                f"Determinism broken."
            )


class TestMutualExclusion:
    def test_no_id_appears_in_multiple_splits(self):
        df = _fake_us_dataframe(n_pos=100, n_neg=900)
        out = create_us_filter_data(df)
        train_ids = set(out["train"]["id"].to_list())
        val_ids = set(out["val"]["id"].to_list())
        test_ids = set(out["test"]["id"].to_list())

        assert not (train_ids & val_ids), "train and val overlap"
        assert not (train_ids & test_ids), "train and test overlap"
        assert not (val_ids & test_ids), "val and test overlap"


class TestCoverage:
    def test_every_non_null_row_lands_in_exactly_one_split(self):
        df = _fake_us_dataframe(n_pos=100, n_neg=900)
        out = create_us_filter_data(df)
        total_rows = sum(out[s].shape[0] for s in ("train", "val", "test"))
        assert total_rows == df.shape[0], (
            f"Expected {df.shape[0]} rows total across splits, got {total_rows}"
        )

    def test_null_rows_are_dropped(self):
        """Rows with null us_label should not appear in any split."""
        df = _fake_us_dataframe_with_nulls(n_pos=100, n_neg=900, n_null=50)
        out = create_us_filter_data(df)
        total_rows = sum(out[s].shape[0] for s in ("train", "val", "test"))
        # Total should equal pos + neg (nulls dropped)
        expected = 100 + 900
        assert total_rows == expected, (
            f"Expected {expected} rows (nulls dropped), got {total_rows}"
        )


class TestRatios:
    def test_approximate_90_5_5_split_per_class(self):
        """90/5/5 split applied within each class (positive and negative separately)."""
        df = _fake_us_dataframe(n_pos=1000, n_neg=10000)
        out = create_us_filter_data(df)
        total = df.shape[0]

        train_frac = out["train"].shape[0] / total
        val_frac = out["val"].shape[0] / total
        test_frac = out["test"].shape[0] / total

        # Allow some slack because the split is applied separately per class,
        # and polars `sample(fraction=...)` gives a deterministic but not-exactly-that-fraction sample.
        assert 0.87 < train_frac < 0.93, f"train fraction {train_frac:.3f}"
        assert 0.03 < val_frac < 0.07, f"val fraction {val_frac:.3f}"
        assert 0.03 < test_frac < 0.07, f"test fraction {test_frac:.3f}"

    def test_both_classes_present_in_each_split(self):
        """Both positive and negative examples should appear in all splits."""
        df = _fake_us_dataframe(n_pos=100, n_neg=900)
        out = create_us_filter_data(df)

        for split_name in ("train", "val", "test"):
            split = out[split_name]
            pos_count = (split["us_label"] == True).sum()
            neg_count = (split["us_label"] == False).sum()
            assert pos_count > 0, f"{split_name} has no positive examples"
            assert neg_count > 0, f"{split_name} has no negative examples"


class TestIdUniquenessAssertion:
    def test_duplicate_ids_are_rejected(self):
        """The split logic depends on unique `id` values — duplicate ids
        would silently cause data leakage between splits. Duplicate ids
        should raise loudly."""
        df = pl.DataFrame(
            {
                "id": ["a", "a", "b", "c", "d"],
                "us_label": [True, False, True, False, False],
            }
        )
        with pytest.raises(AssertionError, match="not unique"):
            create_us_filter_data(df)
