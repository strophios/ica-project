# pattern: Functional Core (pure data label logic)
"""Tests for src.validation.ica_eval — ICA label derivation and anchor reservation."""

from __future__ import annotations

import polars as pl

from src.validation.ica_eval import (
    derive_ica_negatives,
    reconcile_immig_column,
    reserve_anchor_holdout,
    assemble_holdout_ids,
    apply_us_scope_to_ica,
)
from src.validation.schema import validate_gold_set


class TestDeriveIcaNegatives:
    """ICA negative label derivation based on scope gates."""

    def test_us_false_yields_ica_false(self):
        """us_event=False ⟹ ica_event=False regardless of cca_event."""
        df = pl.DataFrame({
            "us_event": [False, False],
            "cca_event": [True, False],
        })
        result = derive_ica_negatives(df)
        assert result["ica_event"].to_list() == [False, False]

    def test_cca_false_yields_ica_false(self):
        """cca_event=False ⟹ ica_event=False regardless of us_event."""
        df = pl.DataFrame({
            "us_event": [True, False],
            "cca_event": [False, False],
        })
        result = derive_ica_negatives(df)
        assert result["ica_event"].to_list() == [False, False]

    def test_both_true_yields_ica_null(self):
        """us_event=True ∧ cca_event=True ⟹ ica_event=null (holistic region)."""
        df = pl.DataFrame({
            "us_event": [True],
            "cca_event": [True],
        })
        result = derive_ica_negatives(df)
        assert result["ica_event"].is_null().to_list() == [True]

    def test_immig_relevant_false_with_both_true_yields_null_not_false(self):
        """Critical: immig_relevant=False MUST NOT auto-set ica_event=False.

        Even when immig_relevant=False, if us_event=True ∧ cca_event=True,
        ica_event remains null for holistic hand-coding (true ICA can be
        relevant only in context).
        """
        df = pl.DataFrame({
            "us_event": [True, True],
            "cca_event": [True, True],
            "immig_relevant": [False, True],
        })
        result = derive_ica_negatives(df)
        # Both rows should have null ica_event; immig_relevant is not consulted
        assert result["ica_event"].to_list() == [None, None]

    def test_preserves_other_columns(self):
        """Derivation doesn't drop or modify other columns."""
        df = pl.DataFrame({
            "id": ["a", "b", "c"],
            "us_event": [True, False, True],
            "cca_event": [True, True, False],
            "immig_relevant": [True, False, True],
            "headline": ["h1", "h2", "h3"],
        })
        result = derive_ica_negatives(df)
        assert result.columns == ["id", "us_event", "cca_event", "immig_relevant", "headline", "ica_event"]
        assert result["id"].to_list() == ["a", "b", "c"]
        assert result["headline"].to_list() == ["h1", "h2", "h3"]

    def test_mixed_scope_gates(self):
        """Multiple rows with various scope gate combinations."""
        df = pl.DataFrame({
            "us_event": [True, True, False, False, True],
            "cca_event": [True, False, True, False, True],
        })
        result = derive_ica_negatives(df)
        # us=T, cca=T -> null (holistic)
        # us=T, cca=F -> False
        # us=F, cca=T -> False
        # us=F, cca=F -> False
        # us=T, cca=T -> null (holistic)
        expected = [None, False, False, False, None]
        assert result["ica_event"].to_list() == expected


class TestReconcileImmigColumn:
    """Legacy immig (0/1) reconciliation into immig_relevant (bool)."""

    def test_immig_1_maps_to_true(self):
        """immig=1 → immig_relevant=True."""
        df = pl.DataFrame({
            "id": ["a"],
            "immig": [1],
        })
        result = reconcile_immig_column(df)
        assert result["immig_relevant"][0] is True
        # immig renamed to immig_advisory
        assert "immig_advisory" in result.columns
        assert result["immig_advisory"][0] == 1

    def test_immig_0_maps_to_false(self):
        """immig=0 → immig_relevant=False."""
        df = pl.DataFrame({
            "id": ["a"],
            "immig": [0],
        })
        result = reconcile_immig_column(df)
        assert result["immig_relevant"][0] is False
        assert "immig_advisory" in result.columns
        assert result["immig_advisory"][0] == 0

    def test_does_not_overwrite_hand_coded_immig_relevant(self):
        """If immig_relevant already exists, reconcile is no-op."""
        df = pl.DataFrame({
            "id": ["a", "b"],
            "immig": [1, 0],
            "immig_relevant": [False, True],  # Hand-coded; should NOT be overwritten
        })
        result = reconcile_immig_column(df)
        # immig_relevant should remain untouched
        assert result["immig_relevant"].to_list() == [False, True]
        # immig is still renamed to advisory
        assert "immig_advisory" in result.columns

    def test_preserves_other_columns(self):
        """Reconciliation doesn't drop other columns."""
        df = pl.DataFrame({
            "id": ["a", "b"],
            "immig": [1, 0],
            "headline": ["h1", "h2"],
            "us_event": [True, False],
        })
        result = reconcile_immig_column(df)
        assert "id" in result.columns
        assert "headline" in result.columns
        assert "us_event" in result.columns

    def test_missing_immig_column_is_noop(self):
        """No immig column present → no changes."""
        df = pl.DataFrame({
            "id": ["a"],
            "headline": ["h1"],
        })
        result = reconcile_immig_column(df)
        assert result.columns == ["id", "headline"]

    def test_immig_source_annotation(self):
        """immig_source='legacy' is added."""
        df = pl.DataFrame({
            "id": ["a"],
            "immig": [1],
        })
        result = reconcile_immig_column(df)
        assert "immig_source" in result.columns
        assert result["immig_source"][0] == "legacy"

    def test_mixed_legacy_and_handcoded(self):
        """Multiple rows with both immig and immig_relevant."""
        df = pl.DataFrame({
            "id": ["a", "b", "c"],
            "immig": [1, 0, 1],
        })
        result = reconcile_immig_column(df)
        # All should reconcile since immig_relevant not present
        assert result["immig_relevant"].to_list() == [True, False, True]
        assert result["immig_advisory"].to_list() == [1, 0, 1]


class TestReserveAnchorHoldout:
    """Anchor holdout reservation with deduplication."""

    def test_dedupes_by_article_id(self):
        """Multiple rows with same article_id are deduplicated."""
        df = pl.DataFrame({
            "article_id": ["art1", "art1", "art2", "art2", "art3"],
            "event_type": ["protest", "protest", "strike", "strike", "march"],
        })
        holdout, train = reserve_anchor_holdout(df, frac=0.5, seed=200)
        # 3 unique articles; 50% holdout = 1-2 split (deterministic)
        assert len(holdout) + len(train) == 3
        assert len(set(holdout) & set(train)) == 0  # No overlap
        assert set(holdout) | set(train) == {"art1", "art2", "art3"}

    def test_deterministic_with_seed(self):
        """Same seed produces same split."""
        df = pl.DataFrame({
            "article_id": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        })
        hold1, train1 = reserve_anchor_holdout(df, frac=0.3, seed=200)
        hold2, train2 = reserve_anchor_holdout(df, frac=0.3, seed=200)
        assert hold1 == hold2
        assert train1 == train2


    def test_frac_30_default(self):
        """Default frac=0.30 reserves ~30% (deterministic rounding)."""
        df = pl.DataFrame({
            "article_id": [f"art{i}" for i in range(100)],
        })
        holdout, train = reserve_anchor_holdout(df, frac=0.30, seed=200)
        # 100 articles * 0.30 = 30
        assert len(holdout) == 30
        assert len(train) == 70

    def test_small_holdout_at_least_one(self):
        """Even with small frac, at least 1 row reserved."""
        df = pl.DataFrame({
            "article_id": ["a", "b"],
        })
        holdout, train = reserve_anchor_holdout(df, frac=0.05, seed=200)
        assert len(holdout) >= 1
        assert len(train) >= 0

    def test_returns_sorted_lists(self):
        """Return values are sorted lists of str."""
        df = pl.DataFrame({
            "article_id": ["z", "a", "m", "b"],
        })
        holdout, train = reserve_anchor_holdout(df, frac=0.5, seed=200)
        assert holdout == sorted(holdout)
        assert train == sorted(train)


class TestAssembleHoldoutIds:
    """Holdout id assembly: union + dedupe."""

    def test_single_list(self):
        """Single id list is deduplicated and returned."""
        result = assemble_holdout_ids(["a", "b", "a", "c"])
        assert result == ["a", "b", "c"]

    def test_multiple_lists_union(self):
        """Multiple lists are unioned (no duplicates across sources)."""
        result = assemble_holdout_ids(
            ["a", "b"],
            ["c", "d"],
            ["e", "f"],
        )
        assert result == ["a", "b", "c", "d", "e", "f"]

    def test_dedupes_across_lists(self):
        """Overlapping ids across lists are deduplicated."""
        result = assemble_holdout_ids(
            ["a", "b", "c"],
            ["b", "c", "d"],
            ["c", "d", "e"],
        )
        assert result == ["a", "b", "c", "d", "e"]
        assert len(result) == 5

    def test_empty_lists_handled(self):
        """Empty lists and empty input are handled."""
        result = assemble_holdout_ids([], ["a"], [])
        assert result == ["a"]

    def test_all_empty(self):
        """No id lists provided returns empty."""
        result = assemble_holdout_ids()
        assert result == []

    def test_sorted_output(self):
        """Output is sorted."""
        result = assemble_holdout_ids(
            ["z", "a"],
            ["m", "b"],
        )
        assert result == ["a", "b", "m", "z"]


class TestApplyUsScopeToIca:
    """US-scope ICA label rule: non-US events cannot be ICA by construction."""

    def test_us_false_sets_ica_false(self):
        """us_event=False ⟹ ica_event=False (scope rule)."""
        df = pl.DataFrame({
            "us_event": [False, False, False, False],
            "ica_event": [True, False, None, True],
        })
        result = apply_us_scope_to_ica(df)
        # All should be False (four rows ica=True/False/null/True -> False)
        assert result["ica_event"].to_list() == [False, False, False, False]

    def test_us_true_preserves_ica_event(self):
        """us_event=True ⟹ ica_event unchanged (scope applies only to negatives)."""
        df = pl.DataFrame({
            "us_event": [True, True, True],
            "ica_event": [True, False, None],
        })
        result = apply_us_scope_to_ica(df)
        # All should remain unchanged
        assert result["ica_event"].to_list() == [True, False, None]

    def test_us_null_preserves_ica_event(self):
        """us_event=null ⟹ ica_event unchanged (scope applies only to False)."""
        df = pl.DataFrame({
            "us_event": [None, None],
            "ica_event": [True, False],
        })
        result = apply_us_scope_to_ica(df)
        assert result["ica_event"].to_list() == [True, False]

    def test_ica_event_intl_preserves_original(self):
        """ica_event_intl column contains original ica_event (operator judgment)."""
        df = pl.DataFrame({
            "us_event": [True, False, False],
            "ica_event": [True, True, False],
        })
        result = apply_us_scope_to_ica(df)
        # ica_event_intl should be the original values
        assert result["ica_event_intl"].to_list() == [True, True, False]
        # ica_event should be scope-adjusted
        assert result["ica_event"].to_list() == [True, False, False]

    def test_idempotence_running_twice_is_noop(self):
        """Running apply_us_scope_to_ica twice is idempotent on ica_event."""
        df = pl.DataFrame({
            "us_event": [True, False, False, True],
            "ica_event": [True, True, False, None],
        })
        result1 = apply_us_scope_to_ica(df)
        result2 = apply_us_scope_to_ica(result1)
        # ica_event should be identical
        assert result1["ica_event"].to_list() == result2["ica_event"].to_list()
        # ica_event_intl should not change (guard against overwrite)
        assert result1["ica_event_intl"].to_list() == result2["ica_event_intl"].to_list()

    def test_mixed_us_scope(self):
        """Multiple rows with various us_event/ica_event combinations."""
        df = pl.DataFrame({
            "us_event": [True, True, False, False, None, None],
            "ica_event": [True, False, True, False, True, False],
        })
        result = apply_us_scope_to_ica(df)
        # us=T, ica=T -> T (preserve)
        # us=T, ica=F -> F (preserve)
        # us=F, ica=T -> F (scope rule)
        # us=F, ica=F -> F (preserve)
        # us=null, ica=T -> T (preserve)
        # us=null, ica=F -> F (preserve)
        expected = [True, False, False, False, True, False]
        assert result["ica_event"].to_list() == expected
        # ica_event_intl preserves originals
        expected_intl = [True, False, True, False, True, False]
        assert result["ica_event_intl"].to_list() == expected_intl

    def test_raises_on_missing_us_event(self):
        """ValueError if us_event column missing."""
        df = pl.DataFrame({
            "ica_event": [True],
        })
        try:
            apply_us_scope_to_ica(df)
            assert False, "should raise ValueError"
        except ValueError as e:
            assert "us_event" in str(e)

    def test_raises_on_missing_ica_event(self):
        """ValueError if ica_event column missing."""
        df = pl.DataFrame({
            "us_event": [True],
        })
        try:
            apply_us_scope_to_ica(df)
            assert False, "should raise ValueError"
        except ValueError as e:
            assert "ica_event" in str(e)

    def test_preserves_other_columns(self):
        """Function doesn't drop or modify other columns."""
        df = pl.DataFrame({
            "id": ["a", "b", "c"],
            "us_event": [True, False, True],
            "ica_event": [True, False, None],
            "headline": ["h1", "h2", "h3"],
        })
        result = apply_us_scope_to_ica(df)
        assert result["id"].to_list() == ["a", "b", "c"]
        assert result["headline"].to_list() == ["h1", "h2", "h3"]


class TestSchemaValidation:
    """Verify that assembled rows validate against the gold schema."""

    def test_derive_negatives_output_validates(self):
        """derive_ica_negatives output passes schema validation."""
        df = pl.DataFrame({
            "id": ["1", "2"],
            "corpus": ["api", "ldc"],
            "year": [1990, 1995],
            "news_desk": ["National", "World"],
            "section_name": ["US", "International"],
            "headline": ["Test", "Article"],
            "lead_paragraph": ["Lead 1", "Lead 2"],
            "sample_stratum": ["random_pre1986", "doca_matched"],
            "us_event": [True, False],
            "cca_event": [True, True],
        })
        result = derive_ica_negatives(df)
        # Should pass schema validation (ica_event is label column, nullable)
        validate_gold_set(result)
        # Verify the label logic
        assert result["ica_event"][0] is None  # us=T, cca=T -> null
        assert result["ica_event"][1] is False  # us=F, cca=T -> False

    def test_reconcile_output_validates(self):
        """reconcile_immig_column output passes schema validation."""
        df = pl.DataFrame({
            "id": ["1"],
            "corpus": ["api"],
            "year": [1990],
            "news_desk": ["National"],
            "section_name": ["US"],
            "headline": ["Test"],
            "lead_paragraph": ["Lead"],
            "sample_stratum": ["random_pre1986"],
            "immig": [1],
        })
        result = reconcile_immig_column(df)
        # Schema validation should pass (immig_relevant is label column)
        validate_gold_set(result)
        assert result["immig_relevant"][0] is True

    def test_integrated_label_logic_validates(self):
        """Combined ica + immig logic produces schema-valid output."""
        df = pl.DataFrame({
            "id": ["1", "2", "3"],
            "corpus": ["api", "api", "api"],
            "year": [1990, 1990, 1990],
            "news_desk": ["National", "National", "National"],
            "section_name": ["US", "US", "US"],
            "headline": ["h1", "h2", "h3"],
            "lead_paragraph": ["l1", "l2", "l3"],
            "sample_stratum": ["random_pre1986", "random_pre1986", "random_pre1986"],
            "us_event": [True, True, False],
            "cca_event": [True, False, True],
            "immig": [1, 0, 1],
        })
        result = (
            df
            .pipe(derive_ica_negatives)
            .pipe(reconcile_immig_column)
        )
        validate_gold_set(result)
        # Row 0: us=T, cca=T -> ica_event=null (holistic), immig_relevant=True
        # Row 1: us=T, cca=F -> ica_event=False, immig_relevant=False
        # Row 2: us=F, cca=T -> ica_event=False, immig_relevant=True
        assert result["ica_event"].to_list() == [None, False, False]
        assert result["immig_relevant"].to_list() == [True, False, True]
