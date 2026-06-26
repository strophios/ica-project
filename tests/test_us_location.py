"""Tests for the US/not-US location heuristic (src/preproc/us_location.py)."""

import polars as pl
import pytest

from src.preproc.us_location import (
    apply_fused_us_gate,
    compute_location_signals,
    gold_first_us_gate,
    is_clearly_foreign,
    is_foreign_place,
    is_us_place,
    load_place_sets,
    location_signals,
    load_location_signals,
    passes_us_gate,
)

PLACES = load_place_sets()


class TestPlacePredicates:
    @pytest.mark.parametrize("value", [
        "United States", "Miami (Fla)", "Brooklyn (NYC)", "California",
        "Los Angeles (Calif)", "U.S.",
    ])
    def test_us_places(self, value):
        assert is_us_place(value, PLACES)
        assert not is_foreign_place(value, PLACES)

    @pytest.mark.parametrize("value", [
        "Cuba", "Havana (Cuba)", "Iran", "Uganda", "Guatemala", "London (England)",
    ])
    def test_foreign_places(self, value):
        assert is_foreign_place(value, PLACES)
        assert not is_us_place(value, PLACES)

    def test_bare_us_city_without_state_is_neither(self):
        # Not in the AP-30 city list and no state parenthetical -> unresolved by
        # the gazetteer (the known limitation the ML filter complements).
        assert not is_us_place("Syracuse", PLACES)
        assert not is_foreign_place("Syracuse", PLACES)


class TestLocationSignals:
    def test_foreign_only_event(self):
        any_us, any_not = location_signals(["Uganda"], "Foreign Desk", "World")
        assert (any_us, any_not) == (False, True)
        assert is_clearly_foreign(any_us, any_not)

    def test_diaspora_has_us_footprint(self):
        # US enclave + foreign homeland -> both signals -> NOT clearly foreign.
        any_us, any_not = location_signals(["Cuba", "Miami (Fla)"], None, None)
        assert (any_us, any_not) == (True, True)
        assert not is_clearly_foreign(any_us, any_not)

    def test_us_desk_signal_without_glocation(self):
        any_us, any_not = location_signals([], "National Desk", "U.S.")
        assert any_us and not any_not

    def test_empty_signals(self):
        assert location_signals([], None, None) == (False, False)


def _kw(*locs):
    return [{"type": "glocations", "value": v, "rank": i + 1, "major": "N"}
            for i, v in enumerate(locs)]


class TestComputeLocationSignals:
    def test_per_article_signals(self):
        df = pl.DataFrame({
            "id": ["foreign", "diaspora", "domestic", "nokw"],
            "keywords": [_kw("Uganda"), _kw("Cuba", "Miami (Fla)"), _kw("California"), None],
            "news_desk": ["Foreign Desk", None, None, "National Desk"],
            "section_name": ["World", None, None, None],
        })
        out = compute_location_signals(df).sort("id")
        d = {r["id"]: (r["any_us"], r["any_not_us"]) for r in out.iter_rows(named=True)}
        assert d["foreign"] == (False, True)     # foreign loc + Foreign Desk
        assert d["diaspora"] == (True, True)      # US enclave + foreign homeland
        assert d["domestic"] == (True, False)     # US state
        assert d["nokw"] == (True, False)         # National Desk, no glocation


class TestFusedGate:
    def test_clearly_foreign_blocked_even_if_ml_passes(self):
        assert not passes_us_gate(0.99, any_us=False, any_not_us=True)

    def test_diaspora_passes_when_ml_passes(self):
        assert passes_us_gate(0.8, any_us=True, any_not_us=True)

    def test_ml_reject_blocks_regardless(self):
        assert not passes_us_gate(0.2, any_us=True, any_not_us=False)

    def test_domestic_passes(self):
        assert passes_us_gate(0.9, any_us=True, any_not_us=False)


class TestApplyFusedUSGate:
    """Truth-table tests for apply_fused_us_gate.

    The gate applies: us = us & ~(any_not_us & ~any_us)
    Which drops clearly-foreign (any_not_us=T, any_us=F) while keeping diaspora.
    """

    def test_us_only_kept(self):
        """US location, no foreign signal: passes through."""
        df = pl.DataFrame({
            "id": ["a"],
            "us": [True],
            "any_us": [True],
            "any_not_us": [False],
        })
        out = apply_fused_us_gate(df)
        assert out["us"].to_list() == [True]

    def test_clearly_foreign_dropped(self):
        """Foreign location signal, no US: dropped by fused gate."""
        df = pl.DataFrame({
            "id": ["b"],
            "us": [True],  # ML passed
            "any_us": [False],  # No US location
            "any_not_us": [True],  # Has foreign location
        })
        out = apply_fused_us_gate(df)
        assert out["us"].to_list() == [False]

    def test_diaspora_kept(self):
        """Both US and foreign locations: diaspora, kept by fused gate."""
        df = pl.DataFrame({
            "id": ["c"],
            "us": [True],  # ML passed
            "any_us": [True],  # Has US location (e.g., Miami)
            "any_not_us": [True],  # Has foreign location (e.g., Cuba)
        })
        out = apply_fused_us_gate(df)
        assert out["us"].to_list() == [True]

    def test_ml_reject_blocks_regardless(self):
        """ML already rejected (us=False): stays rejected."""
        df = pl.DataFrame({
            "id": ["d"],
            "us": [False],  # ML already rejected
            "any_us": [True],
            "any_not_us": [False],
        })
        out = apply_fused_us_gate(df)
        assert out["us"].to_list() == [False]

    def test_missing_required_columns_raises(self):
        """apply_fused_us_gate requires columns us, any_us, any_not_us."""
        df = pl.DataFrame({"id": ["e"], "us": [True]})
        with pytest.raises(ValueError, match="apply_fused_us_gate requires columns"):
            apply_fused_us_gate(df)

    def test_batch_with_mixed_results(self):
        """Batch of articles with mixed outcomes."""
        df = pl.DataFrame({
            "id": ["us_only", "clearly_foreign", "diaspora", "ml_reject"],
            "us": [True, True, True, False],
            "any_us": [True, False, True, True],
            "any_not_us": [False, True, True, False],
        })
        out = apply_fused_us_gate(df)
        # us_only: T & ~(F & F) = T & ~F = T & T = T
        # clearly_foreign: T & ~(T & T) = T & ~T = T & F = F
        # diaspora: T & ~(T & F) = T & ~F = T & T = T
        # ml_reject: F & ~(F & F) = F & ~F = F & T = F
        # When sorted by id: ["clearly_foreign", "diaspora", "ml_reject", "us_only"]
        assert out.sort("id")["us"].to_list() == [False, True, False, True]


class TestLoadLocationSignals:
    """Test load_location_signals empty-overlap case (no API-corpus rows match the given ids).

    When no table ids match any API-corpus row, the function returns an empty DataFrame
    with correct typed schema (string id, boolean any_us/any_not_us), not a Null-typed
    id that would crash the downstream join.
    """

    def test_empty_overlap_returns_typed_empty_frame(self):
        """No ids match API corpus -> returns typed-empty frame suitable for join."""
        # Use ids that definitely won't exist in the API corpus (negative test)
        nonexistent_ids = ["nonexistent_id_xyz_123", "another_fake_id_999"]

        result = load_location_signals(nonexistent_ids)

        # Verify structure: empty but with correct dtypes
        assert result.height == 0
        assert result.width == 3
        assert set(result.columns) == {"id", "any_us", "any_not_us"}

        # Verify dtypes are correct (not Null)
        schema = result.schema
        assert schema["id"] == pl.String
        assert schema["any_us"] == pl.Boolean
        assert schema["any_not_us"] == pl.Boolean

    def test_empty_overlap_join_succeeds(self):
        """Empty-overlap result joins without crashing (schema mismatch prevented)."""
        nonexistent_ids = ["nonexistent_id_xyz_123"]
        signals = load_location_signals(nonexistent_ids)

        # Simulate the join that happens in run_relevance.py and run_cca_doca.py
        test_table = pl.DataFrame({
            "id": ["a", "b", "c"],
            "value": [1, 2, 3],
        })

        # This should not crash with "cannot join on str vs null" error
        result = test_table.join(signals, on="id", how="left")

        # All rows should be present (left join), with nulls filled
        assert result.height == 3
        assert "any_us" in result.columns
        assert "any_not_us" in result.columns

        # After fill_null(False), location signals default to no-signal
        result = result.with_columns(
            pl.col("any_us").fill_null(False),
            pl.col("any_not_us").fill_null(False),
        )
        assert result["any_us"].to_list() == [False, False, False]
        assert result["any_not_us"].to_list() == [False, False, False]


class TestGoldFirstUSGate:
    """Test gold-first US gate: gold label overrides ML when non-null, fallback otherwise."""

    def test_gold_true_overrides_ml_false(self):
        """Gold True beats ML False → True."""
        gold = [True, False]
        ml = [False, False]
        final_gate, coverage = gold_first_us_gate(gold, ml)
        assert final_gate == [True, False]
        assert coverage == 1.0  # All gold non-null

    def test_gold_false_overrides_ml_true(self):
        """Gold False beats ML True → False."""
        gold = [False, True]
        ml = [True, True]
        final_gate, coverage = gold_first_us_gate(gold, ml)
        assert final_gate == [False, True]
        assert coverage == 1.0  # All gold non-null

    def test_gold_null_falls_back_to_ml(self):
        """Gold null → use ML pass/fail (both True and False cases)."""
        gold = [None, None, True, False]
        ml = [True, False, False, False]
        final_gate, coverage = gold_first_us_gate(gold, ml)
        # First two rows use ML since gold is null; last two use gold
        assert final_gate == [True, False, True, False]
        assert coverage == 0.5  # 2/4 gold non-null

    def test_mixed_gold_ml_coverage(self):
        """Coverage fraction computed correctly for mixed cases."""
        gold = [True, None, False, None, True]
        ml = [False, True, True, False, False]
        final_gate, coverage = gold_first_us_gate(gold, ml)
        assert final_gate == [True, True, False, False, True]
        assert coverage == 3.0 / 5.0  # 3/5 gold non-null

    def test_all_gold_null_coverage_zero(self):
        """All gold null → coverage is 0.0."""
        gold = [None, None, None]
        ml = [True, False, True]
        final_gate, coverage = gold_first_us_gate(gold, ml)
        assert final_gate == [True, False, True]
        assert coverage == 0.0

    def test_all_gold_present_coverage_one(self):
        """All gold non-null → coverage is 1.0."""
        gold = [True, False, True, False]
        ml = [False, True, False, True]  # ML ignored
        final_gate, coverage = gold_first_us_gate(gold, ml)
        assert final_gate == [True, False, True, False]
        assert coverage == 1.0

    def test_polars_series_input(self):
        """Accepts polars Series as input."""
        gold_series = pl.Series("gold", [True, None, False])
        ml_series = pl.Series("ml", [False, True, False])
        final_gate, coverage = gold_first_us_gate(gold_series, ml_series)
        assert final_gate == [True, True, False]
        assert coverage == 2.0 / 3.0

    def test_empty_input_coverage_zero(self):
        """Empty input → coverage is 0.0."""
        gold = []
        ml = []
        final_gate, coverage = gold_first_us_gate(gold, ml)
        assert final_gate == []
        assert coverage == 0.0  # 0 / 0 → 0.0
