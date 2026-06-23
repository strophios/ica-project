"""Tests for the US/not-US location heuristic (src/preproc/us_location.py)."""

import polars as pl
import pytest

from src.preproc.us_location import (
    apply_fused_us_gate,
    compute_location_signals,
    is_clearly_foreign,
    is_foreign_place,
    is_us_place,
    load_place_sets,
    location_signals,
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
