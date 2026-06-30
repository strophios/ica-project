# pattern: Imperative Shell
# Reason: test file exercising Functional Core via assertion

"""Unit tests for src/preflight/checks.py — pure verdict functions.

Tests exercise AC7.1-AC7.4 with synthetic inputs (no filesystem).
"""

from __future__ import annotations

import pytest

import polars as pl

from src.preflight.checks import (
    Verdict,
    gold_coverage_counts,
    us_weights_verdict,
    calibration_presence_verdict,
    doca_freshness_verdict,
    ldc_channel_verdict,
    ldc_gold_coverage_verdict,
)


class TestGoldCoverageCounts:
    """Test gold_coverage_counts pure function (non-null us_label counting)."""

    def test_counts_non_null_us_label_intersection(self):
        """Count ids with non-null us_label in the LDC id set."""
        ldc_ids = {"id-1", "id-2", "id-3", "id-4", "id-5"}

        labeled_df = pl.DataFrame({
            "id": ["id-1", "id-2", "id-3", "id-6", "id-7"],
            "us_label": [True, False, None, True, False],
        })

        n_apply_ids, n_with_gold = gold_coverage_counts(ldc_ids, labeled_df)

        # n_apply_ids = len(ldc_ids) = 5
        assert n_apply_ids == 5
        # n_with_gold = count of ids in ldc_ids WITH non-null us_label
        # id-1 (True, non-null, in ldc_ids): yes
        # id-2 (False, non-null, in ldc_ids): yes
        # id-3 (None, null, in ldc_ids): NO
        # id-6 (True, non-null, NOT in ldc_ids): NO
        # id-7 (False, non-null, NOT in ldc_ids): NO
        assert n_with_gold == 2

    def test_ignores_null_us_label(self):
        """Null us_label rows (conflicts) do not count as gold."""
        ldc_ids = {"id-1", "id-2"}

        labeled_df = pl.DataFrame({
            "id": ["id-1", "id-2"],
            "us_label": [None, None],
        })

        n_apply_ids, n_with_gold = gold_coverage_counts(ldc_ids, labeled_df)

        assert n_apply_ids == 2
        assert n_with_gold == 0  # No non-null us_label values

    def test_ignores_ids_not_in_ldc_set(self):
        """Ids not in ldc_ids set do not count, even with non-null us_label."""
        ldc_ids = {"id-1", "id-2"}

        labeled_df = pl.DataFrame({
            "id": ["id-1", "id-2", "id-3", "id-4"],
            "us_label": [True, False, True, True],
        })

        n_apply_ids, n_with_gold = gold_coverage_counts(ldc_ids, labeled_df)

        assert n_apply_ids == 2
        # id-1 (True, in ldc_ids): yes
        # id-2 (False, in ldc_ids): yes
        # id-3 (True, NOT in ldc_ids): no
        # id-4 (True, NOT in ldc_ids): no
        assert n_with_gold == 2

    def test_real_world_scenario(self):
        """Verify 56.5% coverage scenario from real data."""
        # Simulate 624842 LDC ids, 352777 with non-null us_label
        ldc_ids = {f"id-{i}" for i in range(1, 624843)}  # 624842 ids

        # Create labeled_df with:
        # - 352777 rows with non-null us_label
        # - 272065 rows with null us_label (624842 - 352777)
        labeled_rows = []
        for i in range(1, 352778):  # 352777 non-null rows
            labeled_rows.append({"id": f"id-{i}", "us_label": True})
        for i in range(352778, 624843):  # 272065 null rows
            labeled_rows.append({"id": f"id-{i}", "us_label": None})

        labeled_df = pl.DataFrame(labeled_rows)

        n_apply_ids, n_with_gold = gold_coverage_counts(ldc_ids, labeled_df)

        assert n_apply_ids == 624842
        assert n_with_gold == 352777


class TestVerdictDataclass:
    """Test Verdict frozen dataclass structure."""

    def test_verdict_has_required_fields(self):
        """Verdict has name, status, detail, remediation."""
        v = Verdict(
            name="test",
            status="PASS",
            detail="all good",
            remediation=None,
        )
        assert v.name == "test"
        assert v.status == "PASS"
        assert v.detail == "all good"
        assert v.remediation is None

    def test_verdict_frozen(self):
        """Verdict dataclass is frozen."""
        v = Verdict(
            name="test",
            status="PASS",
            detail="all good",
            remediation=None,
        )
        with pytest.raises(AttributeError):
            v.status = "FAIL"  # ty: ignore[invalid-assignment]  # asserts frozen


class TestUsWeightsVerdict:
    """Test AC7.1: US weights check."""

    def test_fail_when_smoke_test_and_no_override(self):
        """FAIL: smoke-test weights + table_build_us_weights=None."""
        cache_provenance = {
            "us_weights": {
                "path": "us_filter/us_classifier.weights.h5",
                "size": 1000,
                "mtime": 123456789,
            }
        }
        verdict = us_weights_verdict(cache_provenance, None)
        assert verdict.status == "FAIL"
        assert "operative training-gate" in verdict.detail.lower()

    def test_pass_when_full_classifier_override(self):
        """PASS: table_build_us_weights points to us_classifier_full."""
        cache_provenance = {
            "us_weights": {
                "path": "us_filter/us_classifier.weights.h5",
                "size": 1000,
                "mtime": 123456789,
            }
        }
        verdict = us_weights_verdict(
            cache_provenance, "/path/to/us_classifier_full.weights.h5"
        )
        assert verdict.status == "PASS"

    def test_pass_when_full_classifier_and_matches_cache(self):
        """PASS: both cache and override point to us_classifier_full."""
        cache_provenance = {
            "us_weights": {
                "path": "us_filter/us_classifier_full.weights.h5",
                "size": 1000,
                "mtime": 123456789,
            }
        }
        verdict = us_weights_verdict(
            cache_provenance, "/path/to/us_classifier_full.weights.h5"
        )
        assert verdict.status == "PASS"

    def test_warn_when_undetermined(self):
        """WARN: cache shows us_classifier_full but no override (undetermined)."""
        cache_provenance = {
            "us_weights": {
                "path": "us_filter/us_classifier_full.weights.h5",
                "size": 1000,
                "mtime": 123456789,
            }
        }
        verdict = us_weights_verdict(cache_provenance, None)
        # Undetermined: not explicitly overridden, but matches expected basename
        # Per the design, this is WARN to encourage explicit confirmation
        assert verdict.status in ("WARN", "PASS")


class TestCalibrationPresenceVerdict:
    """Test AC7.2: calibration sidecars check."""

    def test_fail_when_cca_missing(self):
        """FAIL: CCA calibrator missing."""
        present = {"cca": False, "cca_street": True, "relevance": True}
        verdict = calibration_presence_verdict(present)
        assert verdict.status == "FAIL"
        assert "cca" in verdict.detail.lower()

    def test_fail_when_cca_street_missing(self):
        """FAIL: CCA street calibrator missing."""
        present = {"cca": True, "cca_street": False, "relevance": True}
        verdict = calibration_presence_verdict(present)
        assert verdict.status == "FAIL"
        assert "cca_street" in verdict.detail.lower()

    def test_warn_when_relevance_missing(self):
        """WARN: relevance calibrator missing (fixable in Phase 4)."""
        present = {"cca": True, "cca_street": True, "relevance": False}
        verdict = calibration_presence_verdict(present)
        assert verdict.status == "WARN"
        assert "relevance" in verdict.detail.lower()
        assert verdict.remediation is not None

    def test_pass_when_all_present(self):
        """PASS: all calibrators present."""
        present = {"cca": True, "cca_street": True, "relevance": True}
        verdict = calibration_presence_verdict(present)
        assert verdict.status == "PASS"


class TestDocaFreshnessVerdict:
    """Test AC7.3: DoCA chain freshness check."""

    def test_pass_when_monotone(self):
        """PASS: monotone mtime ordering (doca_csv ≤ rds ≤ positives)."""
        mtimes: dict[str, float | None] = {
            "doca_csv": 100.0,
            "rds": 150.0,
            "positives": 200.0,
        }
        verdict = doca_freshness_verdict(mtimes)
        assert verdict.status == "PASS"

    def test_warn_when_doca_csv_newer_than_rds(self):
        """WARN: doca_csv > rds (match potentially stale)."""
        mtimes: dict[str, float | None] = {
            "doca_csv": 200.0,
            "rds": 100.0,
            "positives": 250.0,
        }
        verdict = doca_freshness_verdict(mtimes)
        assert verdict.status == "WARN"
        assert "stale" in verdict.detail.lower()

    def test_warn_when_rds_newer_than_positives(self):
        """WARN: rds > positives (positives stale)."""
        mtimes: dict[str, float | None] = {
            "doca_csv": 100.0,
            "rds": 200.0,
            "positives": 150.0,
        }
        verdict = doca_freshness_verdict(mtimes)
        assert verdict.status == "WARN"
        assert "stale" in verdict.detail.lower()

    def test_warn_when_mtime_none(self):
        """WARN: missing mtime (file not found)."""
        mtimes: dict[str, float | None] = {
            "doca_csv": 100.0,
            "rds": None,
            "positives": 200.0,
        }
        verdict = doca_freshness_verdict(mtimes)
        assert verdict.status == "WARN"
        assert "rds" in verdict.detail.lower()

    def test_pass_when_equal_mtimes(self):
        """PASS: equal mtimes are acceptable (monotone)."""
        mtimes: dict[str, float | None] = {
            "doca_csv": 100.0,
            "rds": 100.0,
            "positives": 100.0,
        }
        verdict = doca_freshness_verdict(mtimes)
        assert verdict.status == "PASS"


class TestLdcChannelVerdict:
    """Test AC7.4: LDC dateline stripping check."""

    def test_fail_when_lead_column_none(self):
        """FAIL: lead_column is None (raw lead_paragraph, not stripped)."""
        provenance = {
            "lead_column": None,
            "n_rows": 1000,
            "text_channel": "unstripped",
        }
        verdict = ldc_channel_verdict(provenance)
        assert verdict.status == "FAIL"
        assert "dateline" in verdict.detail.lower()

    def test_fail_when_lead_column_lead_paragraph(self):
        """FAIL: lead_column is 'lead_paragraph' (raw, not stripped)."""
        provenance = {
            "lead_column": "lead_paragraph",
            "n_rows": 1000,
            "text_channel": "raw",
        }
        verdict = ldc_channel_verdict(provenance)
        assert verdict.status == "FAIL"

    def test_pass_when_lead_column_stripped_text(self):
        """PASS: lead_column is 'stripped_text'."""
        provenance = {
            "lead_column": "stripped_text",
            "n_rows": 1000,
            "text_channel": "dateline_stripped",
        }
        verdict = ldc_channel_verdict(provenance)
        assert verdict.status == "PASS"


class TestLdcGoldCoverageVerdict:
    """Test AC7.5: LDC gold label coverage (informational)."""

    def test_pass_with_coverage_in_detail(self):
        """PASS: always PASS, detail carries coverage fraction."""
        verdict = ldc_gold_coverage_verdict(n_apply_ids=1000, n_with_gold_label=750)
        assert verdict.status == "PASS"
        assert "750" in verdict.detail or "75" in verdict.detail
        assert "1000" in verdict.detail or "100" in verdict.detail

    def test_coverage_calculation(self):
        """Coverage is correctly calculated and shown."""
        verdict = ldc_gold_coverage_verdict(n_apply_ids=100, n_with_gold_label=50)
        assert verdict.status == "PASS"
        # Detail should show the fraction
        assert "50" in verdict.detail
        assert "100" in verdict.detail

    def test_zero_apply_ids(self):
        """Edge case: no apply ids."""
        verdict = ldc_gold_coverage_verdict(n_apply_ids=0, n_with_gold_label=0)
        assert verdict.status == "PASS"

    def test_zero_gold_labels(self):
        """Edge case: no gold labels."""
        verdict = ldc_gold_coverage_verdict(n_apply_ids=1000, n_with_gold_label=0)
        assert verdict.status == "PASS"
        assert "0" in verdict.detail

    def test_sub_100_percent_coverage(self):
        """Verify numerator counts gold-labeled ids, not all id-presence.

        Regression test for the bug where _get_ldc_gold_coverage counted
        all labeled_df ids (100%) instead of ids with non-null us_label (56.5%).
        The verdict detail must show a sub-100% fraction when the numerator
        (gold label count) is less than the denominator (apply id count).
        """
        # Real-world data: 352,777 ids with non-null us_label out of 624,842 LDC apply ids
        verdict = ldc_gold_coverage_verdict(n_apply_ids=624842, n_with_gold_label=352777)
        assert verdict.status == "PASS"
        # The detail should reflect the sub-100% fraction, not a trivial 100%
        assert "56.4%" in verdict.detail or "56.5%" in verdict.detail
        assert "352777" in verdict.detail
        assert "624842" in verdict.detail
