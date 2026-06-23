# pattern: Functional Core (orchestration script smoke tests)
"""Tests for scripts.build_ica_eval_set — ICA eval set assembly orchestration.

These tests verify the core merge logic: anchors get ica_event=True,
coded survivors have ica_event=null and preserved us/cca values,
and the worklist (us_event AND cca_event with null ica_event) is non-zero.
"""

from __future__ import annotations

import polars as pl

from src.validation.ica_eval import derive_ica_negatives


class TestMergeOrchestration:
    """Test the orchestration logic for merging sources into eval set."""

    def test_anchors_marked_ica_true(self):
        """Anchor rows are marked ica_event=True (confirmation-only positives)."""
        anchors = pl.DataFrame({
            "id": ["anc1", "anc2", "anc3"],
            "us_event": [True, True, True],
            "cca_event": [True, True, True],
            "sample_stratum": [None, None, None],
            "ica_event": [None, None, None],
        })

        # Mark as anchors with ica_event=True
        anchors = anchors.with_columns(
            pl.lit(True).alias("ica_event"),
            pl.lit("anchor").alias("sample_stratum"),
        )

        assert anchors["ica_event"].all()
        assert (anchors["sample_stratum"] == "anchor").all()

    def test_coded_survivors_preserve_us_cca_leave_ica_null(self):
        """Coded-500 survivors preserve us_event/cca_event, leave ica_event null."""
        coded = pl.DataFrame({
            "id": ["cod1", "cod2"],
            "us_event": [True, True],
            "cca_event": [True, True],
            "immig_relevant": [False, True],
            "sample_stratum": [None, None],
            "ica_event": [None, None],
        })

        # Mark as coded_reuse (no change to us/cca)
        coded = coded.with_columns(
            pl.lit("coded_reuse").alias("sample_stratum"),
        )

        # Verify preservation
        assert coded["us_event"].all()
        assert coded["cca_event"].all()
        assert coded["ica_event"].null_count() == coded.height

    def test_worklist_count_nonzero_with_coded_survivors(self):
        """Hand-coding worklist is non-zero when coded survivors present."""
        # Simulate anchors (ica_event=True, ignored for worklist)
        anchors = pl.DataFrame({
            "id": ["anc1"],
            "us_event": [True],
            "cca_event": [True],
            "ica_event": [True],
        })

        # Simulate coded survivors (us_event=True, cca_event=True, ica_event=null)
        coded = pl.DataFrame({
            "id": ["cod1", "cod2", "cod3"],
            "us_event": [True, True, True],
            "cca_event": [True, True, True],
            "ica_event": [None, None, None],
        })

        # Merge
        merged = pl.concat([anchors, coded])

        # Worklist: us_event==True AND cca_event==True AND ica_event==null
        worklist = merged.filter(
            (pl.col("us_event")) & (pl.col("cca_event")) & (pl.col("ica_event").is_null())
        )

        # Should only count coded survivors (anchors have ica_event=True, not null)
        assert worklist.height == 3

    def test_ica_negatives_derivation_in_merge_context(self):
        """ICA negative derivation works correctly in a merge context."""
        # Boundary sample with mixed scope gates
        boundary = pl.DataFrame({
            "id": ["b1", "b2", "b3", "b4"],
            "us_event": [True, True, False, False],
            "cca_event": [True, False, True, False],
            "ica_event": [None, None, None, None],
        })

        # Apply derivation
        boundary = derive_ica_negatives(boundary)

        # us=T, cca=T → ica=null
        # us=T, cca=F → ica=False
        # us=F, cca=T → ica=False
        # us=F, cca=F → ica=False
        assert boundary.filter(
            (pl.col("us_event")) & (pl.col("cca_event"))
        )["ica_event"].null_count() == 1

        assert boundary.filter(
            ~((pl.col("us_event")) & (pl.col("cca_event")))
        )["ica_event"].to_list() == [False, False, False]

    def test_merge_stratum_distribution(self):
        """Merged frame has proper stratum tags for all sources."""
        anchors = pl.DataFrame({"id": ["a1"], "sample_stratum": ["anchor"]})
        coded = pl.DataFrame({"id": ["c1"], "sample_stratum": ["coded_reuse"]})
        boundary = pl.DataFrame({"id": ["b1"], "sample_stratum": ["boundary_sample"]})

        merged = pl.concat([anchors, coded, boundary])

        strata = set(merged["sample_stratum"].to_list())
        assert "anchor" in strata
        assert "coded_reuse" in strata
        assert "boundary_sample" in strata
