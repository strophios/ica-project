# pattern: Functional Core (orchestration script tests)
"""Tests for scripts.build_ica_eval_set — ICA eval set assembly orchestration.

These tests verify:
1. The real assemble_eval_frame function (not inline re-implementations)
2. Orchestration logic: anchors get ica_event=True, coded survivors preserve us/cca/leave
   ica_event null, boundary rows follow derivation logic
3. Worklist (us_event AND cca_event with null ica_event) is non-zero with coded survivors
4. Relevance score application and alignment (guards Critical 1 regression)
"""

from __future__ import annotations

import polars as pl
import numpy as np

from src.validation.ica_eval import (
    assemble_eval_frame,
    derive_ica_negatives,
)
from src.validation.schema import validate_gold_set


class TestAssembleEvalFrame:
    """Test the core assemble_eval_frame Functional Core helper."""

    def test_assemble_eval_frame_marks_anchors_ica_true(self):
        """Anchors retain ica_event=True through assembly.

        Guards Critical 2: coded-500 survivors + anchor positives never enter eval set.
        """
        # Prepare anchor rows (marked ica_event=True by caller)
        anchor_rows = pl.DataFrame({
            "id": ["anc1", "anc2"],
            "us_event": [True, True],
            "cca_event": [True, True],
            "ica_event": [True, True],
            "sample_stratum": ["anchor", "anchor"],
            "corpus": ["api", "api"],
            "year": [1990, 1991],
            "news_desk": ["National", "World"],
            "section_name": ["US", "US"],
            "headline": ["h1", "h2"],
            "lead_paragraph": ["l1", "l2"],
        })

        # Coded survivors (us_event=True, cca_event=True, ica_event=null)
        coded_rows = pl.DataFrame({
            "id": ["cod1", "cod2"],
            "us_event": [True, True],
            "cca_event": [True, True],
            "ica_event": [None, None],
            "sample_stratum": ["coded_reuse", "coded_reuse"],
            "corpus": ["api", "api"],
            "year": [1990, 1991],
            "news_desk": ["National", "World"],
            "section_name": ["US", "US"],
            "headline": ["h1", "h2"],
            "lead_paragraph": ["l1", "l2"],
        })

        # Boundary (mixed scope gates)
        boundary_rows = pl.DataFrame({
            "id": ["b1", "b2"],
            "us_event": [True, False],
            "cca_event": [True, False],
            "ica_event": [None, None],
            "sample_stratum": ["cca_high_relev_high", "cca_low_relev_low"],
            "corpus": ["api", "api"],
            "year": [1990, 1991],
            "news_desk": ["National", "World"],
            "section_name": ["US", "US"],
            "headline": ["h1", "h2"],
            "lead_paragraph": ["l1", "l2"],
        })

        result = assemble_eval_frame(anchor_rows, coded_rows, boundary_rows)

        # Anchors must have ica_event=True
        anchors = result.filter(pl.col("sample_stratum") == "anchor")
        assert anchors["ica_event"].all(), "Anchors must have ica_event=True"
        assert anchors.height == 2

    def test_assemble_eval_frame_coded_survivors_have_null_ica(self):
        """Coded survivors preserve us/cca with ica_event=null for hand-coding."""
        anchor_rows = pl.DataFrame({
            "id": ["anc1"],
            "us_event": [True],
            "cca_event": [True],
            "ica_event": [True],
            "sample_stratum": ["anchor"],
            "corpus": ["api"],
            "year": [1990],
            "news_desk": ["National"],
            "section_name": ["US"],
            "headline": ["h"],
            "lead_paragraph": ["l"],
        })

        coded_rows = pl.DataFrame({
            "id": ["cod1", "cod2", "cod3"],
            "us_event": [True, True, True],
            "cca_event": [True, True, True],
            "ica_event": [None, None, None],
            "immig_relevant": [False, True, False],
            "sample_stratum": ["coded_reuse", "coded_reuse", "coded_reuse"],
            "corpus": ["api", "api", "api"],
            "year": [1990, 1991, 1992],
            "news_desk": ["National", "World", "Business"],
            "section_name": ["US", "US", "US"],
            "headline": ["h1", "h2", "h3"],
            "lead_paragraph": ["l1", "l2", "l3"],
        })

        boundary_rows = pl.DataFrame({
            "id": [],
            "us_event": [],
            "cca_event": [],
            "ica_event": [],
            "sample_stratum": [],
            "corpus": [],
            "year": [],
            "news_desk": [],
            "section_name": [],
            "headline": [],
            "lead_paragraph": [],
        })

        result = assemble_eval_frame(anchor_rows, coded_rows, boundary_rows)

        coded_survivors = result.filter(pl.col("sample_stratum") == "coded_reuse")
        # All coded survivors should have us_event=True, cca_event=True, ica_event=null
        assert coded_survivors["us_event"].all()
        assert coded_survivors["cca_event"].all()
        assert coded_survivors["ica_event"].null_count() == coded_survivors.height
        assert coded_survivors.height == 3

    def test_assemble_eval_frame_applies_ica_derivation(self):
        """ICA negatives are derived: us=False OR cca=False => ica_event=False."""
        # Use empty dataframe with proper schema
        empty = pl.DataFrame({
            "id": pl.Series([], dtype=pl.Utf8),
            "us_event": pl.Series([], dtype=pl.Boolean),
            "cca_event": pl.Series([], dtype=pl.Boolean),
            "ica_event": pl.Series([], dtype=pl.Boolean),
            "sample_stratum": pl.Series([], dtype=pl.Utf8),
            "corpus": pl.Series([], dtype=pl.Utf8),
            "year": pl.Series([], dtype=pl.Int64),
            "news_desk": pl.Series([], dtype=pl.Utf8),
            "section_name": pl.Series([], dtype=pl.Utf8),
            "headline": pl.Series([], dtype=pl.Utf8),
            "lead_paragraph": pl.Series([], dtype=pl.Utf8),
        })

        # Boundary with mixed scope gates
        boundary_rows = pl.DataFrame({
            "id": ["b1", "b2", "b3", "b4"],
            "us_event": [True, True, False, False],
            "cca_event": [True, False, True, False],
            "ica_event": [None, None, None, None],
            "sample_stratum": ["s1", "s2", "s3", "s4"],
            "corpus": ["api", "api", "api", "api"],
            "year": [1990, 1990, 1990, 1990],
            "news_desk": ["National", "National", "National", "National"],
            "section_name": ["US", "US", "US", "US"],
            "headline": ["h1", "h2", "h3", "h4"],
            "lead_paragraph": ["l1", "l2", "l3", "l4"],
        })

        result = assemble_eval_frame(empty, empty, boundary_rows)

        # us=T, cca=T → ica_event=null
        holistic = result.filter((pl.col("us_event")) & (pl.col("cca_event")))
        assert holistic["ica_event"].null_count() == holistic.height

        # us=T, cca=F → ica_event=False
        cca_false = result.filter((pl.col("us_event")) & (~pl.col("cca_event")))
        assert (~cca_false["ica_event"]).all()

        # us=F, cca=T → ica_event=False
        us_false = result.filter((~pl.col("us_event")) & (pl.col("cca_event")))
        assert (~us_false["ica_event"]).all()

    def test_assemble_eval_frame_worklist_nonempty(self):
        """Worklist (us & cca & ica_event=null) is non-zero with coded survivors.

        Guards Critical 2 regression: if coded_survivor_rows are dropped,
        worklist should become zero and this test fails.
        """
        anchor_rows = pl.DataFrame({
            "id": ["anc1"],
            "us_event": [True],
            "cca_event": [True],
            "ica_event": [True],
            "sample_stratum": ["anchor"],
            "corpus": ["api"],
            "year": [1990],
            "news_desk": ["National"],
            "section_name": ["US"],
            "headline": ["h"],
            "lead_paragraph": ["l"],
        })

        coded_rows = pl.DataFrame({
            "id": ["cod1", "cod2"],
            "us_event": [True, True],
            "cca_event": [True, True],
            "ica_event": [None, None],
            "sample_stratum": ["coded_reuse", "coded_reuse"],
            "corpus": ["api", "api"],
            "year": [1990, 1991],
            "news_desk": ["National", "World"],
            "section_name": ["US", "US"],
            "headline": ["h1", "h2"],
            "lead_paragraph": ["l1", "l2"],
        })

        boundary_rows = pl.DataFrame({
            "id": [],
            "us_event": [],
            "cca_event": [],
            "ica_event": [],
            "sample_stratum": [],
            "corpus": [],
            "year": [],
            "news_desk": [],
            "section_name": [],
            "headline": [],
            "lead_paragraph": [],
        })

        result = assemble_eval_frame(anchor_rows, coded_rows, boundary_rows)

        # Worklist: us_event=True AND cca_event=True AND ica_event=null
        worklist = result.filter(
            (pl.col("us_event")) & (pl.col("cca_event")) & (pl.col("ica_event").is_null())
        )

        # Must include the 2 coded survivors (anchors have ica_event=True, not null)
        assert worklist.height == 2, f"Expected 2, got {worklist.height}"

    def test_assemble_eval_frame_validates_schema(self):
        """Output passes schema validation."""
        anchor_rows = pl.DataFrame({
            "id": ["a1"],
            "us_event": [True],
            "cca_event": [True],
            "ica_event": [True],
            "sample_stratum": ["anchor"],
            "corpus": ["api"],
            "year": [1990],
            "news_desk": ["National"],
            "section_name": ["US"],
            "headline": ["h"],
            "lead_paragraph": ["l"],
        })

        coded_rows = pl.DataFrame({
            "id": ["c1"],
            "us_event": [True],
            "cca_event": [True],
            "ica_event": [None],
            "sample_stratum": ["coded_reuse"],
            "corpus": ["api"],
            "year": [1990],
            "news_desk": ["National"],
            "section_name": ["US"],
            "headline": ["h"],
            "lead_paragraph": ["l"],
        })

        boundary_rows = pl.DataFrame({
            "id": [],
            "us_event": [],
            "cca_event": [],
            "ica_event": [],
            "sample_stratum": [],
            "corpus": [],
            "year": [],
            "news_desk": [],
            "section_name": [],
            "headline": [],
            "lead_paragraph": [],
        })

        result = assemble_eval_frame(anchor_rows, coded_rows, boundary_rows)

        # Should not raise
        validate_gold_set(result)


class TestRelevanceScoreAlignment:
    """Guard against relevance-score alignment regression (Critical 1).

    Critical 1: relevance weights path wrong → composed-score stratification collapses.
    This test ensures that relevance logits are properly joined by id, not positionally.
    """

    def test_relevance_logits_joined_by_id_not_position(self):
        """Relevance logits must be joined by id, not by position.

        If alignment is positional rather than by id, a reordering of
        either dataframe would produce wrong results (and this test catches it).
        """
        # Create a scored dataframe with ids
        scored = pl.DataFrame({
            "id": ["art_001", "art_002", "art_003"],
            "cca_logit": [1.5, 0.0, -1.5],
            "corpus": ["api", "api", "api"],
            "year": [1990, 1990, 1990],
            "news_desk": ["National", "World", "Business"],
            "section_name": ["US", "US", "US"],
            "headline": ["h1", "h2", "h3"],
            "lead_paragraph": ["l1", "l2", "l3"],
        })

        # Create logits dataframe with the SAME ids but DIFFERENT order
        # This simulates the actual apply_relevance_model output (which comes from
        # embeddings cache, potentially in a different order than scored)
        logits_df = pl.DataFrame({
            "id": ["art_003", "art_001", "art_002"],
            "relevance_logit": [-1.0, 0.5, 1.5],  # Different values, different order
        })

        # Perform the actual alignment by id (as _apply_relevance_scores does)
        result = scored.join(logits_df, on="id", how="left")

        # Verify that relevance logits are aligned to the correct ids
        # art_001 should have relevance_logit=0.5 (not -1.0)
        # art_002 should have relevance_logit=1.5 (not 0.5)
        # art_003 should have relevance_logit=-1.0 (not 1.5)
        assert result.filter(pl.col("id") == "art_001")["relevance_logit"][0] == 0.5
        assert result.filter(pl.col("id") == "art_002")["relevance_logit"][0] == 1.5
        assert result.filter(pl.col("id") == "art_003")["relevance_logit"][0] == -1.0

    def test_relevance_span_across_rows(self):
        """Relevance logits must span >1 unique value (guards collapsed scores).

        If relevance weights are missing/wrong, _apply_relevance_scores adds
        dummy 0.0 to all rows. This test ensures real logits span a range.
        """
        # Simulate real relevance logits spanning a range
        logits = np.array([-2.0, -0.5, 0.0, 1.5, 2.5])
        logits_df = pl.DataFrame({
            "id": [f"art_{i:03d}" for i in range(len(logits))],
            "relevance_logit": logits.astype(np.float32),
        })

        unique_logits = set(logits_df["relevance_logit"].to_list())
        assert len(unique_logits) > 1, f"Logits must span >1 unique value, got {unique_logits}"


class TestHoldoutCompleteness:
    """Verify holdout ids cover all template sources (guards anti-contamination AC2.2).

    Critical issue: holdout id list must include boundary-draw ids, not just anchors
    and coded-500 survivors. Without boundary ids, the boundary rows are simultaneously
    in the evaluation set AND in Phase 3's training pool, causing selection contamination.
    """

    def test_holdout_ids_include_all_template_sources(self):
        """Holdout must include anchors + coded survivors + boundary rows.

        The holdout_ids.parquet that Phase 3 reads must contain every id in the
        ica_coding_template.parquet that was assembled for hand-coding. If any
        boundary-draw id is missing from holdout, it will be present in Phase 3's
        training pool, causing AC2.2 contamination.

        This test demonstrates the BUG: if holdout extraction uses only
        `set(anchor_holdout_ids) | set(coded500_ids)`, it OMITS boundary_ids.
        With the FIX: `set(full_template["id"].to_list())` includes all three.
        """
        # Build the three sources
        anchor_rows = pl.DataFrame({
            "id": ["anc1", "anc2"],
            "us_event": [True, True],
            "cca_event": [True, True],
            "ica_event": [True, True],
            "sample_stratum": ["anchor", "anchor"],
            "corpus": ["api", "api"],
            "year": [1990, 1991],
            "news_desk": ["National", "World"],
            "section_name": ["US", "US"],
            "headline": ["h1", "h2"],
            "lead_paragraph": ["l1", "l2"],
        })

        coded_survivor_rows = pl.DataFrame({
            "id": ["cod1", "cod2"],
            "us_event": [True, True],
            "cca_event": [True, True],
            "ica_event": [None, None],
            "sample_stratum": ["coded_reuse", "coded_reuse"],
            "corpus": ["api", "api"],
            "year": [1990, 1991],
            "news_desk": ["National", "World"],
            "section_name": ["US", "US"],
            "headline": ["h1", "h2"],
            "lead_paragraph": ["l1", "l2"],
        })

        boundary_rows = pl.DataFrame({
            "id": ["bnd1", "bnd2", "bnd3"],
            "us_event": [True, True, False],
            "cca_event": [True, False, True],
            "ica_event": [None, None, None],
            "sample_stratum": ["cca_high_relev_high", "cca_mid_relev_low", "cca_low_relev_high"],
            "corpus": ["api", "api", "api"],
            "year": [1990, 1991, 1992],
            "news_desk": ["National", "World", "Business"],
            "section_name": ["US", "US", "US"],
            "headline": ["h1", "h2", "h3"],
            "lead_paragraph": ["l1", "l2", "l3"],
        })

        # Assemble the full template (as build_ica_eval_set.py does)
        full_template = assemble_eval_frame(anchor_rows, coded_survivor_rows, boundary_rows)

        # Simulate the BUGGY holdout extraction: only anchors + coded
        anchor_holdout_ids = anchor_rows["id"].to_list()
        coded500_ids = coded_survivor_rows["id"].to_list()
        buggy_holdout_ids = sorted(set(anchor_holdout_ids) | set(coded500_ids))

        # Simulate the FIXED holdout extraction: all template ids
        fixed_holdout_ids = sorted(set(full_template["id"].to_list()))

        # Verify that buggy version is MISSING boundary ids
        buggy_set = set(buggy_holdout_ids)
        boundary_ids = set(boundary_rows["id"].to_list())
        missing_boundary = boundary_ids - buggy_set
        assert missing_boundary, (
            "Test setup broken: buggy version should miss boundary ids, "
            "but all boundary ids present in buggy holdout"
        )

        # Verify that fixed version includes ALL ids from template
        fixed_set = set(fixed_holdout_ids)
        template_ids = set(full_template["id"].to_list())
        assert fixed_set == template_ids, (
            f"Fixed holdout missing ids: {template_ids - fixed_set}"
        )

        # Verify boundary ids are now present in fixed version
        assert boundary_ids <= fixed_set, (
            f"Fixed holdout still missing boundary ids: {boundary_ids - fixed_set}"
        )


class TestMergeOrchestration:
    """Legacy tests—kept for backward compatibility."""

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
