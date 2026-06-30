# pattern: Functional Core (pure stratified sampling)
"""Tests for src.validation.build_ica_coding_template — ICA boundary sampler."""

from __future__ import annotations

import polars as pl

from src.validation.build_ica_coding_template import (
    build_ica_template,
)
from src.validation.schema import validate_gold_set


def _make_synthetic_scored() -> pl.DataFrame:
    """Create a synthetic scored dataframe for ICA template tests.

    Contains:
    - id, year, news_desk, section_name, headline, lead_paragraph
    - cca_logit, relevance_logit (spanning various score bands)
    - No labels (null for us_event, cca_event, etc.)
    """
    rows = []
    for i in range(200):
        # Spread CCA logits from -3 to +3
        cca_logit = -3.0 + 6.0 * (i / 200)
        # Spread relevance logits from -2 to +2
        relevance_logit = -2.0 + 4.0 * ((i + 50) % 200) / 200

        rows.append({
            "id": f"article_{i:03d}",
            "year": 1970 + (i % 20),
            "news_desk": ["National", "World", "Business"][i % 3],
            "section_name": "US",
            "headline": f"Headline {i}",
            "lead_paragraph": f"Lead {i}",
            "cca_logit": cca_logit,
            "relevance_logit": relevance_logit,
        })

    return pl.DataFrame(rows)


class TestBuildIcaTemplate:
    """Build ICA boundary sampler."""

    def test_build_ica_template_schema_conformant(self):
        """Output conforms to schema (labels null, proper stratum, etc.)."""
        scored = _make_synthetic_scored()
        anchor_ids = ["article_000", "article_001"]
        coded500_ids = ["article_002", "article_003"]

        template = build_ica_template(
            scored,
            anchor_ids=anchor_ids,
            coded500_ids=coded500_ids,
            alloc={"cca_high_relev_high": 20, "cca_high_relev_low": 20,
                   "cca_mid_relev_high": 15, "cca_mid_relev_low": 15,
                   "cca_low_relev_high": 15, "cca_low_relev_low": 15},
            seed=200
        )

        # Validate schema
        validate_gold_set(template)

        # All label columns should be null (for hand-coding)
        assert template["us_event"].null_count() == template.height
        assert template["cca_event"].null_count() == template.height
        assert template["immig_relevant"].null_count() == template.height
        assert template["ica_event"].null_count() == template.height

        # sample_stratum should be set
        assert template["sample_stratum"].null_count() == 0

    def test_build_ica_template_exclusion_correctness(self):
        """Anchor and coded-500 ids are excluded from the template."""
        scored = _make_synthetic_scored()
        anchor_ids = ["article_050", "article_051"]
        coded500_ids = ["article_100", "article_101"]

        template = build_ica_template(
            scored,
            anchor_ids=anchor_ids,
            coded500_ids=coded500_ids,
            alloc={"cca_high_relev_high": 50, "cca_high_relev_low": 50},
            seed=200
        )

        # Neither anchor nor coded-500 should appear
        ids_set = set(template["id"].to_list())
        for excluded_id in anchor_ids + coded500_ids:
            assert excluded_id not in ids_set

    def test_build_ica_template_includes_low_relevance_high_cca_stratum(self):
        """Verify that high-CCA/low-relevance stratum is present (contextual ICA signal)."""
        scored = _make_synthetic_scored()
        anchor_ids = []
        coded500_ids = []

        template = build_ica_template(
            scored,
            anchor_ids=anchor_ids,
            coded500_ids=coded500_ids,
            alloc={"cca_high_relev_high": 30, "cca_high_relev_low": 30,
                   "cca_mid_relev_high": 20, "cca_mid_relev_low": 20},
            seed=200
        )

        # At least one row from cca_high_relev_low stratum
        strata = set(template["sample_stratum"].unique().to_list())
        assert "cca_high_relev_low" in strata

    def test_build_ica_template_deterministic_seed(self):
        """Same seed produces same template."""
        scored = _make_synthetic_scored()

        template1 = build_ica_template(
            scored,
            anchor_ids=[],
            coded500_ids=[],
            alloc={"cca_high_relev_high": 30, "cca_high_relev_low": 30},
            seed=200
        )

        template2 = build_ica_template(
            scored,
            anchor_ids=[],
            coded500_ids=[],
            alloc={"cca_high_relev_high": 30, "cca_high_relev_low": 30},
            seed=200
        )

        assert template1.equals(template2)

    def test_build_ica_template_respects_allocation(self):
        """Allocation sizes are met (or capped by availability)."""
        scored = _make_synthetic_scored()
        alloc = {"cca_high_relev_high": 20, "cca_high_relev_low": 15}

        template = build_ica_template(
            scored,
            anchor_ids=[],
            coded500_ids=[],
            alloc=alloc,
            seed=200
        )

        # Total should not exceed sum of alloc (or be capped by availability)
        total = template.height
        max_alloc = sum(alloc.values())
        assert total <= max_alloc

        # Count per stratum
        per_stratum = template.group_by("sample_stratum").agg(pl.len())
        counts_dict = per_stratum.to_dict(as_series=True)

        # Convert to dict mapping stratum -> count
        stratum_to_count = {
            s: c
            for s, c in zip(counts_dict["sample_stratum"], counts_dict["len"])
        }

        # Each stratum count should be <= allocated
        for stratum, count in stratum_to_count.items():
            if stratum in alloc:
                assert count <= alloc[stratum]

    def test_build_ica_template_corpus_api(self):
        """Corpus column is set to 'api'."""
        scored = _make_synthetic_scored()

        template = build_ica_template(
            scored,
            anchor_ids=[],
            coded500_ids=[],
            alloc={"cca_high_relev_high": 20},
            seed=200
        )

        assert (template["corpus"] == "api").all()

    def test_build_ica_template_relevance_stratification_spans_both_high_and_low(self):
        """Relevance stratification produces both high and low bands (guards Critical 1).

        This test guards against the case where relevance scores collapse to a single
        band (e.g., all zeros), which would break the composed-score stratification.
        With real relevance logits spanning the range, we should see both relev_high
        and relev_low strata present in the output.
        """
        scored = _make_synthetic_scored()
        # _make_synthetic_scored spreads relevance logits from -2 to +2,
        # so it should have both high (>=0.5) and low (<0.5) bands.

        template = build_ica_template(
            scored,
            anchor_ids=[],
            coded500_ids=[],
            alloc={
                "cca_high_relev_high": 20,
                "cca_high_relev_low": 20,
                "cca_mid_relev_high": 15,
                "cca_mid_relev_low": 15,
                "cca_low_relev_high": 15,
                "cca_low_relev_low": 15,
            },
            seed=200
        )

        strata = set(template["sample_stratum"].unique().to_list())
        # Both high and low relevance bands must be present
        high_relev_strata = {s for s in strata if "relev_high" in s}
        low_relev_strata = {s for s in strata if "relev_low" in s}
        assert len(high_relev_strata) > 0, f"No high-relevance strata found. Strata: {strata}"
        assert len(low_relev_strata) > 0, f"No low-relevance strata found. Strata: {strata}"
