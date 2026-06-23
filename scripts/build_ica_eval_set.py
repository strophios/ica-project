# pattern: Imperative Shell
"""Assemble a clean joint-ICA evaluation set: anchor holdout + boundary draw + reused coded.

Orchestrates Tasks 1-3 from Phase 2 to emit:
1. An operator coding template (null ica_event across us∧cca region)
2. holdout_ids.parquet (union of reserved anchors + boundary draw + reused coded)

The script:
- Reserves ~30% of anchors as holdout (deduped by article_id)
- Reconciles coded-500 into immig_relevant/immig_advisory
- Re-confirms coded-500 us_event against the current fused US gate
- Draws the composed-score boundary sample (CCA × relevance stratification)
- Merges all sources and applies ICA label logic
- Writes outputs and prints the hand-coding worklist size

Run from project root:
    uv run python -m scripts.build_ica_eval_set
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import src.config as config
from src.validation.ica_eval import (
    derive_ica_negatives,
    reconcile_immig_column,
    reserve_anchor_holdout,
    assemble_holdout_ids,
)
from src.validation.build_ica_coding_template import build_ica_template
from src.validation.schema import validate_gold_set


def _load_anchors() -> pl.DataFrame:
    """Load the anchor ICA articles from relevance/ica_anchors.parquet."""
    anchors_path = Path(config.PROJECT_ROOT) / "relevance" / "ica_anchors.parquet"
    if not anchors_path.exists():
        raise FileNotFoundError(f"Anchors not found at {anchors_path}")
    return pl.read_parquet(anchors_path)


def _load_coded500() -> pl.DataFrame:
    """Load the manually-coded 500 articles from validation/cca_coding_first500_coded.csv."""
    coded_path = config.VALIDATION_DIR / "cca_coding_first500_coded.csv"
    if not coded_path.exists():
        raise FileNotFoundError(f"Coded-500 not found at {coded_path}")
    return pl.read_csv(coded_path)


def _load_scored_candidates() -> pl.DataFrame:
    """Load scored candidate pool for the boundary draw.

    Loads the scored candidates which have cached CCA logits, headline, lead_paragraph,
    and other metadata needed for the boundary draw. Lives in cca_doca/scored_candidates.parquet.
    """
    scored_path = config.CCA_DOCA_DIR / "scored_candidates.parquet"
    if not scored_path.exists():
        raise FileNotFoundError(f"Scored candidates not found at {scored_path}")
    return pl.read_parquet(scored_path)


def _apply_relevance_scores(scored: pl.DataFrame) -> pl.DataFrame:
    """Apply cached relevance head logits to the scored candidate pool.

    If the relevance weights are absent, emit a clear warning and continue
    without relevance scores (boundary draw will use a dummy relevance_logit=0).
    """
    relevance_weights_path = config.CCA_DOCA_DIR / "relevance.weights.h5"

    if not relevance_weights_path.exists():
        print(
            f"WARNING: Relevance weights not found at {relevance_weights_path}. "
            f"Proceeding without relevance scores (using dummy logit=0). "
            f"Phase 3 will train the relevance head.",
            flush=True,
        )
        # Add dummy relevance_logit column (all zeros)
        scored = scored.with_columns(pl.lit(0.0).alias("relevance_logit"))
        return scored

    # Import here to avoid loading keras/TF if weights don't exist
    from src.validation.relevance_slice_eval import apply_relevance_model
    from src.embed_corpus import load_cache

    # Load cached embeddings for the full pool
    embed_cache_dir = config.CCA_EMBED_CACHE_DIR / "full"
    cls = load_cache(str(embed_cache_dir))

    # Apply relevance model
    relevance_logits = apply_relevance_model(cls, weights_path=relevance_weights_path)

    # Map back to scored dataframe by row index
    scored = scored.with_columns(
        pl.Series("relevance_logit", relevance_logits, dtype=pl.Float32)
    )
    return scored


def main():
    """Assemble the clean ICA eval set."""
    print("Phase 2, Task 4: Assemble ICA eval set", flush=True)

    # =========================================================================
    # STEP 1: Reserve anchor holdout (30% of deduped article_ids)
    # =========================================================================
    print("\n[1/5] Loading and reserving anchor holdout...", flush=True)
    anchors = _load_anchors()
    anchor_holdout_ids, anchor_train_ids = reserve_anchor_holdout(
        anchors, frac=0.30, seed=200
    )
    print(
        f"  Anchors: {anchors.height} rows, {anchors['article_id'].n_unique()} unique articles",
        flush=True,
    )
    print(
        f"  Reserved {len(anchor_holdout_ids)} for holdout, "
        f"{len(anchor_train_ids)} for training",
        flush=True,
    )


    # =========================================================================
    # STEP 2: Reconcile coded-500 into immig_relevant/immig_advisory
    # =========================================================================
    print("\n[2/5] Reconciling coded-500...", flush=True)
    coded500 = _load_coded500()
    print(f"  Loaded {coded500.height} coded rows", flush=True)

    # Reconcile immig column (legacy 0/1) into immig_relevant (bool)
    coded500 = reconcile_immig_column(coded500)

    # =========================================================================
    # STEP 3: Re-confirm coded-500 us_event against current US gate
    # =========================================================================
    print("\n[3/5] Re-confirming us_event via current US gate...", flush=True)

    # Filter to rows that have both us_event and cca_event coded
    coded500_valid = coded500.filter(
        pl.col("us_event").is_not_null() & pl.col("cca_event").is_not_null()
    ).clone()

    # Re-compute us_event using the current US gate (combine dateline + location)
    # The coded-500 has headline+lead_paragraph, but we need to construct the
    # text column (headline + "</s>" + lead_paragraph) for the US model
    coded500_valid = coded500_valid.with_columns(
        (
            pl.col("headline").fill_null("")
            + "</s>"
            + pl.col("lead_paragraph").fill_null("")
        ).alias("_text_for_us")
    )

    # Apply the US filter to re-confirm
    try:
        from src.validation.slice_eval import apply_us_model

        texts = coded500_valid["_text_for_us"].to_list()
        us_scores = apply_us_model(texts)
        recomputed_us_logits = us_scores  # These are calibrated [0,1] scores

        # Convert calibrated scores to binary us_event at threshold 0.5
        recomputed_us_event = (recomputed_us_logits >= 0.5).to_list()
        coded500_valid = coded500_valid.with_columns(
            pl.Series("_recomputed_us_event", recomputed_us_event, dtype=pl.Boolean)
        )

        # Drop rows where imported us_event disagrees with recomputed
        original_count = coded500_valid.height
        # Map STRING us_event (FALSE/TRUE) to bool if needed
        coded500_valid = coded500_valid.with_columns(
            pl.col("us_event").cast(pl.Boolean).alias("_imported_us_event")
        )
        coded500_valid = coded500_valid.filter(
            pl.col("_imported_us_event") == pl.col("_recomputed_us_event")
        )
        dropped_count = original_count - coded500_valid.height
        print(f"  Dropped {dropped_count} rows where us_event disagreed", flush=True)

        # Keep the recomputed value
        coded500_valid = coded500_valid.with_columns(
            pl.col("_recomputed_us_event").alias("us_event")
        ).drop("_imported_us_event", "_recomputed_us_event", "_text_for_us")

    except FileNotFoundError as e:
        print(f"  WARNING: US filter weights missing ({e}). Keeping imported us_event.", flush=True)
        coded500_valid = coded500_valid.drop("_text_for_us")

    # =========================================================================
    # STEP 4: Draw the composed-score boundary sample
    # =========================================================================
    print("\n[4/5] Drawing composed-score boundary sample...", flush=True)

    # Load scored candidates and apply relevance scores
    scored = _load_scored_candidates()
    print(f"  Loaded {scored.height} candidate rows from CCA DoCA table", flush=True)

    scored = _apply_relevance_scores(scored)

    # Build the template (excludes anchors and coded-500)
    coded500_ids = coded500_valid["id"].to_list()
    template = build_ica_template(
        scored,
        anchor_ids=anchor_holdout_ids,
        coded500_ids=coded500_ids,
        seed=200,
    )
    print(f"  Boundary draw: {template.height} rows", flush=True)

    # =========================================================================
    # STEP 5: Write the boundary-sample template and holdout id list
    # =========================================================================
    print("\n[5/5] Writing outputs...", flush=True)

    # The template is the boundary sample (already schema-conformant with null labels)
    # Apply ICA label logic: set ica_event=False where us∧cca is False
    template = derive_ica_negatives(template)

    # Validate against the schema
    print("  Validating template against gold set schema...", flush=True)
    validate_gold_set(template)

    # Write outputs
    template_output_path = config.VALIDATION_DIR / "ica_coding_template.parquet"
    holdout_output_path = config.VALIDATION_DIR / "ica_holdout_ids.parquet"

    template.write_parquet(str(template_output_path))
    print(f"  Wrote {template.height} template rows to {template_output_path}", flush=True)

    # Write holdout_ids as a parquet with id column (union of anchors + coded500)
    # These are the ids to exclude from retraining (Phases 3+)
    holdout_ids = assemble_holdout_ids(anchor_holdout_ids, coded500_ids)
    holdout_df = pl.DataFrame({"id": holdout_ids})
    holdout_df.write_parquet(str(holdout_output_path))
    print(f"  Wrote {len(holdout_ids)} holdout ids to {holdout_output_path}", flush=True)

    # =========================================================================
    # Print hand-coding worklist count
    # =========================================================================
    # Count rows where us_event==True AND cca_event==True (need hand-coding of ica_event)
    worklist = template.filter(
        (pl.col("us_event")) & (pl.col("cca_event"))
    )
    worklist_count = worklist.height
    print(f"\n✓ Hand-coding worklist size: {worklist_count} rows", flush=True)
    print("  (Rows with us_event=True AND cca_event=True, ica_event=null)", flush=True)


if __name__ == "__main__":
    main()
