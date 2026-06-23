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

import numpy as np
import polars as pl

import src.config as config
from src.validation.ica_eval import (
    reconcile_immig_column,
    reserve_anchor_holdout,
    assemble_eval_frame,
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
    relevance_weights_path = config.RELEVANCE_DOCA_WEIGHTS

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
    df_cache, cls = load_cache(embed_cache_dir)

    # Apply relevance model (returns logits shape (n,))
    relevance_logits = apply_relevance_model(cls, weights_path=relevance_weights_path)

    # Join relevance logits to scored by id (not positional alignment)
    # df_cache carries id column, so we can use it to align with scored
    # Cast relevance_logits to float32 for schema consistency
    relevance_logits_f32 = np.asarray(relevance_logits, dtype=np.float32)
    logits_df = df_cache.select("id").with_columns(
        pl.Series("relevance_logit", relevance_logits_f32, dtype=pl.Float32)
    )
    scored = scored.join(logits_df, on="id", how="left")
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

    # Re-compute us_event using the current US gate (fused gate + Platt calibration)
    # The coded-500 has headline+lead_paragraph, but we need to construct the
    # text column (headline + "</s>" + lead_paragraph) for the US model
    coded500_valid = coded500_valid.with_columns(
        (
            pl.col("headline").fill_null("")
            + "</s>"
            + pl.col("lead_paragraph").fill_null("")
        ).alias("_text_for_us")
    )

    # Apply the US filter to re-confirm using the operative calibrated US head
    try:
        from src.validation.slice_eval import apply_us_model

        texts = coded500_valid["_text_for_us"].to_list()
        # Use skip_mismatch=True since US_FILTER_CLASSIFIER_FULL_WEIGHTS contains head-only weights
        # (the backbone is separately loaded and frozen, so we only need head weights)
        us_scores = apply_us_model(texts, weights_path=config.US_FILTER_CLASSIFIER_FULL_WEIGHTS, skip_mismatch=True)
        # apply_us_model returns calibrated [0,1] numpy array; threshold at 0.5
        recomputed_us_event = (us_scores >= 0.5).tolist()
        coded500_valid = coded500_valid.with_columns(
            pl.Series("_recomputed_us_event", recomputed_us_event, dtype=pl.Boolean)
        )

        # Drop rows where imported us_event disagrees with recomputed value
        original_count = coded500_valid.height
        # Ensure imported us_event is cast to boolean for comparison
        coded500_valid = coded500_valid.with_columns(
            pl.col("us_event").cast(pl.Boolean).alias("_imported_us_event")
        )
        coded500_valid = coded500_valid.filter(
            pl.col("_imported_us_event") == pl.col("_recomputed_us_event")
        )
        dropped_count = original_count - coded500_valid.height
        print(f"  Dropped {dropped_count} rows where us_event disagreed with recomputed value", flush=True)

        # Keep the recomputed value (always use the current US gate)
        coded500_valid = coded500_valid.with_columns(
            pl.col("_recomputed_us_event").alias("us_event")
        ).drop("_imported_us_event", "_recomputed_us_event", "_text_for_us")

    except FileNotFoundError as e:
        print(f"ERROR: US filter artifacts not found ({e}). Cannot re-confirm us_event. Aborting.", flush=True)
        raise

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
    # STEP 5: Merge anchors, coded-500 survivors, and boundary sample
    # =========================================================================
    print("\n[5/5] Merging sources...", flush=True)

    # Prepare anchor rows: join anchors to scored by id, dedupe, mark ica_event=True
    # Anchors are US ∩ CCA by construction from the ICA definition
    anchor_subset = anchors.filter(
        pl.col("article_id").is_in(anchor_holdout_ids)
    ).unique(subset=["article_id"]).with_columns(
        pl.col("article_id").alias("id")
    ).select(["id", "event_type4", "immigrant_involved"])

    # Join anchors to scored by id (article_ids are the same as scored ids)
    anchor_rows_scored = scored.join(
        anchor_subset,
        on="id",
        how="inner"
    ).with_columns(
        # Anchors are by definition US ∩ CCA ∩ ICA: set all three to True
        pl.lit(True).alias("us_event"),
        pl.lit(True).alias("cca_event"),
        pl.lit(True).alias("ica_event"),
        pl.lit("anchor").alias("sample_stratum"),
        pl.col("event_type4").alias("event_type"),
        pl.col("immigrant_involved").alias("immig_relevant"),
    ).drop("event_type4", "immigrant_involved")

    print(f"  Anchor rows (confirmed positives): {anchor_rows_scored.height}", flush=True)
    anchor_rows = anchor_rows_scored

    # Prepare coded-500 survivors: select schema-required columns, tag stratum
    # Survivors are us_event==True ∧ cca_event==True ∧ passed us re-confirmation
    coded_survivor_rows = coded500_valid.filter(
        (pl.col("us_event")) & (pl.col("cca_event"))
    ).with_columns(
        pl.lit("coded_reuse").alias("sample_stratum"),
    )
    print(f"  Coded-500 survivors (hand-coding worklist): {coded_survivor_rows.height}", flush=True)

    # Assemble the eval frame using the extracted Functional Core helper
    full_template = assemble_eval_frame(
        anchor_rows=anchor_rows,
        coded_survivor_rows=coded_survivor_rows,
        boundary_rows=template,
    )
    print(f"  Total merged rows: {full_template.height}", flush=True)

    # Validate against the schema
    print("  Validating merged template against gold set schema...", flush=True)
    validate_gold_set(full_template)

    # Write outputs
    template_output_path = config.VALIDATION_DIR / "ica_coding_template.parquet"
    holdout_output_path = config.VALIDATION_DIR / "ica_holdout_ids.parquet"

    full_template.write_parquet(str(template_output_path))
    print(f"  Wrote {full_template.height} template rows to {template_output_path}", flush=True)

    # Write holdout_ids as a parquet with id column (union of anchors + coded500 + boundary)
    # These are the ids to exclude from retraining (Phases 3+)
    # The full_template contains all three sources; extract all ids from it
    # This ensures the holdout includes anchors + coded survivors + boundary draw.
    all_reserved_ids = sorted(set(full_template["id"].to_list()))
    holdout_df = pl.DataFrame({"id": all_reserved_ids})
    holdout_df.write_parquet(str(holdout_output_path))
    print(f"  Wrote {len(all_reserved_ids)} holdout ids to {holdout_output_path}", flush=True)

    # =========================================================================
    # Print hand-coding worklist count
    # =========================================================================
    # Count rows that still NEED ica_event hand-coding: us_event==True AND
    # cca_event==True AND ica_event is null. Anchors are us∧cca but already
    # carry ica_event=True (confirmation-only), so they are excluded here.
    worklist = full_template.filter(
        (pl.col("us_event"))
        & (pl.col("cca_event"))
        & (pl.col("ica_event").is_null())
    )
    worklist_count = worklist.height
    print(f"\n✓ Hand-coding worklist size: {worklist_count} rows", flush=True)
    print("  (Rows with us_event=True AND cca_event=True, ica_event=null)", flush=True)


if __name__ == "__main__":
    main()
