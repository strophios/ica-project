# pattern: Imperative Shell (orchestration) + embedded Functional Core (select_combiner, resolve_fusion_inputs)
"""Empirical fusion selection + composed calibration (Phase 4, Task 5).

Orchestrates the fusion combiner choice:
1. Load clean eval set and feature cache
2. Apply US-scope rule (non-US events cannot be ICA by construction)
3. Score calibrated US, CCA, rel heads over eval ids
4. Sanity-check US head on us_event label + combiner heads on conditional-on-US population
5. Pick US gate threshold via recall recipe (anchor DoCA positives)
6. Run StratifiedKFold CV: AND vs LR by PR-AUC on CONDITIONAL-ON-US population
   (isolates combiner fit from US gate's diaspora recall miss)
7. Apply 1-SE decision rule to select combiner
8. Compute composed calibration on conditional-on-US population
9. Save fusion.json + metrics JSON with gold-first gate note

Functional Core: select_combiner (pure 1-SE margin rule decision logic);
resolve_fusion_inputs (pure parameter resolution + overwrite guard).
Imperative Shell: main() orchestration with checkpoint validation.

Run from project root:
    uv run python -m src.fit_fusion
    uv run python -m src.fit_fusion --cache-suffix relevance_train_tuned \
        --cca-weights ../cca_doca/cca_doca_tuned.weights.h5 \
        --rel-weights ../relevance/relevance_tuned.weights.h5 \
        --us-weights ../us_filter/us_classifier_full_tuned.weights.h5 \
        --out ../cca_doca/tuned_fusion
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Literal, cast

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

import src.config as config
from src.artifact_guard import check_no_production_overwrite
from src.calibration.calibrator import platt_fit, platt_transform
from src.calibration.report import calibration_report
from src.calibration.sidecar import (
    calibration_path_for_weights,
    load_calibration,
)
from src.embed_corpus import load_cache
from src.fusion.combiner import (
    combine_and,
    fit_logistic_combiner,
    apply_logistic_combiner,
    FusionConfig,
)
from src.fusion.sidecar import fusion_path_for_weights, save_fusion
from src.validation.cca_slice_eval import apply_cca_model
from src.validation.doca_recall import doca_recall, pick_us_threshold
from src.validation.ica_eval import apply_us_scope_to_ica
from src.validation.relevance_slice_eval import apply_relevance_model
from src.validation.slice_eval import apply_us_model


# main()'s own --cache-suffix default; also the "production cache" identity
# that resolve_fusion_inputs compares an explicit --cache-suffix against.
DEFAULT_CACHE_SUFFIX = "relevance_train"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Functional Core: pure 1-SE decision rule
# ============================================================================


def select_combiner(
    cv_and: dict[str, list[float]],
    cv_lr: dict[str, list[float]],
) -> str:
    """Select combiner via pre-registered 1-SE margin rule (AC4.3).

    Args:
        cv_and: CV results dict with "pr_auc": [fold0, fold1, ...]
        cv_lr: CV results dict with same structure

    Returns:
        "logreg" if LR mean improvement > 1 SE of paired CV diff
        "product" otherwise

    Raises:
        ValueError: If fold counts mismatch, empty, or scores non-numeric
        KeyError: If required "pr_auc" key missing
    """
    # Extract PR-AUC scores
    try:
        and_scores = cv_and["pr_auc"]
        lr_scores = cv_lr["pr_auc"]
    except KeyError as e:
        raise KeyError(f"missing required key: {e}") from e

    # Convert to numpy and validate
    try:
        and_scores = np.asarray(and_scores, dtype=np.float64)
        lr_scores = np.asarray(lr_scores, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"non-numeric fold scores: {e}") from e

    # Check fold counts match and non-empty
    if len(and_scores) == 0:
        raise ValueError("fold lists cannot be empty")
    if len(and_scores) != len(lr_scores):
        raise ValueError(
            f"fold count mismatch: AND {len(and_scores)} vs LR {len(lr_scores)}"
        )

    # Compute paired differences (LR - AND)
    paired_diff = lr_scores - and_scores
    mean_diff = float(np.mean(paired_diff))
    std_diff = float(np.std(paired_diff, ddof=1)) if len(paired_diff) > 1 else 0.0
    se_diff = std_diff / np.sqrt(len(paired_diff))

    # Apply 1-SE margin rule: LR selected iff mean_diff > 1 * SE
    # Ties and marginal improvements (≤1 SE) favor the simpler AND
    if mean_diff > se_diff:
        return "logreg"
    else:
        return "product"


def resolve_fusion_inputs(
    *,
    cache_suffix: str = DEFAULT_CACHE_SUFFIX,
    cca_weights_path: Path | str | None = None,
    rel_weights_path: Path | str | None = None,
    us_weights_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    rel_feature_variant: str | None = None,
) -> dict:
    """Resolve fit_fusion's parameterized inputs and guard against a silent
    production overwrite.

    Each weights path independently defaults to its production artifact
    (`config.CCA_DOCA_WEIGHTS` / `RELEVANCE_DOCA_WEIGHTS` / `US_FILTER_FULL_WEIGHTS`);
    `output_dir` defaults to `config.CCA_DOCA_DIR` (the production
    `ica_fusion.fusion.json` location). fit_fusion.py has five independently
    tunable input sources (one cache + three head-weights paths + the rel
    feature-variant tag) feeding a single shared output artifact, so this
    applies the same overwrite check the six trainers/calibrators use
    (`src.artifact_guard.check_no_production_overwrite`) once per input:
    pointing any ONE of them away from its default while leaving `output_dir`
    at the production default is refused, before any cache/weights I/O happens.

    Args:
        rel_feature_variant: optional rel-branch CLS variant tag (branched-
            encoder fusion refit, `docs/design-plans/2026-08-18-stage4-joint-finetune.md`).
            `None` (default, matching production) = the rel head reads the
            SAME base CLS array as CCA/US, unchanged prior behavior.

    Returns:
        Dict of resolved Paths/str: cache_suffix, cca_weights_path,
        rel_weights_path, us_weights_path, output_dir, rel_feature_variant.

    Raises:
        ValueError: a non-default cache_suffix/cca_weights_path/rel_weights_path/
            us_weights_path/rel_feature_variant is paired with a production
            (default) output_dir.
    """
    cca_weights_path = (
        Path(cca_weights_path) if cca_weights_path is not None else config.CCA_DOCA_WEIGHTS
    )
    rel_weights_path = (
        Path(rel_weights_path) if rel_weights_path is not None else config.RELEVANCE_DOCA_WEIGHTS
    )
    us_weights_path = (
        Path(us_weights_path) if us_weights_path is not None else config.US_FILTER_FULL_WEIGHTS
    )
    output_dir = Path(output_dir) if output_dir is not None else config.CCA_DOCA_DIR

    # check_no_production_overwrite's (cache_suffix, weights_path) shape is
    # reused per input: the "weights_path" compared is always this run's
    # resolved fusion output artifact, and "cache_suffix" is instantiated
    # with each of the five tunable inputs in turn (the real cache suffix,
    # each head-weights path as a string, and the rel-feature-variant tag)
    # against its own production default.
    resolved_output_weights = output_dir / "ica_fusion.weights.h5"
    production_output_weights = config.CCA_DOCA_DIR / "ica_fusion.weights.h5"
    for artifact_label, input_value, production_value in [
        ("fusion cache", cache_suffix, DEFAULT_CACHE_SUFFIX),
        ("fusion CCA weights", str(cca_weights_path), str(config.CCA_DOCA_WEIGHTS)),
        ("fusion relevance weights", str(rel_weights_path), str(config.RELEVANCE_DOCA_WEIGHTS)),
        ("fusion US weights", str(us_weights_path), str(config.US_FILTER_FULL_WEIGHTS)),
        ("fusion rel-feature-variant", rel_feature_variant, None),
    ]:
        check_no_production_overwrite(
            cache_suffix=input_value,
            production_cache_suffix=production_value,
            weights_path=resolved_output_weights,
            production_weights_path=production_output_weights,
            artifact_label=artifact_label,
        )

    return {
        "cache_suffix": cache_suffix,
        "cca_weights_path": cca_weights_path,
        "rel_weights_path": rel_weights_path,
        "us_weights_path": us_weights_path,
        "output_dir": output_dir,
        "rel_feature_variant": rel_feature_variant,
    }


def resolve_head_feature_sources(rel_feature_variant: str | None) -> dict[str, str] | None:
    """The fusion sidecar's `head_feature_sources` record for a given
    --rel-feature-variant tag (branched-encoder apply,
    `docs/design-plans/2026-08-18-stage4-joint-finetune.md`).

    `None` (no variant requested — production/default) -> `None`: no record,
    back-compat with fusion sidecars predating branched-encoder support.
    A tag -> `{"us": "base", "cca": "base", "rel": <tag>}`, matching the
    deployed branched config (US/CCA on the base cache, rel on its branch).
    """
    if rel_feature_variant is None:
        return None
    return {"us": "base", "cca": "base", "rel": rel_feature_variant}


# ============================================================================
# Imperative Shell: orchestration
# ============================================================================


def main(
    eval_csv_path: Path | str = "../../validation/ica_coding_template_coded.csv",
    target_us_recall: float = 0.95,
    us_thresholds: list[float] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    output_dir: Path | str | None = None,
    cache_suffix: str = DEFAULT_CACHE_SUFFIX,
    cca_weights_path: Path | str | None = None,
    rel_weights_path: Path | str | None = None,
    us_weights_path: Path | str | None = None,
    rel_feature_variant: str | None = None,
) -> dict:
    """Orchestrate fusion selection + composed calibration (conditional-on-US population).

    The US head systematically misses diaspora ICA events (US-soil protest about
    foreign topics). To isolate the combiner fit from this recall ceiling, we fit
    on rows where us_event==True (the population the gate targets), not ML-gated
    survivors. This prevents the combiner from being biased by gated-out positives.
    See us-head-retrain-plan.md for the diaspora finding.

    Args:
        eval_csv_path: Hand-coded eval set CSV (relative to project root)
        target_us_recall: Target DoCA recall for US gate (e.g., 0.95)
        us_thresholds: Thresholds to evaluate for recall recipe (e.g., linspace(0.02, 0.7, ...))
        n_splits: Folds for StratifiedKFold CV (default 5)
        random_state: Seed for determinism
        output_dir: Directory for fusion.json + metrics.json (default uses cca_doca dir).
            Passing a non-default cache_suffix/cca_weights_path/rel_weights_path/
            us_weights_path/rel_feature_variant while leaving output_dir at its
            production default raises (see resolve_fusion_inputs) -- pass a
            distinct output_dir for a tuned run.
        cache_suffix: embed cache subdir CCA/rel features + eval-set ids are joined
            against (default: "relevance_train", the production cache)
        cca_weights_path: CCA head weights to score (default: config.CCA_DOCA_WEIGHTS)
        rel_weights_path: relevance head weights to score (default: config.RELEVANCE_DOCA_WEIGHTS)
        us_weights_path: US head weights to score (default: config.US_FILTER_FULL_WEIGHTS)
        rel_feature_variant: optional rel-branch CLS variant tag (branched-encoder
            fusion refit). When set, the rel head's features are read from the
            variant array off the SAME cache join (`cache_suffix`) instead of the
            base CLS array CCA/US use -- no second cache dir. The saved fusion
            sidecar records `head_feature_sources={"us":"base","cca":"base","rel":<tag>}`.
            Default `None` = legacy behavior (rel shares the base array).

    Returns:
        Dict with keys: combiner, tau_us, tau_us_qualified, tau_us_achieved_recall,
                        cv_and_pr_auc_mean/se, cv_lr_pr_auc_mean/se,
                        composed_ece, composed_brier, n_us_true, n_pos_us_true,
                        n_neg_us_true, gate_note
    """
    # Resolve paths
    eval_csv_path = Path(eval_csv_path)
    if not eval_csv_path.is_absolute():
        # PROJECT_ROOT is 00_explorer; validation/ is a sibling to ica_project/ within 00_explorer
        candidates = [
            config.PROJECT_ROOT / eval_csv_path,
            (config.PROJECT_ROOT / "validation" / "ica_coding_template_coded.csv").resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                eval_csv_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"eval CSV not found. Tried: {candidates}"
            )
    elif not eval_csv_path.exists():
        raise FileNotFoundError(f"eval CSV not found: {eval_csv_path}")

    resolved = resolve_fusion_inputs(
        cache_suffix=cache_suffix,
        cca_weights_path=cca_weights_path,
        rel_weights_path=rel_weights_path,
        us_weights_path=us_weights_path,
        output_dir=output_dir,
        rel_feature_variant=rel_feature_variant,
    )
    rel_feature_variant = resolved["rel_feature_variant"]
    cache_suffix = resolved["cache_suffix"]
    cca_weights_path = resolved["cca_weights_path"]
    rel_weights_path = resolved["rel_weights_path"]
    us_weights_path = resolved["us_weights_path"]
    output_dir = resolved["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if us_thresholds is None:
        # Extended down to 0.02 to allow recall recipe best-effort point
        us_thresholds = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    logger.info(f"Loading eval CSV: {eval_csv_path}")
    eval_df = pl.read_csv(eval_csv_path)
    logger.info(f"Loaded {eval_df.shape[0]} rows")

    # Drop null ica_event rows (per task spec)
    eval_df = eval_df.filter(pl.col("ica_event").is_not_null())
    logger.info(f"After dropping null ica_event: {eval_df.shape[0]} rows")

    # Apply US-scope rule: non-US rows cannot be ICA by construction.
    # Preserves ica_event_intl (original operator judgment) and adjusts ica_event.
    eval_df = apply_us_scope_to_ica(eval_df)
    logger.info("Applied US-scope ICA label fix (us_event=False → ica_event=False)")

    # Count label distribution (post-scope)
    n_true = eval_df["ica_event"].sum()
    n_false = (~eval_df["ica_event"]).sum()
    logger.info(f"Label balance (post-scope): {n_true} True, {n_false} False")

    # ========================================================================
    # Step 1: Load features (embed cache) and build feature matrix
    # ========================================================================
    logger.info(f"Loading embed cache meta and CLS features (suffix={cache_suffix})")
    meta_full, cls = load_cache(config.CCA_EMBED_CACHE_DIR / cache_suffix)

    # Map eval ids to emb_row via meta
    meta_by_id = meta_full.with_columns(
        pl.col("id").cast(pl.Utf8).alias("id_str")
    ).select(["id_str", "emb_row"])

    eval_df_join = eval_df.with_columns(
        pl.col("id").cast(pl.Utf8).alias("id_str")
    )
    eval_df_join = eval_df_join.join(meta_by_id, on="id_str", how="left")

    missing_emb = eval_df_join.filter(pl.col("emb_row").is_null()).shape[0]
    if missing_emb > 0:
        logger.warning(
            f"⚠️  {missing_emb} eval rows missing embeddings (not in cache)"
        )
    eval_df_join = eval_df_join.filter(pl.col("emb_row").is_not_null())
    logger.info(
        f"After cache join: {eval_df_join.shape[0]} rows with embeddings"
    )

    # Extract feature matrix in eval row order
    emb_rows = eval_df_join["emb_row"].to_numpy().astype(int)
    features = cls[emb_rows]  # shape (n_eval, hidden_dim)
    logger.info(f"Feature matrix: {features.shape}")

    # Rel-feature-variant (branched-encoder fusion refit): one extra gather
    # off the SAME cache join (emb_rows), no second cache dir.
    rel_features = features
    if rel_feature_variant is not None:
        logger.info(
            f"Loading rel-feature-variant CLS array (tag={rel_feature_variant!r})"
        )
        meta_variant, cls_variant = load_cache(
            config.CCA_EMBED_CACHE_DIR / cache_suffix, variant=rel_feature_variant
        )
        if meta_variant.height != meta_full.height:
            raise ValueError(
                f"rel-feature-variant cache (tag={rel_feature_variant!r}) has "
                f"{meta_variant.height} rows, expected {meta_full.height} (the "
                f"base cache's row count) -- variant and base shards must be "
                f"row-aligned"
            )
        rel_features = cls_variant[emb_rows]
        logger.info(f"Rel feature matrix (variant): {rel_features.shape}")

    # Build texts for US model (headline + "</s>" + lead_paragraph)
    missing_cols = []
    if "headline" not in eval_df_join.columns:
        missing_cols.append("headline")
    if "lead_paragraph" not in eval_df_join.columns:
        missing_cols.append("lead_paragraph")

    if missing_cols:
        raise KeyError(
            f"Cannot score US head: required text columns missing: {missing_cols}. "
            f"Available columns: {list(eval_df_join.columns)}"
        )

    headlines = eval_df_join["headline"].to_list()
    leads = eval_df_join["lead_paragraph"].to_list()
    texts_list = [
        f"{h if h else ''}</s>{lead if lead else ''}"
        for h, lead in zip(headlines, leads)
    ]

    # ========================================================================
    # Step 2: Score calibrated heads
    # ========================================================================
    logger.info(f"Scoring CCA head ({cca_weights_path})")
    cca_logits = apply_cca_model(features, weights_path=cca_weights_path)
    cca_cal = load_calibration(
        calibration_path_for_weights(cca_weights_path)
    )
    p_cca = cca_cal.transform(cca_logits)
    assert np.isfinite(p_cca).all(), "Non-finite CCA probabilities"
    logger.info(f"CCA: mean={p_cca.mean():.3f}, std={p_cca.std():.3f}")

    logger.info(f"Scoring relevance head ({rel_weights_path})")
    rel_logits = apply_relevance_model(rel_features, weights_path=rel_weights_path)
    rel_cal = load_calibration(
        calibration_path_for_weights(rel_weights_path)
    )
    p_rel = rel_cal.transform(rel_logits)
    assert np.isfinite(p_rel).all(), "Non-finite relevance probabilities"
    logger.info(f"Relevance: mean={p_rel.mean():.3f}, std={p_rel.std():.3f}")

    logger.info(f"Scoring US head (calibrated) ({us_weights_path})")
    p_us = apply_us_model(
        texts_list, weights_path=us_weights_path, skip_mismatch=True
    )
    assert np.isfinite(p_us).all(), "Non-finite US probabilities"
    logger.info(f"US: mean={p_us.mean():.3f}, std={p_us.std():.3f}")

    # ========================================================================
    # Step 3: Sanity checks — validate head behavior
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SANITY CHECKS: Head Separation")
    logger.info("=" * 70)

    # US head check: mean p_us for us_event==True vs us_event==False
    # EXPECT clear separation (~0.84 vs ~0.21). If not, US scoring is broken.
    us_event = eval_df_join["us_event"].to_numpy().astype(bool)
    p_us_us_true = p_us[us_event].mean()
    p_us_us_false = p_us[~us_event].mean()
    logger.info(
        f"US head check: mean p_us for us_event==True={p_us_us_true:.3f}, "
        f"us_event==False={p_us_us_false:.3f}"
    )
    if p_us_us_true <= p_us_us_false:
        raise RuntimeError(
            f"US head not separating by us_event! True={p_us_us_true:.3f} "
            f"<= False={p_us_us_false:.3f}. US scoring is broken."
        )
    logger.info("✓ US head separates correctly")

    # Combiner-head check: among us_event==True rows, mean p_cca/p_rel for ica_event True vs False
    # EXPECT True higher for both. (Do NOT expect p_us higher for ica=True — ICA is not
    # monotone in US-ness; many non-immigrant US-CCA events score high on US, so median
    # p_us across all ICA events is similar to non-ICA events. This is the diaspora ceiling.)
    ica_event = eval_df_join["ica_event"].to_numpy().astype(bool)
    us_true_mask = us_event
    if us_true_mask.sum() > 0:
        cca_ica_true = p_cca[us_true_mask & ica_event].mean() if (us_true_mask & ica_event).sum() > 0 else None
        cca_ica_false = p_cca[us_true_mask & ~ica_event].mean() if (us_true_mask & ~ica_event).sum() > 0 else None
        rel_ica_true = p_rel[us_true_mask & ica_event].mean() if (us_true_mask & ica_event).sum() > 0 else None
        rel_ica_false = p_rel[us_true_mask & ~ica_event].mean() if (us_true_mask & ~ica_event).sum() > 0 else None

        logger.info(
            f"Combiner heads (among us_event==True):\n"
            f"  CCA: ica_event==True={cca_ica_true:.3f}, ==False={cca_ica_false:.3f}\n"
            f"  Rel: ica_event==True={rel_ica_true:.3f}, ==False={rel_ica_false:.3f}"
        )
        logger.info(
            "(Note: p_us is NOT expected higher for ica=True. ICA includes diaspora "
            "protests about foreign topics, which score low on US-ness. The US head's "
            "diaspora recall miss is the documented ceiling; gold-first gating rescues it.)"
        )
    logger.info("=" * 70 + "\n")

    # ========================================================================
    # Step 4: Pick US threshold via recall recipe
    # ========================================================================
    logger.info(
        f"Picking US threshold via recall recipe (target recall={target_us_recall})"
    )

    # Get anchors (sample_stratum == "anchor") to evaluate recall
    eval_df_join = eval_df_join.with_columns(
        pl.col("emb_row").cast(pl.Int64).alias("emb_row_int")
    )
    anchors = eval_df_join.filter(pl.col("sample_stratum") == "anchor")
    logger.info(f"Anchors (all ICA-positive by construction): {anchors.shape[0]}")

    # Build scored frame for recall recipe: id, us_score, and optionally doca_id
    # Use id as doca_id for anchors (they are DoCA matches by construction)
    anchor_indices = [i for i, s in enumerate(eval_df_join["sample_stratum"].to_list()) if s == "anchor"]
    recall_df = pl.DataFrame({
        "doca_id": [eval_df_join["id"][i] for i in anchor_indices],
        "us_score": [p_us[i] for i in anchor_indices],
    })
    logger.info(f"Recall frame: {recall_df.shape[0]} anchors with us_score")

    threshold_result = pick_us_threshold(
        recall_df,
        target_recall=target_us_recall,
        thresholds=us_thresholds,
    )
    tau_us = threshold_result.threshold
    tau_us_qualified = threshold_result.qualified
    # ThresholdPickResult carries only threshold + qualified; recompute the
    # achieved recall at the picked threshold for logging/metrics.
    tau_us_achieved_recall = float(doca_recall(recall_df, tau_us)["recall"])
    logger.info(
        f"Selected τ_us={tau_us:.3f}, qualified={tau_us_qualified}, "
        f"achieved_recall={tau_us_achieved_recall:.3f} "
        f"(target was {target_us_recall:.3f})"
    )
    if not tau_us_qualified:
        logger.warning(
            "⚠️  US gate cannot reach target recall (diaspora ceiling). "
            "At apply, gold-first gating bypasses the ML gate for DoCA/dateline-labeled rows."
        )

    # ========================================================================
    # Step 5: Gate survivors and prepare combiner fit population
    # ========================================================================
    # CRITICAL: Fit on CONDITIONAL-ON-US population (us_event==True rows),
    # NOT on ML-gated survivors. This isolates the combiner from the US gate's
    # diaspora recall miss.
    us_true_survivors = eval_df_join.filter(pl.col("us_event"))
    n_us_true = us_true_survivors.shape[0]
    logger.info(
        f"\nCombiner fit population (us_event==True): {n_us_true} rows "
        f"(isolating from US gate's diaspora recall miss)"
    )

    # Extract indices for this population
    us_true_indices = []
    for i in range(len(eval_df_join)):
        if us_event[i]:
            us_true_indices.append(i)
    us_true_indices = np.array(us_true_indices)

    # Subsample everything to US-true rows
    p_cca_us_true = p_cca[us_true_indices]
    p_rel_us_true = p_rel[us_true_indices]
    ica_event_us_true = ica_event[us_true_indices]

    # Guard: ensure no null ica_event in combiner fit population (would silently coerce to False)
    null_ica_rows = us_true_survivors.filter(pl.col("ica_event").is_null())
    assert null_ica_rows.is_empty(), (
        f"Combiner fit population contains {null_ica_rows.shape[0]} rows with null ica_event; "
        "these would silently coerce to False and poison the fit. "
        "All us_event==True rows must have coded (non-null) ica_event labels."
    )

    n_pos_us_true = int(ica_event_us_true.sum())
    n_neg_us_true = int((~ica_event_us_true).sum())
    logger.info(
        f"Label balance (us_event==True): {n_pos_us_true} True, {n_neg_us_true} False"
    )

    # ========================================================================
    # Step 6: StratifiedKFold CV — AND vs LR (on conditional-on-US population)
    # ========================================================================
    logger.info(
        f"\nRunning StratifiedKFold CV on conditional-on-US population "
        f"(n_splits={n_splits})"
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_and_pr_aucs = []
    cv_lr_pr_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(p_cca_us_true, ica_event_us_true)
    ):
        # Split data
        p_cca_train, p_cca_val = p_cca_us_true[train_idx], p_cca_us_true[val_idx]
        p_rel_train, p_rel_val = p_rel_us_true[train_idx], p_rel_us_true[val_idx]
        y_train = ica_event_us_true[train_idx].astype(int)
        y_val = ica_event_us_true[val_idx].astype(int)

        # Compute AND on validation fold
        and_scores_val = combine_and(p_cca_val, p_rel_val)
        and_pr_auc = average_precision_score(y_val, and_scores_val)
        cv_and_pr_aucs.append(and_pr_auc)

        # Fit LR on train fold and apply to val fold
        # Features: z_cca, z_rel (probabilities)
        lr_features_train = np.column_stack([p_cca_train, p_rel_train])
        lr_features_val = np.column_stack([p_cca_val, p_rel_val])

        lr = fit_logistic_combiner(lr_features_train, y_train, random_state=random_state)
        lr_scores_val = apply_logistic_combiner(lr, lr_features_val)
        lr_pr_auc = average_precision_score(y_val, lr_scores_val)
        cv_lr_pr_aucs.append(lr_pr_auc)

        logger.info(
            f"Fold {fold_idx}: AND PR-AUC={and_pr_auc:.4f}, LR PR-AUC={lr_pr_auc:.4f}"
        )

    # Summarize CV results
    and_mean = float(np.mean(cv_and_pr_aucs))
    and_se = float(np.std(cv_and_pr_aucs, ddof=1) / np.sqrt(len(cv_and_pr_aucs)))
    lr_mean = float(np.mean(cv_lr_pr_aucs))
    lr_se = float(np.std(cv_lr_pr_aucs, ddof=1) / np.sqrt(len(cv_lr_pr_aucs)))
    logger.info(f"AND: PR-AUC = {and_mean:.4f} ± {and_se:.4f}")
    logger.info(f"LR:  PR-AUC = {lr_mean:.4f} ± {lr_se:.4f}")

    # ========================================================================
    # Step 7: Apply 1-SE decision rule
    # ========================================================================
    cv_and_result = {"pr_auc": cv_and_pr_aucs}
    cv_lr_result = {"pr_auc": cv_lr_pr_aucs}
    chosen_combiner = select_combiner(cv_and_result, cv_lr_result)
    logger.info(f"Decision rule selected: {chosen_combiner}")

    # ========================================================================
    # Step 8: Fit final LR on full us-true set (for composed score)
    # ========================================================================
    lr_features_full = np.column_stack([p_cca_us_true, p_rel_us_true])
    lr_full = fit_logistic_combiner(
        lr_features_full, ica_event_us_true.astype(int), random_state=random_state
    )
    # CRITICAL: persist both slopes (coef_[0]) and intercept (intercept_[0])
    # to match apply_logistic_combiner's expected format: (coef, intercept)
    lr_slopes = lr_full.coef_[0].tolist()
    lr_intercept = float(lr_full.intercept_[0])
    lr_coefs = lr_slopes + [lr_intercept]  # [slope_cca, slope_rel, intercept]
    logger.info(f"Full LR coefficients (slopes + intercept): {lr_coefs}")

    # ========================================================================
    # Step 9: Compute composed score and calibration
    # ========================================================================
    if chosen_combiner == "product":
        composed_score = combine_and(p_cca_us_true, p_rel_us_true)
        lr_coefs_to_save = None
    else:  # logreg
        composed_score = apply_logistic_combiner(lr_full, lr_features_full)
        lr_coefs_to_save = lr_coefs

    # Calibration report (AC3.3: composed calibration, following label-budget rule)
    composed_cal = calibration_report(composed_score, ica_event_us_true, n_bins=15)
    logger.info(
        f"Composed calibration (ECE/Brier): {composed_cal['ece']:.4f} / {composed_cal['brier']:.4f}"
    )

    # ========================================================================
    # Step 9.5: Optionally fit composed-score Platt (AC3.3 label-budget rule)
    # ========================================================================
    # Per label-budget: fit only if (1) composed score is mis-calibrated (ECE > threshold)
    # AND (2) positive count supports 2 params (EPV > 10). Here: EPV = n_pos / 2.
    # n_pos_us_true positives available; if EPV >= 10, fit.
    composed_platt_ab = None
    composed_platt_fitted = False
    epv_threshold = 10
    epv_composed = n_pos_us_true / 2  # 2 params for Platt
    ece_threshold = 0.12  # mis-calibrated signal (tunable)

    if n_pos_us_true > 0 and epv_composed >= epv_threshold:
        if composed_cal["ece"] > ece_threshold:
            # Fit Platt on the composed score (as logit, not probability)
            # First convert probability to logit for Platt fitting
            composed_score_clipped = np.clip(composed_score, 1e-10, 1.0 - 1e-10)
            composed_logits = np.log(composed_score_clipped / (1.0 - composed_score_clipped))
            A, B = platt_fit(composed_logits, ica_event_us_true)
            composed_platt_ab = [float(A), float(B)]
            composed_platt_fitted = True

            # Re-calibrate with Platt and report post-Platt calibration
            composed_score_calibrated = platt_transform(composed_logits, A, B)
            composed_cal_post = calibration_report(
                composed_score_calibrated, ica_event_us_true, n_bins=15
            )
            logger.info(
                f"Composed Platt fit (EPV={epv_composed:.1f} >= {epv_threshold}): "
                f"ECE {composed_cal['ece']:.4f} → {composed_cal_post['ece']:.4f}, "
                f"Brier {composed_cal['brier']:.4f} → {composed_cal_post['brier']:.4f}"
            )
            composed_cal = composed_cal_post  # Use post-Platt metrics
        else:
            logger.info(
                f"Composed ECE {composed_cal['ece']:.4f} below threshold ({ece_threshold}); "
                f"Platt not needed"
            )
    else:
        decision_note = (
            f"(EPV={epv_composed:.1f} < {epv_threshold} or n_pos={n_pos_us_true})"
        )
        logger.info(
            f"Composed Platt not fit: insufficient label budget {decision_note}"
        )

    # ========================================================================
    # Step 10: Gather calibrator references for head_calibrators
    # ========================================================================
    # Record references to the per-head calibrators this fusion composition uses.
    # These are typically the sidecar stems (without .calibration.json).
    cca_cal_stem = str(calibration_path_for_weights(cca_weights_path)).replace(
        ".calibration.json", ""
    )
    rel_cal_stem = str(calibration_path_for_weights(rel_weights_path)).replace(
        ".calibration.json", ""
    )
    us_cal_stem = str(calibration_path_for_weights(us_weights_path)).replace(
        ".calibration.json", ""
    )

    head_calibrators = {
        "cca": cca_cal_stem,
        "rel": rel_cal_stem,
        "us": us_cal_stem,
    }

    # ========================================================================
    # Step 11: Save fusion.json
    # ========================================================================
    # Type cast needed because select_combiner returns str
    head_feature_sources = resolve_head_feature_sources(rel_feature_variant)
    fusion_cfg = FusionConfig(
        gate_threshold=tau_us,
        combine=cast(Literal["product", "logreg"], chosen_combiner),
        coefs=lr_coefs_to_save,
        score_space="prob",
        includes_us=False,
        composed_platt=composed_platt_ab,
        head_calibrators=head_calibrators,
        head_feature_sources=head_feature_sources,
    )
    fusion_path = fusion_path_for_weights(output_dir / "ica_fusion.weights.h5")
    save_fusion(fusion_cfg, fusion_path)
    logger.info(f"Saved fusion.json: {fusion_path}")

    # Verify round-trip
    from src.fusion.sidecar import load_fusion
    fusion_cfg_loaded = load_fusion(fusion_path)
    assert fusion_cfg_loaded.gate_threshold == tau_us
    assert fusion_cfg_loaded.combine == chosen_combiner
    assert fusion_cfg_loaded.composed_platt == composed_platt_ab
    assert fusion_cfg_loaded.head_calibrators == head_calibrators
    assert fusion_cfg_loaded.head_feature_sources == head_feature_sources
    logger.info("✓ fusion.json round-trip verified (including new fields)")

    # ========================================================================
    # Step 12: Save metrics JSON (with gate_note documenting gold-first + ceiling)
    # ========================================================================
    gate_note = (
        f"Gold-first gating: at apply, DoCA/dateline-labeled rows bypass the ML gate "
        f"and use authoritative US labels. This rescues known positives and diaspora "
        f"events the US head misses. The ML gate (τ_us={tau_us:.3f}) governs only "
        f"non-gold rows. See us-head-retrain-plan.md for diaspora ceiling findings."
    )

    metrics = {
        "chosen_combiner": chosen_combiner,
        "tau_us": tau_us,
        "tau_us_qualified": tau_us_qualified,
        "tau_us_achieved_recall": tau_us_achieved_recall,
        "target_us_recall": target_us_recall,
        "and_pr_auc_mean": and_mean,
        "and_pr_auc_se": and_se,
        "lr_pr_auc_mean": lr_mean,
        "lr_pr_auc_se": lr_se,
        "composed_ece": composed_cal["ece"],
        "composed_brier": composed_cal["brier"],
        "composed_platt_fitted": composed_platt_fitted,
        "n_us_true": n_us_true,
        "n_positive_us_true": n_pos_us_true,
        "n_negative_us_true": n_neg_us_true,
        "gate_note": gate_note,
        "rel_feature_variant": rel_feature_variant,
        "head_feature_sources": head_feature_sources,
    }

    metrics_path = output_dir / "ica_fusion_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Saved metrics.json: {metrics_path}")

    # ========================================================================
    # Final summary
    # ========================================================================
    logger.info("=" * 70)
    logger.info("FUSION SELECTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Chosen combiner:        {chosen_combiner}")
    logger.info(
        f"US gate threshold:      {tau_us:.3f} "
        f"(qualified={tau_us_qualified}, achieved_recall={tau_us_achieved_recall:.3f})"
    )
    logger.info(f"Combiner fit population: {n_us_true} (us_event==True)")
    logger.info(f"  Label balance:        {n_pos_us_true} positive, {n_neg_us_true} negative")
    logger.info(f"AND PR-AUC (CV):         {and_mean:.4f} ± {and_se:.4f}")
    logger.info(f"LR PR-AUC (CV):          {lr_mean:.4f} ± {lr_se:.4f}")
    logger.info(f"Composed ECE/Brier:     {composed_cal['ece']:.4f} / {composed_cal['brier']:.4f}")
    logger.info("=" * 70)

    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser (importable so tests can check flag defaults,
    mirroring `apply_ica.build_arg_parser`)."""
    ap = argparse.ArgumentParser(
        description="Fit the ICA fusion combiner + composed calibration."
    )
    ap.add_argument(
        "--cache-suffix", default=DEFAULT_CACHE_SUFFIX,
        help="embed cache subdir CCA/rel features are joined against "
             "(default: relevance_train)",
    )
    ap.add_argument("--cca-weights", default=None, help="default: config.CCA_DOCA_WEIGHTS")
    ap.add_argument("--rel-weights", default=None, help="default: config.RELEVANCE_DOCA_WEIGHTS")
    ap.add_argument("--us-weights", default=None, help="default: config.US_FILTER_FULL_WEIGHTS")
    ap.add_argument(
        "--out", default=None,
        help="output directory for ica_fusion.fusion.json/ica_fusion_metrics.json "
             "(default: cca_doca/ -- MUST pass a distinct directory when any of "
             "--cache-suffix/--cca-weights/--rel-weights/--us-weights/"
             "--rel-feature-variant is non-default, or the run refuses to start)",
    )
    ap.add_argument(
        "--rel-feature-variant", default=None,
        help="rel-branch CLS variant tag (branched-encoder fusion refit, "
             "docs/design-plans/2026-08-18-stage4-joint-finetune.md). The rel "
             "head's features are read from this variant array off the SAME "
             "--cache-suffix cache; CCA/US stay on the base array. Default: "
             "None = legacy (rel shares the base array). Saves "
             "head_feature_sources={'us':'base','cca':'base','rel':<tag>} into "
             "the fusion sidecar.",
    )
    return ap


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    metrics = main(
        cache_suffix=args.cache_suffix,
        cca_weights_path=args.cca_weights,
        rel_weights_path=args.rel_weights,
        us_weights_path=args.us_weights,
        output_dir=args.out,
        rel_feature_variant=args.rel_feature_variant,
    )
    print("\n✓ Fusion selection complete")
    print(json.dumps(metrics, indent=2))
