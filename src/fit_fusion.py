# pattern: Imperative Shell (orchestration) + embedded Functional Core (select_combiner)
"""Empirical fusion selection + composed calibration (Phase 4, Task 5).

Orchestrates the fusion combiner choice:
1. Load clean eval set and feature cache
2. Score calibrated US, CCA, rel heads over eval ids
3. Pick US gate threshold via recall recipe (anchor DoCA positives)
4. Gate survivors (p_us >= τ_us)
5. Run StratifiedKFold CV: AND vs LR by PR-AUC
6. Apply 1-SE decision rule to select combiner
7. Compute composed calibration and optionally fit final Platt
8. Save fusion.json + metrics JSON

Functional Core: select_combiner (pure 1-SE margin rule decision logic).
Imperative Shell: main() orchestration with checkpoint validation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, cast

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

import src.config as config
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
from src.validation.doca_recall import pick_us_threshold
from src.validation.relevance_slice_eval import apply_relevance_model
from src.validation.slice_eval import apply_us_model


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


# ============================================================================
# Imperative Shell: orchestration
# ============================================================================


def main(
    eval_csv_path: Path | str = "../../validation/ica_coding_template_coded.csv",
    target_us_recall: float = 0.95,
    us_thresholds: list[float] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    include_us_in_lr: bool = False,
    output_dir: Path | str | None = None,
) -> dict:
    """Orchestrate fusion selection + composed calibration.

    Args:
        eval_csv_path: Hand-coded eval set CSV (relative to project root)
        target_us_recall: Target DoCA recall for US gate (e.g., 0.95)
        us_thresholds: Thresholds to evaluate for recall recipe (e.g., linspace(0.3, 0.7, 5))
        n_splits: Folds for StratifiedKFold CV (default 5)
        random_state: Seed for determinism
        include_us_in_lr: If True, fit LR with z_cca + z_rel + z_us (default False)
        output_dir: Directory for fusion.json + metrics.json (default uses cca_doca dir)

    Returns:
        Dict with keys: combiner, tau_us, qualified, cv_and_metrics, cv_lr_metrics,
                        composed_ece, composed_brier, n_gated, n_pos_gated
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

    if output_dir is None:
        output_dir = config.CCA_DOCA_DIR
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if us_thresholds is None:
        us_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    logger.info(f"Loading eval CSV: {eval_csv_path}")
    eval_df = pl.read_csv(eval_csv_path)
    logger.info(f"Loaded {eval_df.shape[0]} rows")

    # Drop null ica_event rows (per task spec)
    eval_df = eval_df.filter(pl.col("ica_event").is_not_null())
    logger.info(f"After dropping null ica_event: {eval_df.shape[0]} rows")

    # Count label distribution
    n_true = eval_df["ica_event"].sum()
    n_false = (~eval_df["ica_event"]).sum()
    logger.info(f"Label balance: {n_true} True, {n_false} False")

    # ========================================================================
    # Step 1: Load features (embed cache) and build feature matrix
    # ========================================================================
    logger.info("Loading embed cache meta and CLS features")
    meta_full, cls = load_cache(config.CCA_EMBED_CACHE_DIR / "relevance_train")

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

    # Build texts for US model (headline + "</s>" + lead_paragraph)
    headline_col = "headline" if "headline" in eval_df_join.columns else None
    lead_col = "lead_paragraph" if "lead_paragraph" in eval_df_join.columns else None

    if headline_col and lead_col:
        headlines = eval_df_join[headline_col].to_list()
        leads = eval_df_join[lead_col].to_list()
        texts_list = [
            f"{h if h else ''}</s>{lead if lead else ''}"
            for h, lead in zip(headlines, leads)
        ]
    else:
        # Fallback: use whatever text columns are available
        logger.warning("⚠️  headline or lead_paragraph columns missing; using fallback")
        texts_list = [""] * eval_df_join.shape[0]

    # ========================================================================
    # Step 2: Score calibrated heads
    # ========================================================================
    logger.info("Scoring CCA head")
    cca_logits = apply_cca_model(features, weights_path=config.CCA_DOCA_WEIGHTS)
    cca_cal = load_calibration(
        calibration_path_for_weights(config.CCA_DOCA_WEIGHTS)
    )
    p_cca = cca_cal.transform(cca_logits)
    assert np.isfinite(p_cca).all(), "Non-finite CCA probabilities"
    logger.info(f"CCA: mean={p_cca.mean():.3f}, std={p_cca.std():.3f}")

    logger.info("Scoring relevance head")
    rel_logits = apply_relevance_model(features, weights_path=config.RELEVANCE_DOCA_WEIGHTS)
    rel_cal = load_calibration(
        calibration_path_for_weights(config.RELEVANCE_DOCA_WEIGHTS)
    )
    p_rel = rel_cal.transform(rel_logits)
    assert np.isfinite(p_rel).all(), "Non-finite relevance probabilities"
    logger.info(f"Relevance: mean={p_rel.mean():.3f}, std={p_rel.std():.3f}")

    logger.info("Scoring US head (calibrated)")
    p_us = apply_us_model(
        texts_list, weights_path=config.US_FILTER_FULL_WEIGHTS, skip_mismatch=True
    )
    assert np.isfinite(p_us).all(), "Non-finite US probabilities"
    logger.info(f"US: mean={p_us.mean():.3f}, std={p_us.std():.3f}")

    # Sanity check: real ICA (ica_event==True) should score higher on all three heads
    ica_event = eval_df_join["ica_event"].to_numpy().astype(bool)
    logger.info("Sanity check: ICA-True vs ICA-False mean scores")
    logger.info(f"  CCA: True={p_cca[ica_event].mean():.3f} vs False={p_cca[~ica_event].mean():.3f}")
    logger.info(f"  Rel: True={p_rel[ica_event].mean():.3f} vs False={p_rel[~ica_event].mean():.3f}")
    logger.info(f"  US:  True={p_us[ica_event].mean():.3f} vs False={p_us[~ica_event].mean():.3f}")

    # ========================================================================
    # Step 3: Pick US threshold via recall recipe
    # ========================================================================
    logger.info(f"Picking US threshold via recall recipe (target recall={target_us_recall})")

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
    qualified = threshold_result.qualified
    logger.info(
        f"Selected τ_us={tau_us:.3f}, qualified={qualified} "
        f"(recall recipe {'met' if qualified else 'not met'})"
    )

    # ========================================================================
    # Step 4: Gate survivors (p_us >= tau_us)
    # ========================================================================
    gated_mask = p_us >= tau_us
    n_gated = int(gated_mask.sum())
    logger.info(f"Gated survivors: {n_gated} rows (p_us >= {tau_us:.3f})")

    if n_gated == 0:
        raise ValueError("No survivors after US gate; cannot proceed")

    # Subsample everything to gated rows
    p_cca_gated = p_cca[gated_mask]
    p_rel_gated = p_rel[gated_mask]
    ica_event_gated = ica_event[gated_mask]
    n_pos_gated = int(ica_event_gated.sum())
    logger.info(f"  Label balance: {n_pos_gated} True, {n_gated - n_pos_gated} False")

    # ========================================================================
    # Step 5: StratifiedKFold CV — AND vs LR
    # ========================================================================
    logger.info(f"Running StratifiedKFold CV (n_splits={n_splits})")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_and_pr_aucs = []
    cv_lr_pr_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(p_cca_gated, ica_event_gated)
    ):
        # Split data
        p_cca_train, p_cca_val = p_cca_gated[train_idx], p_cca_gated[val_idx]
        p_rel_train, p_rel_val = p_rel_gated[train_idx], p_rel_gated[val_idx]
        y_train = ica_event_gated[train_idx].astype(int)
        y_val = ica_event_gated[val_idx].astype(int)

        # Compute AND on validation fold
        and_scores_val = combine_and(p_cca_val, p_rel_val)
        and_pr_auc = average_precision_score(y_val, and_scores_val)
        cv_and_pr_aucs.append(and_pr_auc)

        # Fit LR on train fold and apply to val fold
        # Features: z_cca, z_rel (probabilities)
        lr_features_train = np.column_stack([p_cca_train, p_rel_train])
        lr_features_val = np.column_stack([p_cca_val, p_rel_val])

        # TODO: add include_us_in_lr option if needed (requires tracking original row indices)
        if include_us_in_lr:
            logger.warning("⚠️  include_us_in_lr=True not yet implemented; skipping")

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
    # Step 6: Apply 1-SE decision rule
    # ========================================================================
    cv_and_result = {"pr_auc": cv_and_pr_aucs}
    cv_lr_result = {"pr_auc": cv_lr_pr_aucs}
    chosen_combiner = select_combiner(cv_and_result, cv_lr_result)
    logger.info(f"Decision rule selected: {chosen_combiner}")

    # ========================================================================
    # Step 7: Fit final LR on full gated set (for composed score)
    # ========================================================================
    lr_features_full = np.column_stack([p_cca_gated, p_rel_gated])
    lr_full = fit_logistic_combiner(
        lr_features_full, ica_event_gated.astype(int), random_state=random_state
    )
    lr_coefs = lr_full.coef_[0].tolist()
    logger.info(f"Full LR coefficients: {lr_coefs}")

    # ========================================================================
    # Step 8: Compute composed score and calibration
    # ========================================================================
    if chosen_combiner == "product":
        composed_score = combine_and(p_cca_gated, p_rel_gated)
        lr_coefs_to_save = None
    else:  # logreg
        composed_score = apply_logistic_combiner(lr_full, lr_features_full)
        lr_coefs_to_save = lr_coefs

    # Calibration report
    composed_cal = calibration_report(composed_score, ica_event_gated, n_bins=15)
    logger.info(
        f"Composed calibration: ECE={composed_cal['ece']:.4f}, "
        f"Brier={composed_cal['brier']:.4f}"
    )

    # ========================================================================
    # Step 9: Save fusion.json
    # ========================================================================
    # Type cast needed because select_combiner returns str
    fusion_cfg = FusionConfig(
        gate_threshold=tau_us,
        combine=cast(Literal["product", "logreg"], chosen_combiner),
        coefs=lr_coefs_to_save,
        score_space="prob",
        includes_us=False,
    )
    fusion_path = fusion_path_for_weights(output_dir / "ica_fusion.weights.h5")
    save_fusion(fusion_cfg, fusion_path)
    logger.info(f"Saved fusion.json: {fusion_path}")

    # Verify round-trip
    from src.fusion.sidecar import load_fusion
    fusion_cfg_loaded = load_fusion(fusion_path)
    assert fusion_cfg_loaded.gate_threshold == tau_us
    assert fusion_cfg_loaded.combine == chosen_combiner
    logger.info("✓ fusion.json round-trip verified")

    # ========================================================================
    # Step 10: Save metrics JSON
    # ========================================================================
    metrics = {
        "chosen_combiner": chosen_combiner,
        "tau_us": tau_us,
        "tau_us_qualified": qualified,
        "target_us_recall": target_us_recall,
        "and_pr_auc_mean": and_mean,
        "and_pr_auc_se": and_se,
        "lr_pr_auc_mean": lr_mean,
        "lr_pr_auc_se": lr_se,
        "composed_ece": composed_cal["ece"],
        "composed_brier": composed_cal["brier"],
        "n_gated_survivors": n_gated,
        "n_positive_gated": n_pos_gated,
        "n_negative_gated": n_gated - n_pos_gated,
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
    logger.info(f"US gate threshold:      {tau_us:.3f} (qualified={qualified})")
    logger.info(f"Gated survivors:        {n_gated} ({n_pos_gated} positive)")
    logger.info(f"AND PR-AUC (CV):         {and_mean:.4f} ± {and_se:.4f}")
    logger.info(f"LR PR-AUC (CV):          {lr_mean:.4f} ± {lr_se:.4f}")
    logger.info(f"Composed ECE/Brier:     {composed_cal['ece']:.4f} / {composed_cal['brier']:.4f}")
    logger.info("=" * 70)

    return metrics


if __name__ == "__main__":
    metrics = main()
    print("\n✓ Fusion selection complete")
    print(json.dumps(metrics, indent=2))
