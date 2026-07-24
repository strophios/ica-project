# pattern: Imperative Shell
"""Validate-before-swap: compare the v1 US-head retrain candidate against the
current production head, on data its own val-split selection never touched.

Design: docs/notes/us-head-retrain-plan.md ("Sequencing: validate before swap").
The retrain (src/run_us_pnu.py, `us_filter/us_pnu.weights.h5`) was model-selected
purely on the PNU table's own val split (see that script's module docstring).
This script is the SEPARATE, held-out comparison against the hand-coded ICA eval
set and an LDC dateline-label regression slice -- both untouched by selection.

Three checks:
  1. ICA hand-coded eval set (validation/ica_coding_template_coded.csv, gold
     `us_event`): ROC-AUC + recall at matched-precision operating points for
     both heads (src.validation.matched_operating_point -- valid across the
     scale mismatch below), plus DoCA-anchor recall and its diaspora subset
     (anchors scoring calibrated p_us<0.2 under the CURRENT head -- the failure
     mode that motivated this retrain; docs/notes/us-head-retrain-plan.md
     documents ~26 of ~139).
  2. Dateline-label bulk-performance regression check: P/R/F1 at logit 0 for
     both heads on a LEAK-SAFE LDC dateline-labeled slice (current head's
     reference number, per the design doc: dateline-test F1=0.97). Leak-safety:
     the dateline-TRUE rows were never in PNU-table training at all (only DoCA
     positives are "pos"); the dateline-FALSE rows use the retrain's own held-
     out test split of the "neg" group (never in train/val by construction of
     create_us_pnu_data + assert_holdout_excluded).

CALIBRATION NOTE: the retrain candidate has NO Platt calibrator yet (deferred to
the swap phase). Its scores below are RAW LOGITS. ROC-AUC is scale-invariant;
`recall_at_matched_precision` rank-matches via each head's own precision-recall
curve rather than assuming a shared scale; F1 uses logit>0 (~p=0.5 under the
eventual monotonic calibration) as the retrain's decision boundary.

Run from project root: uv run python -m scripts.eval_us_retrain
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, precision_score, recall_score

import src.cca_config as cca_config
import src.config as config
import src.us_config as us_config
from src.embed_corpus import load_cache
from src.model_setup.assembly import build_feature_inference_model
from src.model_setup.heads import ClassificationHead
from src.run_us_pnu import _load_caches, _load_holdout_ids, attach_emb_rows
from src.data_setup.data import create_us_pnu_data, assert_holdout_excluded
from src.validation.matched_operating_point import recall_at_matched_precision, roc_auc
from src.validation.slice_eval import apply_us_model

EVAL_CSV = config.PROJECT_ROOT / "validation" / "ica_coding_template_coded.csv"
CURRENT_WEIGHTS = config.US_FILTER_FULL_WEIGHTS
NEW_WEIGHTS = config.US_PNU_WEIGHTS
PRECISION_TARGETS = (0.3, 0.5, 0.7)  # matched against the CURRENT head's precision at these thresholds
DIASPORA_THRESHOLD = 0.2  # calibrated p_us cutoff defining the diaspora failure mode
DATELINE_REFERENCE_F1 = 0.97  # design-doc reference number for the current head


# ---------------------------------------------------------------------------
# Pattern-2 feature-mode scorers (shared by both heads' sidecar shapes)
# ---------------------------------------------------------------------------
def _score_features(hidden_dim: int, head_name: str, weights_path: Path, cls: np.ndarray) -> np.ndarray:
    head = ClassificationHead(hidden_dim=hidden_dim, name=head_name)
    model = build_feature_inference_model({head_name: head}, hidden_dim=hidden_dim)
    model.load_weights(str(weights_path), skip_mismatch=False)
    logits = model.predict({"features": cls}, batch_size=512, verbose=0)[head_name]
    logits = np.asarray(logits).reshape(-1)
    if not np.isfinite(logits).all():
        raise ValueError(f"non-finite logits from {weights_path}")
    return logits


def _score_new_head_features(cls: np.ndarray) -> np.ndarray:
    rc = cca_config.RunConfig.from_json(cca_config.config_path_for_weights(NEW_WEIGHTS))
    head_cfg = rc.heads[0]
    return _score_features(head_cfg.hidden_dim, head_cfg.name, NEW_WEIGHTS, cls)


def _score_current_head_features(cls: np.ndarray) -> np.ndarray:
    rc = us_config.UsRunConfig.from_json(us_config.config_path_for_weights(CURRENT_WEIGHTS))
    return _score_features(rc.head.hidden_dim, rc.head.name, CURRENT_WEIGHTS, cls)


# ---------------------------------------------------------------------------
# Check 1: ICA hand-coded eval set
# ---------------------------------------------------------------------------
def _eval_ica_set() -> dict:
    eval_df = pl.read_csv(EVAL_CSV)
    n_raw = eval_df.height

    texts = [
        f"{h if h else ''}</s>{lead if lead else ''}"
        for h, lead in zip(eval_df["headline"].to_list(), eval_df["lead_paragraph"].to_list())
    ]
    # Current head: calibrated probabilities, raw (API/dateline-free) text --
    # exact recipe of scripts/eval_heads_own_terms.py, so the anchor/diaspora
    # numbers reproduce that script's documented ~26/139.
    p_current = apply_us_model(texts, weights_path=CURRENT_WEIGHTS, skip_mismatch=True)
    eval_df = eval_df.with_columns(pl.Series("p_us_current", p_current))

    # New head: raw logits over the relevance_train cached CLS features (same
    # frozen DAPT backbone as every head -- reuse rather than re-run token mode).
    meta_full, cls = load_cache(config.CCA_EMBED_CACHE_DIR / "relevance_train")
    meta_by_id = meta_full.with_columns(pl.col("id").cast(pl.Utf8).alias("id_str")).select(
        ["id_str", "emb_row"]
    )
    eval_df = eval_df.with_columns(pl.col("id").cast(pl.Utf8).alias("id_str")).join(
        meta_by_id, on="id_str", how="left"
    )
    n_missing_emb = eval_df.filter(pl.col("emb_row").is_null()).height
    scored = eval_df.filter(pl.col("emb_row").is_not_null())
    features = cls[scored["emb_row"].to_numpy().astype(int)]
    logit_new = _score_new_head_features(features)
    scored = scored.with_columns(pl.Series("logit_new", logit_new))
    print(f"ICA eval set: {n_raw} rows, {n_missing_emb} missing embeddings, "
          f"{scored.height} scored by both heads")  # LOG

    # --- rank comparison on rows with a coded us_event ---
    coded = scored.filter(pl.col("us_event").is_not_null())
    y = coded["us_event"].to_numpy().astype(bool)
    p_cur = coded["p_us_current"].to_numpy()
    l_new = coded["logit_new"].to_numpy()

    roc_current = roc_auc(p_cur, y)
    roc_new = roc_auc(l_new, y)

    matched_points = []
    for t in PRECISION_TARGETS:
        pred_cur = p_cur >= t
        prec_cur = float(precision_score(y, pred_cur, zero_division=0))
        rec_cur = float(recall_score(y, pred_cur, zero_division=0))
        matched_new = recall_at_matched_precision(l_new, y, target_precision=prec_cur)
        matched_points.append({
            "current_threshold": t, "current_precision": prec_cur, "current_recall": rec_cur,
            "new_matched_precision": matched_new["precision"],
            "new_matched_recall": matched_new["recall"],
            "new_threshold_logit": matched_new["threshold"],
        })

    f1_current = float(f1_score(y, p_cur >= 0.5, zero_division=0))
    f1_new = float(f1_score(y, l_new > 0.0, zero_division=0))

    # --- DoCA-anchor recall + diaspora subset (recomputed on the FULL eval_df,
    # not just embedding-covered rows, so the current-head anchor/diaspora
    # identification matches the design doc's methodology exactly) ---
    anchors = eval_df.filter(pl.col("sample_stratum") == "anchor")
    n_anchors = anchors.height
    diaspora = anchors.filter(pl.col("p_us_current") < DIASPORA_THRESHOLD)
    n_diaspora = diaspora.height
    diaspora_scored = diaspora.filter(pl.col("emb_row").is_not_null())
    n_diaspora_missing_emb = n_diaspora - diaspora_scored.height
    diaspora_logits_new = (
        _score_new_head_features(cls[diaspora_scored["emb_row"].to_numpy().astype(int)])
        if diaspora_scored.height > 0 else np.array([])
    )
    recovery_thresholds = (-2.0, -1.0, 0.0, 1.0)
    diaspora_recovery = [
        {"logit_threshold": t, "n_recovered": int((diaspora_logits_new > t).sum())}
        for t in recovery_thresholds
    ] if diaspora_logits_new.size else []

    non_diaspora_anchors = anchors.filter(pl.col("p_us_current") >= DIASPORA_THRESHOLD)
    non_diaspora_scored = non_diaspora_anchors.filter(pl.col("emb_row").is_not_null())
    non_diaspora_logits_new = (
        _score_new_head_features(cls[non_diaspora_scored["emb_row"].to_numpy().astype(int)])
        if non_diaspora_scored.height > 0 else np.array([])
    )

    return {
        "n_coded_rows": n_raw,
        "n_missing_embeddings": n_missing_emb,
        "n_scored": scored.height,
        "roc_auc_current": roc_current,
        "roc_auc_new": roc_new,
        "f1_current_at_0.5": f1_current,
        "f1_new_at_logit_0": f1_new,
        "matched_precision_points": matched_points,
        "doca_anchors": {
            "n_anchors": n_anchors,
            "n_diaspora": n_diaspora,
            "n_diaspora_missing_embeddings": n_diaspora_missing_emb,
            "diaspora_threshold": DIASPORA_THRESHOLD,
            "diaspora_recovery_by_new_head": diaspora_recovery,
            "diaspora_new_logit_mean": float(diaspora_logits_new.mean()) if diaspora_logits_new.size else None,
            "diaspora_new_logit_median": float(np.median(diaspora_logits_new)) if diaspora_logits_new.size else None,
            "non_diaspora_new_logit_mean": float(non_diaspora_logits_new.mean()) if non_diaspora_logits_new.size else None,
        },
    }


# ---------------------------------------------------------------------------
# Check 2: LDC dateline-label regression check (leak-safe)
# ---------------------------------------------------------------------------
def _eval_dateline_regression() -> dict:
    caches = _load_caches()
    us_train_ldc_meta, us_train_ldc_cls = caches["us_train_ldc"]

    # Positives: ALL dateline-True rows. Never in PNU-table training (the "pos"
    # group there is DoCA-only) -- leak-safe by construction, no exclusion needed.
    pos_meta = us_train_ldc_meta.filter(pl.col("us_label"))

    # Negatives: the retrain's OWN held-out test split of the "neg" group (dateline
    # -resolved-foreign rows) -- reproduce the exact split via the same functions
    # run_us_pnu.py used, so this is provably the held-out slice, not just "should be".
    table = pl.read_parquet(config.US_PNU_TABLE)
    holdout_ids = _load_holdout_ids()
    cache_metas = {name: meta for name, (meta, _cls) in caches.items()}
    resolved_table = attach_emb_rows(table, cache_metas)
    splits = create_us_pnu_data(resolved_table, holdout_ids=holdout_ids)
    assert_holdout_excluded(splits, holdout_ids)
    neg_test = splits["test"]["neg"]

    # Defense-in-depth: confirm zero overlap between the negative test ids used
    # here and the neg train/val ids (belt-and-suspenders on top of the split
    # function's own disjointness).
    neg_train_val_ids = (
        set(splits["train"]["neg"]["id"].to_list()) | set(splits["val"]["neg"]["id"].to_list())
    )
    overlap = set(neg_test["id"].to_list()) & neg_train_val_ids
    if overlap:
        raise ValueError(f"leak: {len(overlap)} dateline-regression negative ids also in train/val")

    pos_rows = pos_meta["emb_row"].to_numpy().astype(int)
    neg_rows = neg_test["emb_row"].to_numpy().astype(int)
    features = np.concatenate([us_train_ldc_cls[pos_rows], us_train_ldc_cls[neg_rows]], axis=0)
    labels = np.concatenate([np.ones(pos_rows.shape[0]), np.zeros(neg_rows.shape[0])]).astype(bool)

    logit_current = _score_current_head_features(features)
    logit_new = _score_new_head_features(features)

    def _prf(logits):
        # NOTE: n_pos (487k dateline-True) vastly outnumbers n_neg (this split's
        # ~5.1k held-out dateline-False) -- at that ratio F1/precision of the
        # POSITIVE class are near-saturated even for a model that rejects almost
        # no negatives (misclassifying every negative only costs ~1% precision).
        # `neg_recall` (specificity -- fraction of the 5.1k known-foreign rows
        # correctly flagged not-US) is the metric that actually discriminates
        # here; report both.
        pred = logits > 0
        tn = int(((~pred) & (~labels)).sum())
        return {
            "precision": float(precision_score(labels, pred, zero_division=0)),
            "recall": float(recall_score(labels, pred, zero_division=0)),
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "neg_recall_specificity": tn / int((~labels).sum()) if int((~labels).sum()) else 0.0,
        }

    return {
        "n_pos": int(pos_rows.shape[0]),
        "n_neg": int(neg_rows.shape[0]),
        "reference_f1_current_head": DATELINE_REFERENCE_F1,
        "current_head": _prf(logit_current),
        "new_head": _prf(logit_new),
    }


def main() -> dict:
    print("[1/2] ICA hand-coded eval set comparison...")  # LOG
    ica_results = _eval_ica_set()
    print("[2/2] Dateline-label regression check...")  # LOG
    dateline_results = _eval_dateline_regression()

    results = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "current_weights": str(CURRENT_WEIGHTS),
        "new_weights": str(NEW_WEIGHTS),
        "note": (
            "New head has no Platt calibrator (deferred to swap phase); its scores "
            "are raw logits. See module docstring CALIBRATION NOTE."
        ),
        "ica_eval": ica_results,
        "dateline_regression": dateline_results,
    }

    out_path = config.CCA_DOCA_DIR / "experiments" / "eval_us_retrain.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}\n")

    print("=== ICA eval set: current vs new (raw logits) ===")
    print(f"ROC-AUC   current={ica_results['roc_auc_current']:.4f}  new={ica_results['roc_auc_new']:.4f}")
    print(f"F1        current@0.5={ica_results['f1_current_at_0.5']:.4f}  "
          f"new@logit0={ica_results['f1_new_at_logit_0']:.4f}")
    for pt in ica_results["matched_precision_points"]:
        print(f"  matched precision~{pt['current_precision']:.3f}: "
              f"current recall={pt['current_recall']:.3f}  "
              f"new recall={pt['new_matched_recall']:.3f} "
              f"(new precision achieved={pt['new_matched_precision']:.3f})")

    anc = ica_results["doca_anchors"]
    print(f"\nDoCA anchors: {anc['n_anchors']} total, {anc['n_diaspora']} diaspora "
          f"(p_us_current<{anc['diaspora_threshold']})")
    for rec in anc["diaspora_recovery_by_new_head"]:
        print(f"  new head logit>{rec['logit_threshold']}: "
              f"{rec['n_recovered']}/{anc['n_diaspora']} diaspora anchors recovered")

    print("\n=== Dateline-label regression (leak-safe) ===")
    dl = dateline_results
    print(f"n_pos={dl['n_pos']} n_neg={dl['n_neg']} (reference F1, current head: "
          f"{dl['reference_f1_current_head']})")
    print(f"current: P={dl['current_head']['precision']:.4f} "
          f"R={dl['current_head']['recall']:.4f} F1={dl['current_head']['f1']:.4f} "
          f"specificity(neg_recall)={dl['current_head']['neg_recall_specificity']:.4f}")
    print(f"new:     P={dl['new_head']['precision']:.4f} "
          f"R={dl['new_head']['recall']:.4f} F1={dl['new_head']['f1']:.4f} "
          f"specificity(neg_recall)={dl['new_head']['neg_recall_specificity']:.4f}")

    return results


if __name__ == "__main__":
    main()
