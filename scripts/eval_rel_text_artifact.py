# pattern: Imperative Shell (matched_rate_threshold / _own_terms_metrics are the
#   pure Functional Core; everything else is I/O -- cache/CSV reads, model builds)
"""Validate-before-swap eval for a text-mode rel artifact.

Scores the hand-coded ICA eval set (`validation/ica_coding_template_coded.csv`,
1,131 rows; text = `headline + "</s>" + lead_paragraph`) with two models and
reports comparable own-terms + vs-ICA + diaspora-slice metrics:

  (a) TEXT-MODE artifact (`--weights`, default `relevance/relevance_text.weights.h5`
      + its `.config.json` sidecar) -- Pattern 2 (fresh backbone + rel head,
      weights loaded by structure), the same shape as
      `src.validation.slice_eval.apply_us_model` /
      `scripts/eval_us_retrain.py`'s cross-config scoring, but for a
      `cca_config.RunConfig` sidecar (the rel head's config family) rather
      than `UsRunConfig`. Scores are RAW LOGITS -- text-mode has no Platt
      calibrator yet (deferred to the swap phase, same caveat as
      `eval_us_retrain.py`'s new US head).
  (b) FROZEN features-mode rel head (`relevance/relevance.weights.h5`, head
      name "rel") over CLS features looked up by id in the `relevance_train`
      embed cache -- the exact mechanism `scripts/eval_heads_own_terms.py`
      uses to score this same eval set. Scores are Platt-calibrated
      probabilities.

If the text-mode weights file is absent (the operator may not have rsynced it
from the cluster yet), the script does NOT crash: it prints the rsync command
needed, skips (a), and still runs + reports (b) alone. The weights path itself
is a CLI arg so multiple run artifacts can be diffed against the frozen
baseline without editing this file.

Because (a) is uncalibrated and (b) is calibrated, no operating point is
comparable at a shared score value. The diaspora-slice comparison instead uses
a RANK-BASED matched-positive-rate threshold: for a target overall flagged
fraction (e.g. 0.20), each model's own threshold is its score's
`(1 - rate)`-quantile over the full scored eval set. This is valid across the
scale mismatch (same rank-based idea as `recall_at_matched_precision` in
`src/validation/matched_operating_point.py`, but matching flagged-rate rather
than precision -- precision needs a shared notion of "positive" cost/base rate
that doesn't help here, whereas flagged-rate directly answers "if operators
review the same number of articles, how many diaspora events surface").

Run from project root:
    uv run python -m scripts.eval_rel_text_artifact
    uv run python -m scripts.eval_rel_text_artifact --weights relevance/relevance_text_eta0.3.weights.h5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

import src.cca_config as cca_config
import src.config as config
from src.calibration.sidecar import calibration_path_for_weights, load_calibration
from src.embed_corpus import load_cache
from src.model_setup.assembly import build_inference_model
from src.model_setup.backbone import load_dapt_backbone
from src.model_setup.heads import ClassificationHead
from src.preproc.preprocessor import ClassifierPreprocessor
from src.validation.relevance_slice_eval import apply_relevance_model

EVAL_CSV = config.PROJECT_ROOT / "validation" / "ica_coding_template_coded.csv"
DEFAULT_TEXT_WEIGHTS = config.RELEVANCE_TEXT_WEIGHTS
FROZEN_WEIGHTS = config.RELEVANCE_DOCA_WEIGHTS
OUT_PATH = config.CCA_DOCA_DIR / "experiments" / "eval_rel_text.json"
MATCHED_POSITIVE_RATES = (0.10, 0.20, 0.30)


# ---------------------------------------------------------------------------
# Functional core
# ---------------------------------------------------------------------------
def matched_rate_threshold(scores: np.ndarray, positive_rate: float) -> float:
    """Pure: the score threshold giving `positive_rate` flagged, rank-based.

    `positive_rate` is the fraction of `scores` at or above the returned
    threshold (approximately -- `np.quantile`'s interpolation means the exact
    flagged fraction can differ slightly from `positive_rate` for small `n`).
    Valid across score scale/calibration mismatches since it only uses each
    model's own rank distribution.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("scores must be non-empty")
    if not (0.0 < positive_rate <= 1.0):
        raise ValueError(f"positive_rate must be in (0, 1]; got {positive_rate}")
    return float(np.quantile(scores, 1.0 - positive_rate))


def own_terms_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    """Pure: ROC-AUC + PR-AUC for scores `p` against boolean labels `y`.

    `p` need only be monotonically related to model confidence (raw logits or
    calibrated probabilities both work -- both metrics are rank-based /
    threshold-swept).
    """
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def diaspora_slice_metrics(
    all_scores: np.ndarray, diaspora_scores: np.ndarray, positive_rates=MATCHED_POSITIVE_RATES
) -> dict:
    """Pure: diaspora recall/mean-score at rank-based matched-positive-rate points.

    `all_scores` is the full scored-eval-set score array (defines the
    thresholds); `diaspora_scores` is the subset restricted to
    `event_type == "Diasporic"`.
    """
    points = []
    for rate in positive_rates:
        thr = matched_rate_threshold(all_scores, rate)
        flagged = diaspora_scores >= thr
        points.append({
            "target_positive_rate": rate,
            "threshold": thr,
            "achieved_positive_rate_overall": float((all_scores >= thr).mean()),
            "diaspora_recall": float(flagged.mean()) if diaspora_scores.size else None,
        })
    return {
        "n_diaspora": int(diaspora_scores.size),
        "mean_score": float(diaspora_scores.mean()) if diaspora_scores.size else None,
        "median_score": float(np.median(diaspora_scores)) if diaspora_scores.size else None,
        "matched_rate_points": points,
    }


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------
def _build_texts(df: pl.DataFrame) -> list[str]:
    return [
        f"{h if h else ''}</s>{lead if lead else ''}"
        for h, lead in zip(df["headline"].to_list(), df["lead_paragraph"].to_list())
    ]


def score_frozen_head(eval_df: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
    """Score `eval_df` with the frozen features-mode rel head.

    Reproduces `scripts/eval_heads_own_terms.py`'s exact mechanism: join eval
    ids to the `relevance_train` embed cache's `emb_row` by id, drop rows
    without a cached embedding, gather CLS features, score + Platt-calibrate.

    Returns (scored_df, calibrated_probabilities) -- `scored_df` is `eval_df`
    filtered to the rows that had a cached embedding (same row order as the
    returned scores).
    """
    meta_full, cls = load_cache(config.CCA_EMBED_CACHE_DIR / "relevance_train")
    meta_by_id = meta_full.with_columns(pl.col("id").cast(pl.Utf8).alias("id_str")).select(
        ["id_str", "emb_row"]
    )
    df = eval_df.with_columns(pl.col("id").cast(pl.Utf8).alias("id_str")).join(
        meta_by_id, on="id_str", how="left"
    )
    n_missing = df.filter(pl.col("emb_row").is_null()).height
    df = df.filter(pl.col("emb_row").is_not_null())
    print(f"[frozen] {eval_df.height} coded, {n_missing} missing embeddings, "
          f"{df.height} scored")  # LOG

    features = cls[df["emb_row"].to_numpy().astype(int)]
    logits = apply_relevance_model(features, weights_path=FROZEN_WEIGHTS)
    cal = load_calibration(calibration_path_for_weights(FROZEN_WEIGHTS))
    p = cal.transform(logits)
    if not np.isfinite(p).all():
        raise ValueError("non-finite frozen-head probabilities")
    return df, p


def score_text_model(eval_df: pl.DataFrame, weights_path: Path) -> np.ndarray:
    """Score `eval_df` with the text-mode rel artifact (Pattern 2 reload).

    Fresh backbone + rel head, loaded by structure from `weights_path` + its
    `cca_config.RunConfig` sidecar -- the token-mode analog of
    `apply_relevance_model` (features-mode) / `apply_us_model` (Pattern 2 for
    `UsRunConfig`). Returns RAW LOGITS (no calibration sidecar exists yet for
    this artifact).
    """
    run_config = cca_config.RunConfig.from_json(cca_config.config_path_for_weights(weights_path))
    head_cfg = run_config.heads[0]

    # Backbone path comes from the PLATFORM-RESOLVED config constant, NOT the
    # artifact's sidecar: sidecars written on the cluster record the cluster's
    # absolute path (/projects/ahd/...), which doesn't exist locally. The DAPT
    # backbone is the same frozen artifact on every platform -- the sidecar's
    # value is provenance, not a load instruction.
    backbone = load_dapt_backbone(config.DAPT_BACKBONE_WEIGHTS)
    run_config.validate_against_backbone(backbone)

    head = ClassificationHead(hidden_dim=head_cfg.hidden_dim, name=head_cfg.name)
    inference_model = build_inference_model(
        backbone=backbone, heads={head_cfg.name: head}, seq_length=run_config.seq_length
    )
    inference_model.load_weights(str(weights_path), skip_mismatch=False)

    preproc = ClassifierPreprocessor(
        SEQ_LENGTH=run_config.seq_length,
        text_key=run_config.text_key,
        label_keys={},
        endpoint_model=True,
        target_dtype=run_config.target_dtype,
    )
    texts = _build_texts(eval_df)
    batch = preproc({run_config.text_key: texts})

    logits = inference_model.predict(batch, batch_size=256, verbose=0)
    if isinstance(logits, dict):
        logits = logits[head_cfg.name]
    logits = np.asarray(logits).reshape(-1)
    if not np.isfinite(logits).all():
        raise ValueError("non-finite text-mode logits")
    return logits


def _model_report(name: str, eval_df: pl.DataFrame, scores: np.ndarray) -> dict:
    """Assemble own-terms / vs-ICA / diaspora-slice metrics for one model's scores."""
    y_rel_raw = eval_df["immig_relevant"].to_numpy()
    rel_mask = ~pl.Series(y_rel_raw).is_null().to_numpy()
    own_terms = own_terms_metrics(scores[rel_mask], y_rel_raw[rel_mask].astype(bool))

    ica_raw = eval_df["ica_event"].to_numpy()
    ica_mask = ~pl.Series(ica_raw).is_null().to_numpy()
    roc_vs_ica = float(roc_auc_score(ica_raw[ica_mask].astype(bool), scores[ica_mask]))

    # fill_null(False) before to_numpy(): event_type has nulls, and a nullable
    # Boolean Series' comparison result converts to an OBJECT numpy array (not
    # bool) when nulls survive -- that fails boolean-mask indexing below.
    diaspora_mask = (eval_df["event_type"] == "Diasporic").fill_null(False).to_numpy()
    diaspora = diaspora_slice_metrics(scores, scores[diaspora_mask])

    return {
        "model": name,
        "n_scored": int(eval_df.height),
        "own_terms_vs_immig_relevant": own_terms,
        "roc_auc_vs_ica_event": roc_vs_ica,
        "diaspora_slice": diaspora,
    }


def _print_report(r: dict) -> None:
    ot = r["own_terms_vs_immig_relevant"]
    print(f"  n_scored={r['n_scored']}  "
          f"own-terms ROC={ot['roc_auc']:.3f} PR-AUC={ot['pr_auc']:.3f} "
          f"(n={ot['n']} base_rate={ot['base_rate']:.3f})  "
          f"vs-ICA ROC={r['roc_auc_vs_ica_event']:.3f}")  # LOG
    d = r["diaspora_slice"]
    print(f"  diaspora slice: n={d['n_diaspora']} mean_score={d['mean_score']:.4f} "
          f"median_score={d['median_score']:.4f}")  # LOG
    for pt in d["matched_rate_points"]:
        print(f"    matched positive-rate~{pt['target_positive_rate']:.2f} "
              f"(achieved={pt['achieved_positive_rate_overall']:.3f}): "
              f"diaspora_recall={pt['diaspora_recall']:.3f}")  # LOG


def main(weights_path: Path | str = DEFAULT_TEXT_WEIGHTS) -> dict:
    weights_path = Path(weights_path)
    eval_df = pl.read_csv(EVAL_CSV)
    n_raw = eval_df.height

    results: dict = {
        "eval_csv": str(EVAL_CSV),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_coded_rows": n_raw,
        "frozen_weights": str(FROZEN_WEIGHTS),
        "text_weights": str(weights_path),
        "note": (
            "Frozen head scores are Platt-calibrated probabilities; text-mode "
            "scores are RAW LOGITS (no calibration sidecar yet, deferred to "
            "swap phase). ROC-AUC/PR-AUC are scale-invariant across this "
            "mismatch; the diaspora-slice comparison uses rank-based "
            "matched-positive-rate thresholds, also scale-invariant. Absolute "
            "score values (mean_score) are NOT comparable across models."
        ),
    }

    print("[1/2] Scoring with the FROZEN features-mode rel head...")  # LOG
    frozen_df, p_frozen = score_frozen_head(eval_df)
    results["frozen"] = _model_report("frozen_features_mode", frozen_df, p_frozen)

    if not weights_path.is_file():
        sidecar_path = cca_config.config_path_for_weights(weights_path)
        rsync_hint = (
            f"rsync -avz <cluster_host>:<project_root>/relevance/"
            f"{weights_path.name} {weights_path}"
        )
        msg = (
            f"text-mode weights not found at {weights_path} -- the operator may not "
            f"have transferred it yet. To fetch it: {rsync_hint} "
            f"(and the matching sidecar, {sidecar_path.name}). "
            "Reporting the frozen-baseline side only."
        )
        print(f"[2/2] SKIPPED: {msg}")  # LOG
        results["text"] = None
        results["text_skipped_reason"] = msg
    else:
        print("[2/2] Scoring with the TEXT-MODE rel artifact...")  # LOG
        p_text = score_text_model(eval_df, weights_path)
        results["text"] = _model_report("text_mode", eval_df, p_text)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH}\n")  # LOG

    print("=== frozen (features-mode, calibrated) ===")
    _print_report(results["frozen"])
    if results["text"] is not None:
        print("\n=== text-mode (raw logits, uncalibrated) ===")
        _print_report(results["text"])
    else:
        print("\n=== text-mode: SKIPPED (see text_skipped_reason in the JSON) ===")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Validate-before-swap eval: text-mode rel artifact vs. the frozen features-mode rel head."
    )
    ap.add_argument("--weights", type=str, default=str(DEFAULT_TEXT_WEIGHTS),
                     help="path to the text-mode rel .weights.h5 to evaluate "
                          "(its .config.json sidecar must sit alongside it)")
    args = ap.parse_args()
    main(weights_path=args.weights)
