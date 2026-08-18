# pattern: Imperative Shell (extract_sweep_params / natural_balance_val_population /
#   compose_scores are the pure Functional Core; everything else is I/O -- model
#   build, cache/CSV reads, sidecar parsing)
"""Score one joint CCA+rel text-mode artifact -> compact JSON.

The per-cell scorer for the stage-4 joint fine-tune sweep
(docs/design-plans/2026-08-18-stage4-joint-finetune.md, Components item 4):
each cluster sweep job trains a joint two-head text-mode artifact
(`src/run_joint_text.py`), then runs this to fit Platt calibrators on the
artifact's own val split and report the composed-product proxy against the
gold ICA eval set -- selection happens over these JSONs; only the winning
artifact travels.

CPU-forced by default on every platform (`--allow-gpu` opts out): calibration
is NOT rank-invariant for the composed product, so both the Platt-fit inputs
(the val-split logits) and the gold-set logits must come from the SAME
execution path (`docs/notes/metal-execution-findings.md` deployment rules --
the same rule `eval_rel_sweep_point.py` follows for rank metrics).

Pipeline:
  1. Rebuild the two-head inference model Pattern-2 from the artifact +
     its `.config.json` sidecar (fresh backbone + fresh `cca`/`rel` heads,
     weights loaded by structure -- the two-head analog of
     `eval_rel_text_artifact.score_text_model`).
  2. Score the joint table's VAL split at NATURAL BALANCE (recomputed via
     `src.data_setup.data.create_joint_text_data`, the same split the
     trainer used -- no Ratio-Batch stream sampling). Fit Platt per head:
     cca vs `cca_label`; rel vs `rel_label==1` EXCLUDING `reliable_neg` rows
     (those carry `rel_label=0` by construction but are a different
     population than ordinary unlabeled background -- fitting on them would
     bias B toward an artificially foreign-heavy negative pool).
  3. Score the 1,131-row gold eval CSV -> calibrated `p_cca`, `p_rel` ->
     composed = `p_cca * p_rel`.
  4. Metrics to JSON: composed ROC vs `ica_event` (null-masked) + diaspora
     recall @0.30/@0.10 as PRIMARY; per-head own-terms ROC (cca vs
     `cca_event`, rel vs `immig_relevant`) as guardrails; rel-solo vs-ICA ROC
     as a diagnostic; sweep params from the sidecar; calibration A/B per
     head; n's; generated_utc; cpu_forced flag.

Run from the repo root:
    uv run python -m scripts.eval_joint_sweep_point \
        --weights ../relevance/joint_sweep/joint_N1_lam050_s200.weights.h5 \
        --out ../relevance/joint_sweep/joint_N1_lam050_s200.eval.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Functional core
# ---------------------------------------------------------------------------
def extract_sweep_params(sidecar: dict) -> dict:
    """Pure: pull the sweep-relevant provenance fields out of a joint
    trainer's `.config.json` sidecar dict.

    `heads` is looked up by name ("cca"/"rel") rather than positionally --
    `RunConfig.heads` preserves declaration order (cca, rel per
    `_default_joint_text_config`), but keying by name is robust to that
    ever changing. Missing fields (a malformed/partial sidecar, or the
    "no heads" edge case exercised by tests) resolve to `None` rather than
    raising -- this is a reporting helper, not a validator.
    """
    heads_by_name = {h.get("name"): h for h in sidecar.get("heads", [])}
    cca = heads_by_name.get("cca", {})
    rel = heads_by_name.get("rel", {})
    return {
        "unfreeze_top_n": sidecar.get("unfreeze_top_n"),
        "layer_multipliers": sidecar.get("layer_multipliers"),
        "hard_freeze": sidecar.get("hard_freeze"),
        "seed": sidecar.get("seed"),
        "epochs": sidecar.get("epochs"),
        "cca_loss_weight": cca.get("loss_weight"),
        "rel_loss_weight": rel.get("loss_weight"),
    }


def natural_balance_val_population(splits: dict) -> pl.DataFrame:
    """Pure: reconstruct the val split's natural-balance population from
    `create_joint_text_data`'s grouped `{"cca_pos", "rel_pos", "unl"}` output.

    A row positive for both heads appears in BOTH `cca_pos` and `rel_pos`
    (the deliberate Ratio-Batch overlap, see `create_joint_text_data`'s
    docstring); natural-balance scoring wants each val row counted exactly
    once, so the three group frames are concatenated and deduped by `id`
    (`keep="first"` -- arbitrary but deterministic, since a duplicated row's
    other columns are identical across its two group appearances).
    """
    val = splits["val"]
    union = pl.concat([val["cca_pos"], val["rel_pos"], val["unl"]])
    return union.unique(subset="id", keep="first", maintain_order=True)


def compose_scores(p_cca: np.ndarray, p_rel: np.ndarray) -> np.ndarray:
    """Pure: the composed ICA proxy score, elementwise `p_cca * p_rel`."""
    return np.asarray(p_cca) * np.asarray(p_rel)


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------
def main(weights: str, out: str, allow_gpu: bool = False) -> dict:
    import tensorflow as tf

    if not allow_gpu:
        tf.config.set_visible_devices([], "GPU")

    from scripts.eval_rel_text_artifact import (
        _build_texts,
        diaspora_slice_metrics,
        own_terms_metrics,
    )
    import src.cca_config as cca_config
    import src.config as config
    from src.calibration.calibrator import PlattCalibrator
    from src.data_setup.data import create_joint_text_data
    from src.model_setup.assembly import build_inference_model
    from src.model_setup.backbone import load_dapt_backbone
    from src.model_setup.heads import ClassificationHead
    from src.preproc.preprocessor import ClassifierPreprocessor
    from src.run_joint_text import derive_rel_target

    weights_path = Path(weights)
    out_path = Path(out)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {out_path}")

    sidecar_path = cca_config.config_path_for_weights(weights_path)
    sidecar = json.loads(Path(sidecar_path).read_text())
    run_config = cca_config.RunConfig.from_json(sidecar_path)
    if run_config.head_names != ("cca", "rel"):
        raise ValueError(
            f"expected a two-head (cca, rel) joint artifact; sidecar declares "
            f"heads {run_config.head_names}"
        )
    cca_head_cfg, rel_head_cfg = run_config.heads

    # --- Pattern 2 reload: fresh backbone + fresh heads, weights by structure ---
    # Backbone path comes from the PLATFORM-RESOLVED config constant, NOT the
    # sidecar (a cluster-written sidecar records a cluster-only path) -- same
    # convention as eval_rel_text_artifact.score_text_model.
    backbone = load_dapt_backbone(config.DAPT_BACKBONE_WEIGHTS)
    run_config.validate_against_backbone(backbone)
    heads = {
        cca_head_cfg.name: ClassificationHead(hidden_dim=cca_head_cfg.hidden_dim, name=cca_head_cfg.name),
        rel_head_cfg.name: ClassificationHead(hidden_dim=rel_head_cfg.hidden_dim, name=rel_head_cfg.name),
    }
    inference_model = build_inference_model(
        backbone=backbone, heads=heads, seq_length=run_config.seq_length
    )
    inference_model.load_weights(str(weights_path), skip_mismatch=False)

    preproc = ClassifierPreprocessor(
        SEQ_LENGTH=run_config.seq_length,
        text_key=run_config.text_key,
        label_keys={},
        endpoint_model=True,
        target_dtype=run_config.target_dtype,
    )

    def _score(texts: list[str]) -> dict[str, np.ndarray]:
        batch = preproc({run_config.text_key: texts})
        logits = inference_model.predict(batch, batch_size=256, verbose=0)
        scored = {}
        for name in (cca_head_cfg.name, rel_head_cfg.name):
            arr = np.asarray(logits[name]).reshape(-1)
            if not np.isfinite(arr).all():
                raise ValueError(f"non-finite {name!r} logits")
            scored[name] = arr
        return scored

    # --- val split (natural balance, no Ratio-Batch sampling) + Platt fit ---
    print("[1/3] Recomputing the val split + scoring it...")  # LOG
    table = pl.read_parquet(config.JOINT_TEXT_TABLE)
    table = derive_rel_target(table)
    holdout_ids = pl.read_parquet(config.ICA_HOLDOUT_IDS)["id"].to_list()
    splits = create_joint_text_data(table, holdout_ids=holdout_ids)
    val_df = natural_balance_val_population(splits)

    val_logits = _score(val_df["headline_with_lead"].to_list())
    cal_cca = PlattCalibrator.fit(
        val_logits["cca"], val_df["cca_label"].to_numpy(),
        fit_population="joint_val_natural_balance",
    )
    rel_fit_mask = ~val_df["reliable_neg"].to_numpy()
    cal_rel = PlattCalibrator.fit(
        val_logits["rel"][rel_fit_mask],
        (val_df["rel_label"].to_numpy() == 1)[rel_fit_mask],
        fit_population="joint_val_natural_balance_excl_reliable_neg",
    )
    print(f"  val n={val_df.height}  cca calib A={cal_cca.A:.4f} B={cal_cca.B:.4f}  "
          f"rel calib A={cal_rel.A:.4f} B={cal_rel.B:.4f} (n={cal_rel.n})")  # LOG

    # --- gold eval CSV: score, calibrate, compose ---
    print("[2/3] Scoring the gold ICA eval set...")  # LOG
    eval_csv = config.PROJECT_ROOT / "validation" / "ica_coding_template_coded.csv"
    eval_df = pl.read_csv(eval_csv, infer_schema_length=None)
    texts = _build_texts(eval_df)
    gold_logits = _score(texts)
    p_cca = cal_cca.transform(gold_logits["cca"])
    p_rel = cal_rel.transform(gold_logits["rel"])
    composed = compose_scores(p_cca, p_rel)

    # --- metrics ---
    print("[3/3] Computing metrics...")  # LOG

    def _masked(col_name):
        raw = eval_df[col_name].to_numpy()
        mask = ~pl.Series(raw).is_null().to_numpy()
        return raw[mask].astype(bool), mask

    ica_bool, ica_mask = _masked("ica_event")
    composed_vs_ica = own_terms_metrics(composed[ica_mask], ica_bool)

    diaspora_mask = (eval_df["event_type"] == "Diasporic").fill_null(False).to_numpy()
    diaspora = diaspora_slice_metrics(composed, composed[diaspora_mask], positive_rates=(0.30, 0.10))

    cca_bool, cca_mask = _masked("cca_event")
    cca_own_terms = own_terms_metrics(p_cca[cca_mask], cca_bool)

    rel_bool, rel_mask = _masked("immig_relevant")
    rel_own_terms = own_terms_metrics(p_rel[rel_mask], rel_bool)

    rel_solo_vs_ica = own_terms_metrics(p_rel[ica_mask], ica_bool)

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": str(weights_path),
        "eval_csv": str(eval_csv),
        "n_eval_rows": eval_df.height,
        "n_val_rows": val_df.height,
        "cpu_forced": not allow_gpu,
        "sweep_params": extract_sweep_params(sidecar),
        "calibration": {
            "cca": {"A": cal_cca.A, "B": cal_cca.B, "n": cal_cca.n,
                    "fit_population": cal_cca.fit_population},
            "rel": {"A": cal_rel.A, "B": cal_rel.B, "n": cal_rel.n,
                    "fit_population": cal_rel.fit_population},
        },
        # PRIMARY
        "composed_vs_ica_event": composed_vs_ica,
        "diaspora_slice_composed": diaspora,
        # Guardrails
        "own_terms_guardrails": {
            "cca_vs_cca_event": cca_own_terms,
            "rel_vs_immig_relevant": rel_own_terms,
        },
        # Diagnostic
        "rel_solo_vs_ica_event": rel_solo_vs_ica,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")  # LOG
    print(
        f"composed ROC {composed_vs_ica['roc_auc']:.4f}  "
        f"cca own-terms ROC {cca_own_terms['roc_auc']:.4f}  "
        f"rel own-terms ROC {rel_own_terms['roc_auc']:.4f}  "
        f"rel-solo vs-ICA ROC {rel_solo_vs_ica['roc_auc']:.4f}"
    )  # LOG
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score one joint CCA+rel text-mode artifact (sweep point).")
    ap.add_argument("--weights", required=True, help="joint two-head text-mode .weights.h5")
    ap.add_argument("--out", required=True, help="output JSON path (must not exist)")
    ap.add_argument(
        "--allow-gpu",
        action="store_true",
        help="skip the CPU-force (calibration/rank metrics must be execution-portable; "
        "leave unset unless you know why)",
    )
    args = ap.parse_args()
    main(args.weights, args.out, allow_gpu=args.allow_gpu)
