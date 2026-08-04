# pattern: Imperative Shell
"""Quantify the tensorflow-metal scoring bug's effect on the eval metrics (2026-08-04).

Runs the own-terms scoring recipe (mirrors scripts/eval_heads_own_terms.py:
relevance_train cache features; calibrated CCA/rel features-mode; US text-mode)
plus the composed gated product, twice:

  - device=default : whatever the platform gives (local MPS = the DISTORTED
                     path that produced the memo numbers)
  - device=cpu     : GPU hidden before any op initializes (EXACT math)

and prints per-head own-terms ROC/PR, vs-ICA ROC, and composed ICA ROC/PR
side by side with deltas. The composed score uses the fusion sidecar's gate
threshold and raw product (composed-Platt omitted: monotone, rank-invariant).

Usage (from project root):
  uv run python -m scripts.compare_scoring_paths              # orchestrates both
  uv run python -m scripts.compare_scoring_paths --device cpu --out /tmp/x.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def score_once(device: str, out_path: Path) -> None:
    """Single scoring run on the chosen device; writes metrics JSON."""
    import tensorflow as tf

    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")

    import numpy as np
    import polars as pl
    from sklearn.metrics import average_precision_score, roc_auc_score

    import src.config as config
    from src.calibration.sidecar import calibration_path_for_weights, load_calibration
    from src.embed_corpus import load_cache
    from src.fusion.sidecar import load_fusion
    from src.validation.cca_slice_eval import apply_cca_model
    from src.validation.relevance_slice_eval import apply_relevance_model
    from src.validation.slice_eval import apply_us_model

    eval_df = pl.read_csv(config.PROJECT_ROOT / "validation" / "ica_coding_template_coded.csv")
    meta_full, cls = load_cache(config.CCA_EMBED_CACHE_DIR / "relevance_train")
    meta_by_id = meta_full.with_columns(
        pl.col("id").cast(pl.Utf8).alias("id_str")
    ).select(["id_str", "emb_row"])
    eval_df = eval_df.with_columns(pl.col("id").cast(pl.Utf8).alias("id_str")).join(
        meta_by_id, on="id_str", how="left"
    ).filter(pl.col("emb_row").is_not_null())
    features = cls[eval_df["emb_row"].to_numpy().astype(int)]

    cca_cal = load_calibration(calibration_path_for_weights(config.CCA_DOCA_WEIGHTS))
    p_cca = cca_cal.transform(apply_cca_model(features, weights_path=config.CCA_DOCA_WEIGHTS))
    rel_cal = load_calibration(calibration_path_for_weights(config.RELEVANCE_DOCA_WEIGHTS))
    p_rel = rel_cal.transform(
        apply_relevance_model(features, weights_path=config.RELEVANCE_DOCA_WEIGHTS)
    )
    texts = [
        f"{h if h else ''}</s>{lead if lead else ''}"
        for h, lead in zip(eval_df["headline"].to_list(), eval_df["lead_paragraph"].to_list())
    ]
    p_us = apply_us_model(texts, weights_path=config.US_FILTER_FULL_WEIGHTS, skip_mismatch=True)

    fusion = load_fusion(str(config.CCA_DOCA_DIR / "ica_fusion.fusion.json"))
    composed = np.where(p_us >= fusion.gate_threshold, p_cca * p_rel, 0.0)

    out: dict = {"device": device, "heads": {}, "composed": {}}
    for name, p, label_col in [
        ("us", p_us, "us_event"),
        ("cca", p_cca, "cca_event"),
        ("rel", p_rel, "immig_relevant"),
    ]:
        y_raw = eval_df[label_col].to_numpy()
        mask = ~pl.Series(y_raw).is_null().to_numpy()
        y = y_raw[mask].astype(bool)
        ica_raw = eval_df["ica_event"].to_numpy()
        ica_mask = mask & ~pl.Series(ica_raw).is_null().to_numpy()
        out["heads"][name] = {
            "roc_auc": float(roc_auc_score(y, p[mask])),
            "pr_auc": float(average_precision_score(y, p[mask])),
            "roc_auc_vs_ica": float(
                roc_auc_score(ica_raw[ica_mask].astype(bool), p[ica_mask])
            ),
        }
    ica_raw = eval_df["ica_event"].to_numpy()
    ica_mask = ~pl.Series(ica_raw).is_null().to_numpy()
    y_ica = ica_raw[ica_mask].astype(bool)
    out["composed"] = {
        "roc_auc": float(roc_auc_score(y_ica, composed[ica_mask])),
        "pr_auc": float(average_precision_score(y_ica, composed[ica_mask])),
        "gated_in_frac": float((p_us >= fusion.gate_threshold).mean()),
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[{device}] wrote {out_path}")


def orchestrate() -> None:
    """Run both devices in fresh subprocesses (device choice must precede tf init)."""
    results = {}
    with tempfile.TemporaryDirectory() as td:
        for device in ["default", "cpu"]:
            out = Path(td) / f"{device}.json"
            subprocess.run(
                [sys.executable, "-m", "scripts.compare_scoring_paths",
                 "--device", device, "--out", str(out)],
                check=True,
            )
            results[device] = json.loads(out.read_text())

    a, b = results["default"], results["cpu"]
    print("\nmetric                         default(MPS)   cpu(true)    delta")
    print("-" * 66)
    for name in ["us", "cca", "rel"]:
        for metric in ["roc_auc", "pr_auc", "roc_auc_vs_ica"]:
            va, vb = a["heads"][name][metric], b["heads"][name][metric]
            print(f"{name+' '+metric:<30} {va:>10.4f} {vb:>11.4f} {vb-va:>+9.4f}")
    for metric in ["roc_auc", "pr_auc", "gated_in_frac"]:
        va, vb = a["composed"][metric], b["composed"][metric]
        print(f"{'composed '+metric:<30} {va:>10.4f} {vb:>11.4f} {vb-va:>+9.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["default", "cpu"], default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.device is None:
        orchestrate()
    else:
        score_once(args.device, Path(args.out))
