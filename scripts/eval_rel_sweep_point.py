# pattern: Imperative Shell
"""Score one text-mode rel artifact on the gold eval set -> compact JSON.

The per-point scorer for the stage-3 depth sweep (branched-encoder ladder,
`docs/notes/branched-encoder-strategy.md`): each cluster sweep job trains a
text-mode rel artifact, then runs this to emit rank metrics (own-terms,
vs-ICA, diaspora matched-rate points) plus sweep provenance as a small JSON —
selection happens over JSONs; only the winning artifact travels.

Unlike `eval_rel_text_artifact.py` (fixed output path, frozen-head comparison
baked in), this takes explicit --weights/--out and scores exactly one
artifact. Scoring is CPU-forced by default on every platform: rank metrics
must be execution-portable, and the 1,131-row eval costs ~2 min on CPU
(`docs/notes/metal-execution-findings.md` deployment rules).

Run from the repo root:
    uv run python -m scripts.eval_rel_sweep_point \
        --weights ../relevance/sweep/rel_text_N2_graded_s200.weights.h5 \
        --out ../relevance/sweep/rel_text_N2_graded_s200.eval.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main(weights: str, out: str, allow_gpu: bool = False) -> dict:
    import tensorflow as tf

    if not allow_gpu:
        tf.config.set_visible_devices([], "GPU")

    import polars as pl

    import src.cca_config as cca_config
    import src.config as config
    from scripts.eval_rel_text_artifact import _model_report, score_text_model

    weights_path = Path(weights)
    out_path = Path(out)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {out_path}")

    eval_csv = config.PROJECT_ROOT / "validation" / "ica_coding_template_coded.csv"
    eval_df = pl.read_csv(eval_csv, infer_schema_length=None)

    scores = score_text_model(eval_df, weights_path)
    report = _model_report(weights_path.stem, eval_df, scores)

    sidecar = json.loads(
        Path(cca_config.config_path_for_weights(weights_path)).read_text()
    )
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": str(weights_path),
        "eval_csv": str(eval_csv),
        "n_rows": eval_df.height,
        "cpu_forced": not allow_gpu,
        "sweep_params": {
            k: sidecar.get(k)
            for k in ("unfreeze_top_n", "layer_multipliers", "seed", "hard_freeze", "epochs")
        },
        "head_loss": sidecar.get("heads", [{}])[0].get("loss"),
        "report": report,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")  # LOG
    print(
        f"own-terms ROC {report['own_terms_vs_immig_relevant']['roc_auc']:.4f}  "
        f"vs-ICA ROC {report['roc_auc_vs_ica_event']:.4f}"
    )  # LOG
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score one text-mode rel artifact (sweep point).")
    ap.add_argument("--weights", required=True, help="text-mode rel .weights.h5")
    ap.add_argument("--out", required=True, help="output JSON path (must not exist)")
    ap.add_argument(
        "--allow-gpu",
        action="store_true",
        help="skip the CPU-force (rank metrics must be execution-portable; "
        "leave unset unless you know why)",
    )
    args = ap.parse_args()
    main(args.weights, args.out, allow_gpu=args.allow_gpu)
