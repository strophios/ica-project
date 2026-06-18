# pattern: Imperative Shell
"""Out-of-sample CCA eval: score LDC 1995-2007 against the NYT cca_descriptor tag.

The model trained on the API corpus 1960-1995 (DoCA-confirmed positives). This scores
it on the LDC corpus 1995-2007 -- a LATER period AND a DIFFERENT source -- to test
temporal + cross-source generalization.

The reference label is the NYT `cca_descriptor` tag, which is NOISY and over-generous
(it's exactly what the DoCA-matching was meant to improve on). So the honest read is
rank-based: ROC-AUC / PR-AUC of the model score vs the descriptor measure whether the
model's ORDERING still tracks collective-action signal out of period -- a "does it fall
apart" check, not an accuracy. Raw precision/recall vs the descriptor conflate model
errors with descriptor errors and are reported only for context.

Run from project root (after embedding ldc_9507):
    uv run python -m src.validation.cca_oos_eval
    uv run python -m src.validation.cca_oos_eval --weights ../cca_doca/cca_doca_street.weights.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

import src.config as config
from src.embed_corpus import load_cache
from src.validation.cca_slice_eval import apply_cca_model


def _load_reference() -> pl.DataFrame:
    """LDC corpus reference: id, headline, cca_descriptor. id cast to str for a
    dtype-safe join against the cache (the API/LDC id-dtype split has bitten before)."""
    ref = pl.scan_parquet(
        f"{config.PROJECT_ROOT}/ldc_corpus/**/*.parquet", hive_partitioning=True
    ).select(["id", "headline", "cca_descriptor"]).collect()
    # null cca_descriptor = untagged = not a descriptor-positive (negative reference).
    return ref.with_columns(
        pl.col("id").cast(pl.Utf8),
        pl.col("cca_descriptor").fill_null(False),
    )


def main(weights_path=None, suffix="ldc_9507", top_k=20):
    weights_path = config.CCA_DOCA_WEIGHTS if weights_path is None else Path(weights_path)
    meta, cls = load_cache(config.CCA_EMBED_CACHE_DIR / suffix)
    logits = apply_cca_model(cls, weights_path)
    scored = (
        meta.select(["id", "year"])
        .with_columns(pl.col("id").cast(pl.Utf8), pl.Series("cca_logit", logits))
    )
    df = scored.join(_load_reference(), on="id", how="inner")
    print(f"weights={weights_path.name}  scored={scored.height}  joined={df.height}")  # LOG

    y = df["cca_descriptor"].cast(pl.Int8).to_numpy()
    s = df["cca_logit"].to_numpy()
    n_pos = int(y.sum())
    print(f"descriptor positives: {n_pos} ({100 * y.mean():.2f}% of joined)")  # LOG
    print(f"ROC-AUC = {roc_auc_score(y, s):.4f}   "
          f"PR-AUC = {average_precision_score(y, s):.4f}  "
          f"(base rate {y.mean():.4f})")  # LOG
    print(f"median cca_logit:  descriptor+ = {np.median(s[y == 1]):+.2f}   "
          f"descriptor- = {np.median(s[y == 0]):+.2f}")  # LOG

    print("\n  thr | flagged | precision-vs-desc | recall-vs-desc")  # LOG
    for t in (0.0, 1.0, 1.5, 2.0):
        flagged = s >= t
        nf = int(flagged.sum())
        prec = float((y[flagged] == 1).mean()) if nf else 0.0
        rec = float((flagged & (y == 1)).sum() / max(1, n_pos))
        print(f"  {t:>3} | {nf:>7} | {prec:>17.3f} | {rec:>14.3f}")  # LOG

    # Demo material: the highest-scored articles (descriptor status in parens). A
    # high-scoring descriptor-negative is a candidate event the over-generous tag
    # still missed -- or a false positive; read the headlines, don't trust the tag.
    top = df.sort("cca_logit", descending=True).head(top_k)
    print(f"\nTop-{top_k} highest-scored (desc = NYT tag):")  # LOG
    for r in top.iter_rows(named=True):
        hl = (r["headline"] or "")[:88]
        print(f"  [{r['year']}] desc={str(r['cca_descriptor']):<5} "
              f"logit={r['cca_logit']:+.2f}  {hl}")  # LOG


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Out-of-sample CCA eval on LDC 1995-2007.")
    ap.add_argument("--weights", default=None, help="CCA weights (default: CCA_DOCA_WEIGHTS)")
    ap.add_argument("--suffix", default="ldc_9507", help="embedding cache subdir")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()
    main(weights_path=args.weights, suffix=args.suffix, top_k=args.top_k)
