# pattern: Imperative Shell
"""Score a full embedding cache with a CCA model and report the threshold-count
distribution over the US-restricted population (the yield-projection input).

For the characterization doc's "what would we get" section: how many articles a
given CCA model flags at each logit threshold, over the US-restricted corpus the
model expects. Multiply a count by that threshold's gold reweighted precision to
project expected true events.

Run from project root:
    uv run python -m scripts.cca_corpus_scores --suffix full --weights <path> \
        --out cca_doca/experiments/corpus_scores_<tag>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import src.config as config
from src.embed_corpus import load_cache
from src.validation.cca_slice_eval import apply_cca_model

_THRESHOLDS = (0.0, 0.5, 1.0, 1.5, 2.0)


def main(suffix: str, weights: str, out: str, us_threshold: float = 0.0) -> None:
    meta, cls = load_cache(config.CCA_EMBED_CACHE_DIR / suffix)
    n_total = meta.height
    us_mask = (meta["us_logit"] >= us_threshold).to_numpy()
    n_us = int(us_mask.sum())
    feats = cls[us_mask]
    logits = apply_cca_model(feats, Path(weights))

    dist = {
        str(t): {"n_flagged": int((logits >= t).sum()),
                 "frac_of_us": float((logits >= t).mean())}
        for t in _THRESHOLDS
    }
    record = {
        "suffix": suffix, "weights": str(weights), "us_threshold": us_threshold,
        "n_total": n_total, "n_us_restricted": n_us,
        "us_fraction": n_us / n_total if n_total else 0.0,
        "threshold_counts": dist,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(record, indent=2))
    print(f"n_total={n_total}  US-restricted={n_us} ({100*n_us/max(n_total,1):.1f}%)")  # LOG
    for t in _THRESHOLDS:
        d = dist[str(t)]
        print(f"  logit>={t}: flagged={d['n_flagged']:>8}  ({100*d['frac_of_us']:.2f}% of US)")  # LOG
    print(f"Wrote {out}")  # LOG


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Corpus score distribution for a CCA model.")
    ap.add_argument("--suffix", default="full", help="embedding cache subdir")
    ap.add_argument("--weights", required=True, help="CCA model weights .h5")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--us-threshold", type=float, default=0.0)
    args = ap.parse_args()
    main(suffix=args.suffix, weights=args.weights, out=args.out,
         us_threshold=args.us_threshold)
