# pattern: Imperative Shell
"""Evaluate the trained CCA model against the hand-coded gold set.

Re-scores the coded gold ids with the CURRENT model weights -- the template's
`cca_score` is stale (it came from a model that trained on these ids as noisy
negatives; the honest base model holds them out, so they must be re-scored) --
then reports BOTH:

  - RAW precision/recall (unweighted): directly comparable across experiments
    (the stratification bias is identical every run);
  - IPW-REWEIGHTED precision/recall: the corpus operating-point estimate, weighting
    each coded row by `corpus_band / gold_band` for its sampling stratum.

The gold set is score-band stratified, so raw recall (and raw precision below the
top band boundary) is biased high; the reweighted columns correct it but are
high-variance wherever a heavily-weighted band has few coded rows. Per-threshold
support counts are printed/recorded so unreliable cells are visible -- the two
estimates converge and both become trustworthy at high thresholds (the high band,
sampled densely, dominates there).

A self-contained run record is written to `CCA_DOCA_DIR/experiments/<run_id>.json`
(the seed of the experiment registry).

Run from project root (after coding cca_coding_first500_coded.csv):
    uv run python -m src.validation.run_cca_eval
    uv run python -m src.validation.run_cca_eval --coded <csv> --suffix train250k
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

import polars as pl

import numpy as np

import src.config as config
import src.cca_config as cca_config
from src.build_cca_doca_table import label_and_restrict
from src.data_setup.data import create_cca_doca_data
from src.embed_corpus import load_cache
from src.validation.build_cca_coding_template import assign_score_band
from src.validation.cca_slice_eval import (
    apply_cca_model,
    band_ipw_weights,
    evaluate_cca_slice,
    evaluate_cca_slice_weighted,
    recall_at_thresholds,
)

# Logit-space thresholds. 0.0 == prob 0.5; 1.0 is the high-band boundary, where the
# raw and reweighted estimates converge (high band is densely sampled).
_THRESHOLD_GRID = (-1.0, 0.0, 0.5, 1.0, 1.5, 2.0)
_EVENT_TYPE_THRESHOLDS = (0.0, 1.0)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _parse_cca_event(coded: pl.DataFrame) -> pl.DataFrame:
    """Coerce `cca_event` to Boolean with null for uncoded/NA (defense-in-depth).

    polars usually infers TRUE/FALSE as Boolean and NA as null, but the column is
    hand-edited, so normalize explicitly at this input boundary.
    """
    if coded.schema["cca_event"] == pl.Boolean:
        return coded
    up = pl.col("cca_event").cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    return coded.with_columns(
        pl.when(up == "TRUE").then(True)
        .when(up == "FALSE").then(False)
        .otherwise(None)
        .alias("cca_event")
    )


def _attach_gold_scores(
    coded: pl.DataFrame, meta: pl.DataFrame, cls: np.ndarray, weights_path: Path
) -> pl.DataFrame:
    """Attach a fresh `cca_logit` by scoring gold ids from the cached features.

    Joins gold ids to the cache's `emb_row`, gathers CLS vectors, and applies the
    current model. Replaces any stale `cca_logit`/`cca_score` from the template.
    """
    joined = coded.drop(
        [c for c in ("cca_logit", "cca_score", "emb_row") if c in coded.columns]
    ).join(meta.select("id", "emb_row"), on="id", how="left")

    missing = int(joined["emb_row"].null_count())
    if missing:
        raise ValueError(
            f"{missing} gold ids not found in cache; cannot re-score"
        )
    feats = cls[joined["emb_row"].to_numpy()]
    logits = apply_cca_model(feats, weights_path)
    return joined.with_columns(pl.Series("cca_logit", logits))


def _doca_test_recall(
    meta: pl.DataFrame, cls: np.ndarray, weights_path: Path,
    thresholds: tuple[float, ...], us_threshold: float = 0.0,
) -> list[dict[str, float | int]]:
    """Recall over the held-out test-split DoCA positives (trustworthy recall proxy).

    Reconstructs the deterministic 90/5/5 split (seed=200) and scores the TEST
    positives -- never trained on, so leakage-honest. The positive split is
    independent of the gold-set holdout (held-out ids are all unlabeled), so no
    holdout is passed here. Recall = fraction of confirmed CCA events scored above
    each logit threshold.
    """
    positives = pl.read_parquet(config.CCA_DOCA_POSITIVES)["id"].to_list()
    table = label_and_restrict(meta, positives, us_threshold)
    splits = create_cca_doca_data(table)
    test_pos = splits["test"]["pos"]
    feats = cls[test_pos["emb_row"].to_numpy()]
    logits = apply_cca_model(feats, weights_path)
    return recall_at_thresholds(logits, thresholds)


def _corpus_band_counts() -> dict[str, int]:
    """Band the scored unlabeled background (the population the gold set sampled)."""
    scored = pl.read_parquet(config.CCA_DOCA_DIR / "scored_candidates.parquet")
    banded = (
        scored.filter(pl.col("cca_label") == 0)
        .with_columns(assign_score_band(pl.col("cca_logit")).alias("_band"))
        .group_by("_band")
        .len()
    )
    return dict(zip(banded["_band"].to_list(), banded["len"].to_list()))


def _gold_band_counts(coded: pl.DataFrame) -> dict[str, int]:
    """Coded rows (non-null label) per sampling stratum -- the IPW denominator."""
    g = (
        coded.filter(pl.col("cca_event").is_not_null())
        .group_by("sample_stratum")
        .len()
    )
    return dict(zip(g["sample_stratum"].to_list(), g["len"].to_list()))


def _per_event_type_recall(
    coded: pl.DataFrame, threshold: float, score_col: str = "cca_logit"
) -> dict[str, dict[str, float | int]]:
    """Recall within each coded `event_type` (of true positives, fraction caught)."""
    pos = coded.filter(pl.col("cca_event") == True)  # noqa: E712 (polars mask)
    out: dict[str, dict[str, float | int]] = {}
    grp = pos.group_by("event_type").agg(
        pl.len().alias("n"),
        (pl.col(score_col) >= threshold).sum().alias("caught"),
    )
    for row in grp.iter_rows(named=True):
        etype = row["event_type"] if row["event_type"] is not None else "NA"
        n, caught = int(row["n"]), int(row["caught"])
        out[etype] = {"n": n, "caught": caught,
                      "recall": (caught / n) if n else 0.0}
    return out


def run_eval(
    coded_path: Path, suffix: str, weights_path: Path
) -> dict:
    """Full eval -> a run-record dict (raw + IPW metrics, per-type, provenance)."""
    meta, cls = load_cache(config.CCA_EMBED_CACHE_DIR / suffix)
    coded = _parse_cca_event(pl.read_csv(coded_path))
    coded = _attach_gold_scores(coded, meta, cls, Path(weights_path))

    gold_counts = _gold_band_counts(coded)
    corpus_counts = _corpus_band_counts()
    weights = band_ipw_weights(gold_counts, corpus_counts)
    coded = coded.with_columns(
        pl.col("sample_stratum").replace_strict(weights).alias("ipw")
    )

    sweep = []
    for thr in _THRESHOLD_GRID:
        raw = evaluate_cca_slice(coded, threshold=thr)
        wtd = evaluate_cca_slice_weighted(coded, threshold=thr)
        sweep.append({"threshold": thr, "raw": raw, "reweighted": wtd})

    per_type = {
        str(thr): _per_event_type_recall(coded, thr)
        for thr in _EVENT_TYPE_THRESHOLDS
    }
    doca_recall = _doca_test_recall(meta, cls, Path(weights_path), _THRESHOLD_GRID)

    run_config = cca_config.RunConfig.from_json(
        cca_config.config_path_for_weights(Path(weights_path))
    )
    n_coded = int(coded["cca_event"].is_not_null().sum())
    n_pos = int(coded["cca_event"].fill_null(False).sum())
    return {
        "weights_path": str(weights_path),
        "coded_path": str(coded_path),
        "cache_suffix": suffix,
        "git_commit": _git_commit(),
        "prior": run_config.heads[0].loss.prior,
        "n_coded": n_coded,
        "n_positive": n_pos,
        "gold_band_counts": gold_counts,
        "corpus_band_counts": corpus_counts,
        "ipw_weights": weights,
        "threshold_sweep": sweep,
        "per_event_type_recall": per_type,
        "doca_test_recall": doca_recall,
    }


def _print_report(record: dict) -> None:
    print(f"\nCCA gold-set eval  (commit {record['git_commit']}, prior {record['prior']})")  # LOG
    print(f"  coded n={record['n_coded']}  positives={record['n_positive']}  "
          f"cache={record['cache_suffix']}")  # LOG
    print("  IPW weights: " + "  ".join(
        f"{b.split('_')[-1]}={w:.1f}" for b, w in record["ipw_weights"].items()))  # LOG

    doca = {d["threshold"]: d for d in record["doca_test_recall"]}
    doca_n = record["doca_test_recall"][0]["n"] if record["doca_test_recall"] else 0
    print(f"\n  thr |        RAW (P / R / F1)        |   REWEIGHTED (P / R / F1)     | "
          f"support tp/fp/fn | DoCA-R (n={doca_n})")  # LOG
    print("  " + "-" * 112)  # LOG
    for s in record["threshold_sweep"]:
        r, w = s["raw"], s["reweighted"]
        dr = doca.get(s["threshold"], {}).get("recall", float("nan"))
        print(f"  {s['threshold']:+.1f} | "
              f"{r['precision']:.3f} / {r['recall']:.3f} / {r['f1']:.3f}         | "
              f"{w['precision']:.3f} / {w['recall']:.3f} / {w['f1']:.3f}         | "
              f"   {w['support_tp']:>3}/{w['support_fp']:>3}/{w['support_fn']:>3}     | "
              f"{dr:.3f}")  # LOG

    for thr, types in record["per_event_type_recall"].items():
        print(f"\n  per-event-type recall @ logit {thr}:")  # LOG
        for etype, m in sorted(types.items(), key=lambda kv: -kv[1]["n"]):
            print(f"    {etype:<14} {m['caught']:>3}/{m['n']:<3}  recall={m['recall']:.3f}")  # LOG


def main(coded_path: str | None = None, suffix: str = "train250k",
         weights_path: str | None = None) -> None:
    coded = Path(coded_path) if coded_path else (
        config.VALIDATION_DIR / "cca_coding_first500_coded.csv"
    )
    weights = Path(weights_path) if weights_path else config.CCA_DOCA_WEIGHTS
    record = run_eval(coded, suffix, weights)
    _print_report(record)

    out_dir = config.CCA_DOCA_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"eval_{stamp}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nWrote run record -> {out}")  # LOG


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate CCA model on the coded gold set.")
    ap.add_argument("--coded", default=None, help="coded gold-set CSV (default: first500)")
    ap.add_argument("--suffix", default="train250k", help="embedding cache subdir")
    ap.add_argument("--weights", default=None, help="weights .h5 (default: CCA_DOCA_WEIGHTS)")
    args = ap.parse_args()
    main(coded_path=args.coded, suffix=args.suffix, weights_path=args.weights)
