# pattern: Imperative Shell
"""Apply assembled IcaModel over the API corpus (1960-1995) and LDC (1996-2007).

Produces ranked ICA candidates, per-head scores, and US/CCA per-year outputs.

For API: ML US gate (no gold-first, API has no dateline labels).
For LDC: gold-first US gate (gold dateline us_label overrides ML where available).

CLI Usage:
    # Apply over API corpus (full cache 1960-1995 locally)
    uv run python -m src.apply_ica --corpus api

    # Apply over LDC corpus (ldc_9507 cache, 1996-2007, with gold-first gating)
    uv run python -m src.apply_ica --corpus ldc

    # With custom cache suffix (default: 'full' for api, 'ldc_9507' for ldc)
    uv run python -m src.apply_ica --corpus ldc --cache-suffix ldc_9607_stripped

    # Smoke test (limit rows)
    uv run python -m src.apply_ica --corpus api --limit 100
"""

from __future__ import annotations

import argparse
import logging
from typing import Literal

import numpy as np
import polars as pl

import src.config as config
from src.assemble_ica import IcaModel
from src.embed_corpus import load_cache
from src.preproc.us_location import gold_first_us_gate


logger = logging.getLogger(__name__)


def _filter_years(
    meta: pl.DataFrame, cls_features: np.ndarray, years: tuple[int, int]
) -> tuple[pl.DataFrame, np.ndarray]:
    """Restrict (meta, cls) to an inclusive year range, keeping row alignment.

    Uses the `emb_row` index into the ORIGINAL cls matrix (the load_cache
    contract). Year is cast to Int64 — cache metas carry string years for the
    API corpus and ints for synthetic/LDC caches. No-op if no year column.
    """
    if "year" not in meta.columns:
        return meta, cls_features
    lo, hi = years
    filtered = meta.filter(pl.col("year").cast(pl.Int64).is_between(lo, hi))
    emb_rows = filtered["emb_row"].to_numpy()
    return filtered, cls_features[emb_rows]


def assert_scoring_integrity(model, features: np.ndarray, atol: float = 0.05) -> None:
    """Guard: the compiled predict path must match direct head computation.

    2026-08-04 finding: tensorflow-metal (local MPS) mis-executes the
    ClassificationHead dropout sub-path inside compiled predict graphs —
    deterministic per process, wrong vs true math (us mean shift ~+1.6 on
    real CLS features), while CUDA and CPU compute correctly. Every score
    product computed on MPS before this date carries that distortion. This
    check runs on a small sample before any candidates are written and fails
    loudly rather than letting a distorted stack score a corpus.

    Tolerance calibration (two well-separated scales): benign cross-kernel
    numerics between compiled predict and the direct eager call — CUDA
    TF32/fusion/batching differences — measure maxabs ~5e-3 on the cluster;
    the MPS bug signature is >= ~0.5 logits. atol=0.05 sits an order of
    magnitude above the former and below the latter (the first cluster apply
    tripped a 1e-3 atol on exactly that benign noise).
    """
    sample = features[: min(len(features), 256)]
    pred = model.model.predict({"features": sample}, verbose=0)
    for name, head in (("us", model.us_head), ("cca", model.cca_head), ("rel", model.rel_head)):
        direct = np.asarray(head(sample)).reshape(-1)
        got = np.asarray(pred[name]).reshape(-1)
        maxabs = float(np.max(np.abs(got - direct)))
        if maxabs > atol:
            raise RuntimeError(
                f"scoring integrity check failed for head '{name}': compiled "
                f"predict differs from direct computation (maxabs {maxabs:.4f} "
                f"> atol {atol}). Known cause: the tensorflow-metal dropout "
                f"mis-execution (2026-08-04) — run apply on CUDA or CPU, not MPS."
            )
    logger.info("scoring integrity check passed (predict == direct for us/cca/rel)")


def apply_ica_api(
    cache_suffix: str = "full",
    limit: int | None = None,
    out_name: str | None = None,
    years: tuple[int, int] | None = None,
) -> None:
    """Apply IcaModel over API corpus (1960-1995 or whatever years cache covers).

    Reads the cached CLS embeddings and metadata, runs IcaModel predictions,
    writes per-year score outputs, and produces a ranked ICA candidates file.

    Args:
        cache_suffix: cache subdirectory name (default "full")
        limit: optional row limit for smoke testing
        out_name: candidates filename (default "api_1960_1995.parquet" —
            unchanged legacy name; the forward run passes "api_1996_2025.parquet")
        years: optional inclusive (lo, hi) restriction on the cache rows
    """
    if out_name is None:
        out_name = "api_1960_1995.parquet"
    logger.info(f"Loading API cache: {cache_suffix}")
    cache_dir = config.CCA_EMBED_CACHE_DIR / cache_suffix
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache dir not found: {cache_dir}")

    meta, cls_features = load_cache(cache_dir)
    logger.info(f"Loaded {meta.height} rows, {cls_features.shape[1]}d features")

    if years is not None:
        meta, cls_features = _filter_years(meta, cls_features, years)
        logger.info(f"Filtered to {years[0]}-{years[1]}: {meta.height} rows")

    # Apply limit if specified (smoke test)
    if limit is not None:
        meta = meta.head(limit)
        cls_features = cls_features[:limit]
        logger.info(f"Limited to {limit} rows")

    # Load and apply IcaModel
    logger.info("Loading IcaModel")
    model = IcaModel()
    assert_scoring_integrity(model, cls_features)

    logger.info("Running predictions on API corpus")
    result = model.predict_ica_from_features(cls_features)

    # Attach scores to metadata
    output = meta.with_columns(
        us_score=pl.Series("us_score", result["us"], dtype=pl.Float32),
        cca_score=pl.Series("cca_score", result["cca"], dtype=pl.Float32),
        rel_score=pl.Series("rel_score", result["rel"], dtype=pl.Float32),
        ica_score=pl.Series("ica_score", result["ica_score"], dtype=pl.Float32),
        gated=pl.Series("gated", result["us"] >= model.fusion_config.gate_threshold, dtype=pl.Boolean),
    )

    # Write per-year US and CCA scores
    output_dir_us = config.US_FILTER_SCORES_DIR
    output_dir_cca = config.CCA_DOCA_SCORES_DIR
    output_dir_us.mkdir(parents=True, exist_ok=True)
    output_dir_cca.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing per-year US scores to {output_dir_us}")
    for year in sorted(output["year"].unique()):
        year_df = output.filter(pl.col("year") == year).select(["id", "us_score"])
        year_path = output_dir_us / f"{year}.parquet"
        year_df.write_parquet(year_path)
        logger.info(f"  {year}: {year_df.height} rows")

    logger.info(f"Writing per-year CCA scores to {output_dir_cca}")
    for year in sorted(output["year"].unique()):
        year_df = output.filter(pl.col("year") == year).select(["id", "cca_score"])
        year_path = output_dir_cca / f"{year}.parquet"
        year_df.write_parquet(year_path)
        logger.info(f"  {year}: {year_df.height} rows")

    # Write ranked ICA candidates (sorted by ica_score descending)
    candidates_dir = config.ICA_CANDIDATES_DIR
    candidates_dir.mkdir(parents=True, exist_ok=True)

    candidates_output = output.select([
        "id", "year", "us_score", "cca_score", "rel_score", "ica_score", "gated"
    ]).sort("ica_score", descending=True)

    candidates_path = candidates_dir / out_name
    candidates_output.write_parquet(candidates_path)
    logger.info(f"Wrote ranked ICA candidates to {candidates_path}: {candidates_output.height} rows")

    # Print summary statistics
    n_gated_in = output["gated"].sum()
    pct_gated_in = 100 * n_gated_in / output.height if output.height > 0 else 0
    print("\nAPI apply summary:")
    print(f"  Total rows scored: {output.height}")
    print(f"  Gated in (us_score >= {model.fusion_config.gate_threshold}): {n_gated_in} ({pct_gated_in:.1f}%)")
    print(f"  ICA score range: [{output['ica_score'].min():.4f}, {output['ica_score'].max():.4f}]")
    if candidates_output.height > 0:
        top_5 = candidates_output.head(5)
        print("  Top-5 ICA candidates (by ica_score):")
        for row in top_5.to_dicts():
            print(f"    id={row['id']}, year={row['year']}, ica_score={row['ica_score']:.4f}")


def apply_ica_ldc(
    cache_suffix: str = "ldc_9507",
    limit: int | None = None,
    out_name: str | None = None,
    years: tuple[int, int] = (1996, 2007),
) -> None:
    """Apply IcaModel over LDC corpus (1996-2007) with gold-first US gating.

    Reads the cached CLS embeddings and metadata, loads gold dateline us_label,
    applies gold-first gating (gold overrides ML where available), runs predictions,
    and writes LDC candidates with gate provenance.

    Args:
        cache_suffix: cache subdirectory name (default "ldc_9507")
        limit: optional row limit for smoke testing
        out_name: candidates filename (default "ldc_1996_2007.parquet")
        years: inclusive (lo, hi) restriction (default the 1996-2007 expansion
            window — the pre-parameterization hardcoded behavior)
    """
    if out_name is None:
        out_name = "ldc_1996_2007.parquet"
    logger.info(f"Loading LDC cache: {cache_suffix}")
    cache_dir = config.CCA_EMBED_CACHE_DIR / cache_suffix
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache dir not found: {cache_dir}")

    meta, cls_features = load_cache(cache_dir)
    logger.info(f"Loaded {meta.height} rows, {cls_features.shape[1]}d features")

    # Filter to the target year range if a year column exists
    meta, cls_features = _filter_years(meta, cls_features, years)
    logger.info(f"Filtered to {years[0]}-{years[1]}: {meta.height} rows")

    # Apply limit if specified (smoke test)
    if limit is not None:
        meta = meta.head(limit)
        cls_features = cls_features[:limit]
        logger.info(f"Limited to {limit} rows")

    # Load IcaModel
    logger.info("Loading IcaModel")
    model = IcaModel()
    assert_scoring_integrity(model, cls_features)

    # Run predictions (with ML US gate initially)
    logger.info("Running predictions on LDC corpus")
    result = model.predict_ica_from_features(cls_features)

    # Load gold dateline us_label (join by id)
    logger.info(f"Loading gold US labels from {config.US_FILTER_LABELED_PARQUET}")
    gold_labels_df = pl.read_parquet(
        config.US_FILTER_LABELED_PARQUET, columns=["id", "us_label"]
    )

    # Guard against duplicate ids in gold labels (would misalign scores on join)
    n_rows = gold_labels_df.height
    n_unique = gold_labels_df.select(pl.col("id").unique().count()).item()
    if n_unique < n_rows:
        raise ValueError(
            f"gold labels parquet has duplicate ids: {n_rows} rows, {n_unique} unique ids"
        )

    # Join gold labels to metadata (outer join to keep all rows, unmatched get null)
    meta_with_gold = meta.select("id").join(gold_labels_df, on="id", how="left")

    # Extract gold labels and ML US pass for gold-first gating
    gold_label_list = meta_with_gold["us_label"].to_list()
    ml_us_pass = result["us"] >= model.fusion_config.gate_threshold

    # Apply gold-first gating
    final_gate, gold_coverage = gold_first_us_gate(gold_label_list, ml_us_pass)
    final_gate_array = np.array(final_gate, dtype=bool)

    # Re-score with gate_override
    logger.info("Applying gold-first US gate to ICA scores")
    result_gated = model.predict_ica_from_features(cls_features, gate_override=final_gate_array)

    # Determine gate source (gold vs ml for each row)
    gate_source = [
        "gold" if g is not None else "ml"
        for g in gold_label_list
    ]

    # Attach scores to metadata
    output = meta.with_columns(
        us_score=pl.Series("us_score", result["us"], dtype=pl.Float32),
        cca_score=pl.Series("cca_score", result["cca"], dtype=pl.Float32),
        rel_score=pl.Series("rel_score", result["rel"], dtype=pl.Float32),
        ica_score=pl.Series("ica_score", result_gated["ica_score"], dtype=pl.Float32),
        gated=pl.Series("gated", final_gate, dtype=pl.Boolean),
        gate_source=pl.Series("gate_source", gate_source, dtype=pl.Utf8),
    )

    # Write LDC candidates (sorted by ica_score descending)
    candidates_dir = config.ICA_CANDIDATES_DIR
    candidates_dir.mkdir(parents=True, exist_ok=True)

    candidates_output = output.select([
        "id", "year", "us_score", "cca_score", "rel_score", "ica_score", "gated", "gate_source"
    ]).sort("ica_score", descending=True)

    candidates_path = candidates_dir / out_name
    candidates_output.write_parquet(candidates_path)
    logger.info(f"Wrote LDC ICA candidates to {candidates_path}: {candidates_output.height} rows")

    # Print summary statistics
    n_gold = sum(1 for g in gold_label_list if g is not None)
    n_ml = len(gold_label_list) - n_gold
    n_gated_in = sum(final_gate)
    pct_gated_in = 100 * n_gated_in / len(final_gate) if final_gate else 0

    print("\nLDC apply summary:")
    print(f"  Total rows scored: {len(final_gate)}")
    print("  Gold-first gating:")
    print(f"    Gold labels available: {n_gold} ({100*gold_coverage:.1f}%)")
    print(f"    ML fallback: {n_ml} ({100*(1-gold_coverage):.1f}%)")
    print(f"  Gated in total: {n_gated_in} ({pct_gated_in:.1f}%)")
    print(f"  ICA score range: [{output['ica_score'].min():.4f}, {output['ica_score'].max():.4f}]")
    if candidates_output.height > 0:
        top_5 = candidates_output.head(5)
        print("  Top-5 ICA candidates (by ica_score):")
        for row in top_5.to_dicts():
            print(f"    id={row['id']}, year={row['year']}, ica_score={row['ica_score']:.4f}, gate_source={row['gate_source']}")


def main(
    corpus: Literal["api", "ldc"] = "api",
    cache_suffix: str | None = None,
    limit: int | None = None,
    out_name: str | None = None,
    years: tuple[int, int] | None = None,
) -> None:
    """Apply IcaModel over API or LDC corpus.

    Args:
        corpus: "api" or "ldc"
        cache_suffix: cache subdirectory name (auto-default based on corpus)
        limit: optional row limit for smoke testing
        out_name: candidates filename (default: the per-corpus legacy name)
        years: optional inclusive (lo, hi) year restriction (LDC default 1996-2007)
    """
    logging.basicConfig(level=logging.INFO)

    if cache_suffix is None:
        cache_suffix = "full" if corpus == "api" else "ldc_9507"

    if corpus == "api":
        apply_ica_api(cache_suffix=cache_suffix, limit=limit,
                      out_name=out_name, years=years)
    elif corpus == "ldc":
        ldc_kwargs = {} if years is None else {"years": years}
        apply_ica_ldc(cache_suffix=cache_suffix, limit=limit,
                      out_name=out_name, **ldc_kwargs)
    else:
        raise ValueError(f"Unknown corpus: {corpus}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser (importable so tests can check flag defaults)."""
    parser = argparse.ArgumentParser(
        description="Apply assembled IcaModel over API/LDC corpora"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        choices=["api", "ldc"],
        default="api",
        help="Corpus to apply over (default: api)",
    )
    parser.add_argument(
        "--cache-suffix",
        type=str,
        default=None,
        help="Cache subdirectory suffix (default: 'full' for api, 'ldc_9507' for ldc)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit rows for smoke testing",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default=None,
        help="Candidates parquet filename (default: legacy per-corpus name; "
             "forward run: api_1996_2025.parquet)",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Inclusive year range 'LO-HI' to score (default: whole cache for "
             "api, 1996-2007 for ldc)",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    years = None
    if args.years is not None:
        lo, hi = args.years.split("-")
        years = (int(lo), int(hi))
    main(
        corpus=args.corpus,
        cache_suffix=args.cache_suffix,
        limit=args.limit,
        out_name=args.out_name,
        years=years,
    )
