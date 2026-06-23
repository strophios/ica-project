# pattern: Imperative Shell

"""Assembly pre-flight gate: verifies multi-head ICA assembly prerequisites.

Gathers real I/O (out-of-repo artifacts, run sidecars, calibration files)
and calls Functional Core verdict functions. Prints a verdict table;
exits nonzero if any FAIL verdict is returned.

Verifies AC7.1-AC7.4 before multi-head retrain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

import src.config as config
from src.calibration.sidecar import calibration_path_for_weights
from src.preflight.checks import (
    calibration_presence_verdict,
    doca_freshness_verdict,
    ldc_channel_verdict,
    ldc_gold_coverage_verdict,
    us_weights_verdict,
)


def _read_provenance_json(cache_path: Path) -> dict | None:
    """Read the latest provenance.NNN.json from a cache subdirectory.

    Args:
        cache_path: Path to a specific cache subdirectory (e.g., full, ldc_9507)

    Returns:
        Parsed provenance dict, or None if file doesn't exist
    """
    if not cache_path.exists():
        return None

    # Find all provenance.*.json files
    provenance_files = sorted(cache_path.glob("provenance.*.json"))
    if not provenance_files:
        return None

    # Read the latest (highest numbered)
    latest = provenance_files[-1]
    try:
        return json.loads(latest.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _stat_file(path: Path) -> float | None:
    """Get mtime of a file, or None if it doesn't exist."""
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _resolve_operative_us_weights() -> str | None:
    """Resolve the operative training-gate US weights.

    Checks the CCA/relevance run sidecars (.config.json) and the table-build
    invocation for a --us-weights reference. Returns a weights path basename
    if found, or None if undetermined.

    Returns:
        Path to operative US weights, or None if undetermined
    """
    # Check for CCA run config (cca_doca.config.json)
    cca_config_path = config.CCA_DOCA_DIR / "cca_doca.config.json"
    if cca_config_path.exists():
        try:
            cfg = json.loads(cca_config_path.read_text())
            # Check for us_weights reference in config
            if "us_weights_path" in cfg:
                return cfg["us_weights_path"]
        except (json.JSONDecodeError, OSError):
            pass

    # Check for relevance run config (relevance.config.json)
    relevance_config_path = config.CCA_DOCA_DIR / "relevance.config.json"
    if relevance_config_path.exists():
        try:
            cfg = json.loads(relevance_config_path.read_text())
            # Check for us_weights reference in config
            if "us_weights_path" in cfg:
                return cfg["us_weights_path"]
        except (json.JSONDecodeError, OSError):
            pass

    # No override found
    return None


def _check_calibration_presence() -> dict[str, bool]:
    """Check if calibration sidecars exist.

    Returns:
        dict with keys cca, cca_street, relevance; values are presence bools
    """
    return {
        "cca": calibration_path_for_weights(
            config.CCA_DOCA_DIR / "cca_doca.weights.h5"
        ).exists(),
        "cca_street": calibration_path_for_weights(
            config.CCA_DOCA_DIR / "cca_doca_street.weights.h5"
        ).exists(),
        "relevance": calibration_path_for_weights(
            config.CCA_DOCA_DIR / "relevance.weights.h5"
        ).exists(),
    }


def _get_doca_mtimes() -> dict[str, float | None]:
    """Get mtimes for the DoCA chain files.

    Returns:
        dict with keys doca_csv, rds, positives; values are mtime or None
    """
    # doca.csv: at project root (PROJECT_ROOT is the data dir, not git repo)
    doca_csv_path = config.PROJECT_ROOT / "doca.csv"

    # cca_matches_good.rds: from config
    rds_path = config.DOCA_CCA_MATCHES

    # cca_doca_positives.parquet: from config
    positives_path = config.CCA_DOCA_POSITIVES

    return {
        "doca_csv": _stat_file(doca_csv_path),
        "rds": _stat_file(rds_path),
        "positives": _stat_file(positives_path),
    }


def _get_ldc_gold_coverage() -> tuple[int, int]:
    """Get LDC 1996-2007 apply id count and gold label coverage.

    Returns:
        (n_apply_ids, n_with_gold_label) tuple
    """
    # Load ldc_labeled.parquet (has us_label column)
    ldc_labeled_path = config.US_FILTER_LABELED_PARQUET

    if not ldc_labeled_path.exists():
        return 0, 0

    try:
        labeled_df = pl.read_parquet(ldc_labeled_path)
    except (OSError, pl.exceptions.ComputeError):
        return 0, 0

    # Filter to LDC 1996-2007 (Hive-partitioned by publication_year)
    ldc_corpus_path = config.LDC_CORPUS

    if not ldc_corpus_path.exists():
        return 0, 0

    # Read the Hive-partitioned dataset for years 1996-2007 only
    try:
        # Collect ids from each year partition via scan_parquet + filter
        ldc_ids_set = set()
        for year in range(1996, 2008):  # 1996-2007 inclusive
            year_partition = ldc_corpus_path / f"publication_year={year}"
            if year_partition.exists():
                # Lazy load and extract unique ids
                year_df = pl.scan_parquet(str(year_partition / "*.parquet")).select("id").collect()
                ldc_ids_set.update(year_df.select("id").to_series().to_list())

        # Count gold labels in that range: ids with non-null us_label (conflict rows
        # have null us_label and don't count as gold). The numerator measures the count
        # of ids with actual gold us_label values, not merely id-presence.
        labeled_df_gold = labeled_df.filter(pl.col("us_label").is_not_null())
        labeled_ids_gold = set(labeled_df_gold.select("id").to_series().to_list())
        ldc_labeled = ldc_ids_set & labeled_ids_gold

        return len(ldc_ids_set), len(ldc_labeled)
    except (OSError, pl.exceptions.ComputeError, KeyError):
        # Catch expected I/O and schema issues; unexpected errors propagate
        return 0, 0


def main() -> int:
    """Gather real inputs and print verdict table.

    Returns:
        0 if all verdicts are PASS/WARN, nonzero if any FAIL
    """
    verdicts = []

    # =====================================================================
    # AC7.1: US weights verdict
    # =====================================================================

    # Read cache provenance for the operative cache (e.g., "full")
    cache_provenance: dict | None = None
    cache_dir = config.CCA_EMBED_CACHE_DIR
    if cache_dir.exists():
        # Try to read from "full" subdirectory
        full_cache = cache_dir / "full"
        cache_provenance = _read_provenance_json(full_cache)

    if cache_provenance is None:
        cache_provenance = {"us_weights": {"path": ""}}

    # Resolve operative US weights from config/table-build
    operative_us_weights = _resolve_operative_us_weights()

    verdicts.append(us_weights_verdict(cache_provenance, operative_us_weights))

    # =====================================================================
    # AC7.2: Calibration presence verdict
    # =====================================================================

    calibration_present = _check_calibration_presence()
    verdicts.append(calibration_presence_verdict(calibration_present))

    # =====================================================================
    # AC7.3: DoCA freshness verdict
    # =====================================================================

    doca_mtimes = _get_doca_mtimes()
    verdicts.append(doca_freshness_verdict(doca_mtimes))

    # =====================================================================
    # AC7.4: LDC channel verdict
    # =====================================================================

    # Read ldc_9507 provenance
    # Note: lead_column is injected post-hoc by embed_corpus.py:339 and thus
    # not present in older provenance. Older provenance defaults to None → FAIL
    # (correct, as older runs used raw lead_paragraph, not stripped_text).
    ldc_provenance: dict | None = None
    if cache_dir.exists():
        ldc_9507_cache = cache_dir / "ldc_9507"
        ldc_provenance = _read_provenance_json(ldc_9507_cache)

    if ldc_provenance is None:
        ldc_provenance = {"lead_column": None}

    verdicts.append(ldc_channel_verdict(ldc_provenance))

    # =====================================================================
    # AC7.5: LDC gold coverage verdict
    # =====================================================================

    n_apply_ids, n_with_gold = _get_ldc_gold_coverage()
    verdicts.append(ldc_gold_coverage_verdict(n_apply_ids, n_with_gold))

    # =====================================================================
    # Print verdict table
    # =====================================================================

    print("\nMulti-Head ICA Assembly — Pre-Flight Verdicts\n")
    print(
        f"{'Verdict':<25} {'Status':<8} {'Detail':<60} {'Remediation':<50}"
    )
    print("-" * 145)

    has_fail = False
    for v in verdicts:
        status = v.status
        if status == "FAIL":
            has_fail = True

        # Truncate long strings for readability
        detail = v.detail[:59] if v.detail else ""
        remediation = v.remediation[:49] if v.remediation else ""

        print(
            f"{v.name:<25} {status:<8} {detail:<60} {remediation:<50}"
        )

    print()

    # =====================================================================
    # Exit code
    # =====================================================================

    if has_fail:
        print("ABORT: At least one FAIL verdict. Address remediation(s) above.")
        return 1

    print(
        "OK: No FAIL verdicts. Proceed with assembly."
        " (WARN verdicts may require action per phase plan.)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
