# pattern: Functional Core

"""Pure pre-flight verdict functions for multi-head ICA assembly.

No I/O. Takes already-loaded inputs and returns structured verdicts.
Verifies AC7.1-AC7.4 pre-flight facts before any retrain.
"""

from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class Verdict:
    """Pre-flight verdict: atomic check result (name, status, detail,
    remediation)."""

    name: str
    status: Literal["PASS", "WARN", "FAIL"]
    detail: str
    remediation: str | None = None


def us_weights_verdict(
    cache_provenance: dict, table_build_us_weights: str | None
) -> Verdict:
    """AC7.1: Verify operative training-gate US weights are us_classifier_full.

    The operative gate depends on whether the table-build invocation overrides the
    cache's weights. If table_build_us_weights is None (no override), check the
    cache's us_weights.path basename. If the operative weights are the smoke-test
    us_classifier.weights.h5, return FAIL; us_classifier_full.weights.h5, return
    PASS; undetermined, return WARN.

    Args:
        cache_provenance: dict with us_weights.path (string basename)
        table_build_us_weights: path to override weights (or None if no override)

    Returns:
        Verdict with status PASS/WARN/FAIL
    """
    # Extract the cache's us_weights.path basename
    cache_path = cache_provenance.get("us_weights", {}).get("path", "")
    cache_basename = cache_path.split("/")[-1] if cache_path else ""

    # Determine the operative weights
    operative_path = (
        table_build_us_weights if table_build_us_weights else cache_basename
    )

    # Classify
    if operative_path:
        operative_basename = operative_path.split("/")[-1]
        if "us_classifier_full" in operative_basename:
            return Verdict(
                name="us_weights",
                status="PASS",
                detail=f"operative training-gate weights: {operative_basename}",
                remediation=None,
            )
        elif operative_basename == "us_classifier.weights.h5":
            return Verdict(
                name="us_weights",
                status="FAIL",
                detail=(
                    f"operative training-gate weights are smoke-test "
                    f"{operative_basename}, not us_classifier_full"
                ),
                remediation="Phase 3: retrain with us_classifier_full",
            )
        else:
            return Verdict(
                name="us_weights",
                status="WARN",
                detail=(
                    f"operative training-gate weights undetermined: "
                    f"{operative_basename}"
                ),
                remediation="confirm operative weights via config or table-build log",
            )
    else:
        return Verdict(
            name="us_weights",
            status="WARN",
            detail=(
                "operative training-gate weights undetermined "
                "(no cache, no override)"
            ),
            remediation="confirm cache provenance and table-build invocation",
        )


def calibration_presence_verdict(present: dict[str, bool]) -> Verdict:
    """AC7.2: Verify calibration sidecars are present (or defer to Phase 4).

    Keys: cca, cca_street, relevance. CCA sidecars are required (FAIL if missing);
    relevance is optional (WARN if missing, to be fit in Phase 4).

    Args:
        present: dict[str, bool] with keys cca, cca_street, relevance

    Returns:
        Verdict with status PASS/WARN/FAIL
    """
    missing_required = []
    missing_optional = []

    if not present.get("cca", False):
        missing_required.append("cca")
    if not present.get("cca_street", False):
        missing_required.append("cca_street")
    if not present.get("relevance", False):
        missing_optional.append("relevance")

    if missing_required:
        missing_str = ", ".join(missing_required)
        return Verdict(
            name="calibration_presence",
            status="FAIL",
            detail=f"missing required calibration sidecars: {missing_str}",
            remediation=f"Phase 4: fit calibrators for {missing_str}",
        )

    if missing_optional:
        return Verdict(
            name="calibration_presence",
            status="WARN",
            detail=f"missing optional calibration sidecar: {missing_optional[0]}",
            remediation="Phase 4: fit relevance calibrator",
        )

    return Verdict(
        name="calibration_presence",
        status="PASS",
        detail="all calibration sidecars present",
        remediation=None,
    )


def doca_freshness_verdict(mtimes: dict[str, float | None]) -> Verdict:
    """AC7.3: Verify DoCA chain freshness (monotone mtime ordering).

    Keys: doca_csv, rds, positives. Warns if doca_csv > rds (match stale) or
    rds > positives (positives stale); passes if monotone doca_csv <= rds <= positives;
    warns if any mtime is None.

    Args:
        mtimes: dict[str, float | None] with keys doca_csv, rds, positives

    Returns:
        Verdict with status PASS/WARN/FAIL
    """
    doca_csv = mtimes.get("doca_csv")
    rds = mtimes.get("rds")
    positives = mtimes.get("positives")

    # Check for None (missing mtime)
    missing = [k for k, v in mtimes.items() if v is None]
    if missing:
        return Verdict(
            name="doca_freshness",
            status="WARN",
            detail=(
                f"missing mtime for {', '.join(missing)}; "
                "cannot verify DoCA chain freshness"
            ),
            remediation=(
                "check file existence and modification dates: "
                + ", ".join(missing)
            ),
        )

    # Check monotone ordering
    stale_signals = []
    if doca_csv is not None and rds is not None and doca_csv > rds:
        stale_signals.append("doca_csv > rds (match potentially stale)")
    if rds is not None and positives is not None and rds > positives:
        stale_signals.append("rds > positives (positives stale)")

    if stale_signals:
        return Verdict(
            name="doca_freshness",
            status="WARN",
            detail="; ".join(stale_signals),
            remediation=(
                "re-run doca export pipeline: "
                "r/doca/export_cca_positives.R → cca_doca_positives.parquet"
            ),
        )

    return Verdict(
        name="doca_freshness",
        status="PASS",
        detail="DoCA chain mtimes monotone (doca_csv ≤ rds ≤ positives)",
        remediation=None,
    )


def ldc_channel_verdict(provenance: dict) -> Verdict:
    """AC7.4: Verify LDC text channel is dateline-stripped.

    Checks that lead_column == 'stripped_text' (not None or 'lead_paragraph').

    Args:
        provenance: dict with lead_column field

    Returns:
        Verdict with status PASS/FAIL
    """
    lead_column = provenance.get("lead_column")

    if lead_column != "stripped_text":
        return Verdict(
            name="ldc_channel",
            status="FAIL",
            detail=(
                f"LDC text channel uses '{lead_column}' (raw lead_paragraph), "
                "not dateline-stripped text"
            ),
            remediation="Phase 6: re-embed LDC cache with lead_column='stripped_text'",
        )

    return Verdict(
        name="ldc_channel",
        status="PASS",
        detail="LDC text channel is dateline-stripped",
        remediation=None,
    )


def ldc_gold_coverage_verdict(n_apply_ids: int, n_with_gold_label: int) -> Verdict:
    """Informational: LDC gold label coverage over apply ids.

    Always returns PASS. Detail carries the coverage fraction so Phase 6 can size
    the gold-vs-ML-fallback split. (Not a numbered AC check.)

    Args:
        n_apply_ids: count of LDC 1996-2007 apply ids
        n_with_gold_label: count of those with gold us_label in ldc_labeled.parquet

    Returns:
        Verdict with status PASS
    """
    if n_apply_ids == 0:
        fraction_str = "N/A (no apply ids)"
    else:
        fraction = n_with_gold_label / n_apply_ids
        fraction_str = f"{n_with_gold_label}/{n_apply_ids} ({100*fraction:.1f}%)"

    return Verdict(
        name="ldc_gold_coverage",
        status="PASS",
        detail=f"LDC gold us_label coverage: {fraction_str}",
        remediation=None,
    )
