# pattern: Functional Core (metrics), Imperative Shell (main)
"""Free heuristic audit metrics: error rate vs dateline labels, lead similarity."""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


# === Simple US heuristic (desk/section based) ===
# Simplified from nyt_location_checking.R us_assign()
DESK_US = {
    "National Desk",
    "Metropolitan Desk",
    "Connecticut Weekly Desk",
    "Westchester Weekly Desk",
    "Long Island Weekly Desk",
    "New York Region",
    "New Jersey Weekly Desk",
    "The City Weekly Desk",
}

SECTION_US = {
    "U.S.",
    "New York",
    "New York and Region",
    "Washington",
}

DESK_NON_US = {"Foreign Desk"}
SECTION_NON_US = {"World"}


def simple_us_assign(dsk: str | None, online_sections: str | None) -> bool:
    """Simple US assignment from desk and section (simplified us_assign from R).

    Returns True if any US desk/section indicator is present.
    """
    if dsk:
        dsk_lower = dsk.lower()
        if dsk_lower in {d.lower() for d in DESK_US}:
            return True
        if dsk_lower in {d.lower() for d in DESK_NON_US}:
            return False

    if online_sections:
        sections = [s.strip() for s in online_sections.split(";")]
        sections_lower = [s.lower() for s in sections]
        if any(s in sections_lower for s in {d.lower() for d in SECTION_US}):
            return True
        if any(s in sections_lower for s in {d.lower() for d in SECTION_NON_US}):
            return False

    # Default: assume US if no non-US indicator
    return False


def _norm(s: str | None) -> str:
    """Normalize text: lowercase, alphanumeric+space only.

    Matches the R normalization in api_ldc_join.R.
    """
    if s is None:
        return ""
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def heuristic_error_rate(heuristic_us: Sequence[bool | None], dateline_us: Sequence[bool | None]) -> float:
    """Disagreement rate of the heuristic vs dateline labels.

    Computed over rows where BOTH are non-null.
    Returns NaN if no valid pairs exist.

    Args:
        heuristic_us: List of boolean or None values from us_assign() heuristic.
        dateline_us: List of boolean or None values from dateline labels.

    Returns:
        Float in [0, 1] representing fraction of disagreement.
        Returns float("nan") if no pairs where both are non-null.
    """
    pairs = [
        (h, d)
        for h, d in zip(heuristic_us, dateline_us)
        if h is not None and d is not None
    ]
    if not pairs:
        return float("nan")
    return sum(h != d for h, d in pairs) / len(pairs)


def lead_similarity(stripped_texts: Sequence[str | None], api_leads: Sequence[str | None]) -> float:
    """Mean normalized similarity (difflib SequenceMatcher) of LDC stripped lead vs API lead.

    Both texts are normalized before comparison (lowercase, alphanumeric+space only).

    Args:
        stripped_texts: LDC stripped lead paragraphs (or None).
        api_leads: API lead paragraphs (or None).

    Returns:
        Float in [0, 1] representing mean similarity ratio.
        Returns float("nan") if lists are empty.
    """
    sims = [
        SequenceMatcher(None, _norm(s), _norm(a)).ratio()
        for s, a in zip(stripped_texts, api_leads)
    ]
    if not sims:
        return float("nan")
    return sum(sims) / len(sims)


# === Imperative Shell: read and report ===
CAVEAT_STR = (
    "NOTE: Error rates and similarities conditional on joinability; "
    "articles that failed to match on headline+date are not represented here. "
    "The heuristic AC6.1 and AC6.2 figures are biased toward more joinable/matchable articles."
)


def audit_matched_parquet(matched_path: Path | str) -> None:
    """Read matched parquet and report AC6.1, AC6.2, and AC6.5 metrics.

    Args:
        matched_path: Path to api_ldc_matched.parquet.
    """
    if pl is None:
        print("Error: polars not available", file=sys.stderr)
        sys.exit(1)

    matched_path = Path(matched_path)
    if not matched_path.exists():
        print(f"Error: {matched_path} not found", file=sys.stderr)
        sys.exit(1)

    # Read parquet
    df = pl.read_parquet(matched_path)

    print("=== Free Heuristic Audit Report (AC6.1, AC6.2, AC6.5) ===\n")

    # Compute heuristic labels from desk/section (AC6.1 uses simple_us_assign)
    heuristic_labels = [
        simple_us_assign(row["ldc_dsk"], row["ldc_online_sections"])
        for row in df.iter_rows(named=True)
    ]

    # AC6.1: error rate (heuristic vs dateline labels)
    error_rate = heuristic_error_rate(
        heuristic_labels,
        df["ldc_us_label"].to_list(),
    )
    print(f"AC6.1 Heuristic Error Rate (vs dateline labels): {error_rate:.4f}")

    # AC6.2: lead similarity
    sim = lead_similarity(
        df["ldc_stripped_text"].to_list(),
        df["api_lead"].to_list(),
    )
    print(f"AC6.2 Mean Lead Similarity (LDC stripped vs API lead): {sim:.4f}")

    # Report counts
    print(f"\nMatched pairs: {len(df)}")

    # AC6.5: joinability caveat
    print(f"\nAC6.5 Caveat (biased-by-joinability):\n{CAVEAT_STR}")


if __name__ == "__main__":
    # Allow passing path as argument or use default
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/us_filter/audit/api_ldc_matched.parquet"
    audit_matched_parquet(path)
