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

    The parquet is produced by r/audit/api_ldc_join.R and contains ldc_heuristic_us
    computed from LDC-side descriptors (LDC2008T19/data/parsed_to_rds fields + LDC online_sections/dsk)
    using the us_assign() heuristic from nyt_location_checking.R.

    AC6.1 (heuristic error rate) is computed against dateline-labeled rows only to avoid
    circularity: desk-derived labels in ldc_label_source == "heuristic" would be compared
    against heuristic-derived verdicts, conflating the sources.

    Args:
        matched_path: Path to api_ldc_matched.parquet (from api_ldc_join.R).
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

    # AC6.1: error rate (heuristic vs dateline labels only)
    # Filter to rows where ldc_label_source == "dateline" to avoid circularity.
    # Heuristic-sourced labels would be desk/section/keywords derived; comparing
    # heuristic verdicts against those would be circular.
    df_dateline = df.filter(pl.col("ldc_label_source") == "dateline")
    dateline_count = len(df_dateline)

    if dateline_count > 0:
        error_rate = heuristic_error_rate(
            df_dateline["ldc_heuristic_us"].to_list(),
            df_dateline["ldc_us_label"].to_list(),
        )
        print(f"AC6.1 Heuristic Error Rate (vs dateline labels, n={dateline_count}): {error_rate:.4f}")
    else:
        print("AC6.1 Heuristic Error Rate: No dateline-labeled rows in matched set")

    # AC6.2: lead similarity
    # Compute over a deterministic random sample if the full set is very large
    full_sim_list = list(zip(df["ldc_stripped_text"].to_list(), df["api_lead"].to_list()))
    if len(full_sim_list) > 20000:
        # Use deterministic sample (seed 200) for performance
        import random
        random.seed(200)
        sample_list = random.sample(full_sim_list, min(20000, len(full_sim_list)))
        stripped_texts = [s for s, _ in sample_list]
        api_leads = [a for _, a in sample_list]
        sim = lead_similarity(stripped_texts, api_leads)
        print(f"AC6.2 Mean Lead Similarity (LDC stripped vs API lead, {len(sample_list)} sample, seed=200): {sim:.4f}")
    else:
        sim = lead_similarity(
            df["ldc_stripped_text"].to_list(),
            df["api_lead"].to_list(),
        )
        print(f"AC6.2 Mean Lead Similarity (LDC stripped vs API lead, full set): {sim:.4f}")

    # Report counts
    print(f"\nMatched pairs: {len(df)}")
    print(f"Dateline-labeled for AC6.1: {dateline_count}")

    # Report heuristic verdict distribution
    heuristic_dist = df.select("ldc_heuristic_us").group_by("ldc_heuristic_us").agg(pl.len())
    print("\nHeuristic Verdict Distribution:")
    for row in heuristic_dist.sort("len", descending=True).to_dicts():
        print(f"  {row['ldc_heuristic_us']}: {row['len']}")

    # AC6.5: joinability caveat
    print(f"\nAC6.5 Caveat (biased-by-joinability):\n{CAVEAT_STR}")


if __name__ == "__main__":
    # Allow passing path as argument or use default
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/us_filter/audit/api_ldc_matched.parquet"
    audit_matched_parquet(path)
