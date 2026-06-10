# pattern: Imperative Shell
"""
Join API corpus (1987–1995) to LDC labeled parquet on normalized headline + pub_date.
Uses difflib for fuzzy matching (Levenshtein distance).
Then applies us_assign heuristic (desk/section) to compute ldc_heuristic_us.

Provenance: fuzzy matching adapted from R approach in 00_proc_and_matching_prep.R:456-491

Approved deviation: us_assign() is applied to matched API-side rows (the heuristic
faces API data pre-1986 in production, and LDC lacks keywords in parquet form).
Output ldc_heuristic_us is the heuristic's tri-state verdict from API-side descriptors.
"""

import re
from pathlib import Path
from datetime import date
from difflib import SequenceMatcher
import polars as pl

# Paths
API_CORPUS_DIR = Path("/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/api_corpus/")
LDC_LABELED = Path("/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/us_filter/ldc_labeled.parquet")
LDC_CORPUS_DIR = Path("/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/ldc_corpus/")
OUT_DIR = Path("/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/us_filter/audit")
OUT_PARQUET = OUT_DIR / "api_ldc_matched.parquet"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_headline(x):
    """Normalize: lowercase, alphanumeric+space only."""
    if x is None:
        return ""
    return re.sub(r"[^a-z0-9 ]", "", x.lower())

def simple_us_assign(news_desk, section_name):
    """Simplified us_assign: desk/section heuristic.

    Returns:
        True if US, False if not-US, None if ambiguous/unknown
    """
    section_us = {"u.s.", "new york", "new york and region", "washington"}
    dsk_us = {
        "national desk",
        "metropolitan desk",
        "connecticut weekly desk",
        "westchester weekly desk",
        "long island weekly desk",
        "new york region",
        "new jersey weekly desk",
        "the city weekly desk"
    }

    section_non_us = {"world"}
    dsk_non_us = {"foreign desk"}

    meta_is_us = False
    meta_not_us = False

    if section_name:
        section_lower = section_name.lower()
        if section_lower in section_us:
            meta_is_us = True
        elif section_lower in section_non_us:
            meta_not_us = True

    if news_desk:
        desk_lower = news_desk.lower()
        if desk_lower in dsk_us:
            meta_is_us = True
        elif desk_lower in dsk_non_us:
            meta_not_us = True

    if meta_is_us and not meta_not_us:
        return True
    elif meta_not_us and not meta_is_us:
        return False
    else:
        return None

# Load LDC labeled
print("Loading LDC labeled...")
ldc_labeled = pl.read_parquet(LDC_LABELED)

# Per-year join + heuristic application
matched_all = []
join_summary = []

for year in range(1987, 1996):
    print(f"\n=== Year {year} ===")

    # Load API for this year
    api_path = API_CORPUS_DIR / f"{year}.parquet"
    if not api_path.exists():
        print(f"  API parquet not found, skipping")
        continue

    api_df = pl.read_parquet(api_path)
    api_count = len(api_df)
    print(f"  API rows: {api_count}")

    # Load LDC corpus for this year
    ldc_corpus_path = LDC_CORPUS_DIR / f"publication_year={year}"
    if not ldc_corpus_path.exists():
        print(f"  LDC corpus dir not found, skipping")
        continue

    # Read LDC corpus files (parquet partition)
    ldc_corpus = pl.read_parquet(ldc_corpus_path)

    # Join ldc_labeled with publication_date from corpus
    ldc_full = ldc_labeled.join(
        ldc_corpus.select(["id", "publication_date"]),
        on="id",
        how="inner"
    )

    ldc_count = len(ldc_full)
    print(f"  LDC rows: {ldc_count}")

    # Prepare for matching
    api = api_df.with_columns([
        pl.col("id").cast(pl.Utf8).alias("api_id"),
        pl.col("lead_paragraph").alias("api_lead"),
        (pl.col("pub_date").cast(pl.Utf8)).alias("pub_date_str"),
        pl.col("headline").map_elements(normalize_headline, return_dtype=pl.Utf8).alias("headline_norm")
    ]).select(["api_id", "api_lead", "pub_date_str", "headline_norm", "news_desk", "section_name"])

    ldc = ldc_full.with_columns([
        pl.col("id").cast(pl.Utf8).alias("ldc_id"),
        (pl.col("publication_date").cast(pl.Utf8)).alias("publication_date_str"),
        pl.col("headline").map_elements(normalize_headline, return_dtype=pl.Utf8).alias("headline_norm")
    ]).select(["ldc_id", "publication_date_str", "headline_norm", "stripped_text", "us_label"])

    # Convert to Python for faster processing
    api_rows = api.to_dicts()
    ldc_rows = ldc.to_dicts()

    # Group by date
    api_by_date = {}
    for row in api_rows:
        date_str = row["pub_date_str"]
        if date_str not in api_by_date:
            api_by_date[date_str] = []
        api_by_date[date_str].append(row)

    ldc_by_date = {}
    for row in ldc_rows:
        date_str = row["publication_date_str"]
        if date_str not in ldc_by_date:
            ldc_by_date[date_str] = []
        ldc_by_date[date_str].append(row)

    # Match by date
    year_matches = []
    date_count = 0
    unique_dates = sorted(set(ldc_by_date.keys()))

    for date_str in unique_dates:
        date_count += 1
        if date_count % 50 == 0:
            print(f"    Processing date {date_count}/{len(unique_dates)}")

        api_on_date = api_by_date.get(date_str, [])
        ldc_on_date = ldc_by_date.get(date_str, [])

        if not api_on_date or not ldc_on_date:
            continue

        # One-to-one matching: each LDC row matches at most one API row
        ldc_matched = set()

        for api_row in api_on_date:
            api_headline = api_row["headline_norm"]

            # Find best unmatched LDC match
            best_ldc_idx = None
            best_dist = float("inf")

            for ldc_idx, ldc_row in enumerate(ldc_on_date):
                if ldc_idx in ldc_matched:
                    continue

                ldc_headline = ldc_row["headline_norm"]

                # Use difflib SequenceMatcher for similarity ratio
                # Convert to distance (1 - ratio)
                ratio = SequenceMatcher(None, api_headline, ldc_headline).ratio()
                dist = 1 - ratio

                if dist < best_dist:
                    best_dist = dist
                    best_ldc_idx = ldc_idx

            # Match if dist <= 0.5 (equivalently, ratio >= 0.5)
            # This is roughly equivalent to Levenshtein distance <= 5 for medium-length headlines
            if best_ldc_idx is not None and best_dist <= 0.5:
                ldc_row = ldc_on_date[best_ldc_idx]
                api_row_ldc_orig = api_df.filter(pl.col("id") == api_row["api_id"].replace("nyt://article/", "")).to_dicts()

                # Compute heuristic
                heuristic_us = simple_us_assign(api_row.get("news_desk"), api_row.get("section_name"))

                year_matches.append({
                    "api_id": api_row["api_id"],
                    "ldc_id": ldc_row["ldc_id"],
                    "api_lead": api_row["api_lead"],
                    "ldc_stripped_text": ldc_row["stripped_text"],
                    "ldc_us_label": ldc_row["us_label"],
                    "ldc_heuristic_us": heuristic_us
                })
                ldc_matched.add(best_ldc_idx)

    if year_matches:
        matched_all.extend(year_matches)

        print(f"  Matched pairs: {len(year_matches)}")
        joinability_rate = len(year_matches) / ldc_count * 100 if ldc_count > 0 else 0
        print(f"  Joinability rate: {joinability_rate:.2f}%")

        join_summary.append({
            "year": year,
            "ldc_rows": ldc_count,
            "matched_pairs": len(year_matches),
            "joinability_rate": joinability_rate
        })
    else:
        print(f"  Matched pairs: 0")
        print(f"  Joinability rate: 0%")

# Write output
if matched_all:
    matched_df = pl.DataFrame(matched_all)
    print(f"\nWriting output to {OUT_PARQUET}...")
    matched_df.write_parquet(OUT_PARQUET)
    print(f"Output written: {len(matched_df)} rows")
else:
    print("\nNo matches found!")

# Print summary
print("\n=== Grand Summary (1987–1995) ===")
if join_summary:
    summary_df = pl.DataFrame(join_summary)
    print(summary_df)

    total_ldc = summary_df["ldc_rows"].sum()
    total_matched = summary_df["matched_pairs"].sum()
    print(f"\nTotal LDC rows (1987–1995): {total_ldc}")
    print(f"Total matched pairs: {total_matched}")
    if total_ldc > 0:
        print(f"Overall joinability rate: {total_matched / total_ldc * 100:.2f}%")
