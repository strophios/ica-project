# pattern: Imperative Shell (translate_holdout_to_ldc is the pure core)
"""
Translate the API-space ICA-eval holdout ids to their LDC-space twins.

`assert_holdout_excluded` (src/data_setup/data.py) does a literal id-set
intersection, so it can only catch a leaked holdout id in the id space it's
given. The 1,131 ICA-eval holdout ids are all `nyt://` API ids; any of the
US-head retrain's LDC-space rows (the 345 LDC-format DoCA positives, the
dateline-negative pool) need the SAME anchors excluded, but by their LDC twin,
not the API id. `us_filter/audit/api_ldc_matched.parquet` is the cross-match
table that makes the translation possible -- only holdout ids that happen to
have a matched LDC row translate; the rest contribute nothing.

Run from project root:
    uv run python -m src.build_holdout_ids_ldc
"""

from __future__ import annotations

from collections.abc import Collection

import polars as pl

import src.config as config


# ---------------------------------------------------------------------------
# Functional core: id-space translation (pure, unit-tested)
# ---------------------------------------------------------------------------
def translate_holdout_to_ldc(
    holdout_ids: Collection[str], matched: pl.DataFrame
) -> list[str]:
    """Translate API-space holdout ids to their LDC-space twins.

    `matched` must carry `api_id` (`nyt://...` string) and `ldc_id` (any dtype;
    cast to Utf8 for the unified id space). Holdout ids with no row in `matched`
    contribute nothing -- expected: most of the 1,131 ICA-eval anchors have no
    LDC twin (they were never cross-matched), so the returned set is small.

    Returns a sorted, de-duplicated list of LDC ids (Utf8).
    """
    ids = set(holdout_ids)
    sub = matched.filter(pl.col("api_id").is_in(list(ids)))
    ldc_ids = sub.select(pl.col("ldc_id").cast(pl.Utf8)).to_series().unique().to_list()
    return sorted(ldc_ids)


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------
def main() -> None:
    holdout = pl.read_parquet(config.ICA_HOLDOUT_IDS)["id"].to_list()
    matched = pl.read_parquet(config.US_FILTER_API_LDC_MATCHED)

    ldc_ids = translate_holdout_to_ldc(holdout, matched)
    print(f"translated {len(holdout)} api holdout ids -> {len(ldc_ids)} ldc twins")  # LOG

    out = pl.DataFrame({"id": ldc_ids}, schema={"id": pl.Utf8})
    config.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(config.ICA_HOLDOUT_IDS_LDC)
    print(f"wrote {config.ICA_HOLDOUT_IDS_LDC}")  # LOG


if __name__ == "__main__":
    main()
