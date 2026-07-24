# pattern: Imperative Shell (join_ldc_positives_to_stripped_text is the pure core)
"""
Build the embed-source parquet for the 345 LDC-format DoCA positives.

15,614 DoCA-matched articles total; 345 of them never matched an API article
(`cca_doca_positives.parquet` carries them by their LDC file id, `NNNNNNN.xml`,
instead of an `nyt://` id). The US-head retrain's positive set needs ALL 15,614
embedded on the stripped-text channel, so these 345 need a source parquet with
the columns `embed_corpus.py` expects (`id`, `headline`, `stripped_text`) before
they can be embedded into their own small cache.

Run from project root:
    uv run python -m src.build_us_pos_ldc345_source
    uv run python -m src.embed_corpus --full \
        --source-pattern us_filter/us_pos_ldc345_source.parquet \
        --lead-column stripped_text --no-year \
        --out-suffix us_pos_ldc345 --stamp 20260724
"""

from __future__ import annotations

from collections.abc import Collection

import polars as pl

import src.config as config


# ---------------------------------------------------------------------------
# Functional core: id-space join (pure, unit-tested)
# ---------------------------------------------------------------------------
def join_ldc_positives_to_stripped_text(
    ldc_positive_ids: Collection[str],
    ldc_corpus: pl.DataFrame,
    ldc_labeled: pl.DataFrame,
) -> pl.DataFrame:
    """Resolve LDC-format (`NNNNNNN.xml`) DoCA positive ids to their
    dateline-stripped training text.

    `ldc_positive_ids` -- the `id` values from `cca_doca_positives.parquet` that
    are NOT `nyt://` API ids (DoCA events with no API match, identified instead
    by their LDC file id). `ldc_corpus` must carry `id` (the integer id
    `ldc_labeled` keys on) and `file_id` (the `NNNNNNN.xml` form) -- this is the
    join that translates file-id -> integer id. `ldc_labeled` must carry `id`,
    `headline`, `stripped_text`.

    Raises ValueError if any input id fails to resolve at either join -- a
    silent drop would silently shrink the positive set. Returns columns `id`
    (Int64), `headline`, `stripped_text`, one row per input id (deduplicated).
    """
    ids = list(dict.fromkeys(ldc_positive_ids))
    step1 = ldc_corpus.filter(pl.col("file_id").is_in(ids)).select("id", "file_id")
    missing1 = set(ids) - set(step1["file_id"].to_list())
    if missing1:
        raise ValueError(
            f"{len(missing1)} ids not found in ldc_corpus.file_id "
            f"(e.g. {sorted(missing1)[:5]})"
        )
    step2 = step1.join(
        ldc_labeled.select(pl.col("id").cast(pl.Int64), "headline", "stripped_text"),
        on="id",
        how="inner",
    )
    missing2 = set(step1["id"].to_list()) - set(step2["id"].to_list())
    if missing2:
        raise ValueError(
            f"{len(missing2)} ids not found in ldc_labeled "
            f"(e.g. {sorted(missing2)[:5]})"
        )
    return step2.select("id", "headline", "stripped_text")


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------
def main() -> None:
    positives = pl.read_parquet(config.CCA_DOCA_POSITIVES)
    ldc_ids = positives.filter(~pl.col("id").str.starts_with("nyt://"))["id"].to_list()
    print(f"LDC-format DoCA positives: {len(ldc_ids)}")  # LOG

    ldc_corpus = (
        pl.scan_parquet(f"{config.LDC_CORPUS}/**/*.parquet", hive_partitioning=True)
        .select("id", "file_id")
        .collect()
    )
    ldc_labeled = pl.read_parquet(config.US_FILTER_LABELED_PARQUET)

    out = join_ldc_positives_to_stripped_text(ldc_ids, ldc_corpus, ldc_labeled)
    print(f"resolved {out.height}/{len(ldc_ids)} to stripped text")  # LOG

    config.US_FILTER_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(config.US_POS_LDC345_SOURCE)
    print(f"wrote {config.US_POS_LDC345_SOURCE}")  # LOG


if __name__ == "__main__":
    main()
