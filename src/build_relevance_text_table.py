# pattern: Imperative Shell (assemble_headline_with_lead is the pure text-assembly
#   core; label derivation itself is delegated to already-pure helpers in
#   build_cca_doca_table.label_and_restrict / preproc.us_location)
"""Build a TEXT-BEARING relevance training table for the text-mode / encoder-
unfreeze rel-first training path (docs/notes/encoder-unfreeze-strategy.md).

`src/run_relevance.py` trains features-mode on cached CLS vectors, so it never
needs the underlying article text. Text-mode training (this table's consumer,
`src/run_relevance_text.py`) does. This script reproduces run_relevance.py's
population/label derivation EXACTLY -- same candidates, same fused US gate, same
US-restricted positives, same reliable negatives, same ICA-eval holdout exclusion
-- but attaches `headline` / `lead_paragraph` / `headline_with_lead` from
`api_corpus/*.parquet` by id instead of reading a CLS embedding cache.

Belt-and-suspenders holdout (mirrors src/build_us_pnu_table.py): the ICA-eval
holdout is dropped from the WHOLE table at BUILD time, here. Downstream,
`run_relevance_text.py` passes the SAME holdout ids to `create_relevance_data`
again (a no-op re-filter) and calls `assert_holdout_excluded` -- a second,
independent check, not reliance on this script alone.

Persisted to `relevance/relevance_text_table.parquet` (config.RELEVANCE_TEXT_TABLE).

Run from project root:
    uv run python -m src.build_relevance_text_table
"""

from __future__ import annotations

import glob
from collections.abc import Collection

import polars as pl

import src.config as config
from src.build_cca_doca_table import label_and_restrict
from src.embed_corpus import load_cache_meta
from src.preproc.us_location import apply_fused_us_gate, load_location_signals

RELEVANCE_DIR = config.PROJECT_ROOT / "relevance"
_RELEVANCE_CACHE_SUFFIX = "relevance_train"  # matches run_relevance.py's default
_US_THRESHOLD = 0.5  # calibrated probability gate; matches run_relevance.py's default


# ---------------------------------------------------------------------------
# Functional core: text assembly (pure, unit-tested)
# ---------------------------------------------------------------------------
def assemble_headline_with_lead(
    df: pl.DataFrame, lead_column: str = "lead_paragraph"
) -> pl.DataFrame:
    """Pure: fill-null/"NA"-string handling + `headline</s>lead` concatenation.

    Mirrors `src.data_setup.data.data_from_parquet`'s text-assembly logic
    exactly (duplicated rather than reused -- that function is coupled to a
    full-corpus-read Imperative Shell and isn't separable from it without a
    broader refactor out of this task's scope). Requires `headline` and
    `lead_column` columns; adds `headline_with_lead`.
    """
    df = df.with_columns(
        pl.col("headline").fill_null(""),
        pl.col(lead_column).fill_null(""),
    )
    df = df.with_columns(
        pl.when(pl.col("headline") == "NA")
        .then(pl.lit(""))
        .otherwise(pl.col("headline"))
        .alias("headline"),
        pl.when(pl.col(lead_column) == "NA")
        .then(pl.lit(""))
        .otherwise(pl.col(lead_column))
        .alias(lead_column),
    )
    return df.with_columns(
        (pl.col("headline") + "</s>" + pl.col(lead_column)).alias("headline_with_lead")
    )


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------
def read_api_text(ids: Collection[str], corpus_dir=None) -> pl.DataFrame:
    """Shell: read (id, headline, lead_paragraph) rows for `ids` from api_corpus.

    Targeted per-file filtered read (mirrors `us_location.load_location_signals`)
    rather than a full-corpus scan via `data_from_parquet` -- the population here
    (~266k ids) is a small fraction of the full 1960-1995 API corpus.
    """
    corpus_dir = corpus_dir or config.API_CORPUS_DIR
    want = set(ids)
    parts = []
    for f in sorted(glob.glob(str(corpus_dir / "*.parquet"))):
        d = pl.read_parquet(
            f, columns=["id", "headline", "lead_paragraph"]
        ).filter(pl.col("id").is_in(list(want)))
        if d.height:
            parts.append(d)
    if not parts:
        return pl.DataFrame(
            schema={"id": pl.String, "headline": pl.String, "lead_paragraph": pl.String}
        )
    return pl.concat(parts)


def _report(raw_count: int, table: pl.DataFrame, n_holdout_dropped: int) -> None:
    n = table.height
    n_pos = int(table.filter(pl.col("cca_label") == 1).height)
    n_us = int(table.filter(pl.col("us")).height)
    n_neg = int(table.filter(pl.col("reliable_neg")).height)
    empty_headline = int((table["headline"] == "").sum())
    empty_lead = int((table["lead_paragraph"] == "").sum())
    print(f"cache population (pre-holdout) = {raw_count}")  # LOG
    print(f"holdout dropped = {n_holdout_dropped}")  # LOG
    print(f"table rows = {n} (= {raw_count} - {n_holdout_dropped}: "
          f"{'MATCHES' if n == raw_count - n_holdout_dropped else 'MISMATCH'})")  # LOG
    print(f"  positives(US-restricted)={n_pos}  US-gated={n_us}  reliable_neg={n_neg}")  # LOG
    print(f"  empty headline={empty_headline}  empty lead_paragraph={empty_lead} "
          f"(pre-existing corpus nulls, not a join failure)")  # LOG


def main(cache_suffix: str = _RELEVANCE_CACHE_SUFFIX, threshold: float = _US_THRESHOLD) -> None:
    meta = load_cache_meta(config.CCA_EMBED_CACHE_DIR / cache_suffix)
    raw_count = meta.height

    positives = pl.read_parquet(RELEVANCE_DIR / "candidates.parquet")["id"].to_list()
    table = label_and_restrict(meta, positives, threshold)

    # Fused US gate (identical to run_relevance.py): ML filter passes AND not
    # clearly foreign.
    signals = load_location_signals(table["id"].to_list())
    table = table.join(signals, on="id", how="left").with_columns(
        pl.col("any_us").fill_null(False), pl.col("any_not_us").fill_null(False)
    )
    n_us_ml = int(table["us"].sum())
    table = apply_fused_us_gate(table)
    print(f"fused US gate: {int(table['us'].sum())}/{n_us_ml} of ML-gated kept "
          f"(clearly-foreign removed)")  # LOG

    # Relevance-specific: positives must ALSO be US (see run_relevance.py's
    # module docstring for the CCA-vs-relevance distinction here).
    n_pos_all = int((table["cca_label"] == 1).sum())
    table = table.with_columns(
        ((pl.col("cca_label") == 1) & pl.col("us")).cast(pl.Int8).alias("cca_label")
    )
    n_pos_us = int((table["cca_label"] == 1).sum())
    print(f"positives US-restricted: {n_pos_us}/{n_pos_all} kept")  # LOG

    reliable_neg_ids = pl.read_parquet(RELEVANCE_DIR / "reliable_negatives.parquet")["id"].to_list()
    table = table.with_columns(pl.col("id").is_in(reliable_neg_ids).alias("reliable_neg"))
    n_neg = int(table["reliable_neg"].sum())
    print(f"reliable negatives: {n_neg}")  # LOG

    # Belt-and-suspenders holdout exclusion at BUILD time (mirrors
    # build_us_pnu_table.py). run_relevance_text.py re-applies + re-verifies.
    holdout_ids = pl.read_parquet(config.ICA_HOLDOUT_IDS)["id"].to_list()
    before = table.height
    table = table.filter(pl.col("id").is_in(holdout_ids).not_())
    n_holdout_dropped = before - table.height
    if any(hid in set(table["id"].to_list()) for hid in holdout_ids):
        raise ValueError("holdout ids survived the drop filter -- programming error")

    # Attach text.
    text_rows = read_api_text(table["id"].to_list())
    missing = set(table["id"].to_list()) - set(text_rows["id"].to_list())
    if missing:
        raise ValueError(
            f"{len(missing)} ids not found in api_corpus text "
            f"(e.g. {sorted(missing)[:5]}); the inventory's '0 missing' claim "
            f"no longer holds -- investigate before training on this table"
        )
    table = table.join(text_rows, on="id", how="left")
    table = assemble_headline_with_lead(table)

    _report(raw_count, table, n_holdout_dropped)

    RELEVANCE_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(config.RELEVANCE_TEXT_TABLE)
    print(f"Wrote {config.RELEVANCE_TEXT_TABLE} ({table.height} rows)")  # LOG


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the text-bearing relevance training table.")
    ap.add_argument("--suffix", default=_RELEVANCE_CACHE_SUFFIX, help="embedding cache subdir (labels/us_logit source)")
    ap.add_argument("--threshold", type=float, default=_US_THRESHOLD, help="calibrated US probability gate")
    args = ap.parse_args()
    main(cache_suffix=args.suffix, threshold=args.threshold)
