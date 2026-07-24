# pattern: Imperative Shell (label_* / sample_unlabeled / assemble_pnu_table / _as_splits_view are the pure core)
"""
Build the v1 US-head retrain P/N/U training table (stripped channel).

Design: docs/notes/us-head-retrain-plan.md (the retrain rationale) and the v1
scoping brief (this table's exact source rules). Four sources, each pinned to
the embedding cache that carries its CLS vectors:

  POSITIVES (api)  -- all API-format DoCA positives, from `train250k`
  POSITIVES (ldc)  -- the 345 LDC-format DoCA positives, from `us_pos_ldc345`
                      (built by build_us_pos_ldc345_source.py + embed_corpus.py)
  RELIABLE NEGATIVES -- LDC rows with us_label==False AND label_source=="dateline"
                      (dateline-resolved foreign only; heuristic negatives are
                      left out of N entirely for v1 -- they fall into the
                      unlabeled mass conceptually but are not added to U either)
  UNLABELED        -- a 250,000-row random sample (seed=200) from `full`
                      (API 1960-1975, unrestricted), excluding ids already in P

The ICA-eval holdout anchors are excluded in BOTH id spaces: directly (API ids,
`validation/ica_holdout_ids.parquet`) and via translation (LDC ids, produced by
`build_holdout_ids_ldc.py` / recomputed here from `api_ldc_matched.parquet` so
this script doesn't depend on that artifact having been (re-)run first).
Holdout exclusion is belt-and-suspenders: dropped at build time AND re-verified
by `assert_holdout_excluded` on a synthetic splits-shaped view before writing.

Output (`us_filter/us_pnu_table.parquet`): `id` (Utf8, unified -- API ids keep
their `nyt://` form, LDC ids are their bare decimal string), `cache` (which
embed_cache subdir carries this row's CLS vector: train250k | us_pos_ldc345 |
us_train_ldc | full), `pnu_label` (pos | neg | unl), `source` (doca_api |
doca_ldc | ldc_dateline_neg | api_unlabeled), `year` (Utf8, null where the
source cache doesn't carry one).

Run from project root:
    uv run python -m src.build_us_pnu_table
"""

from __future__ import annotations

from collections.abc import Collection

import polars as pl

import src.config as config
from src.build_holdout_ids_ldc import translate_holdout_to_ldc
from src.data_setup.data import assert_holdout_excluded
from src.embed_corpus import load_cache_meta

_TABLE_COLUMNS = ["id", "cache", "pnu_label", "source", "year"]


# ---------------------------------------------------------------------------
# Functional core: source selection + labeling (pure, unit-tested)
# ---------------------------------------------------------------------------
def label_api_positives(
    cache_meta: pl.DataFrame,
    positive_ids: Collection[str],
    holdout_ids: Collection[str] = (),
) -> pl.DataFrame:
    """Tag `train250k` rows whose id is an API-format DoCA positive as pos/doca_api.

    `positive_ids` -- the `nyt://...` DoCA positive ids (duplicates ok).
    `holdout_ids` -- ICA-eval anchor ids to exclude (API space, no translation
    needed here). Every non-holdout positive id must resolve in `cache_meta`
    (train250k was built to include every API DoCA positive) -- raises
    ValueError enumerating any missing id, since a silent drop would silently
    shrink the positive set.
    """
    keep = [i for i in dict.fromkeys(positive_ids) if i not in set(holdout_ids)]
    sub = cache_meta.filter(pl.col("id").is_in(keep))
    missing = set(keep) - set(sub["id"].to_list())
    if missing:
        raise ValueError(
            f"{len(missing)} DoCA positive ids not found in train250k cache "
            f"(e.g. {sorted(missing)[:5]})"
        )
    return (
        sub.select(pl.col("id").cast(pl.Utf8), pl.col("year").cast(pl.Utf8))
        .with_columns(
            pl.lit("train250k").alias("cache"),
            pl.lit("pos").alias("pnu_label"),
            pl.lit("doca_api").alias("source"),
        )
        .select(_TABLE_COLUMNS)
    )


def label_ldc_positives(
    cache_meta: pl.DataFrame, holdout_ids: Collection[str] = ()
) -> pl.DataFrame:
    """Tag every row of the `us_pos_ldc345` cache as pos/doca_ldc.

    All rows in this cache are DoCA positives by construction (it was built
    from exactly the 345 LDC-format positives) -- the only filter is holdout
    exclusion. `holdout_ids` must already be in LDC space (translated).
    """
    sub = cache_meta.filter(
        ~pl.col("id").cast(pl.Utf8).is_in(list(set(holdout_ids)))
    )
    return (
        sub.select(pl.col("id").cast(pl.Utf8))
        .with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("year"),
            pl.lit("us_pos_ldc345").alias("cache"),
            pl.lit("pos").alias("pnu_label"),
            pl.lit("doca_ldc").alias("source"),
        )
        .select(_TABLE_COLUMNS)
    )


def label_dateline_negatives(
    ldc_labeled: pl.DataFrame, holdout_ids: Collection[str] = ()
) -> pl.DataFrame:
    """Select dateline-resolved-foreign LDC rows as reliable negatives.

    Only `label_source == "dateline"` rows with `us_label == False` qualify.
    Heuristic-sourced negatives (`label_source == "heuristic"`) are explicitly
    NOT included -- the v1 design treats them as unlabeled mass, not N; adding
    them to N would inject the heuristic's noise into the reliable-negative
    signal the retrain depends on. `holdout_ids` must already be in LDC space.
    """
    sub = ldc_labeled.filter(
        pl.col("us_label").not_() & (pl.col("label_source") == "dateline")
    )
    sub = sub.filter(~pl.col("id").cast(pl.Utf8).is_in(list(set(holdout_ids))))
    return (
        sub.select(pl.col("id").cast(pl.Utf8))
        .with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("year"),
            pl.lit("us_train_ldc").alias("cache"),
            pl.lit("neg").alias("pnu_label"),
            pl.lit("ldc_dateline_neg").alias("source"),
        )
        .select(_TABLE_COLUMNS)
    )


def sample_unlabeled(
    cache_meta: pl.DataFrame,
    exclude_ids: Collection[str],
    n: int,
    seed: int = 200,
) -> pl.DataFrame:
    """Deterministic random sample of `n` rows from the `full` cache.

    Excludes any id already claimed elsewhere (`exclude_ids` -- callers pass the
    union of the raw API DoCA-positive ids and the raw API holdout ids, so a
    positive or a holdout anchor never doubles as unlabeled background). Returns
    every remaining row if fewer than `n` are available.
    """
    pool = cache_meta.filter(~pl.col("id").is_in(list(set(exclude_ids))))
    sampled = pool if n >= pool.height else pool.sample(n=n, seed=seed)
    return (
        sampled.select(pl.col("id").cast(pl.Utf8), pl.col("year").cast(pl.Utf8))
        .with_columns(
            pl.lit("full").alias("cache"),
            pl.lit("unl").alias("pnu_label"),
            pl.lit("api_unlabeled").alias("source"),
        )
        .select(_TABLE_COLUMNS)
    )


def assemble_pnu_table(
    pos_api: pl.DataFrame,
    pos_ldc: pl.DataFrame,
    neg: pl.DataFrame,
    unl: pl.DataFrame,
) -> pl.DataFrame:
    """Concatenate the four labeled/tagged sources into one table.

    Verifies no id collision across the pos/neg/unl union -- a defense-in-depth
    check: a positive or reliable negative doubling as unlabeled background
    would poison the training signal. Raises ValueError enumerating any
    duplicate ids.
    """
    table = pl.concat([pos_api, pos_ldc, neg, unl], how="vertical")
    if table["id"].n_unique() != table.height:
        dupes = table.filter(pl.col("id").is_duplicated())["id"].unique().to_list()
        raise ValueError(
            f"{len(dupes)} ids appear in more than one source "
            f"(e.g. {sorted(dupes)[:5]})"
        )
    return table


def validate_ids_present(
    candidate_ids: Collection[str], cache_ids: Collection[str]
) -> None:
    """Raise ValueError if any candidate id is absent from a target cache's ids.

    Defense-in-depth for `label_dateline_negatives`: the reliable-negative ids
    come from `ldc_labeled.parquet`, but the table points consumers at the
    `us_train_ldc` embed cache for their CLS vectors -- this confirms every
    negative id is actually embedded there (rather than trusting the "exactly
    the non-null us_label rows" invariant to hold forever).
    """
    missing = set(candidate_ids) - set(cache_ids)
    if missing:
        raise ValueError(
            f"{len(missing)} ids not found in target cache "
            f"(e.g. {sorted(missing)[:5]})"
        )


def _as_splits_view(table: pl.DataFrame) -> dict:
    """Wrap the table's pos/neg/unl groups in the shape `assert_holdout_excluded`
    expects (`{"train": {"pos","neg","unl"}, "val": {...}}`).

    This table has no train/val/test split of its own (that's the downstream
    training script's job) -- the "train" bucket here is the WHOLE table, so
    reusing the existing leakage-guard machinery gives a second, independent
    check that no holdout id survived the per-source filters above. "val" is
    empty (nothing to check there, but the shape is required).
    """
    groups = {
        "pos": table.filter(pl.col("pnu_label") == "pos"),
        "neg": table.filter(pl.col("pnu_label") == "neg"),
        "unl": table.filter(pl.col("pnu_label") == "unl"),
    }
    empty = {name: group.head(0) for name, group in groups.items()}
    return {"train": groups, "val": empty}


def _report(table: pl.DataFrame) -> None:
    print(f"rows={table.height}")  # LOG
    counts = (
        table.group_by(["source", "pnu_label"])
        .len()
        .sort(["pnu_label", "source"])
    )
    for row in counts.iter_rows(named=True):
        print(f"  {row['pnu_label']:>3} / {row['source']:<16} {row['len']}")  # LOG


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------
def main() -> None:
    positives = pl.read_parquet(config.CCA_DOCA_POSITIVES)
    api_pos_ids = positives.filter(pl.col("id").str.starts_with("nyt://"))["id"].to_list()

    holdout_api = pl.read_parquet(config.ICA_HOLDOUT_IDS)["id"].to_list()
    matched = pl.read_parquet(config.US_FILTER_API_LDC_MATCHED)
    holdout_ldc = translate_holdout_to_ldc(holdout_api, matched)
    print(f"holdout translation: {len(holdout_api)} api ids -> {len(holdout_ldc)} ldc twins")  # LOG

    train250k_meta = load_cache_meta(config.CCA_EMBED_CACHE_DIR / "train250k")
    ldc345_meta = load_cache_meta(config.CCA_EMBED_CACHE_DIR / "us_pos_ldc345")
    full_meta = load_cache_meta(config.CCA_EMBED_CACHE_DIR / "full")
    us_train_ldc_meta = load_cache_meta(config.CCA_EMBED_CACHE_DIR / "us_train_ldc")
    ldc_labeled = pl.read_parquet(config.US_FILTER_LABELED_PARQUET)

    pos_api = label_api_positives(train250k_meta, api_pos_ids, holdout_api)
    pos_ldc = label_ldc_positives(ldc345_meta, holdout_ldc)
    neg = label_dateline_negatives(ldc_labeled, holdout_ldc)
    validate_ids_present(
        neg["id"].to_list(), us_train_ldc_meta["id"].cast(pl.Utf8).to_list()
    )

    exclude_for_u = set(api_pos_ids) | set(holdout_api)
    unl = sample_unlabeled(full_meta, exclude_for_u, n=250_000, seed=200)

    table = assemble_pnu_table(pos_api, pos_ldc, neg, unl)

    all_holdout = set(holdout_api) | set(holdout_ldc)
    assert_holdout_excluded(_as_splits_view(table), all_holdout)

    config.US_FILTER_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(config.US_PNU_TABLE)
    _report(table)
    print(f"Wrote {config.US_PNU_TABLE} ({table.height} rows)")  # LOG


if __name__ == "__main__":
    main()
