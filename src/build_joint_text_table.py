# pattern: Imperative Shell (union_population and apply_rel_label are the
#   pure core; cca_label derivation is delegated to build_cca_doca_table's
#   already-pure label_and_restrict; text assembly is delegated to
#   build_relevance_text_table's assemble_headline_with_lead/read_api_text)
"""Build the UNION CCA+rel text-bearing training table for the joint CCA+rel
fine-tune (docs/design-plans/2026-08-18-stage4-joint-finetune.md item 2).

`src/run_joint_text.py` (a separate build-order item) trains two
`ClassificationHead`s -- `cca` and `rel` -- off a single shared encoder, so it
needs one table carrying BOTH heads' labels over the union of their two
(overlapping but distinct) populations: CCA's `train250k` cache and rel's
`relevance_train` cache.

Population and label derivation, precisely:

  * Rows = union by `id` of the two cache metas (`load_cache_meta`). On ids
    present in both, the `us_logit` / `year` columns are taken from the REL
    side -- it is the fused-gate-calibrated probability channel the rel head's
    US threshold was tuned on, whereas the CCA-side `us_logit` is the raw
    (uncalibrated) US-head logit emitted by `embed_corpus.py`. Provenance
    columns `in_cca_pop` / `in_rel_pop` record which cache(s) an id came from.
  * `cca_label` = id in DoCA positives (`config.CCA_DOCA_POSITIVES`), kept
    REGARDLESS of the US gate -- DoCA events are US by construction (see
    `build_cca_doca_table.py`'s report text). Computed by
    `build_cca_doca_table.label_and_restrict`, which also produces the raw
    (pre-fused) `us` gate column from the union's `us_logit`.
  * The fused US gate (`preproc.us_location.apply_fused_us_gate`, ML pass AND
    not-clearly-foreign) is then applied to `us`, exactly as
    `build_relevance_text_table.main` does.
  * `rel_label` = id in `relevance/candidates.parquet` AND the (now fused)
    `us` gate -- rel positives must pass the fused gate, same US-restriction
    step as `build_relevance_text_table`.
  * `reliable_neg` = id in `relevance/reliable_negatives.parquet` (carried for
    parity with the rel-only table; inert at eta=0).
  * The clean-ICA holdout (`config.ICA_HOLDOUT_IDS`) is dropped at build time
    with the same belt-and-suspenders drop-then-verify as
    `build_relevance_text_table.py` -- this table is the leakage contract's
    fifth consumer (US head, CCA head, rel head, ICA eval, and now this one;
    `run_joint_text.py` re-verifies with `assert_holdout_excluded`).

Persisted to `relevance/joint_text_table.parquet` (config.JOINT_TEXT_TABLE).

Run from project root:
    uv run python -m src.build_joint_text_table
"""

from __future__ import annotations

import polars as pl

import src.config as config
from src.build_cca_doca_table import label_and_restrict
from src.build_relevance_text_table import assemble_headline_with_lead, read_api_text
from src.embed_corpus import load_cache_meta
from src.preproc.us_location import apply_fused_us_gate, load_location_signals

RELEVANCE_DIR = config.PROJECT_ROOT / "relevance"
# The two production caches by design -- not a CLI knob (see the design doc).
_CCA_CACHE_SUFFIX = "train250k"
_REL_CACHE_SUFFIX = "relevance_train"
_US_THRESHOLD = 0.5  # rel-side threshold convention; matches build_relevance_text_table


# ---------------------------------------------------------------------------
# Functional core: union + rel-label derivation (pure, unit-tested)
# ---------------------------------------------------------------------------
def union_population(cca_meta: pl.DataFrame, rel_meta: pl.DataFrame) -> pl.DataFrame:
    """Pure: union the CCA and rel cache metas by id, one row per id.

    On ids present in both, `us_logit` and `year` are taken from `rel_meta`
    (the fused-gate-calibrated probability channel the rel US threshold was
    tuned on) rather than `cca_meta` (a raw, uncalibrated logit on a different
    scale). Adds boolean provenance columns `in_cca_pop` / `in_rel_pop`.
    Each input is deduped to one row per id before the union (defensive --
    upstream cache metas are expected unique, but a duplicate would otherwise
    silently fan out the join).
    """
    cols = ["id", "us_logit", "year"]
    cca = cca_meta.select(cols).unique(subset="id", keep="first", maintain_order=True)
    rel = rel_meta.select(cols).unique(subset="id", keep="first", maintain_order=True)

    merged = cca.join(rel, on="id", how="full", suffix="_rel", coalesce=True)
    return merged.with_columns(
        in_cca_pop=pl.col("us_logit").is_not_null(),
        in_rel_pop=pl.col("us_logit_rel").is_not_null(),
    ).with_columns(
        us_logit=pl.coalesce(["us_logit_rel", "us_logit"]),
        year=pl.coalesce(["year_rel", "year"]),
    ).select("id", "us_logit", "year", "in_cca_pop", "in_rel_pop")


def apply_rel_label(table: pl.DataFrame, rel_candidate_ids) -> pl.DataFrame:
    """Pure: `rel_label` (Int8 0/1) = id in `rel_candidate_ids` AND `table["us"]`.

    Must be called AFTER the fused US gate has finalized the `us` column
    (`apply_fused_us_gate`) -- rel positives must pass the fused gate, same
    US-restriction as `build_relevance_text_table.main`.
    """
    pos_list = list(dict.fromkeys(rel_candidate_ids))
    return table.with_columns(
        ((pl.col("id").is_in(pos_list)) & pl.col("us")).cast(pl.Int8).alias("rel_label")
    )


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------
def _report(
    union_n: int, n_cca_only: int, n_rel_only: int, n_overlap: int,
    table: pl.DataFrame, n_holdout_dropped: int,
) -> None:
    n = table.height
    empty_headline = int((table["headline"] == "").sum())
    empty_lead = int((table["lead_paragraph"] == "").sum())
    print(f"union population = {union_n} "
          f"(cca-only={n_cca_only}, rel-only={n_rel_only}, overlap={n_overlap})")  # LOG
    print(f"holdout dropped = {n_holdout_dropped}")  # LOG
    print(f"table rows = {n} (= {union_n} - {n_holdout_dropped}: "
          f"{'MATCHES' if n == union_n - n_holdout_dropped else 'MISMATCH'})")  # LOG
    print(f"  empty headline={empty_headline}  empty lead_paragraph={empty_lead} "
          f"(pre-existing corpus nulls, not a join failure)")  # LOG


def main(threshold: float = _US_THRESHOLD) -> None:
    cca_meta = load_cache_meta(config.CCA_EMBED_CACHE_DIR / _CCA_CACHE_SUFFIX)
    rel_meta = load_cache_meta(config.CCA_EMBED_CACHE_DIR / _REL_CACHE_SUFFIX)
    population = union_population(cca_meta, rel_meta)
    union_n = population.height
    n_overlap = int((population["in_cca_pop"] & population["in_rel_pop"]).sum())
    n_cca_only = int((population["in_cca_pop"] & ~population["in_rel_pop"]).sum())
    n_rel_only = int((~population["in_cca_pop"] & population["in_rel_pop"]).sum())

    doca_positive_ids = pl.read_parquet(config.CCA_DOCA_POSITIVES)["id"].to_list()
    table = label_and_restrict(population, doca_positive_ids, threshold)
    n_cca_pos = int((table["cca_label"] == 1).sum())
    print(f"cca_label positives (kept regardless of US gate) = {n_cca_pos}")  # LOG

    # Fused US gate (identical to build_relevance_text_table.main): ML filter
    # passes AND not clearly foreign.
    signals = load_location_signals(table["id"].to_list())
    table = table.join(signals, on="id", how="left").with_columns(
        pl.col("any_us").fill_null(False), pl.col("any_not_us").fill_null(False)
    )
    n_us_ml = int(table["us"].sum())
    table = apply_fused_us_gate(table)
    print(f"fused US gate (thr={threshold}): {int(table['us'].sum())}/{n_us_ml} "
          f"of ML-gated kept (clearly-foreign removed)")  # LOG

    # rel_label: candidates ALSO restricted to the (now fused) US gate.
    rel_candidate_ids = pl.read_parquet(RELEVANCE_DIR / "candidates.parquet")["id"].to_list()
    n_rel_candidates_in_pop = int(table["id"].is_in(rel_candidate_ids).sum())
    table = apply_rel_label(table, rel_candidate_ids)
    n_rel_pos = int((table["rel_label"] == 1).sum())
    print(f"rel_label positives (US-restricted) = {n_rel_pos}/{n_rel_candidates_in_pop} kept "
          f"({n_rel_candidates_in_pop - n_rel_pos} dropped by US restriction)")  # LOG

    reliable_neg_ids = pl.read_parquet(RELEVANCE_DIR / "reliable_negatives.parquet")["id"].to_list()
    table = table.with_columns(pl.col("id").is_in(reliable_neg_ids).alias("reliable_neg"))
    n_neg = int(table["reliable_neg"].sum())
    print(f"reliable negatives: {n_neg}")  # LOG

    # Belt-and-suspenders holdout exclusion at BUILD time (mirrors
    # build_relevance_text_table.py / build_us_pnu_table.py). run_joint_text.py
    # re-applies + re-verifies via assert_holdout_excluded.
    holdout_ids = pl.read_parquet(config.ICA_HOLDOUT_IDS)["id"].to_list()
    before = table.height
    table = table.filter(pl.col("id").is_in(holdout_ids).not_())
    n_holdout_dropped = before - table.height
    if any(hid in set(table["id"].to_list()) for hid in holdout_ids):
        raise ValueError("holdout ids survived the drop filter -- programming error")

    # Attach text. `read_api_text` concatenates raw api_corpus rows across
    # files with no dedupe, so the same known duplicate-id family that
    # `load_location_signals` guards against (see docs/notes/
    # metal-execution-findings.md) can fan out this join too -- guard the
    # boundary here rather than editing the shared sibling helper.
    text_rows = read_api_text(table["id"].to_list())
    before_text = text_rows.height
    text_rows = text_rows.unique(subset="id", keep="first", maintain_order=True)
    if text_rows.height != before_text:
        print(f"read_api_text: dropped {before_text - text_rows.height} duplicate id "
              f"row(s) (api_corpus duplicate-id family; kept first occurrence)")  # LOG
    missing = set(table["id"].to_list()) - set(text_rows["id"].to_list())
    if missing:
        raise ValueError(
            f"{len(missing)} ids not found in api_corpus text "
            f"(e.g. {sorted(missing)[:5]}); investigate before training on this table"
        )
    table = table.join(text_rows, on="id", how="left")
    table = assemble_headline_with_lead(table)

    _report(union_n, n_cca_only, n_rel_only, n_overlap, table, n_holdout_dropped)

    RELEVANCE_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(config.JOINT_TEXT_TABLE)
    print(f"Wrote {config.JOINT_TEXT_TABLE} ({table.height} rows)")  # LOG


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the union CCA+rel text-bearing joint table.")
    ap.add_argument("--threshold", type=float, default=_US_THRESHOLD,
                     help="US probability/logit gate threshold (rel-side convention)")
    args = ap.parse_args()
    main(threshold=args.threshold)
