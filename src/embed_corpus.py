# pattern: Imperative Shell (pure selection/provenance helpers at top; model + I/O in main)
"""
Embedding extractor: cache frozen-DAPT CLS vectors (+ raw US logit) for the API corpus.

The whole CCA/DoCA retrain rests on a frozen backbone, so the RoBERTa forward
pass is a pure function `text -> 768-d CLS`. This script runs it ONCE over a
chosen article set and caches `(id, year, us_logit, CLS)`; everything downstream
(US restriction, DEDPUL, CCA head training, gold-set scoring) then runs on the
cached vectors in minutes. See docs/notes/cca-doca-retrain-design.md.

The US head shares the same DAPT backbone, so we co-emit its **raw logit** in the
same forward pass (the Platt calibration sidecar does not exist yet, so we cache
the logit and threshold on logit 0.0 downstream — never the calibrated path).

Cache layout (under CCA_EMBED_CACHE_DIR / <suffix>/):
    provenance.json
    shard_{i:03d}_cls.npy        float32 (rows, 768)
    shard_{i:03d}_meta.parquet   columns: id, year, us_logit  (row-aligned to cls)

Usage (from project root):
    # unblock run: all DoCA positives + a stratified unlabeled sample
    uv run python -m src.embed_corpus --include-ids <positives.parquet> \
        --sample-n 235000 --stamp 20260615 --out-suffix train250k
    # canonical full cache (overnight)
    uv run python -m src.embed_corpus --full --stamp 20260615 --out-suffix full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import keras
import tensorflow as tf

import src.cca_config as cca_config
import src.config as config
from src.data_setup.data import data_from_parquet
from src.preproc.preprocessor import ClassifierPreprocessor
from src.model_setup.backbone import build_grafted_backbone, load_dapt_backbone, resolve_backbone_path
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_inference_model
from src.us_config import UsRunConfig, config_path_for_weights
from src.extract_tuned_backbone import expected_tuned_groups


# ---------------------------------------------------------------------------
# Functional core: article selection + provenance (pure, unit-tested)
# ---------------------------------------------------------------------------
def stratified_sample_by_year(df: pl.DataFrame, n: int, seed: int = 200) -> pl.DataFrame:
    """Proportional per-year sample of ~n rows (preserves the corpus year mix).

    Deterministic under `seed`. Returns all rows if n >= len; empty if n <= 0.
    Requires a `year` column.
    """
    if n <= 0 or df.height == 0:
        return df.head(0)
    if n >= df.height:
        return df
    frac = n / df.height
    parts = [part.sample(fraction=frac, seed=seed) for part in df.partition_by("year")]
    return pl.concat(parts)


def select_articles(
    corpus: pl.DataFrame,
    include_ids: list[str] | None,
    sample_n: int,
    full: bool,
    seed: int = 200,
) -> pl.DataFrame:
    """Choose the article set to embed.

    - full=True: the entire corpus.
    - otherwise: force-include every `include_ids` row present in the corpus
      (the DoCA positives, so training is not starved of positives), then add a
      stratified-by-year sample of `sample_n` rows drawn from the REMAINDER.

    Deterministic under `seed`. The result has unique ids (includes are unique;
    the sample is drawn disjointly from the remainder).
    """
    if full:
        return corpus
    include_list = list(dict.fromkeys(include_ids or []))
    included = corpus.filter(pl.col("id").is_in(include_list))
    remainder = corpus.filter(~pl.col("id").is_in(include_list))
    sampled = stratified_sample_by_year(remainder, sample_n, seed)
    return pl.concat([included, sampled])


def dedupe_by_id(df: pl.DataFrame) -> pl.DataFrame:
    """Resolve duplicate ids deterministically (post-1995 API pull overlaps).

    Preference per id: a copy with a non-empty effective lead (detected as
    `headline_with_lead` NOT ending in the bare separator "</s>"), then the
    earliest year (when a `year` column exists), then lexical text order — so
    the survivor is invariant under input permutation. Run BEFORE any year
    filter: split year-range embed jobs then agree globally on which copy is
    canonical (a cross-year dup's survivor lands in exactly one job's range).
    """
    sort_cols = (
        ["id", "_lead_empty"]
        + (["year"] if "year" in df.columns else [])
        + ["headline_with_lead"]
    )
    # Empty/null ids are untraceable junk (2025 transform artifacts: 13 rows
    # with no headline/lead/abstract) — drop them rather than collapse to one.
    df = df.filter(pl.col("id").is_not_null() & (pl.col("id") != ""))
    return (
        df.with_columns(
            pl.col("headline_with_lead").str.ends_with("</s>").alias("_lead_empty")
        )
        .sort(sort_cols)
        .unique(subset="id", keep="first", maintain_order=True)
        .drop("_lead_empty")
    )


def _stat_path(p: Path) -> dict:
    """Pure-ish (filesystem read-only): path/exists/size/mtime stat record,
    the shared provenance leaf used for every weights-file reference (base
    backbone, override, US weights, branch donors)."""
    p = Path(p)
    return {"path": str(p), "exists": p.exists(),
            "size": p.stat().st_size if p.exists() else None,
            "mtime": int(p.stat().st_mtime) if p.exists() else None}


def parse_branch_spec(raw: str) -> tuple[str, str, int]:
    """Pure: parse one `--branch` CLI value into `(variant, donor_path, top_n)`.

    Syntax: `<variant>=<donor_path>[:<top_n>]`; `top_n` defaults to 1 (the
    deployed rel-branch donor, `relevance/tuned_backbone.job8823087.weights.h5`,
    was trained with top-1 unfreeze and is a backbone-only file with no
    `RunConfig` sidecar to read it from -- see `_resolve_branch_groups`).
    If `donor_path` itself contains a `:` (unusual on POSIX), the LAST `:`
    is taken as the top_n separator.
    """
    if "=" not in raw:
        raise ValueError(
            f"invalid --branch spec {raw!r}: expected '<variant>=<donor_path>[:<top_n>]'"
        )
    variant, rest = raw.split("=", 1)
    variant = variant.strip()
    if not variant:
        raise ValueError(f"invalid --branch spec {raw!r}: empty variant name")
    if ":" in rest:
        donor_path, top_n_str = rest.rsplit(":", 1)
        try:
            top_n = int(top_n_str)
        except ValueError as e:
            raise ValueError(
                f"invalid --branch spec {raw!r}: top_n {top_n_str!r} is not an int"
            ) from e
    else:
        donor_path, top_n = rest, 1
    if not donor_path:
        raise ValueError(f"invalid --branch spec {raw!r}: empty donor path")
    return variant, donor_path, top_n


def parse_branch_specs(raw_list: list[str] | None) -> dict[str, tuple[str, int]]:
    """Pure: parse repeated `--branch` values into `{variant: (donor_path, top_n)}`.

    Raises on a duplicate variant tag (ambiguous: which donor wins?) or a
    variant colliding with a base output key (`"cls"`/`"us"`).
    """
    if not raw_list:
        return {}
    specs: dict[str, tuple[str, int]] = {}
    for raw in raw_list:
        variant, donor_path, top_n = parse_branch_spec(raw)
        if variant in ("cls", "us"):
            raise ValueError(
                f"--branch variant {variant!r} collides with a base output key "
                f"('cls'/'us')"
            )
        if variant in specs:
            raise ValueError(f"duplicate --branch variant {variant!r}")
        specs[variant] = (donor_path, top_n)
    return specs


def _resolve_branch_groups(donor_path: Path, top_n: int | None) -> tuple[set[str], int]:
    """Resolve which backbone layer groups a branch donor tunes.

    `top_n` explicit (not `None`): `expected_tuned_groups(top_n)` directly.
    `top_n` is `None`: fall back to the donor's own `.config.json` sidecar
    (a `cca_config.RunConfig`'s `unfreeze_top_n`) when one exists alongside
    it. Raises `ValueError` if neither is available -- a bare backbone-only
    donor (e.g. `extract_tuned_backbone.py`'s output, which is NOT a full
    training artifact and carries no sidecar) can't be resolved without an
    explicit top_n.
    """
    if top_n is not None:
        return expected_tuned_groups(top_n), top_n
    donor_path = Path(donor_path)
    cfg_path = cca_config.config_path_for_weights(donor_path)
    if cfg_path.exists():
        run_config = cca_config.RunConfig.from_json(cfg_path)
        return expected_tuned_groups(run_config.unfreeze_top_n), run_config.unfreeze_top_n
    raise ValueError(
        f"no explicit top_n given and no RunConfig sidecar found at {cfg_path} "
        f"for donor {donor_path} -- pass an explicit top_n "
        f"(--branch <variant>=<donor_path>:<top_n>)"
    )


def provenance_record(
    *,
    backbone_weights: Path,
    us_weights: Path,
    seq_length: int,
    text_channel: str,
    stamp: str,
    n_rows: int,
    n_included: int,
    sample_n: int,
    full: bool,
    backbone_weights_override: Path | None = None,
) -> dict:
    """Provenance for a cache run: what produced these embeddings.

    Records the source weights' mtime+size so a stale/swapped backbone is
    detectable. `stamp` is passed in (not read from the clock) for reproducibility.

    `backbone_weights` is the backbone ACTUALLY used to produce this cache's
    CLS vectors (the US sidecar's own `backbone_weights_path` by default, or
    `--backbone-weights`'s value when given -- e.g. a fine-tuned backbone from
    `extract_tuned_backbone.py`) -- this field's meaning ("what produced these
    embeddings") is unchanged; only the concrete value can now differ from the
    US sidecar's recorded path. `backbone_weights_override` is purely additive:
    `None` when no override was passed (the default, unchanged-behavior case),
    else a stat of the override path -- an explicit, auditable record that a
    non-default backbone was requested, independent of what `backbone_weights`
    resolved to.
    """
    return {
        "backbone_weights": _stat_path(Path(backbone_weights)),
        "backbone_weights_override": (
            _stat_path(Path(backbone_weights_override))
            if backbone_weights_override is not None
            else None
        ),
        "us_weights": _stat_path(Path(us_weights)),
        "seq_length": seq_length,
        "text_channel": text_channel,
        "stamp": stamp,
        "n_rows": n_rows,
        "n_included": n_included,
        "sample_n": sample_n,
        "full": full,
    }


# ---------------------------------------------------------------------------
# Cache I/O (thin shell helpers; reused by downstream phases)
# ---------------------------------------------------------------------------
def _shard_paths(cache_dir: Path, idx: int) -> tuple[Path, Path]:
    return (cache_dir / f"shard_{idx:03d}_cls.npy",
            cache_dir / f"shard_{idx:03d}_meta.parquet")


def _variant_shard_path(cache_dir: Path, idx: int, variant: str) -> Path:
    """Per-variant CLS array path for one shard: DOTTED (`shard_000_cls.<variant>.npy`,
    e.g. `shard_000_cls.rel_branch.npy`), distinct from the base
    `shard_000_cls.npy` -- must never end in the bare `_cls.npy` suffix, which
    would corrupt the append-mode shard-offset glob (see `_count_existing_shards`)."""
    return cache_dir / f"shard_{idx:03d}_cls.{variant}.npy"


def write_shard(
    cache_dir: Path,
    idx: int,
    cls: np.ndarray,
    meta: pl.DataFrame,
    variants: dict[str, np.ndarray] | None = None,
) -> None:
    """Write one row-aligned shard (CLS matrix + metadata).

    `variants`: optional extra per-shard CLS arrays (branched-embed stage-4,
    `docs/design-plans/2026-08-18-stage4-joint-finetune.md`), e.g. a
    grafted-backbone's CLS vectors for a `rel_branch` variant. Each array
    must row-align with `cls`/`meta`, same as the base array. `None`
    (default) writes exactly the two files this function always wrote --
    byte-identical prior behavior.
    """
    if cls.shape[0] != meta.height:
        raise ValueError(
            f"shard {idx}: cls rows {cls.shape[0]} != meta rows {meta.height}"
        )
    cls_path, meta_path = _shard_paths(cache_dir, idx)
    np.save(cls_path, cls.astype(np.float32, copy=False))
    meta.write_parquet(meta_path)
    if variants:
        for variant, arr in variants.items():
            if arr.shape[0] != meta.height:
                raise ValueError(
                    f"shard {idx}: variant {variant!r} rows {arr.shape[0]} != "
                    f"meta rows {meta.height}"
                )
            np.save(
                _variant_shard_path(cache_dir, idx, variant),
                arr.astype(np.float32, copy=False),
            )


def load_cache_meta(cache_dir: Path) -> pl.DataFrame:
    """Load only shard metadata (id, year, us_logit) with a contiguous `emb_row`.

    Skips the (large) CLS matrix — use when only labels/scores are needed (table
    build, US thresholding). `emb_row` ordering matches `load_cache` (both iterate
    shards in zero-padded filename order), so an `emb_row` from here indexes the
    same row `load_cache` would return.
    """
    cache_dir = Path(cache_dir)
    metas = sorted(cache_dir.glob("shard_*_meta.parquet"))
    if not metas:
        raise FileNotFoundError(f"no shards found in {cache_dir}")
    return pl.concat([pl.read_parquet(p) for p in metas]).with_row_index("emb_row")


def load_cache(
    cache_dir: Path, variant: str | None = None
) -> tuple[pl.DataFrame, np.ndarray]:
    """Load all shards in order; return (meta, cls) with a contiguous `emb_row`.

    `meta` carries the shard columns plus `emb_row` (0..N-1), the row index into
    the returned `cls` matrix. Raises if no shards or if a shard is misaligned.

    `variant`: `None` (default) returns the base CLS array -- byte-identical
    to prior behavior, the two-tuple contract every existing caller relies on.
    Passing a variant name returns that variant's CLS array instead (same
    `meta`/alignment), concatenated across shards in the same order. A shard
    missing that variant's array raises `FileNotFoundError` naming the shard
    (a legacy or partially-branched cache must fail loudly, not silently mix
    variant and base rows).
    """
    cache_dir = Path(cache_dir)
    metas = sorted(cache_dir.glob("shard_*_meta.parquet"))
    if not metas:
        raise FileNotFoundError(f"no shards found in {cache_dir}")
    meta_parts, cls_parts = [], []
    for meta_path in metas:
        idx = int(meta_path.name.split("_")[1])
        if variant is None:
            cls_path, _ = _shard_paths(cache_dir, idx)
        else:
            cls_path = _variant_shard_path(cache_dir, idx, variant)
        if not cls_path.exists():
            variant_desc = f"variant {variant!r} " if variant is not None else ""
            raise FileNotFoundError(
                f"shard {idx}: missing {variant_desc}cls array at {cls_path}"
            )
        m = pl.read_parquet(meta_path)
        c = np.load(cls_path)
        if c.shape[0] != m.height:
            raise ValueError(f"shard {idx} misaligned: {c.shape[0]} vs {m.height}")
        meta_parts.append(m)
        cls_parts.append(c)
    meta = pl.concat(meta_parts).with_row_index("emb_row")
    cls = np.concatenate(cls_parts, axis=0)
    return meta, cls


def _count_existing_shards(cache_dir: Path) -> int:
    """Count already-written shards for append-mode shard-offset resolution.

    Counts meta parquets (one per shard, unambiguous), NOT a `_cls.npy` glob
    -- the brief-identified fragility this replaces: a per-variant array
    (`shard_000_cls.<variant>.npy`) doesn't actually match `shard_*_cls.npy`
    (glob `*` doesn't span the required literal `_cls.npy` tail when a
    `.<variant>` segment sits in between), but counting the one-per-shard
    meta file is the unambiguous source of truth regardless. Behavior-
    identical to the old glob for any well-formed (non-branched) cache.
    """
    return len(list(Path(cache_dir).glob("shard_*_meta.parquet")))


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------
def _build_embed_model(
    us_weights: Path,
    backbone_weights: Path | str | None = None,
    branch_specs: dict[str, tuple[str, int | None]] | None = None,
):
    """Build a model {cls, us_logit, [cls.<variant>, ...]} on the weighted DAPT
    backbone + US head, plus one grafted branch backbone per `branch_specs` entry.

    Mirrors slice_eval.apply_us_model's Pattern-2 construction (load UsRunConfig
    sidecar, fresh head, build_inference_model, load_weights) but taps the CLS
    vector and emits the RAW US logit (no calibrator).

    `backbone_weights`: optional override for which backbone checkpoint to load
    (e.g. a fine-tuned backbone from `extract_tuned_backbone.py`, for the
    rel-first encoder-unfreeze re-embed step -- see
    `docs/notes/encoder-unfreeze-strategy.md`). Default `None` uses the US
    sidecar's own `backbone_weights_path`, unchanged from prior behavior. The
    US head's weights are always loaded from `us_weights` regardless -- only
    the encoder producing the CLS vector / us_logit input changes.

    `branch_specs`: optional `{variant: (donor_weights_path, top_n)}` (stage-4
    branched embed model, `docs/design-plans/2026-08-18-stage4-joint-finetune.md`).
    Each variant gets its OWN grafted backbone instance (base backbone + the
    donor's top-`top_n` transformer layers, via `build_grafted_backbone`),
    run through the SAME tokenization inputs as the base backbone (one
    tokenization pass, two-plus encoder passes). The US head always reads the
    BASE cls (`ica_fusion`'s deployed config keeps `us` on `"base"`).

    Ordering: branch backbones are constructed AFTER the us-weights load and
    the base-override reapplication below -- same rule as the
    "CRITICAL ORDER FIX" comment on the base override (a branch backbone is a
    separate instance, not wired into `inf`, so it can't be clobbered by
    `inf.load_weights`, but keeping every backbone-resolution step in this
    one place, in this order, avoids relying on that instance-identity
    argument holding forever).

    Returns `(model, us_cfg, backbone_path, branch_provenance)` --
    `backbone_path` is the resolved base-backbone path actually loaded, for
    provenance. `branch_provenance` is `{variant: {"donor": _stat_path(...),
    "groups": [...], "unfreeze_top_n": ..., "graft_verification": {...}}}`
    (empty dict when `branch_specs` is `None`/empty).
    """
    us_cfg = UsRunConfig.from_json(config_path_for_weights(us_weights))
    us_head = ClassificationHead(hidden_dim=us_cfg.head.hidden_dim, name=us_cfg.head.name)
    # Resolve here (not just inside load_dapt_backbone) so provenance records
    # the path actually loaded, not a machine-foreign sidecar path.
    backbone_path = (
        Path(backbone_weights)
        if backbone_weights is not None
        else resolve_backbone_path(us_cfg.backbone_weights_path)
    )
    backbone = load_dapt_backbone(backbone_path)
    us_cfg.validate_against_backbone(backbone)
    # Build + load weights via the inference model (populates backbone + head).
    inf = build_inference_model(
        backbone=backbone, heads={us_cfg.head.name: us_head}, seq_length=us_cfg.seq_length
    )
    inf.load_weights(str(us_weights), skip_mismatch=False)
    # CRITICAL ORDER FIX (2026-07-29): `us_weights` may be a FULL-model save
    # (us_classifier.weights.h5, the 503MB text-mode smoke, is the default) --
    # loading it over the inference model OVERWRITES the backbone, silently
    # undoing any `backbone_weights` override loaded above. Harmless for every
    # production cache (the smoke's backbone is exact frozen DAPT, so the
    # clobber was a no-op) but it voided the first tuned re-embed: all three
    # "tuned" caches of 2026-07-29 were actually DAPT embeds. Re-applying the
    # override AFTER the us-weights load guarantees the requested backbone is
    # the one that embeds.
    if backbone_weights is not None:
        backbone.load_weights(str(backbone_path), skip_mismatch=False)

    # Branch backbones (stage-4 branched embed model): built AFTER the
    # us-weights load + base-override reapplication above -- see the
    # ordering note in this function's docstring.
    branch_backbones: dict[str, object] = {}
    branch_provenance: dict[str, dict] = {}
    for variant, (donor_path_raw, top_n) in (branch_specs or {}).items():
        donor_path = Path(donor_path_raw)
        groups, resolved_top_n = _resolve_branch_groups(donor_path, top_n)
        branch_backbone, diffs = build_grafted_backbone(backbone_path, donor_path, groups)
        branch_backbones[variant] = branch_backbone
        branch_provenance[variant] = {
            "donor": _stat_path(donor_path),
            "groups": sorted(groups),
            "unfreeze_top_n": resolved_top_n,
            "graft_verification": diffs,
        }

    # Output graph on the now-weighted instances.
    tok = keras.Input(shape=(us_cfg.seq_length,), dtype="int32", name="token_ids")
    pad = keras.Input(shape=(us_cfg.seq_length,), dtype="int32", name="padding_mask")
    seq_out = backbone({"token_ids": tok, "padding_mask": pad})
    cls = seq_out[:, 0, :]
    us_logit = us_head(cls)
    outputs = {"cls": cls, "us": us_logit}
    for variant, branch_backbone in branch_backbones.items():
        branch_seq_out = branch_backbone({"token_ids": tok, "padding_mask": pad})
        outputs[f"cls.{variant}"] = branch_seq_out[:, 0, :]
    model = keras.Model(
        inputs={"token_ids": tok, "padding_mask": pad},
        outputs=outputs,
    )
    return model, us_cfg, backbone_path, branch_provenance


def main(
    include_ids_path=None,
    sample_n=0,
    full=False,
    stamp="unstamped",
    out_suffix="run",
    shard_size=250_000,
    batch_size=256,
    years=None,
    append=False,
    corpus_subdir="api_corpus",
    year_column="year",
    lead_column="lead_paragraph",
    label_column=None,
    include_year=True,
    limit=None,
    source_pattern=None,
    backbone_weights=None,
    lead_fallback_column=None,
    dedupe_ids=False,
    branch_specs=None,
):
    keras.config.set_dtype_policy(config.DTYPE_POLICY)
    keras.utils.set_random_seed(200)

    cache_dir = config.CCA_EMBED_CACHE_DIR / out_suffix
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Append mode (split a big embed into parts that share one cache): continue
    # shard numbering after the shards already present, so a later run extends the
    # same canonical cache rather than overwriting shard_000.
    shard_offset = _count_existing_shards(cache_dir) if append else 0

    # Load corpus (id, headline, <lead_column>, [year], [label_column],
    # headline_with_lead). The API corpus carries `year`; the LDC corpus partitions
    # on `publication_year` (normalized to `year`). The US-filter training source
    # (`us_filter/ldc_labeled.parquet`) has NO year column and a `stripped_text`
    # lead channel — so for that case: include_year=False, lead_column="stripped_text",
    # label_column="us_label".
    addl = [year_column] if include_year else []
    if label_column is not None:
        addl.append(label_column)
    corpus = data_from_parquet(
        config.PROJECT_ROOT, corpus_subdir, addl_columns=addl,
        lead_column=lead_column, pattern=source_pattern,
        lead_fallback_column=lead_fallback_column,
    )
    if include_year and year_column != "year":
        corpus = corpus.rename({year_column: "year"})
    # Dedupe BEFORE the year filter (see dedupe_by_id: split year-range jobs
    # must agree globally on the canonical copy of a cross-year duplicate).
    if dedupe_ids:
        before = corpus.height
        corpus = dedupe_by_id(corpus)
        print(f"dedupe_ids: {before} -> {corpus.height} rows "
              f"({before - corpus.height} duplicate rows dropped)")  # LOG
    # Drop unlabeled rows when a label channel is requested (mirrors
    # create_us_filter_data's null-label drop) — embedding them would waste the pass.
    if label_column is not None:
        kept = corpus.filter(pl.col(label_column).is_not_null())
        print(f"label filter {label_column} non-null: {kept.height}/{corpus.height}")  # LOG
        corpus = kept
    if years is not None:
        if not include_year:
            raise ValueError("--years requires a year column (include_year is False)")
        lo, hi = years
        corpus = corpus.filter(pl.col("year").cast(pl.Int64).is_between(lo, hi))
        print(f"year filter {lo}-{hi}: {corpus.height} articles")  # LOG
    # Smoke-test slice: head(limit) BEFORE selection, no year needed (unlike the
    # stratified sample path) — for verifying a new corpus/channel cheaply.
    if limit is not None:
        corpus = corpus.head(limit)
        print(f"limit: head({limit}) -> {corpus.height} articles")  # LOG

    include_ids = None
    if include_ids_path is not None:
        include_ids = pl.read_parquet(include_ids_path)["id"].to_list()

    selected = select_articles(corpus, include_ids, sample_n, full)
    # Defense-in-depth: the split logic downstream requires unique ids.
    if selected["id"].n_unique() != selected.height:
        raise ValueError(
            f"selected set has duplicate ids: {selected.height} rows, "
            f"{selected['id'].n_unique()} unique"
        )
    n_included = 0 if include_ids is None else selected.filter(
        pl.col("id").is_in(include_ids)
    ).height
    print(f"Embedding {selected.height} articles "
          f"(included={n_included}, sample_n={sample_n}, full={full})")  # LOG

    model, us_cfg, backbone_path, branch_provenance = _build_embed_model(
        config.US_FILTER_CLASSIFIER_WEIGHTS, backbone_weights=backbone_weights,
        branch_specs=branch_specs,
    )
    variant_names = list(branch_provenance.keys())
    preproc = ClassifierPreprocessor(
        SEQ_LENGTH=us_cfg.seq_length, text_key=us_cfg.text_key,
        label_keys={}, endpoint_model=True, target_dtype=us_cfg.target_dtype,
    )

    n = selected.height
    n_shards = (n + shard_size - 1) // shard_size
    for idx in range(n_shards):
        chunk = selected.slice(idx * shard_size, shard_size)
        ds = (
            tf.data.Dataset.from_tensor_slices(
                {us_cfg.text_key: chunk[us_cfg.text_key].to_list()}
            )
            .batch(batch_size)
            .map(preproc, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
        preds = model.predict(ds, verbose=0)
        cls = np.asarray(preds["cls"], dtype=np.float32)
        us_logit = np.asarray(preds["us"], dtype=np.float32).reshape(-1)
        variant_arrays = {}
        for variant in variant_names:
            arr = np.asarray(preds[f"cls.{variant}"], dtype=np.float32)
            if not np.isfinite(arr).all():
                raise ValueError(
                    f"shard {idx}: non-finite embeddings for variant {variant!r}"
                )
            variant_arrays[variant] = arr
        if not np.isfinite(cls).all() or not np.isfinite(us_logit).all():
            raise ValueError(f"shard {idx}: non-finite embeddings/logits produced")
        meta_cols = ["id"] + (["year"] if include_year else [])
        if label_column is not None:
            meta_cols.append(label_column)
        meta = chunk.select(meta_cols).with_columns(
            pl.Series("us_logit", us_logit)
        )
        write_shard(cache_dir, shard_offset + idx, cls, meta, variants=variant_arrays or None)
        # Vigilance spot-check, per shard.
        variant_std = " ".join(
            f"cls_std.{v}={variant_arrays[v].std():.4f}" for v in variant_names
        )
        print(f"  shard {shard_offset + idx}: rows={cls.shape[0]} "
              f"cls_std={float(cls.std()):.4f} us_logit[min/mean/max]="
              f"{us_logit.min():.2f}/{us_logit.mean():.2f}/{us_logit.max():.2f}"
              + (f" {variant_std}" if variant_std else ""))  # LOG

    prov = provenance_record(
        backbone_weights=backbone_path,
        us_weights=config.US_FILTER_CLASSIFIER_WEIGHTS,
        seq_length=us_cfg.seq_length, text_channel=us_cfg.text_key,
        stamp=stamp, n_rows=n, n_included=n_included, sample_n=sample_n, full=full,
        backbone_weights_override=Path(backbone_weights) if backbone_weights is not None else None,
    )
    prov["years"] = list(years) if years is not None else None
    prov["shard_offset"] = shard_offset
    prov["lead_column"] = lead_column
    prov["lead_fallback_column"] = lead_fallback_column
    prov["label_column"] = label_column
    prov["dedupe_ids"] = dedupe_ids
    prov["branches"] = branch_provenance
    (cache_dir / f"provenance.{shard_offset:03d}.json").write_text(
        json.dumps(prov, indent=2)
    )
    print(f"Wrote {n_shards} shards (offset {shard_offset}, {n} rows) to {cache_dir}")  # LOG


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser (split out from `__main__` so tests can check
    flag defaults without invoking `main()`'s model-loading side effects)."""
    ap = argparse.ArgumentParser(description="Embed the API corpus (CLS + US logit).")
    ap.add_argument("--include-ids", default=None,
                    help="parquet with an `id` column to force-include (e.g. DoCA positives)")
    ap.add_argument("--sample-n", type=int, default=0,
                    help="stratified-by-year sample size drawn from non-included rows")
    ap.add_argument("--full", action="store_true", help="embed the entire corpus")
    ap.add_argument("--stamp", required=True, help="reproducibility stamp (e.g. YYYYMMDD)")
    ap.add_argument("--out-suffix", required=True, help="cache subdir name under CCA_EMBED_CACHE_DIR")
    ap.add_argument("--shard-size", type=int, default=250_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--years", default=None,
                    help="inclusive year range 'LO-HI' to embed (e.g. 1960-1975)")
    ap.add_argument("--append", action="store_true",
                    help="continue shard numbering after existing shards in the cache dir")
    ap.add_argument("--corpus", default="api_corpus",
                    help="corpus subdir under the data root (e.g. api_corpus, ldc_corpus)")
    ap.add_argument("--year-column", default="year",
                    help="name of the year column in the corpus (LDC: publication_year)")
    ap.add_argument("--lead-column", default="lead_paragraph",
                    help="text column feeding headline+lead (US filter: stripped_text)")
    ap.add_argument("--lead-fallback-column", default=None,
                    help="column supplying the post-separator text where the lead is "
                         "empty (post-1995 API embed: abstrct — the coalesce policy of "
                         "roadmap §A item 1; NOT for pre-1996 eras without the "
                         "pre-registered channel experiment)")
    ap.add_argument("--label-column", default=None,
                    help="label column to carry into shard meta + drop nulls "
                         "(US-filter training cache: us_label)")
    ap.add_argument("--no-year", dest="include_year", action="store_false",
                    help="source has no year column (e.g. us_filter/ldc_labeled.parquet)")
    ap.add_argument("--limit", type=int, default=None,
                    help="embed only the first N rows (smoke test; no stratification)")
    ap.add_argument("--source-pattern", default=None,
                    help="exact parquet path under the data root (overrides --corpus "
                         "glob; e.g. us_filter/ldc_labeled.parquet)")
    ap.add_argument("--dedupe-ids", dest="dedupe_ids", action="store_true",
                    help="deterministically resolve duplicate ids before embedding "
                         "(required for the 1996-2025 API corpus: 911 pull-overlap "
                         "dup ids; default off keeps the loud duplicate-id guard)")
    ap.add_argument("--backbone-weights", default=None,
                    help="backbone .weights.h5 to load instead of the US sidecar's own "
                         "backbone_weights_path (e.g. a fine-tuned backbone from "
                         "extract_tuned_backbone.py). Default: unchanged prior behavior.")
    ap.add_argument("--branch", action="append", default=None,
                    help="variant=donor_backbone_weights[:top_n] (repeatable): grafts the "
                         "donor's top-N transformer layers onto the base backbone, adding "
                         "a cls.<variant> array to every shard (stage-4 branched embed "
                         "model). top_n defaults to 1. E.g. --branch "
                         "rel_branch=../relevance/tuned_backbone.job8823087.weights.h5:1")
    return ap


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    years = None
    if args.years is not None:
        lo, hi = args.years.split("-")
        years = (int(lo), int(hi))
    main(
        include_ids_path=args.include_ids, sample_n=args.sample_n, full=args.full,
        stamp=args.stamp, out_suffix=args.out_suffix,
        shard_size=args.shard_size, batch_size=args.batch_size,
        years=years, append=args.append,
        corpus_subdir=args.corpus, year_column=args.year_column,
        lead_column=args.lead_column, label_column=args.label_column,
        include_year=args.include_year, limit=args.limit,
        source_pattern=args.source_pattern,
        backbone_weights=args.backbone_weights,
        lead_fallback_column=args.lead_fallback_column,
        dedupe_ids=args.dedupe_ids,
        branch_specs=parse_branch_specs(args.branch),
    )
