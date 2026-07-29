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

import src.config as config
from src.data_setup.data import data_from_parquet
from src.preproc.preprocessor import ClassifierPreprocessor
from src.model_setup.backbone import load_dapt_backbone
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_inference_model
from src.us_config import UsRunConfig, config_path_for_weights


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
    def _stat(p: Path) -> dict:
        return {"path": str(p), "exists": p.exists(),
                "size": p.stat().st_size if p.exists() else None,
                "mtime": int(p.stat().st_mtime) if p.exists() else None}

    return {
        "backbone_weights": _stat(Path(backbone_weights)),
        "backbone_weights_override": (
            _stat(Path(backbone_weights_override))
            if backbone_weights_override is not None
            else None
        ),
        "us_weights": _stat(Path(us_weights)),
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


def write_shard(cache_dir: Path, idx: int, cls: np.ndarray, meta: pl.DataFrame) -> None:
    """Write one row-aligned shard (CLS matrix + metadata)."""
    if cls.shape[0] != meta.height:
        raise ValueError(
            f"shard {idx}: cls rows {cls.shape[0]} != meta rows {meta.height}"
        )
    cls_path, meta_path = _shard_paths(cache_dir, idx)
    np.save(cls_path, cls.astype(np.float32, copy=False))
    meta.write_parquet(meta_path)


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


def load_cache(cache_dir: Path) -> tuple[pl.DataFrame, np.ndarray]:
    """Load all shards in order; return (meta, cls) with a contiguous `emb_row`.

    `meta` carries the shard columns plus `emb_row` (0..N-1), the row index into
    the returned `cls` matrix. Raises if no shards or if a shard is misaligned.
    """
    cache_dir = Path(cache_dir)
    metas = sorted(cache_dir.glob("shard_*_meta.parquet"))
    if not metas:
        raise FileNotFoundError(f"no shards found in {cache_dir}")
    meta_parts, cls_parts = [], []
    for meta_path in metas:
        idx = int(meta_path.name.split("_")[1])
        cls_path, _ = _shard_paths(cache_dir, idx)
        m = pl.read_parquet(meta_path)
        c = np.load(cls_path)
        if c.shape[0] != m.height:
            raise ValueError(f"shard {idx} misaligned: {c.shape[0]} vs {m.height}")
        meta_parts.append(m)
        cls_parts.append(c)
    meta = pl.concat(meta_parts).with_row_index("emb_row")
    cls = np.concatenate(cls_parts, axis=0)
    return meta, cls


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------
def _build_embed_model(us_weights: Path, backbone_weights: Path | str | None = None):
    """Build a dual-output model {cls, us_logit} on the weighted DAPT backbone + US head.

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

    Returns `(model, us_cfg, backbone_path)` -- `backbone_path` is the
    resolved path actually loaded, for provenance.
    """
    us_cfg = UsRunConfig.from_json(config_path_for_weights(us_weights))
    us_head = ClassificationHead(hidden_dim=us_cfg.head.hidden_dim, name=us_cfg.head.name)
    backbone_path = (
        Path(backbone_weights) if backbone_weights is not None else Path(us_cfg.backbone_weights_path)
    )
    backbone = load_dapt_backbone(backbone_path)
    us_cfg.validate_against_backbone(backbone)
    # Build + load weights via the inference model (populates backbone + head).
    inf = build_inference_model(
        backbone=backbone, heads={us_cfg.head.name: us_head}, seq_length=us_cfg.seq_length
    )
    inf.load_weights(str(us_weights), skip_mismatch=False)
    # Dual-output graph on the now-weighted instances.
    tok = keras.Input(shape=(us_cfg.seq_length,), dtype="int32", name="token_ids")
    pad = keras.Input(shape=(us_cfg.seq_length,), dtype="int32", name="padding_mask")
    seq_out = backbone({"token_ids": tok, "padding_mask": pad})
    cls = seq_out[:, 0, :]
    us_logit = us_head(cls)
    model = keras.Model(
        inputs={"token_ids": tok, "padding_mask": pad},
        outputs={"cls": cls, "us": us_logit},
    )
    return model, us_cfg, backbone_path


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
):
    keras.config.set_dtype_policy(config.DTYPE_POLICY)
    keras.utils.set_random_seed(200)

    cache_dir = config.CCA_EMBED_CACHE_DIR / out_suffix
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Append mode (split a big embed into parts that share one cache): continue
    # shard numbering after the shards already present, so a later run extends the
    # same canonical cache rather than overwriting shard_000.
    shard_offset = len(list(cache_dir.glob("shard_*_cls.npy"))) if append else 0

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
    )
    if include_year and year_column != "year":
        corpus = corpus.rename({year_column: "year"})
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

    model, us_cfg, backbone_path = _build_embed_model(
        config.US_FILTER_CLASSIFIER_WEIGHTS, backbone_weights=backbone_weights
    )
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
        if not np.isfinite(cls).all() or not np.isfinite(us_logit).all():
            raise ValueError(f"shard {idx}: non-finite embeddings/logits produced")
        meta_cols = ["id"] + (["year"] if include_year else [])
        if label_column is not None:
            meta_cols.append(label_column)
        meta = chunk.select(meta_cols).with_columns(
            pl.Series("us_logit", us_logit)
        )
        write_shard(cache_dir, shard_offset + idx, cls, meta)
        # Vigilance spot-check, per shard.
        print(f"  shard {shard_offset + idx}: rows={cls.shape[0]} "
              f"cls_std={float(cls.std()):.4f} us_logit[min/mean/max]="
              f"{us_logit.min():.2f}/{us_logit.mean():.2f}/{us_logit.max():.2f}")  # LOG

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
    prov["label_column"] = label_column
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
    ap.add_argument("--backbone-weights", default=None,
                    help="backbone .weights.h5 to load instead of the US sidecar's own "
                         "backbone_weights_path (e.g. a fine-tuned backbone from "
                         "extract_tuned_backbone.py). Default: unchanged prior behavior.")
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
    )
