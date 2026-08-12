# pattern: Imperative Shell (build_headline_lead_texts / cosine_similarity_stats /
#   assert_cosine_ok / diaspora_recall_at / graft_pass_fail are the pure
#   Functional Core, colocated here rather than split into a second file --
#   this is a one-off experiment script, mirroring the layout of
#   scripts/eval_rel_text_artifact.py; everything else -- filesystem stats,
#   backbone construction, weight grafting, CLS extraction, cache I/O, head
#   scoring -- is the Imperative Shell)
"""Stage-1 graft test (docs/notes/branched-encoder-strategy.md, experiment
ladder item 1): verify that the rel-first top-1-unfreeze fine-tune
(job8823087, "only transformer_layer_11 received a real learning rate") is
reproducible by COMPOSING pristine-DAPT layers 0-10 with the tuned
transformer_layer_11, rather than needing the full tuned encoder.

Pass ⇒ the branched-encoder frame is real: a K=1 branch (shared frozen trunk
+ a per-head top-layer swap) gets the rel gain at ~(12+1)/12 of one encoder
pass, not two full passes. Fail ⇒ the ~2.3e-3 sub-branch AdamW-weight-decay
drift below the unfrozen layer is load-bearing, not noise, and the next step
is a hard-freeze (`layer.trainable=False`) re-run so the graft is exact by
construction.

Three legs scored over the same 1,131-row hand-coded eval set
(`validation/ica_coding_template_coded.csv`):
  1. control   -- production rel head on locally-embedded PRISTINE-DAPT CLS.
                  Validates the harness end-to-end against the recorded
                  eval-set numbers (own-terms ROC 0.829 / vs-ICA 0.783 /
                  diaspora recall 0.382 @ 0.30 review rate).
  2. reference -- tuned head (relevance_tuned.weights.h5) on locally-embedded
                  FULL-TUNED CLS. The same-session target the graft leg is
                  measured against (prose record: 0.836 / 0.853 / 0.662).
  3. graft     -- the SAME tuned head, on CLS from a GRAFTED encoder: a
                  fresh pristine-DAPT backbone with ONLY transformer_layer_11
                  overwritten from the tuned backbone. This is the test.

Pre-registered pass/fail (graft vs the SAME-SESSION reference leg, not the
prose record): |Δ vs-ICA ROC| <= 0.01 AND |Δ diaspora recall @ 0.30| <=
2/n_diaspora (n_diaspora expected 68 -> ~0.0294). Own-terms ROC delta is
reported as a guardrail, not gated.

A harness self-check runs before any leg is trusted: per-row cosine
similarity between each locally-embedded CLS matrix and the matching
production embed cache (relevance_train / relevance_train_tuned), id-joined.
This must be ~1.0 (cluster-vs-local exactness was verified 2026-07-29) --
if it isn't, `assert_cosine_ok` raises rather than let a silently-broken
text/tokenization channel produce misleading leg metrics.

CPU-only (see the `tf.config.set_visible_devices` call immediately after the
`tensorflow` import, below): the tensorflow-metal GPU `predict` path distorts
metrics by ~+-0.01, the same order as this experiment's pass band
(precedent: scripts/compare_scoring_paths.py).

Run from project root (module form required -- `python scripts/...` puts
scripts/ on sys.path, breaking the `src.*` imports):
    uv run python -m scripts.graft_test

Writes `cca_doca/experiments/graft_test.json` (overwrite-safe: falls back to
a timestamped sibling filename if that path is already taken).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf

# Must run BEFORE any model construction / other project imports that build
# models -- device visibility has to be set before TensorFlow initializes
# its runtime. See module docstring.
tf.config.set_visible_devices([], "GPU")

import keras  # noqa: E402  (see CPU-only ordering note above)
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import src.config as config  # noqa: E402
from src.calibration.sidecar import calibration_path_for_weights, load_calibration  # noqa: E402
from src.embed_corpus import load_cache  # noqa: E402
from src.extract_tuned_backbone import expected_tuned_groups, layer_diff_summary  # noqa: E402
from src.model_setup.backbone import load_dapt_backbone  # noqa: E402
from src.preproc.preprocessor import ClassifierPreprocessor  # noqa: E402
from src.validation.relevance_slice_eval import apply_relevance_model  # noqa: E402
from scripts.eval_rel_text_artifact import diaspora_slice_metrics, own_terms_metrics  # noqa: E402

EVAL_CSV = config.PROJECT_ROOT / "validation" / "ica_coding_template_coded.csv"
DAPT_BACKBONE_WEIGHTS = config.DAPT_BACKBONE_WEIGHTS
TUNED_BACKBONE_WEIGHTS = config.PROJECT_ROOT / "relevance" / "tuned_backbone.job8823087.weights.h5"
CONTROL_REL_WEIGHTS = config.RELEVANCE_DOCA_WEIGHTS
REFERENCE_REL_WEIGHTS = config.PROJECT_ROOT / "relevance" / "relevance_tuned.weights.h5"
CACHE_DAPT_SUFFIX = "relevance_train"
CACHE_TUNED_SUFFIX = "relevance_train_tuned"
SEQ_LENGTH = 128
TEXT_KEY = "headline_with_lead"
UNFREEZE_TOP_N = 1  # job8823087: only the top transformer layer was unfrozen
TOP_LAYER_GROUP = next(iter(expected_tuned_groups(UNFREEZE_TOP_N)))
MATCHED_POSITIVE_RATE = 0.30
DEFAULT_OUT = config.CCA_DOCA_DIR / "experiments" / "graft_test.json"

# Pre-registered pass/fail band (branched-encoder-strategy.md stage 1):
# graft vs the SAME-SESSION reference leg, not the prose record below.
VS_ICA_ROC_BAND = 0.01
DIASPORA_RECALL_NUMERATOR_BAND = 2  # anchors; denominator = n_diaspora observed this run

# Prose-record numbers (encoder-unfreeze-strategy.md / branched-encoder-strategy.md)
# -- sanity targets for the control/reference legs. NOT the pass/fail gate.
EXPECTED = {
    "control": {"own_terms_roc": 0.829, "vs_ica_roc": 0.783, "diaspora_recall_030": 0.382},
    "reference": {"own_terms_roc": 0.836, "vs_ica_roc": 0.853, "diaspora_recall_030": 0.662},
}
SANITY_ROC_TOL = 0.01


# ---------------------------------------------------------------------------
# Functional core
# ---------------------------------------------------------------------------
def build_headline_lead_texts(headlines: list, leads: list) -> list[str]:
    """Pure: the exact text channel the embed caches were built with.

    `f"{headline}</s>{lead}"`, nulls coalesced to empty string.
    """
    if len(headlines) != len(leads):
        raise ValueError(
            f"headlines/leads length mismatch: {len(headlines)} vs {len(leads)}"
        )
    return [
        f"{h if h else ''}</s>{lead if lead else ''}" for h, lead in zip(headlines, leads)
    ]


def cosine_similarity_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Pure: per-row cosine similarity summary between two row-aligned matrices."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.shape[0] == 0:
        raise ValueError("cosine_similarity_stats requires at least one row")
    num = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    n_zero_norm = int(np.sum(denom == 0.0))
    denom_safe = np.where(denom == 0.0, np.nan, denom)
    cos = num / denom_safe
    return {
        "n": int(a.shape[0]),
        "min": float(np.nanmin(cos)),
        "mean": float(np.nanmean(cos)),
        "p01": float(np.nanpercentile(cos, 1)),
        "n_zero_norm": n_zero_norm,
    }


def assert_cosine_ok(stats: dict, leg_name: str, min_mean: float = 0.999) -> None:
    """Pure (raises, no I/O): hard-fail the harness self-check.

    A broken text/tokenization channel or a stale/mismatched cache shows up
    here as low cosine similarity; this must trip BEFORE any leg's metrics
    are trusted (see module docstring's harness self-check).
    """
    if stats["mean"] < min_mean:
        raise ValueError(
            f"harness self-check FAILED for {leg_name}: mean cosine {stats['mean']:.6f} "
            f"< {min_mean} (min={stats['min']:.6f}, p01={stats['p01']:.6f}, n={stats['n']}). "
            f"Local embedding does not reproduce the cached CLS -- stop, do not trust "
            f"downstream leg metrics."
        )


def diaspora_recall_at(diaspora_slice: dict, target_rate: float) -> float | None:
    """Pure: pull `diaspora_recall` out of a `diaspora_slice_metrics` result
    at the matched-positive-rate point equal to `target_rate`."""
    for pt in diaspora_slice["matched_rate_points"]:
        if pt["target_positive_rate"] == target_rate:
            return pt["diaspora_recall"]
    raise ValueError(f"no matched_rate_point at target_positive_rate={target_rate}")


def graft_pass_fail(reference: dict, graft: dict, n_diaspora: int) -> dict:
    """Pure: the pre-registered graft-vs-(same-session)reference comparison.

    Raises if either leg has no diaspora rows to compute a recall from
    (`diaspora_recall_at` returns `None` in that case) -- `n_diaspora == 0`
    would otherwise divide-by-zero on `diaspora_band` below AND make the
    recall delta meaningless, not just unavailable, so this must fail loudly
    rather than propagate a `None`/NaN through the pass/fail booleans.
    """
    if n_diaspora <= 0:
        raise ValueError(f"n_diaspora must be positive; got {n_diaspora}")
    d_vs_ica = graft["vs_ica_roc"] - reference["vs_ica_roc"]
    d_own_terms = graft["own_terms"]["roc_auc"] - reference["own_terms"]["roc_auc"]
    r_ref = diaspora_recall_at(reference["diaspora_slice"], MATCHED_POSITIVE_RATE)
    r_graft = diaspora_recall_at(graft["diaspora_slice"], MATCHED_POSITIVE_RATE)
    if r_ref is None or r_graft is None:
        raise ValueError(
            f"diaspora recall unavailable at rate={MATCHED_POSITIVE_RATE} "
            f"(reference={r_ref}, graft={r_graft}) despite n_diaspora={n_diaspora} "
            f"-- diaspora_slice_metrics/diaspora_recall_at contract violated"
        )
    d_diaspora = r_graft - r_ref
    diaspora_band = DIASPORA_RECALL_NUMERATOR_BAND / n_diaspora
    vs_ica_pass = abs(d_vs_ica) <= VS_ICA_ROC_BAND
    diaspora_pass = abs(d_diaspora) <= diaspora_band
    return {
        "delta_vs_ica_roc": d_vs_ica,
        "delta_own_terms_roc": d_own_terms,
        "delta_diaspora_recall_030": d_diaspora,
        "reference_diaspora_recall_030": r_ref,
        "graft_diaspora_recall_030": r_graft,
        "n_diaspora": n_diaspora,
        "vs_ica_roc_band": VS_ICA_ROC_BAND,
        "diaspora_recall_band": diaspora_band,
        "vs_ica_pass": vs_ica_pass,
        "diaspora_pass": diaspora_pass,
        "overall_pass": vs_ica_pass and diaspora_pass,
    }


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------
def file_stat(path: Path) -> dict:
    """Stat a provenance artifact path (mtime/size) for the output record."""
    path = Path(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else None,
        "mtime_utc": (
            datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
            if path.exists() else None
        ),
    }


def default_out_path() -> Path:
    """Overwrite-safe output path: DEFAULT_OUT, or a timestamped sibling if
    that filename is already taken."""
    if not DEFAULT_OUT.exists():
        return DEFAULT_OUT
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    alt = DEFAULT_OUT.with_name(f"graft_test.{stamp}.json")
    print(f"{DEFAULT_OUT} already exists -- writing to {alt} instead")  # LOG
    return alt


def load_eval_frame() -> pl.DataFrame:
    return pl.read_csv(EVAL_CSV)


def build_cls_model(backbone) -> keras.Model:
    """Bare CLS-extraction graph over a (weight-loaded) backbone instance.

    Mirrors embed_corpus._build_embed_model's tap point (seq_out[:, 0, :])
    without the US head -- this experiment only needs the CLS vector.
    """
    tok = keras.Input(shape=(SEQ_LENGTH,), dtype="int32", name="token_ids")
    pad = keras.Input(shape=(SEQ_LENGTH,), dtype="int32", name="padding_mask")
    seq_out = backbone({"token_ids": tok, "padding_mask": pad})
    cls = seq_out[:, 0, :]
    return keras.Model(inputs={"token_ids": tok, "padding_mask": pad}, outputs=cls)


def extract_cls(backbone, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Run `texts` through `backbone`'s CLS extraction graph; return (n, 768)."""
    preproc = ClassifierPreprocessor(
        SEQ_LENGTH=SEQ_LENGTH, text_key=TEXT_KEY, label_keys={},
        endpoint_model=True, target_dtype="float32",
    )
    batch = preproc({TEXT_KEY: texts})
    model = build_cls_model(backbone)
    cls = np.asarray(model.predict(batch, batch_size=batch_size, verbose=0), dtype=np.float32)
    if not np.isfinite(cls).all():
        raise ValueError("non-finite CLS features produced")
    return cls


def build_graft_backbone():
    """Load DAPT, tuned, and grafted backbones.

    Graft mechanics: a fresh pristine-DAPT instance with ONLY
    `TOP_LAYER_GROUP`'s weights overwritten from the tuned backbone (Pattern
    A in-process mutation -- the CLS-extraction model built from `graft`
    afterward reads the mutated instance).

    Returns `(dapt, tuned, graft, diff_vs_dapt, diff_vs_tuned)` -- the two
    diff summaries are `layer_diff_summary` per-group max|delta| tables
    (`src.extract_tuned_backbone`), verifying the graft differs from DAPT
    ONLY in `TOP_LAYER_GROUP` and matches the tuned backbone EXACTLY
    (0.0 delta) there.
    """
    dapt = load_dapt_backbone(DAPT_BACKBONE_WEIGHTS)
    tuned = load_dapt_backbone(TUNED_BACKBONE_WEIGHTS)
    graft = load_dapt_backbone(DAPT_BACKBONE_WEIGHTS)

    dapt_paths = [w.path for w in dapt.weights]
    tuned_paths = [w.path for w in tuned.weights]
    graft_paths = [w.path for w in graft.weights]
    if not (dapt_paths == tuned_paths == graft_paths):
        raise ValueError("backbone weight ordering diverged across the three instances")

    graft.get_layer(TOP_LAYER_GROUP).set_weights(
        tuned.get_layer(TOP_LAYER_GROUP).get_weights()
    )

    dapt_arrays = [np.asarray(w.numpy()) for w in dapt.weights]
    tuned_arrays = [np.asarray(w.numpy()) for w in tuned.weights]
    graft_arrays = [np.asarray(w.numpy()) for w in graft.weights]

    diff_vs_dapt = layer_diff_summary(graft_paths, graft_arrays, dapt_arrays)
    diff_vs_tuned = layer_diff_summary(graft_paths, graft_arrays, tuned_arrays)
    return dapt, tuned, graft, diff_vs_dapt, diff_vs_tuned


def score_rel_head(cls: np.ndarray, weights_path: Path) -> np.ndarray:
    """Rel-head scores over `cls` features.

    Platt-calibrated when the weights carry a `.calibration.json` sidecar; raw
    logits otherwise (all metrics in this script are rank-based, so a missing
    calibration changes nothing downstream -- scratch diagnostic heads from the
    2026-08-12 metal-artifact investigation have no sidecar yet).
    """
    logits = apply_relevance_model(cls, weights_path=weights_path)
    cal_path = calibration_path_for_weights(weights_path)
    if Path(cal_path).exists():
        scores = load_calibration(cal_path).transform(logits)
    else:
        print(f"  no calibration sidecar for {weights_path}; using raw logits")  # LOG
        scores = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError(f"non-finite scores from {weights_path}")
    return scores


def joined_cache_cls(
    eval_df_indexed: pl.DataFrame, cache_suffix: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """id-join `eval_df_indexed` (must carry an `orig_row` column) against an
    embed cache.

    Returns `(orig_rows, cached_cls, n_missing)` -- `orig_rows` indexes the
    caller's original (un-joined) eval frame / local CLS array, so the two
    can be compared row-for-row via `cls_local[orig_rows]` vs `cached_cls`.
    """
    meta_full, cls = load_cache(config.CCA_EMBED_CACHE_DIR / cache_suffix)
    meta_by_id = meta_full.with_columns(
        pl.col("id").cast(pl.Utf8).alias("id_str")
    ).select(["id_str", "emb_row"])
    df = eval_df_indexed.with_columns(pl.col("id").cast(pl.Utf8).alias("id_str")).join(
        meta_by_id, on="id_str", how="left"
    )
    n_missing = df.filter(pl.col("emb_row").is_null()).height
    df = df.filter(pl.col("emb_row").is_not_null())
    orig_rows = df["orig_row"].to_numpy().astype(int)
    cached = cls[df["emb_row"].to_numpy().astype(int)]
    print(
        f"  cache join {cache_suffix}: {eval_df_indexed.height} rows, "
        f"{n_missing} missing, {df.height} matched"
    )  # LOG
    return orig_rows, cached, n_missing


def leg_report(name: str, eval_df: pl.DataFrame, scores: np.ndarray) -> dict:
    """Own-terms / vs-ICA / diaspora-slice metrics for one leg's scores.

    Mirrors scripts/eval_rel_text_artifact.py:_model_report (same label
    columns, same null-masking, same diaspora slice mechanism) so
    graft-test numbers are directly comparable to the prose-record numbers.
    """
    y_rel_raw = eval_df["immig_relevant"].to_numpy()
    rel_mask = ~pl.Series(y_rel_raw).is_null().to_numpy()
    own_terms = own_terms_metrics(scores[rel_mask], y_rel_raw[rel_mask].astype(bool))

    ica_raw = eval_df["ica_event"].to_numpy()
    ica_mask = ~pl.Series(ica_raw).is_null().to_numpy()
    vs_ica_roc = float(roc_auc_score(ica_raw[ica_mask].astype(bool), scores[ica_mask]))

    # fill_null(False): event_type nulls make a Boolean-comparison Series
    # convert to an OBJECT numpy array, which breaks boolean-mask indexing.
    diaspora_mask = (eval_df["event_type"] == "Diasporic").fill_null(False).to_numpy()
    diaspora = diaspora_slice_metrics(scores, scores[diaspora_mask])

    return {
        "leg": name,
        "n_scored": int(eval_df.height),
        "own_terms": own_terms,
        "vs_ica_roc": vs_ica_roc,
        "diaspora_slice": diaspora,
    }


def _print_summary(results: dict) -> None:
    print("\n=== per-leg metrics ===")  # LOG
    hdr = f"{'leg':<10} {'own-ROC':>8} {'own-PR':>7} {'vs-ICA':>7} {'diasp@30':>9} {'n_diasp':>8}"
    print(hdr)  # LOG
    print("-" * len(hdr))  # LOG
    for name in ("control", "reference", "graft"):
        leg = results["legs"][name]
        r030 = diaspora_recall_at(leg["diaspora_slice"], MATCHED_POSITIVE_RATE)
        r030_s = f"{r030:>9.3f}" if r030 is not None else f"{'n/a':>9}"
        print(
            f"{name:<10} {leg['own_terms']['roc_auc']:>8.3f} "
            f"{leg['own_terms']['pr_auc']:>7.3f} {leg['vs_ica_roc']:>7.3f} "
            f"{r030_s} {leg['diaspora_slice']['n_diaspora']:>8d}"
        )  # LOG

    print("\n=== control/reference vs prose-record expectations (sanity, not the gate) ===")  # LOG
    for name in ("control", "reference"):
        c = results["sanity_checks"][name]
        print(
            f"{name}: vs-ICA observed={c['observed_vs_ica_roc']:.3f} "
            f"expected={c['expected']['vs_ica_roc']:.3f} within_tol={c['vs_ica_within_tol']}"
        )  # LOG

    print("\n=== graft vs SAME-SESSION reference (the pre-registered gate) ===")  # LOG
    cmp = results["graft_vs_reference"]
    print(
        f"delta vs-ICA ROC     = {cmp['delta_vs_ica_roc']:+.4f}  "
        f"(band +/-{cmp['vs_ica_roc_band']:.3f})  pass={cmp['vs_ica_pass']}"
    )  # LOG
    print(
        f"delta diaspora@0.30  = {cmp['delta_diaspora_recall_030']:+.4f}  "
        f"(band +/-{cmp['diaspora_recall_band']:.4f}, n_diaspora={cmp['n_diaspora']})  "
        f"pass={cmp['diaspora_pass']}"
    )  # LOG
    print(
        f"delta own-terms ROC  = {cmp['delta_own_terms_roc']:+.4f}  (guardrail, not gated)"
    )  # LOG
    print(f"OVERALL PASS = {cmp['overall_pass']}")  # LOG


def main(
    out_path: Path | str | None = None,
    reference_rel_weights: Path | str | None = None,
) -> dict:
    started = datetime.now(timezone.utc)
    out_path = Path(out_path) if out_path is not None else default_out_path()
    reference_rel_weights = (
        Path(reference_rel_weights)
        if reference_rel_weights is not None
        else REFERENCE_REL_WEIGHTS
    )

    keras.config.set_dtype_policy(config.DTYPE_POLICY)
    keras.utils.set_random_seed(200)

    eval_df = load_eval_frame()
    n_rows = eval_df.height
    eval_df_indexed = eval_df.with_row_index("orig_row")
    texts = build_headline_lead_texts(
        eval_df["headline"].to_list(), eval_df["lead_paragraph"].to_list()
    )
    print(f"[0/5] eval set: {n_rows} rows")  # LOG

    print("[1/5] Building DAPT / tuned / grafted backbones + layer-diff verification...")  # LOG
    dapt, tuned, graft, diff_vs_dapt, diff_vs_tuned = build_graft_backbone()
    print("  diff vs DAPT (max |delta| per group):")  # LOG
    for g in sorted(diff_vs_dapt, key=lambda x: (x != TOP_LAYER_GROUP, x)):
        tag = " <- GRAFTED" if g == TOP_LAYER_GROUP else ""
        print(f"    {g:24s} {diff_vs_dapt[g]:.3e}{tag}")  # LOG
    print("  diff vs tuned (max |delta| per group; expect 0.0 at the grafted layer):")  # LOG
    for g in sorted(diff_vs_tuned, key=lambda x: (x != TOP_LAYER_GROUP, x)):
        tag = " <- GRAFTED" if g == TOP_LAYER_GROUP else ""
        print(f"    {g:24s} {diff_vs_tuned[g]:.3e}{tag}")  # LOG

    print("[2/5] Extracting CLS features (control=DAPT, reference=tuned, graft)...")  # LOG
    cls_dapt = extract_cls(dapt, texts)
    cls_tuned = extract_cls(tuned, texts)
    cls_graft = extract_cls(graft, texts)

    print("[3/5] Harness self-check: cosine vs the production embed caches...")  # LOG
    dapt_rows, dapt_cached, dapt_missing = joined_cache_cls(eval_df_indexed, CACHE_DAPT_SUFFIX)
    tuned_rows, tuned_cached, tuned_missing = joined_cache_cls(eval_df_indexed, CACHE_TUNED_SUFFIX)
    cosine_dapt = cosine_similarity_stats(cls_dapt[dapt_rows], dapt_cached)
    cosine_tuned = cosine_similarity_stats(cls_tuned[tuned_rows], tuned_cached)
    print(
        f"  DAPT  cosine: min={cosine_dapt['min']:.6f} mean={cosine_dapt['mean']:.6f} "
        f"p01={cosine_dapt['p01']:.6f} (n={cosine_dapt['n']}, missing={dapt_missing})"
    )  # LOG
    print(
        f"  tuned cosine: min={cosine_tuned['min']:.6f} mean={cosine_tuned['mean']:.6f} "
        f"p01={cosine_tuned['p01']:.6f} (n={cosine_tuned['n']}, missing={tuned_missing})"
    )  # LOG
    assert_cosine_ok(cosine_dapt, "control (DAPT vs relevance_train cache)")
    assert_cosine_ok(cosine_tuned, "reference (tuned vs relevance_train_tuned cache)")

    print("[4/5] Scoring the three legs...")  # LOG
    p_control = score_rel_head(cls_dapt, CONTROL_REL_WEIGHTS)
    p_reference = score_rel_head(cls_tuned, reference_rel_weights)
    p_graft = score_rel_head(cls_graft, reference_rel_weights)

    control = leg_report("control", eval_df, p_control)
    reference = leg_report("reference", eval_df, p_reference)
    graft_leg = leg_report("graft", eval_df, p_graft)

    n_diaspora = graft_leg["diaspora_slice"]["n_diaspora"]
    comparison = graft_pass_fail(reference, graft_leg, n_diaspora)

    control_check = {
        "expected": EXPECTED["control"],
        "observed_own_terms_roc": control["own_terms"]["roc_auc"],
        "observed_vs_ica_roc": control["vs_ica_roc"],
        "observed_diaspora_recall_030": diaspora_recall_at(
            control["diaspora_slice"], MATCHED_POSITIVE_RATE
        ),
        "vs_ica_within_tol": (
            abs(control["vs_ica_roc"] - EXPECTED["control"]["vs_ica_roc"]) <= SANITY_ROC_TOL
        ),
    }
    reference_check = {
        "expected": EXPECTED["reference"],
        "observed_own_terms_roc": reference["own_terms"]["roc_auc"],
        "observed_vs_ica_roc": reference["vs_ica_roc"],
        "observed_diaspora_recall_030": diaspora_recall_at(
            reference["diaspora_slice"], MATCHED_POSITIVE_RATE
        ),
        "vs_ica_within_tol": (
            abs(reference["vs_ica_roc"] - EXPECTED["reference"]["vs_ica_roc"]) <= SANITY_ROC_TOL
        ),
    }

    finished = datetime.now(timezone.utc)
    results: dict = {
        "generated_utc": finished.isoformat(timespec="seconds"),
        "runtime_seconds": (finished - started).total_seconds(),
        "eval_csv": str(EVAL_CSV),
        "n_rows": n_rows,
        "provenance": {
            "dapt_backbone_weights": file_stat(DAPT_BACKBONE_WEIGHTS),
            "tuned_backbone_weights": file_stat(TUNED_BACKBONE_WEIGHTS),
            "control_rel_weights": file_stat(CONTROL_REL_WEIGHTS),
            "reference_rel_weights": file_stat(reference_rel_weights),
            "cache_dapt": str(config.CCA_EMBED_CACHE_DIR / CACHE_DAPT_SUFFIX),
            "cache_tuned": str(config.CCA_EMBED_CACHE_DIR / CACHE_TUNED_SUFFIX),
            "top_layer_group": TOP_LAYER_GROUP,
            "unfreeze_top_n": UNFREEZE_TOP_N,
            "seq_length": SEQ_LENGTH,
        },
        "layer_diff_summary": {
            "graft_vs_dapt": diff_vs_dapt,
            "graft_vs_tuned": diff_vs_tuned,
            "expected_grafted_group": TOP_LAYER_GROUP,
        },
        "cosine_self_check": {
            "dapt_vs_relevance_train_cache": {**cosine_dapt, "n_missing": dapt_missing},
            "tuned_vs_relevance_train_tuned_cache": {**cosine_tuned, "n_missing": tuned_missing},
        },
        "legs": {"control": control, "reference": reference, "graft": graft_leg},
        "sanity_checks": {"control": control_check, "reference": reference_check},
        "graft_vs_reference": comparison,
        "pre_registered_bands": {
            "vs_ica_roc_band": VS_ICA_ROC_BAND,
            "diaspora_recall_numerator_band": DIASPORA_RECALL_NUMERATOR_BAND,
            "note": (
                "Pass/fail is graft vs the SAME-SESSION reference leg (tuned head on "
                "the full tuned encoder), per docs/notes/branched-encoder-strategy.md "
                "stage 1. control/reference vs EXPECTED (prose-record) numbers are a "
                "harness sanity check, not the pass/fail gate."
            ),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[5/5] wrote {out_path}")  # LOG

    _print_summary(results)
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stage-1 graft test (branched-encoder ladder).")
    ap.add_argument("--out", default=None, help="output JSON path (default: experiments dir)")
    ap.add_argument(
        "--rel-weights",
        default=None,
        help="reference rel head weights (default: the production relevance_tuned artifact; "
        "pass the scratch_diag CPU-trained probe to score against a correct-math head)",
    )
    args = ap.parse_args()
    main(out_path=args.out, reference_rel_weights=args.rel_weights)
