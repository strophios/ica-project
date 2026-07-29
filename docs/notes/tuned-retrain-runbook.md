# Tuned-cache retrain runbook — US + CCA negative-transfer check

*Created 2026-07-29. Companion to `docs/notes/encoder-unfreeze-strategy.md` (the
rel-first sequential encoder-unfreeze decision). Once the tuned embed caches
(`cca_doca/embed_cache/{train250k_tuned, us_train_ldc_tuned,
relevance_train_tuned}`) land from the cluster job, this is the exact command
sequence to retrain US + CCA features-mode on them and compare against the
production heads — WITHOUT touching any production artifact. Every trainer/
calibrator below already has (or now has, per the accompanying diff) the knobs
this runbook uses; see the "What changed" section at the bottom for the diff
summary and the "Gaps" section for what's explicitly NOT fixed here.*

## Naming convention used throughout

| Artifact | Production path | Tuned path (this runbook's convention) |
|---|---|---|
| US weights | `us_filter/us_classifier_full.weights.h5` | `us_filter/us_classifier_full_tuned.weights.h5` |
| CCA weights | `cca_doca/cca_doca.weights.h5` | `cca_doca/cca_doca_tuned.weights.h5` |
| Relevance weights | `relevance/relevance.weights.h5` | `relevance/relevance_tuned.weights.h5` |
| Tuned backbone | — | `relevance/tuned_backbone.<jobtag>.weights.h5` (from `extract_tuned_backbone.py`, already produced upstream of this runbook) |
| Tuned embed caches | — | `cca_doca/embed_cache/{us_train_ldc_tuned, train250k_tuned, relevance_train_tuned}` (already produced upstream, on-cluster) |

Config sidecars (`.config.json`) and calibration sidecars (`.calibration.json`)
derive automatically from each `.weights.h5` path (`config_path_for_weights`,
`calibration_path_for_weights`) — naming the weights path correctly is
sufficient; you never name the sidecars directly.

Every trainer/calibrator below refuses to run if you pass a non-default
`--suffix`/cache while leaving the weights path at its production default
(`check_no_production_overwrite` in `src/artifact_guard.py`) — so forgetting
`--out`/`--weights` on a tuned run fails loudly instead of silently
clobbering production.

## Step 0 — confirm the caches landed

```
ls cca_doca/embed_cache/us_train_ldc_tuned
ls cca_doca/embed_cache/train250k_tuned
ls cca_doca/embed_cache/relevance_train_tuned
```

## Step 1 — retrain US (tuned cache)

Find the tuned backbone path the caches were embedded with (produced by
`extract_tuned_backbone.py` upstream of this runbook) and pass it via
`--backbone-weights` so the sidecar records it — this is load-bearing for
Step 4's `eval_heads_own_terms.py` US scoring (see "Backbone bookkeeping"
below).

```
uv run python -m src.run_us_features \
    --suffix us_train_ldc_tuned \
    --out us_filter/us_classifier_full_tuned.weights.h5 \
    --backbone-weights relevance/tuned_backbone.<jobtag>.weights.h5
```

## Step 2 — retrain CCA (tuned cache)

Check the production sidecar for the prior currently in force
(`cca_doca/cca_doca.config.json` → `heads[0].loss.prior`; CLAUDE.md records
π≈0.02 as of the harmonized retrain, but the sidecar is authoritative) and
reuse it, so the comparison isolates the encoder change from a prior change.

```
uv run python -m src.run_cca_doca \
    --prior <production prior> \
    --suffix train250k_tuned \
    --out cca_doca/cca_doca_tuned.weights.h5 \
    --us-weights us_filter/us_classifier_full_tuned.weights.h5
```

`--us-weights` re-scores the US-restriction gate against the just-retrained
tuned US head (applied to the tuned cache's CLS, both are already on the tuned
representation) rather than whatever `us_logit` got baked into the
`train250k_tuned` cache at embed time — use it so the CCA retrain's positive/
unlabeled split matches what the tuned US head would actually gate at apply
time. Omit it only if you specifically want to hold the gate fixed while
varying just the CCA head.

## Step 3 — retrain relevance (tuned cache)

```
uv run python -m src.run_relevance \
    --prior <production prior> \
    --suffix relevance_train_tuned \
    --out relevance/relevance_tuned.weights.h5
```

**Gap (not fixed here):** `run_relevance.py` has no `--us-weights` rescore
knob — unlike `run_cca_doca.py`, it calls `label_and_restrict` directly on the
cache's own `us_logit` (`src/build_cca_doca_table.py:24-35`, the shared
`(pl.col("us_logit") >= threshold).alias("us")` helper both `run_cca_doca.py`
and `run_relevance.py` use), because `relevance_train`'s `us_logit` is baked
in at cache-*build* time by `build_relevance_table` (see the module
docstring) rather than re-scored at train time the way `run_cca_doca.py`'s
`_rescore_us_restriction` (`src/run_cca_doca.py:87-103`) does when
`--us-weights` is passed. Whatever `us_logit` the
`relevance_train_tuned` cache carries is whatever US head scored it when the
cache was built — if that wasn't the tuned US head from Step 1, the relevance
retrain's US-restriction is gated by a head other than the one this runbook
retrained. Needed fix if this matters: either add a `--us-weights` rescore
knob mirroring `run_cca_doca.py`'s `_rescore_us_restriction` (~17 lines,
`src/run_cca_doca.py:87-103`), or confirm out-of-band that
`relevance_train_tuned` was built against the tuned US head.

## Step 4 — gold evals of both vs. production baselines

Two existing eval scripts serve this; both were extended (or already
supported) the needed params:

**CCA-specific:** `src/validation/run_cca_eval.py` — already fully
parameterized (`--suffix`, `--weights`), no changes needed.

```
uv run python -m src.validation.run_cca_eval \
    --suffix train250k_tuned --weights cca_doca/cca_doca_tuned.weights.h5
uv run python -m src.validation.run_cca_eval \
    --suffix train250k --weights cca_doca/cca_doca.weights.h5   # production baseline, for the same-session comparison
```

**All three heads, own-terms:** `scripts/eval_heads_own_terms.py` — extended
in this change with `--cca-weights`/`--rel-weights`/`--us-weights`/
`--cache-suffix`/`--out` (all independently defaulting to production; `--out`
defaults to the production JSON, so pass a distinct path for the tuned run or
you'll overwrite the production comparison file).

```
uv run python -m scripts.eval_heads_own_terms \
    --cca-weights cca_doca/cca_doca_tuned.weights.h5 \
    --rel-weights relevance/relevance_tuned.weights.h5 \
    --us-weights us_filter/us_classifier_full_tuned.weights.h5 \
    --cache-suffix relevance_train_tuned \
    --out cca_doca/experiments/eval_heads_own_terms_tuned.json
```

Note `--cache-suffix` here must be a cache that has ALL of headline,
lead_paragraph, and emb_row for the eval-set ids — i.e. it plays the same role
`relevance_train_tuned` plays for production `relevance_train` (both CCA and
rel features come from the same cache in this script). If the tuned CCA and
tuned relevance caches are actually two separate caches with different row
populations, this script can't join both from one `--cache-suffix` as-is —
confirm `relevance_train_tuned` is the single unified cache before relying on
this, or treat that as a second instance of the fit_fusion.py gap below.

**Backbone bookkeeping (why Step 1's `--backbone-weights` matters here):**
`apply_us_model` (used inside `eval_heads_own_terms.py` for the US head)
re-embeds `headline`+`lead_paragraph` text through a *token-mode* backbone
loaded from the US sidecar's own `backbone_weights_path`
(`src/validation/slice_eval.py:67-69`). If Step 1 didn't record the tuned
backbone there, this call silently re-embeds through the *production* DAPT
backbone — a features/backbone mismatch with the just-trained tuned US head,
producing incoherent (not obviously wrong) scores. Passing
`--backbone-weights` in Step 1 is what prevents this.

**No such gap for CCA/rel:** `apply_cca_model`/`apply_relevance_model` score
already-embedded `features` arrays directly (no backbone reload), so they're
automatically correct for whichever cache's CLS you pass in.

## Step 5 — calibrate all three tuned heads

```
uv run python -m src.calibrate_us_filter \
    --suffix us_train_ldc_tuned --out us_filter/us_classifier_full_tuned.weights.h5
uv run python -m src.calibrate_cca \
    --suffix train250k_tuned --weights cca_doca/cca_doca_tuned.weights.h5
uv run python -m src.calibrate_relevance \
    --suffix relevance_train_tuned --out relevance/relevance_tuned.weights.h5
```

Each writes its `.calibration.json` next to the given weights path
(`calibration_path_for_weights` — verified in
`tests/test_tuned_calibration_knobs.py::TestCalibrationSidecarDerivesNextToTunedWeights`),
so the tuned sidecars never collide with the production ones.

## Step 6 — fit_fusion on tuned scores

**Gap (not fixed here, scope-contained per task instructions):**
`src/fit_fusion.py` has NO CLI and NO per-head weights/cache parameters —
everything is hardcoded:

- cache: `config.CCA_EMBED_CACHE_DIR / "relevance_train"` (`src/fit_fusion.py:207`)
- CCA weights: `config.CCA_DOCA_WEIGHTS` (`src/fit_fusion.py:258`, `:260`, `:549`)
- relevance weights: `config.RELEVANCE_DOCA_WEIGHTS` (`src/fit_fusion.py:267`, `:269`, `:552`)
- US weights: `config.US_FILTER_FULL_WEIGHTS` (`src/fit_fusion.py:277`, `:555`)

`output_dir` (where `ica_fusion.fusion.json`/`ica_fusion_metrics.json` land)
IS already a `main()` parameter defaulting to `config.CCA_DOCA_DIR` — so a
tuned run that forgets to pass `output_dir` will silently overwrite the
**production fusion artifact**, the one gap in this file with the same shape
as the overwrite risk the trainers/calibrators now guard against.

Needed fix to run this step for real: add
`cca_weights_path`/`rel_weights_path`/`us_weights_path`/`cache_suffix`
parameters to `main()` (mirroring what this change did to
`eval_heads_own_terms.py` — same hardcoded-constant shape, same fix pattern),
each independently defaulting to its production path, and thread them into
the four call sites above. Until that lands, a tuned fusion fit requires
either a manual edit of `fit_fusion.py` (temporarily, not committed) with
`output_dir` explicitly pointed away from `config.CCA_DOCA_DIR`, or a
standalone script that reimplements `main()`'s steps 1–11 with tuned paths
substituted (steps 1–9 are the reusable logic; `select_combiner` is already a
pure, independently-testable function — see `src/fit_fusion.py:67-118`).

## Step 7 — composed eval

Unlike Step 4/6, this step is NOT blocked by the fit_fusion.py gap for a
first read on transfer: `IcaModel.__init__` (`src/assemble_ica.py:58-63`)
already takes `us_weights_path`/`cca_weights_path`/`rel_weights_path`/
`fusion_path` as independent constructor parameters (all defaulting to
production), and `predict_ica_from_features` is **features-mode for all
three heads** (`src/assemble_ica.py:207-230` — no token-mode backbone reload
anywhere in `IcaModel`, unlike the `apply_us_model` path in Step 4), so once
you have tuned weights + calibrations (Steps 1–5) you can construct a tuned
`IcaModel` directly. The one thing you can't get around Step 6's gap for is
the `fusion_path` itself — a tuned combiner/gate has to have been *fit* on
tuned scores first. Two options:

- **Full comparison** (needs Step 6's fix): fit a tuned `fusion.json` first,
  then pass `fusion_path=<tuned fusion.json>` below.
- **Partial comparison** (works today, answers "did the heads individually
  regress" but not "does the fused decision regress"): reuse the *production*
  `fusion.json` (same gate threshold, same combiner) with the tuned heads —
  valid as a first-pass negative-transfer signal since the combiner's
  coefficients are small (product-AND or a ≤3-param LR) and unlikely to be
  the dominant source of any regression, but not a substitute for a properly
  refit tuned fusion once Step 6 is addressed.

```python
# uv run python (adjust cache_suffix to whatever Step 4 confirmed is unified)
import numpy as np
import polars as pl
from src.assemble_ica import IcaModel
from src.embed_corpus import load_cache
import src.config as config

model = IcaModel(
    us_weights_path="us_filter/us_classifier_full_tuned.weights.h5",
    cca_weights_path="cca_doca/cca_doca_tuned.weights.h5",
    rel_weights_path="relevance/relevance_tuned.weights.h5",
    fusion_path="cca_doca/ica_fusion_tuned.fusion.json",  # from Step 6, once fixed;
                                                            # or omit for the partial
                                                            # comparison (production fusion)
)

eval_df = pl.read_csv(config.VALIDATION_DIR / "ica_coding_template_coded.csv")
meta, cls = load_cache(config.CCA_EMBED_CACHE_DIR / "relevance_train_tuned")
joined = eval_df.join(
    meta.select(["id", "emb_row"]), on="id", how="inner"
)
features = cls[joined["emb_row"].to_numpy()]
scores = model.predict_ica_from_features(features)

from sklearn.metrics import roc_auc_score, average_precision_score
y = joined["ica_event"].to_numpy().astype(bool)
print("tuned composed ROC-AUC:", roc_auc_score(y, scores["ica_score"]))
print("tuned composed PR-AUC:", average_precision_score(y, scores["ica_score"]))
```

Compare against the production numbers in
`ml_memo/ica_model_state_2026-06.md` (or re-run the same snippet with
`IcaModel()` defaults for an apples-to-apples same-session baseline).

## Gaps summary (explicitly not fixed in this change)

1. **`run_relevance.py` has no `--us-weights` rescore knob** (Step 3) — the
   relevance retrain's US-restriction is whatever `us_logit` the
   `relevance_train_tuned` cache was built with, not necessarily gated by the
   tuned US head from Step 1.
2. **`src/fit_fusion.py` is fully hardcoded** (Step 6) — no CLI, no per-head
   weights/cache params. `output_dir` is the sole parameterized escape hatch,
   and it's easy to forget, risking a silent overwrite of the production
   `ica_fusion.fusion.json`/`ica_fusion_metrics.json`.
3. **No standalone "composed eval on the gold set" CLI** — Step 7 uses an
   inline snippet because `apply_ica.py` hardcodes `IcaModel()` at both call
   sites (`src/apply_ica.py:70`, `:170`) and is a full-corpus batch-apply tool,
   not a gold-set eval tool; `IcaModel` itself is already parameterized enough
   to not need changes for this use case.
4. **Population-alignment assumption in Step 4** — `eval_heads_own_terms.py`
   joins CCA, relevance, AND (via `apply_us_model`, cache-independent text)
   the US head all against ONE `--cache-suffix`. This holds today because
   `relevance_train` is a superset cache covering both CCA and relevance eval
   rows; it needs re-confirming for whatever unified tuned cache Step 4 is
   pointed at.

## What changed (this diff)

- New `src/artifact_guard.py` (`check_no_production_overwrite`, Functional
  Core) — shared by all six trainers/calibrators below.
- `src/run_us_features.py`: `DEFAULT_SUFFIX` constant, guard call, new
  `--backbone-weights` knob (bookkeeping — records the backbone that
  produced the cache in the sidecar, for token-mode eval consumers).
- `src/run_cca_doca.py`, `src/run_relevance.py`: `DEFAULT_SUFFIX` constant +
  guard call (both already had `--suffix`/`--out` knobs; `--us-weights`
  already existed on `run_cca_doca.py` only).
- `src/calibrate_us_filter.py`, `src/calibrate_cca.py`,
  `src/calibrate_relevance.py`: `DEFAULT_SUFFIX` constant + guard call (all
  three already had `--suffix`/`--out`(`--weights`) knobs).
- `scripts/eval_heads_own_terms.py`: `main()` gained
  `cca_weights`/`rel_weights`/`us_weights`/`cache_suffix`/`out_path` params
  (each independently defaulting to production) + matching CLI flags; output
  JSON now records which weights/cache were used.
- Tests: `tests/test_artifact_guard.py` (pure guard logic),
  `tests/test_tuned_cache_knobs.py` (trainer wiring),
  `tests/test_tuned_calibration_knobs.py` (calibrator wiring),
  `tests/test_eval_heads_own_terms.py` (eval script argument resolution). All
  use a monkeypatched first-post-guard-I/O-call marker exception, since the
  tuned caches don't exist locally to run real training/eval against.
- **Not touched:** `src/fit_fusion.py`, `src/apply_ica.py` — see Gaps above.
