# CCA/DoCA Retrain Implementation Plan — Phase 0: Embedding-cache infrastructure

**Goal:** Factor the frozen DAPT backbone into a one-time CLS-embedding cache and add a
features-mode training path that reuses the full instrumented stack, so all downstream training runs
on cached vectors in minutes.

**Architecture:** Embedding extractor (Imperative Shell) tokenizes via `ClassifierPreprocessor`, runs
a `backbone → CLS` + US-head dual-output model, and caches `(id, 768-float CLS, us_logit)`. A
features-mode assembly variant swaps the token-input + backbone front-end for a `(hidden_dim,)`
features `Input`, leaving `ClassificationHead` / `LayerLRModel` / diagnostics unchanged.

**Tech Stack:** Keras 3 + TensorFlow, keras_hub RoBERTa, polars, numpy.

**Scope:** Phase 0 of 5 (+ stretch). Design: `docs/notes/cca-doca-retrain-design.md`.

**Codebase verified:** 2026-06-15 (read `assembly.py`, `heads.py`, `data.py`, `backbone.py`,
`preprocessor.py`, `run_cca_classification.py`, `cca_metrics.py`, `config.py`).

---

## Acceptance Criteria Coverage

### cca-doca.AC0: Embedding cache
- **cca-doca.AC0.1:** The extractor, given a set of API articles, writes a cache of `(id, 768-float CLS)`
  plus `us_logit`, with a provenance record identifying the backbone weights, seq_length, and text channel.
- **cca-doca.AC0.2:** A features-mode endpoint model built from a `(768,)` input trains a
  `ClassificationHead` and, with diagnostics enabled, populates the loss-component / batch-balance /
  prediction-distribution trackers (parity with the token-mode path).
- **cca-doca.AC0.3:** Building a features-mode model whose feature dim ≠ head `hidden_dim` raises a clear error.

---

## Notes for the implementer

- Run all commands with `uv run` from the project root. Do not activate the venv.
- Local is `float32`/MPS; apply `keras.config.set_dtype_policy(config.DTYPE_POLICY)` in scripts.
- The US filter shares the *same* frozen DAPT backbone; its weights file
  (`config.US_FILTER_CLASSIFIER_WEIGHTS`) is a backbone+head endpoint checkpoint. Loading it into a
  `backbone + us_head` inference model restores the (unchanged, frozen) DAPT backbone plus the trained
  US head — see `src/validation/slice_eval.py:apply_us_model` for the existing construction pattern to mirror.
- **Vigilance:** after the extractor runs, spot-check that `us_logit` values are sane (not all-equal,
  not NaN/Inf) and that CLS vectors are non-degenerate (nonzero variance across rows). Treat anomalies
  as blocking.

---

<!-- START_SUBCOMPONENT_A (tasks 1-1) -->
<!-- START_TASK_1 -->
### Task 1: Add config paths for the DoCA match file and the embedding cache

**Files:**
- Modify: `src/config.py` (paths section, after the US-filter block ~line 119)

**Implementation:**
Add three `Path` constants (platform-conditional only where needed; the DoCA file lives outside the
repo data root):
- `DOCA_CCA_MATCHES`: the DoCA→NYT match RDS. Local: `PROJECT_ROOT.parent / "LDC2008T19" / "data" / "cca_matches_good.rds"`
  (PROJECT_ROOT is `.../00_explorer`, so `.parent` is `.../00_ML_data_expansion`). Add a comment that
  this is an external, non-checked-in artifact and that the cluster path must be set when the cluster returns.
- `CCA_EMBED_CACHE_DIR`: `PROJECT_ROOT / "cca_embed_cache"` — the CLS embedding cache.
- (Reuse existing `US_FILTER_CLASSIFIER_WEIGHTS`, `API_CORPUS_DIR`, `DAPT_BACKBONE_WEIGHTS`.)

**Verification:**
Run: `uv run python -c "import src.config as c; print(c.DOCA_CCA_MATCHES, c.DOCA_CCA_MATCHES.exists()); print(c.CCA_EMBED_CACHE_DIR)"`
Expected: prints the DoCA path with `True`, and the cache dir path.

**Commit:** `feat(cca-doca): add config paths for DoCA matches + embedding cache`
<!-- END_TASK_1 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 2-3) -->
<!-- START_TASK_2 -->
### Task 2: Embedding extractor script

**Verifies:** cca-doca.AC0.1

**Files:**
- Create: `src/embed_corpus.py` (Imperative Shell — FCIS `# pattern:` header)

**Implementation:**
A `main(...)` (guarded by `if __name__ == "__main__"`) that:
1. Applies dtype policy + seed (mirror `run_cca_classification.py:50-55`).
2. Loads the API corpus via `src.data_setup.data.data_from_parquet(config.PROJECT_ROOT, "api_corpus",
   addl_columns=["year"], lead_column="lead_paragraph")` → gives `id`, `headline`, `lead_paragraph`,
   `year`, `headline_with_lead`.
3. Selects the article set:
   - `--sample-n N --stratify-by-year`: draw N rows stratified by `year` (deterministic, seed 200,
     polars `group_by("year").map_groups` proportional sample) — the ~250k unblock run.
   - `--full`: all rows (the overnight canonical cache).
   - `--ids-from PATH`: restrict to ids in a parquet (used later by gold-set/stretch).
4. Builds the dual-output extractor model on the *same* backbone + US-head instances:
   - `backbone = load_dapt_backbone(config.DAPT_BACKBONE_WEIGHTS)`
   - Build US head + inference model and load US weights exactly as `slice_eval.apply_us_model` does
     (read `UsRunConfig` sidecar via `config_path_for_weights(config.US_FILTER_CLASSIFIER_WEIGHTS)`,
     fresh `ClassificationHead(hidden_dim=..., name="us")`, `build_inference_model`,
     `load_weights(..., skip_mismatch=False)`). This populates the shared `backbone` + `us_head` weights.
   - Construct the cache model from the now-weighted instances:
     `tok = Input((seq,), "int32", "token_ids"); pad = Input((seq,), "int32", "padding_mask");
      out = backbone({"token_ids": tok, "padding_mask": pad}); cls = out[:, 0, :];
      us_logit = us_head(cls); model = keras.Model({...}, {"cls": cls, "us": us_logit})`.
5. Tokenizes with `ClassifierPreprocessor(SEQ_LENGTH=seq, text_key="headline_with_lead",
   label_keys={}, endpoint_model=True)` over a `tf.data` pipeline (batch, map, prefetch — mirror the
   `_finite_predict_dataset` pattern in `run_cca_classification.py:452-458`), `model.predict`.
6. Writes per-shard outputs under `config.CCA_EMBED_CACHE_DIR`: a float32 `.npy` matrix of CLS
   vectors and a row-aligned parquet of `(id, year, us_logit)`; plus a `provenance.json`
   (backbone weights path + mtime + size, seq_length=128, text_channel="headline_with_lead",
   created date passed in via `--stamp` since `Date.now()` is unavailable in some contexts — accept a
   CLI stamp arg). Row order in the `.npy` MUST match the parquet (assert lengths equal).

**Testing:**
Light — this is an Imperative Shell script. A focused test in `tests/test_embed_corpus.py` covering
the pure helper(s): the stratified-sample helper (deterministic, proportional by year) and the
cache-write/read round-trip (write a tiny `(id, vec, us_logit)` set, read back, assert alignment).
Do NOT test keras/backbone forward output values.

**Verification:**
Run: `uv run python -m src.embed_corpus --sample-n 512 --stratify-by-year --stamp 20260615 --out-suffix smoke`
Expected: writes a `.npy` (512×768) + parquet (512 rows) + provenance.json; script prints shapes.
Then: `uv run python -c "import numpy as np, polars as pl; ..."` to assert 512×768, finite, nonzero variance.

**Commit:** `feat(cca-doca): embedding extractor (CLS + US logit cache)`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Kick off the embedding runs

**Files:** none (operational)

**Implementation:**
1. **Unblock run (~250k, ~1.5h):** `uv run python -m src.embed_corpus --sample-n 250000
   --stratify-by-year --stamp <today> --out-suffix train250k` in the background. This feeds Phases 1–3.
2. **Canonical full cache (overnight, ~13–20h):** `uv run python -m src.embed_corpus --full
   --stamp <today> --out-suffix full` started in the background once the smoke run is verified.

**Verification:**
Monitor: the unblock run completes and writes the expected row count; spot-check `us_logit`
distribution (not degenerate) and CLS variance. The full run is checked the next morning.

**Commit:** none (no code).
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_SUBCOMPONENT_C (tasks 4-5) -->
<!-- START_TASK_4 -->
### Task 4: Features-mode assembly

**Verifies:** cca-doca.AC0.2, cca-doca.AC0.3

**Files:**
- Modify: `src/model_setup/assembly.py` (add `build_feature_endpoint_model`; mirror
  `build_endpoint_model:69-216`)

**Implementation:**
`build_feature_endpoint_model(heads, hidden_dim, target_dtype="float32", layer_multipliers=None,
group_fn=None, diagnostics=None)`:
- Same unique-name and diagnostics-key checks as `build_endpoint_model:142-161`.
- Inputs: `features = keras.Input(shape=(hidden_dim,), dtype=..., name="features")` plus one
  `"<head>_targets"` Input per head (identical to `build_endpoint_model:170-175`).
- Call each head directly on `features` (NO `[:, 0, :]` slice — features are already the CLS vector):
  `outputs[name] = head(features, targets=target_inputs[f"{name}_targets"])`.
- Diagnostics: gather constituent trainable vars from the heads only (no backbone), then `build_trackers`
  and return a `LayerLRModel` exactly as `build_endpoint_model:195-216`.
- AC0.3: raise `ValueError` if any head's `hidden_dim` ≠ the `hidden_dim` arg (the head's intermediate
  Dense is built lazily, so validate against the configured value explicitly).
- Add a companion `build_feature_inference_model(heads, hidden_dim)` (features Input, no targets) for predict.

**Testing:** `tests/test_assembly.py` — add cases mirroring the existing endpoint tests:
- cca-doca.AC0.2: a features-mode model with `diagnostics` enabled, after a `train_step` on a small
  synthetic `{features, cca_targets}` batch, exposes the loss-component / batch-balance /
  prediction-distribution trackers in `model.metrics` (parity with token-mode).
- cca-doca.AC0.3: constructing with mismatched feature dim raises `ValueError`.

**Verification:** Run `uv run pytest tests/test_assembly.py -q`. Expected: all pass.

**Commit:** `feat(cca-doca): features-mode assembly reusing head + diagnostics`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Features-mode dataset builder

**Verifies:** cca-doca.AC0.2 (supports it)

**Files:**
- Modify: `src/data_setup/data.py` (add `dataset_from_embeddings`)

**Implementation:**
`dataset_from_embeddings(shuffle_buffer, batch_size, data, weights=None, head_name="cca", seed=200)`
where `data` is a list of `(features_array, labels_array)` groups (pos, unl) or a single group:
- Build `tf.data.Dataset.from_tensor_slices({"features": feats, f"{head_name}_targets": labels})`
  per group; ratio-batch via `tf.data.Dataset.sample_from_datasets(..., weights=...)` exactly as
  `dataset_create:217-224`; `.shuffle().repeat().batch(drop_remainder=True).prefetch`.
- No preprocessor map needed — entries are already numeric (cast targets to float32 at array-build time).

**Testing:** `tests/test_data_loading.py` (or a new `tests/test_embeddings_dataset.py`): a batch from
two synthetic groups with weights `[0.1, 0.9]` yields the documented dict keys and the expected
positive fraction in expectation.

**Verification:** Run `uv run pytest tests/test_embeddings_dataset.py -q`. Expected: pass.

**Commit:** `feat(cca-doca): tf.data builder over cached embeddings`
<!-- END_TASK_5 -->
<!-- END_SUBCOMPONENT_C -->

---

## Phase 0 done when
- Extractor produces a provenance-tagged cache; smoke run verified (AC0.1).
- Features-mode endpoint model trains a head with all diagnostics firing; dim-mismatch raises (AC0.2, AC0.3).
- `uv run pytest` green; `ruff check` clean.
- The ~250k unblock embed has been launched (and ideally completed); the full overnight embed launched.
