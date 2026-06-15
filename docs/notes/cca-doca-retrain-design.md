# CCA/DoCA Retrain — Design Plan

*Created: 2026-06-15. Status: active. Design doc for the MVP CCA classifier retrain on the
NYT Archive API corpus using DoCA-confirmed positives. Feeds the phased implementation plan at
`docs/implementation-plans/2026-06-15-cca-doca-retrain/`. Read alongside the top-level `CLAUDE.md`
(project contracts) and the US-filter section therein.*

## Goal

A cleanly trained, **evaluable** CCA (collective action) classifier we can show collaborators this
week — solid, not perfect. Concretely:

1. Trained on the **NYT Archive API corpus (1960–1995)** — the full DoCA period — rather than the
   LDC corpus (1987–2007).
2. Positives = **DoCA-matched articles** (confirmed collective-action events), replacing the
   overgenerous NYT-descriptor labels the current CCA head uses.
3. **US-restricted** training population (the deployment target is US events; DoCA events are US,
   so a global unlabeled pool would let the model learn "US-ness" as a CCA proxy).
4. A **DEDPUL-re-estimated class prior** for this new population.
5. Real **precision / recall / F1** from a hand-coded gold set — the current eval path is
   predict-only and produces no metrics.

**Stretch:** apply the US→CCA pipeline over the full corpus to surface a sample of *discovered*
CCA events outside the training overlap.

## Why this supersedes the current CCA path

The current CCA classifier (`src/run_cca_classification.py`) trains on `ldc_corpus` with
`cca_label = cca | cca_descriptor` (NYT indexer descriptors). Those descriptors are imprecise and
overgenerous — a known limitation. Two data improvements made outside this repo change the picture:

- The **NYT Archive API pull** gives headline+lede for the full DoCA period (1960–1995), not just
  the LDC window (1987–2007).
- **DoCA→NYT matching** yields confirmed positives keyed directly to API articles.

So for the MVP we swap noisy descriptor positives for confirmed DoCA positives and train on the
larger, deployment-aligned period. Refining the descriptor definitions remains future work (it
matters for the immigration head), but it is **not on the MVP path**.

## Grounded facts (verified 2026-06-15)

**DoCA match artifact** — `~/immigration_project/00_ML_data_expansion/LDC2008T19/data/cca_matches_good.rds`
(outside the repo; not yet in `src/config.py`):
- 23,615 event–article match rows; **15,627 unique `article_id`**; 19,568 rows `match_quality == "succeeded"`.
- `article_id` is in `nyt://article/...` form — **identical to the API corpus `id`**, so positives
  join *directly* onto the API corpus; the fuzzy matching (DoCA event title ↔ NYT headline) is already done.
- Carries `eventid`, `match_quality`, `match_dist`, DoCA event-form codes (`street`, `lawsuit`,
  `conventional`, `boycott`), `immigrant_involved` (370 true — useful later for the immigration head), `keywords`.
- ~400 matches resolved to LDC (not API) articles when the API match failed/was worse; these drop
  out naturally on the inner join to the API corpus. Acceptable for the MVP.

**API corpus** — `api_corpus/` (`config.API_CORPUS_DIR`): 3,699,431 rows, 1960–1995, 36 per-year
parquet. Columns include `id`, `headline`, `lead_paragraph`, `year`, `news_desk`, `section_name`,
`keywords`. **No `doca_id`, no US scores** joined in — both are separate joins.

**US filter**: trained (`us_filter/us_classifier.weights.h5` + `.config.json`), but **never applied
to the API corpus** (`us_filter/api_us_scores/` does not exist) and the **Platt calibration sidecar
is missing**. Same frozen DAPT backbone as the CCA head; reads `headline</s>lead_paragraph` on the
API corpus (no datelines there, so `stripped_text == lead_paragraph`).

**Eval gap**: `src/eval_cca_classifier.py` is predict-only (logit dump for hand review). All
`src/validation/` tooling targets the US filter's `us_event` label; there is **no CCA slice-eval**.

**Local throughput (M1, float32/MPS, seq 128)**: measured **~50 articles/sec** for the RoBERTa-base
forward pass, flat across batch size (compute-bound).

## Core architectural idea: the frozen-backbone embedding cache

Every model in this plan uses a **frozen** DAPT backbone (US filter, the L/U classifier for DEDPUL,
the CCA head). The backbone is therefore a *pure function* `text → 768-d CLS vector`, and the
`ClassificationHead` already consumes exactly that: `assembly.py` computes `cls = backbone_out[:, 0, :]`
and the head docstring requires `(batch, hidden_dim)` features.

So we factor the one expensive primitive — the RoBERTa forward pass — into a **one-time embedding
cache**, and train heads on cached vectors:

- **Embed once** (`text → CLS`), cache `(id, 768-vec)` to disk, co-emitting the US head's logit in
  the same forward pass (US filter shares the backbone).
- **Train/score on cached vectors**: US restriction (threshold cached US scores), DEDPUL, CCA head
  training, the π sensitivity sweep, and gold-set scoring all become trivial CPU/MPS work (minutes),
  because the only heavy step is amortized.

### Doing it right, not as a bypass

The hard requirement (operator directive): the cached-vector path must **reuse the existing
instrumented stack**, not sit beside it as a bare `Dense` model that loses our training-quality
diagnostics. The design:

- **Features-mode assembly** — a variant of `build_endpoint_model` whose input is a `(hidden_dim,)`
  features `Input` instead of `token_ids/padding_mask → backbone`. *Everything downstream is
  unchanged*: same `ClassificationHead`, `LayerLRModel`, target inputs, `group_fn`, and the
  diagnostics wiring (`build_trackers`).
- **All diagnostics survive** because they operate at the head/loss level, not the backbone:
  per-group gradient norms, the FLPU loss-component tracker (`positive_risk` / `negative_risk` /
  `correction_triggered`), batch-balance, and the prediction-distribution collapse signal. The
  frozen-encoder case (no backbone gradients) is one the instrumentation already handles.
- **Config reuse**: `RunConfig` + JSON sidecar, with provenance recorded (which DAPT weights,
  seq_length, text channel produced the cache); `validate_against_backbone` is repurposed to assert
  the cached feature dim matches the head's `hidden_dim`.
- **Reuse dividend**: this path amortizes the backbone pass for *all* future heads (immigration,
  multi-head ICA), π sweeps, and re-runs. It is infrastructure, not throwaway MVP scaffolding.

### The cache is powerful-now, not eternal

The embedding cache is valid **only while the backbone is frozen**. When we eventually do full deep
training — unfreezing the backbone with per-layer learning rates once the head structure is final —
the embeddings are invalidated. Therefore:

- The **token-mode path** (`build_endpoint_model` with `token_ids/padding_mask → backbone`) is
  **retained unchanged**; full fine-tuning via `LayerLRModel`'s `freeze_encoder=False` /
  `layer_multipliers` is its home.
- The cache and the features-mode path are explicitly scoped and provenance-tagged as the
  **frozen-probe accelerator**.

### Cache storage (finalize in Phase 0)

Proposed: a `float32` matrix `.npy` of CLS vectors plus a row-aligned parquet sidecar of
`(id, year, us_logit)`, per shard, under a new `config` path (e.g. `CCA_EMBED_CACHE_DIR`), with a
provenance JSON (backbone weights path + mtime/size, seq_length, text channel, creation date).
`us_logit` is cached so US restriction is a pure threshold with no second forward pass.

## Compute strategy and budget

The Explorer cluster is down for maintenance this week; we work locally (M1) plus optional cheap
cloud. Because the only heavy step is the embedding pass:

- **Primary MVP — entirely local.** A ~250k year-stratified embed (~1.5 h at 50/s) unblocks
  building and iterating the features-mode pipeline, DEDPUL, the first CCA train, and gold-set
  scoring — all of which then run in minutes on cached vectors.
- **Canonical full cache — local overnight.** Embed all 3.7M articles once (~13–20 h, $0), banking
  the artifact for the stretch goal and final runs. Kicked off in the background so it never gates
  iteration.
- **Cloud is optional**, reserved for the stretch full-corpus embed if the overnight run is
  inconvenient (~30–60 min, low single-digit $ — well within the stated budget).

## Knobs (locked)

- **Training/DEDPUL unlabeled sample**: ~250k year-stratified (leaves ~150–250k US after filtering).
  Rationale: positives (~13–15k) are the scarce resource and the head is tiny (768→768→1); DEDPUL's
  1-D density estimate stabilizes by tens of thousands of unlabeled. Representativeness (year
  stratification) matters more than raw count; returns past ~250k are marginal.
- **Gold set**: draw a **target of 3,000** once, in a deterministic stratified order whose every
  prefix is itself approximately stratified (round-robin over era × desk × score-band). **MVP floor
  = first 500 coded**; coding more later just walks down the same list (no re-draw, no collision).

## Phases

Five phases plus a stretch. Each ends in a verifiable state.

- **Phase 0 — Embedding-cache infrastructure.** Embedding-extraction script (frozen backbone → CLS,
  co-emit US logit, provenance); features-mode assembly variant; features-mode dataset builder;
  `RunConfig` provenance + feature-dim validation; tests mirroring `tests/test_assembly.py`. Run the
  ~250k stratified embed; kick off the full-corpus embed overnight.
  *Done when:* features-mode endpoint model trains a head on cached vectors with all diagnostics
  firing; extractor produces a provenance-tagged cache; tests pass.
- **Phase 1 — DoCA-labeled, US-restricted training table.** Add a config path for the DoCA match
  file; join `cca_matches_good` (succeeded) → API `id` → `cca_label`; US-restrict via cached US
  scores; `create_cca_doca_data` split (PU 90/5/5, unique-id assertion, within-split shuffle, no
  leakage).
  *Done when:* split returns the documented pos/unl counts; invariants asserted; positives all carry
  DoCA provenance.
- **Phase 2 — DEDPUL prior re-estimation.** Train the L/U head on cached embeddings (labeled = DoCA
  positives, unlabeled = US-restricted sample); run DEDPUL EM; record π and sanity-check against the
  naive labeled rate.
  *Done when:* a π in (0,1) is produced, recorded with provenance, and within a sane band of the
  labeled rate.
- **Phase 3 — CCA retrain (features-mode) + spot-check.** Train the CCA head features-mode with
  FLPU/nnPU at the re-estimated π, frozen probe, full diagnostics, config sidecar; small π
  sensitivity check; spot-check training health.
  *Done when:* weights + sidecar written; diagnostics show no collapse/anomaly; top-scored articles
  look sane.
- **Phase 4 — Gold set + CCA eval tooling.** Score-stratified, prefix-stratified coding template
  (target 3,000, floor 500); `apply_cca_model` + `evaluate_cca_slice` (P/R/F1 at threshold);
  `doca_recall` as a secondary (topic-skewed) diagnostic.
  *Done when:* template is schema-valid and prefix-stratified; eval computes P/R/F1 from coded
  labels. (MVP metrics require ≥500 hand-coded — a human step.)
- **Phase 5 (stretch) — full-corpus discovered events.** Gated on the overnight full cache. Apply
  US → CCA over 1960–1995; produce a sample of discovered CCA events for the demo.
  *Done when:* per-year scored output exists; a curated discovered-events sample is produced.

## Acceptance criteria (`cca-doca`)

- **AC0 (embedding cache).**
  - AC0.1: The extractor, given a set of API articles, writes a cache of `(id, 768-float CLS)` plus
    `us_logit`, with a provenance record identifying the backbone weights, seq_length, and text channel.
  - AC0.2: A features-mode endpoint model built from a `(768,)` input trains a `ClassificationHead`
    and, with diagnostics enabled, populates the loss-component / batch-balance / prediction-distribution
    trackers (parity with the token-mode path's diagnostics).
  - AC0.3: Building a features-mode model whose feature dim ≠ head `hidden_dim` raises a clear error.
- **AC1 (training table).**
  - AC1.1: An API article whose `id` is in `cca_matches_good` (succeeded) is labeled positive; one
    that is not is unlabeled.
  - AC1.2: After US restriction, the unlabeled pool contains only articles scoring US at the chosen
    threshold; positives are retained.
  - AC1.3: `create_cca_doca_data` produces disjoint train/val/test splits (unique `id`, no leakage),
    PU-separated, deterministic under seed 200.
- **AC2 (prior).**
  - AC2.1: DEDPUL produces a finite π ∈ (0,1) for the new population, recorded with provenance.
  - AC2.2: π is within a defensible band of the naive labeled positive rate (flagged, not silently
    accepted, if not).
- **AC3 (retrain).**
  - AC3.1: Training writes `*.weights.h5` + `*.config.json` sidecar reloadable by structure.
  - AC3.2: Diagnostics over the run show no distribution collapse (prediction std not ≈0) and FLPU
    `correction_triggered` behavior is sane.
- **AC4 (eval).**
  - AC4.1: The coding template is `validate_gold_set`-conformant with null CCA labels and a
    prefix-stratified ordering; the first 500 rows are themselves approximately stratified.
  - AC4.2: `evaluate_cca_slice` computes precision/recall/F1 at a threshold from a coded set,
    dropping null-label rows.
- **AC5 (stretch).**
  - AC5.1: A per-year scored output over the full corpus exists with US-gated CCA scores; a curated
    discovered-events sample is produced.

## Verification & vigilance discipline

We deliberately skipped the multi-round code-review finalization gate (lean mode), so we substitute
**active empirical vigilance** — anomalies are blocking until understood:

- After any training step, inspect the diagnostics: FLPU `positive_risk`/`negative_risk`/
  `correction_triggered`, `BatchLabelBalanceTracker` (positive fraction), prediction distribution
  (`std` not ≈0; `frac_above` not pinned at 0/1), per-group gradient norms.
- Sanity-check DEDPUL's π against the naive labeled rate; investigate large gaps before trusting it.
- Eyeball top-scored positives and unlabeled (analogue of the existing `pos_top_*.csv` /
  `unl_top_*.csv` dumps) for face validity.
- Treat any warning (dtype, NaN/Inf, shape, leakage assertion) as a stop-and-investigate, not noise.
- Run the test suite (`uv run pytest`) per phase; honor the pre-commit lint gate (`ruff check`).

## Deferred / out of scope (MVP)

- ALUM/VAT (only a non-functional PyTorch sketch exists in `src/test_module.py`).
- Refining NYT descriptor label definitions (relevant to the immigration head, not this retrain).
- Full hyperparameter search; unfreezing the backbone / per-layer-LR deep training (the token-mode
  path is retained for it).
- Unifying the standalone US filter into the shared-encoder multi-head.
