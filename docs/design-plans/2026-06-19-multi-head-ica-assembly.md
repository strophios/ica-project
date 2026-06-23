# Multi-Head ICA Assembly Design

## Summary

The assembly takes three already-trained, independently-trained binary classifiers — a US/not-US
geographic filter, a collective-action-event (CCA) head, and an immigrant-relevance head — and
combines them into a single inference artifact that outputs a composed ICA ("Immigrant Collective
Action") score for any article. The shared encoder (a RoBERTa model domain-adapted to NYT news
text) is deliberately kept frozen across all three heads: because negative transfer is a
gradient-level phenomenon among trainable parameters, a frozen encoder with independent heads
carries none of that risk, and joint fine-tuning is deferred as a separately-measured experiment.
The practical effect is that the entire fusion operates in "features mode" — each head scores
pre-computed 768-dimensional encoder embeddings rather than running a forward pass through the
encoder — which keeps retrain and evaluation cycles to minutes rather than hours.

The approach has two distinguishing structural choices. First, US is a hard gate rather than a
co-equal term in the fusion: the CCA and relevance heads were trained on US-restricted data, so
their scores estimate quantities *conditional on US*, and applying them to non-US articles is a
distributional extrapolation. The US classifier is therefore applied as a threshold filter first;
the CCA and relevance scores are only combined on the survivors. Second, before the gate and
combiner can be tuned honestly, both the CCA and relevance heads are retrained from scratch on a
harmonized training population (the same fused US gate applied to both, the same held-out
evaluation ids excluded from both), so that the clean evaluation set — a mix of hand-coded known
positives and boundary-region candidates — is genuinely unseen by either head. This "harmonized
retrain" ensures the empirical fusion selection and the final composed-score calibration are not
contaminated by the data choices that produced the heads being combined.

## Definition of Done

- A single multi-head **inference artifact** exists — frozen DAPT encoder → `{us, cca, rel}`
  heads → a declarative `fusion.json` — that maps an article (raw text or cached CLS features)
  to three calibrated per-head probabilities and one composed **ICA score**, and reloads
  reproducibly across processes.
- The CCA and relevance heads are **retrained on a harmonized training table** that applies the
  same fused US gate to both, with a single clean joint-ICA eval slice held out of *both* heads.
- A **clean, hand-coded joint-ICA eval set** exists (held-out anchors + a stratified boundary
  draw), unseen by either head, conforming to a documented schema.
- The **fusion rule** (US gate threshold + combiner) is selected empirically against the clean
  eval set under a pre-registered AND-vs-LR margin rule, and the composed score's calibration is
  validated (and corrected if needed).
- The assembled model is **applied** to `api_corpus` (1960–1995, in-period) and LDC 1996–2007
  (out-of-DoCA-period expansion test), producing `api_us_scores/`, `api_cca_scores/`, and a
  ranked `ica_candidates` set for both.
- The **pre-flight verifications** pass: US-logit provenance, per-head calibration sidecars,
  DoCA freshness propagation, and the LDC dateline/text-channel consistency.

## Acceptance Criteria
<!-- DRAFTED BELOW — pending user validation before finalization -->

### multi-head-ica-assembly.AC1: Harmonized retrain on a shared population
- **multi-head-ica-assembly.AC1.1 Success:** CCA and relevance heads are retrained from a single
  harmonized table that applies the *same* fused US gate (`us_location` + ML filter) to both.
- **multi-head-ica-assembly.AC1.2 Success:** The clean ICA eval ids are absent from both heads'
  train *and* val pools (verified: `eval_ids ∩ (train ∪ val) = ∅` for each head).
- **multi-head-ica-assembly.AC1.3 Edge:** ~30% of the 466 anchors are reserved into the eval
  slice and excluded from training positives in both heads.
- **multi-head-ica-assembly.AC1.4 Failure:** If any eval id leaks into either training pool, the
  retrain aborts rather than training on it.

### multi-head-ica-assembly.AC2: Clean joint-ICA evaluation set
- **multi-head-ica-assembly.AC2.1 Success:** The eval set carries a joint-ICA label
  (US ∩ CCA ∩ immigrant-relevant) per row and validates against a documented schema.
- **multi-head-ica-assembly.AC2.2 Success:** Boundary candidates were drawn stratified near the
  current heads' decision boundary (selection-by-old-score does not contaminate the retrain).
- **multi-head-ica-assembly.AC2.3 Failure:** Rows missing any of the three component judgments
  are rejected by schema validation.

### multi-head-ica-assembly.AC3: Per-head and composed calibration
- **multi-head-ica-assembly.AC3.1 Success:** Each head has a Platt calibrator fit on
  natural-balance data; all three `*.calibration.json` sidecars are present with A/B recorded.
- **multi-head-ica-assembly.AC3.2 Success:** The gate threshold `τ_us` is the largest threshold
  meeting the target anchor/DoCA recall (the recall recipe).
- **multi-head-ica-assembly.AC3.3 Success:** The composed ICA score's calibration is reported
  (reliability / ECE / Brier) on the clean eval set; a final 2-param Platt is fit if mis-calibrated.

### multi-head-ica-assembly.AC4: Empirical fusion selection
- **multi-head-ica-assembly.AC4.1 Success:** Calibrated-AND baseline and the ≤3-param logistic
  challenger are both evaluated by cross-validation on the clean eval set.
- **multi-head-ica-assembly.AC4.2 Success:** The chosen combiner is recorded in `fusion.json`
  (gate threshold, calibrator refs, combine rule, score space).
- **multi-head-ica-assembly.AC4.3 Decision:** The LR ships only if it beats AND by more than the
  pre-registered CV-noise margin; otherwise the parameter-free AND ships.

### multi-head-ica-assembly.AC5: Assembled multi-head artifact
- **multi-head-ica-assembly.AC5.1 Success:** A single inference model builds with the frozen
  backbone + `{us, cca, rel}` heads loaded by structure (Pattern 2).
- **multi-head-ica-assembly.AC5.2 Success:** The artifact scores cached features (and raw text)
  to three logits + the composed ICA score.
- **multi-head-ica-assembly.AC5.3 Success:** Cross-process reload reproduces scores within
  tolerance (the `artifact_check` analogue).
- **multi-head-ica-assembly.AC5.4 Failure:** Head-name collision or a missing head weight raises
  at assembly time, not silently.

### multi-head-ica-assembly.AC6: Apply and dataset expansion
- **multi-head-ica-assembly.AC6.1 Success:** Scoring `api_corpus` (1960–1995) writes
  `us_filter/api_us_scores/` + `cca_doca/api_cca_scores/` and a ranked `ica_candidates` parquet.
- **multi-head-ica-assembly.AC6.2 Success:** Scoring LDC 1996–2007 (`ldc_9507` cache) yields
  out-of-DoCA-period ICA candidates.
- **multi-head-ica-assembly.AC6.3 Edge:** LDC US gating uses the dateline-stripped text channel
  consistent with US-head training (datelines exist on LDC, unlike the API corpus).

### multi-head-ica-assembly.AC7: Pre-flight verifications (cross-cutting)
- **multi-head-ica-assembly.AC7.1:** The cached `us_logit` is confirmed produced by
  `us_classifier_full`, not the deprecated smoke-test weights.
- **multi-head-ica-assembly.AC7.2:** CCA and relevance calibration sidecars are confirmed present
  (or fit if missing).
- **multi-head-ica-assembly.AC7.3:** DoCA freshness (`doca.csv`/`tmp.R` → match → positives) is
  confirmed propagated, or the edit is explicitly accepted as incidental.
- **multi-head-ica-assembly.AC7.4:** The `ldc_9507` cache's `us_logit` is confirmed computed on
  dateline-stripped LDC text.

## Glossary

- **ICA (Immigrant Collective Action)**: The target event type — an article reporting on
  collective action (a protest, strike, boycott, etc.) that involves immigrants. ICA is the
  intersection of the three component signals: US location, collective action, and immigrant
  relevance.
- **DoCA (Dynamics of Collective Action)**: A hand-coded dataset of protest events in the United
  States, 1960–1995, used as the source of labeled positive examples for the CCA head. DoCA
  matches form the "anchor" positives for the ICA evaluation set.
- **DAPT (Domain-Adaptive Pre-Training)**: A fine-tuning pass of the base RoBERTa language model on
  unlabeled NYT headline/lede pairs, adapting its representations to the news domain before any
  classification training. All three heads share the same frozen DAPT encoder.
- **CCA**: The collective-action-event head/label — whether an article reports any protest-type
  collective action event (demonstrations, strikes, boycotts, etc.), regardless of immigrant
  involvement. One of the three component heads.
- **PU learning / nnPU / FLPU**: Positive-Unlabeled learning trains a classifier when only positive
  examples are confirmed-labeled and the remaining corpus is unlabeled (it may contain unidentified
  positives). Non-negative PU (nnPU, Kiryo 2017) corrects a sign-flip pathology in the standard PU
  risk estimator. FLPU is this project's variant that wraps nnPU in focal loss to down-weight easy
  examples.
- **Platt scaling / calibration**: A post-hoc two-parameter logistic transformation (σ(A·logit + B))
  that maps a model's raw output logits to well-calibrated probabilities. Calibration means that
  among articles scored 0.7, roughly 70% should truly be positive. Platt scaling is the cheapest
  calibration method and is used as a seam between the heads' logits and the fusion arithmetic.
- **ECE / Brier**: Expected Calibration Error (average gap between predicted confidence and observed
  frequency, across bins) and Brier score (mean squared error between predicted probability and
  binary outcome). Both measure how well-calibrated a probability forecast is; lower is better.
- **Anchors**: A set of ~466 confirmed ICA-positive articles derived from DoCA records. These are
  the reliable-positive spine of the evaluation set; roughly 30% are reserved as held-out test
  positives and excluded from both heads' training.
- **Fused US gate**: The combination of two US-location signals — a text-based machine-learning US
  classifier (scores from the US head) and a rule-based geographic signal derived from datelines and
  location strings (`us_location` module). The fused gate is stricter than either alone and is
  applied consistently to both CCA and relevance training populations in this design.
- **Features mode**: Running classification heads directly on pre-computed 768-d CLS embeddings from
  the encoder, bypassing the encoder forward pass entirely. Possible only because the encoder is
  frozen; enables minute-scale retrains and sweeps.
- **Pattern 2**: Cross-process weight loading — constructing a fresh head object with the correct
  architecture and loading saved `.weights.h5` weights into it by structural position, rather than
  by user-assigned name. Used here because each head was trained in a separate process.
- **Artifact triple**: The project convention that a trained model is represented as three files:
  `*.weights.h5` (weights), `*.config.json` (hyperparameters and architecture), and
  `*.calibration.json` (Platt A/B). This design extends the triple with a fourth file, `fusion.json`.
- **Negative transfer**: In multi-task learning with a shared encoder, gradient updates from one
  task can degrade performance on another. With a frozen encoder, all heads are trained
  independently and no gradients flow through the shared parameters, so negative transfer cannot
  occur.
- **EPV / Harrell's rule (events-per-variable)**: A heuristic for logistic regression model
  complexity: roughly 10–15 outcome events per estimated parameter are needed for stable fits. With
  ~466 positive anchors, Harrell's rule caps the fusion logistic regression at 3–4 coefficients,
  which is why the document refers to a "≤3-param" combiner.
- **Cascade**: The hierarchical structure in which one classifier (the US gate) filters the
  population before subsequent classifiers (CCA, relevance) are applied. Errors at the gate are
  unrecoverable downstream, which is why the gate threshold is tuned for high recall.

## Architecture

The assembly composes three already-trained, standalone binary classifiers into one hierarchical
ICA ("immigrant collective action") decision, on a **frozen** shared encoder. Freezing is the
load-bearing choice: negative transfer is a gradient-level phenomenon among *trainable* shared
parameters, so a frozen encoder with independent heads carries no negative-transfer risk. A
top-N joint encoder fine-tune is deliberately deferred as a separately-measured ceiling-lift
experiment, not part of this effort.

**The artifact.** A single inference model built through the existing multi-head path
(`build_inference_model` / `build_feature_inference_model`, which already accept
`dict[str, ClassificationHead]`):

```
load_dapt_backbone(DAPT_BACKBONE_WEIGHTS)         # frozen, shared
  → CLS (768-d)
  → { "us":  ClassificationHead,   # weights ← us_classifier_full
      "cca": ClassificationHead,   # weights ← cca_doca (or _street)
      "rel": ClassificationHead }  # weights ← relevance (renamed from "cca")
  → { "us": logit, "cca": logit, "rel": logit }
```

Heads load by *structure* (Pattern 2): each is constructed fresh and its trained `.weights.h5`
loaded in. Because the frozen encoder is identical to the one that produced the embedding cache,
head weights trained features-mode are valid on the encoder's live output too.

**The fusion is a declarative spec, not a trained network layer.** A `fusion.json` sidecar
(extending the project's artifact-triple convention to a fourth file) holds the gate threshold,
per-head calibrator references, and the composition rule. This keeps the empirical sweep
(AND ↔ LR ↔ soft-US) a config change and a re-score, not a retrain.

**Fusion topology — hierarchical cascade.** The target factorizes as
`P(ICA) = P(US) · P(CCA ∧ relevant | US)`. This is principled, not stylistic: the CCA and
relevance heads are trained on US-restricted data, so their scores estimate quantities
*conditional on US* and are out-of-support off the US subpopulation. Therefore US is a **gate**,
not a co-equal fusion term:

```
gate:   pass  ⇔  calib_us(logit_us) ≥ τ_us        # hard, recall-tuned (errors unrecoverable)
score:  ICA   =  combine(calib_cca, calib_rel [, calib_us])   on gated survivors
                 combine ∈ { product (AND, 0-param),  σ(LR(scores)) (≤3 wt) }
```

The combiner is chosen empirically: the parameter-free calibrated-AND is the baseline; a ≤3-param
logistic regression on the calibrated scores is the interaction test (a real CCA×relevance
interaction shows up as the LR beating AND); an optional `p_us` term tests a graded — rather than
binary — US contribution on survivors. Harrell's events-per-variable rule at ~466 positives caps
the learned combiner at 3–4 coefficients, so the LR is simultaneously the interaction test and the
richest combiner the labels support. A learned deep fusion head would require Approach C (weak-
supervision label expansion), held as an escalation if the interaction is real but unlearnable at
this budget.

**Data flow.** Canonical path is features-mode over the cached 768-d CLS embeddings — seconds-to-
minutes, no encoder forward pass — used for the fusion fit, threshold tuning, and all evaluation.
A token-mode inference path (frozen backbone over raw text) is left available for future un-cached
articles (the eventual full-NYT expansion) but is not built or tested here.

## Existing Patterns

Investigation confirmed the assembly is mostly *composition* of existing, validated machinery
rather than new modeling:

- **Multi-head wiring already exists.** `src/model_setup/assembly.py` —
  `build_endpoint_model` / `build_inference_model` and their features-mode counterparts
  `build_feature_endpoint_model` / `build_feature_inference_model` — already accept
  `dict[str, ClassificationHead]` and assert unique head names. They have only ever been
  instantiated with a single head.
- **Shared head spine.** `src/model_setup/heads.py:ClassificationHead` (dropout → dense 768→768
  relu → dropout → dense→1) is the one head class serving US, CCA, and relevance. Head `name` is
  keyword-only and required; Keras routes by name at the call site.
- **Features-mode on a shared cache.** `src/embed_corpus.py` produces one CLS cache
  (`cca_doca/embed_cache/<suffix>/`) carrying `(id, year, us_logit, CLS)`; CCA
  (`run_cca_doca.py`), relevance (`run_relevance.py`), and the real US filter
  (`run_us_features.py`) all consume it. This is the de-facto shared frozen encoder.
- **Holdout machinery exists on both heads.** `run_relevance.py` takes `--holdout-ids` →
  `data.create_relevance_data(table, holdout_ids=...)`, and imports `_load_holdout_ids` from
  `run_cca_doca.py`; the same mechanism gates the CCA retrain.
- **Fused US gate.** `src/preproc/us_location.py:compute_location_signals` yields
  `(any_us, any_not_us)`; `run_relevance.py` fuses it with the ML filter as
  `us = ml_pass & ~(any_not_us & ~any_us)` (a hard row filter). Today CCA gates on dateline-only
  `us_logit ≥ τ`; harmonizing both onto the fused gate is part of this design.
- **Calibration + artifact triple.** `src/calibration/` (`platt_fit`/`platt_transform`,
  `PlattCalibrator`, `calibration_path_for_weights`) and `src/calibrate_cca.py` /
  `src/calibrate_us_filter.py` establish `*.weights.h5` + `*.config.json` + `*.calibration.json`.
  `src/validation/artifact_check.py:reload_and_score` is the cross-process reload proof.
- **Validation instruments.** `src/validation/schema.py` (gold-set schema),
  `build_coding_template.py` (stratified hand-coding sampler), `doca_recall.py`, `slice_eval.py`.

**Divergence noted:** the relevance head's config currently names itself `"cca"` (cosmetic, per
`run_relevance.py:14-17`); this design renames it to `"rel"` so the three heads coexist in one
artifact. The `fusion.json` sidecar is a new artifact type, extending the existing triple.

## Implementation Phases

<!-- START_PHASE_1 -->
### Phase 1: Pre-flight verifications
**Goal:** Establish the factual ground the rest of the plan stands on, before any retrain.

**Components:**
- US-logit provenance check — confirm the cached `us_logit` in `relevance_train` / `full` /
  `ldc_9507` was produced by `us_classifier_full`, not the smoke-test `us_classifier.weights.h5`
  (read `embed_corpus.py` provenance / cache `provenance.json`).
- Calibration-sidecar audit — confirm `*.calibration.json` exists for CCA (both tracks) and
  relevance; flag relevance as a fit-needed if absent.
- DoCA freshness check — confirm `doca.csv`/`tmp.R` edits propagated to `cca_matches_good.rds` →
  `cca_doca_positives.parquet`, or record the edit as incidental.
- LDC channel check — confirm the `ldc_9507` cache's `us_logit` was computed on dateline-stripped
  LDC text.

**Dependencies:** None.

**Done when:** Each of AC7.1–AC7.4 is resolved with an explicit recorded answer (pass, or a
remediation noted). Operational verification; no new code beyond small inspection scripts.

**Covers:** AC7.1, AC7.2, AC7.3, AC7.4.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Clean joint-ICA evaluation set
**Goal:** Produce an honest, contamination-free joint-ICA gold set and the held-out id list.

**Components:**
- Eval-id reservation — reserve ~30% of `relevance/ica_anchors.parquet` (the 466) as held-out
  positives.
- Boundary candidate draw — a stratified sampler (extending `build_coding_template.py`) over
  current-head scores near the composed decision boundary, schema-conforming with null labels.
- Hand-coding (★ operator-gated) — code each candidate for US / CCA / immigrant-relevance and the
  joint label; validate with `src/validation/schema.py`.
- The combined held-out id list (anchors + coded boundary) becomes the `--holdout-ids` input.

**Dependencies:** Phase 1.

**Done when:** A schema-valid joint-ICA eval set exists with held-out anchors + coded boundary
rows, and the held-out id list is materialized.

**Covers:** AC2.1, AC2.2, AC2.3.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Harmonized retrain of CCA + relevance
**Goal:** Retrain both heads on one population/gate with the eval slice held out of both.

**Components:**
- Harmonized training table — apply the *same* fused US gate (`us_location`) to both heads'
  positives and background (CCA currently gates on dateline-only `us_logit`).
- Retrain CCA (`run_cca_doca.py`) and relevance (`run_relevance.py`) features-mode with the
  Phase-2 held-out ids removed from train and val.
- Head rename — relevance head config `"cca" → "rel"`.
- Leakage guard — assert `eval_ids ∩ (train ∪ val) = ∅` for each head before fit.

**Dependencies:** Phase 2 (held-out id list).

**Done when:** Both heads retrained (minutes, features-mode), sidecars written, and the leakage
guard passes. Tests verify the holdout intersection is empty and the rename took effect.

**Covers:** AC1.1, AC1.2, AC1.3, AC1.4.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: Calibration and empirical fusion selection
**Goal:** Calibrate each head, tune the gate, choose the combiner, validate composed calibration.

**Components:**
- Per-head Platt calibrators fit on natural-balance data (`src/calibration/`, `calibrate_*.py`);
  ensure all three `*.calibration.json` sidecars exist.
- Gate threshold `τ_us` from the recall recipe (largest threshold ≥ target recall).
- Combiner sweep on the clean eval set — calibrated-AND baseline vs ≤3-param LR (+ optional
  soft-`p_us`), cross-validated; pre-registered margin decision rule.
- Composed-score recalibration check (reliability / ECE / Brier); optional final 2-param Platt.
- `fusion.json` writer (gate threshold, calibrator refs, combine rule, score space).

**Dependencies:** Phase 3 (retrained heads), Phase 2 (clean eval set).

**Done when:** `fusion.json` records the selected combiner under the margin rule, and the
composed-score calibration is reported. Tests verify the decision rule (LR ships only past
margin) and `fusion.json` round-trips.

**Covers:** AC3.1, AC3.2, AC3.3, AC4.1, AC4.2, AC4.3.
<!-- END_PHASE_4 -->

<!-- START_PHASE_5 -->
### Phase 5: Assemble the multi-head artifact
**Goal:** One serializable inference object that produces per-head + composed ICA scores.

**Components:**
- Assembly entrypoint — build `build_inference_model({"us","cca","rel"})` on the frozen backbone,
  load each head's weights by structure, attach the `fusion.json` composition.
- Reload proof — a cross-process reload-and-score check (the `artifact_check.py` analogue) over
  the composed score.

**Dependencies:** Phase 4 (`fusion.json` + calibrated heads).

**Done when:** The artifact scores cached features (and raw text) to three logits + the composed
ICA score, raises on head-name collision / missing weights, and reproduces scores across processes.
Tests verify AC5.1–AC5.4.

**Covers:** AC5.1, AC5.2, AC5.3, AC5.4.
<!-- END_PHASE_5 -->

<!-- START_PHASE_6 -->
### Phase 6: Full-corpus and out-of-period apply
**Goal:** Produce the ICA candidate sets — the dataset-expansion deliverable.

**Components:**
- Finish `embed_cache/full` part 2 (1976–1995) so the in-period corpus is fully cached.
- Apply over `api_corpus` (1960–1995): per-head scores → gate → composed ICA → write
  `us_filter/api_us_scores/`, `cca_doca/api_cca_scores/`, ranked `ica_candidates`.
- Apply over LDC 1996–2007 (`ldc_9507` cache), using the dateline-stripped channel for US gating.
- Fold in the `run_us_classification.py` greedy-glob fix if that path is touched.

**Dependencies:** Phase 5 (assembled artifact).

**Done when:** Score outputs and ranked candidate sets exist for both corpora; the LDC pass uses
the stripped channel. Tests verify output schema and the LDC channel selection.

**Covers:** AC6.1, AC6.2, AC6.3.
<!-- END_PHASE_6 -->

## Additional Considerations

**Operator-gated step on the critical path.** Phase 2's hand-coding is the only human bottleneck;
every other phase is automatable and fast (features-mode retrains in minutes). The plan is ordered
so the hand-coding draw can be prepared early.

**Cascade calibration is not free.** A product of individually-calibrated marginals is not a
calibrated joint, and the cascade induces dependence (stage 2 sees only US-passed rows). Hence the
explicit composed-score recalibration in Phase 4 — do not assume multiplication preserves
calibration.

**Deferred (tracked, not in scope):** the top-N encoder joint fine-tune (ceiling-lift experiment,
on the three component tasks, never the 466 anchors); typed multi-label relevance heads
(roadmap B4); Approach C weak-supervision label expansion (escalation only if the LR can't beat
AND); relevance operating-threshold re-tuning and richer location signal (roadmap B1–B3).
