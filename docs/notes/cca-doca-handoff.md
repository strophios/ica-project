# CCA/DoCA Retrain — Handoff & Tuning Plan

> **Historical arc detail.** The live roadmap / deferred list (incl. the Saturday
> Tier) is consolidated in `docs/notes/roadmap.md`; this doc retains the CCA/DoCA
> narrative and reasoning.

*Created 2026-06-15 (evening). Branch `cca-doca-retrain`. Read with
`docs/notes/cca-doca-retrain-design.md` (design) and the `docs/implementation-plans/2026-06-15-cca-doca-retrain/`
phase files. This doc captures where today's build landed and the plan for tomorrow:
code the gold set, then dial the model in with a proper eval/experiment harness.*

## Where we are (end of day 1)

The full features-mode pipeline landed and a first model trained successfully:

- **Embedding cache** (`src/embed_corpus.py`): frozen DAPT backbone → CLS, co-emits raw US logit.
  The 250k unblock cache is at `cca_doca/embed_cache/train250k/` (15,269 positives + 235k stratified).
- **Training table / split** (`src/build_cca_doca_table.py`, `data.create_cca_doca_data`):
  DoCA-labeled, US-restricted (logit 0.0; positives kept regardless).
- **Prior** (`src/run_cca_doca_prior.py`): DEDPUL π ≈ **0.02** (= positive rate in the unlabeled
  background; the labeled-rate "floor" intuition does NOT apply under force-include sampling).
- **Trained model** (`src/run_cca_doca.py`): `cca_doca.weights.h5` + sidecar. Held-out separation:
  positives median logit +0.18 vs unlabeled −2.54; ~55% recall at 1.1% background-positive @ logit 0.
- **Face validity** (`src/score_cca_doca.py`): top-scored are unmistakable collective action;
  "discovered" unlabeled are real protests but many FOREIGN (US gate too loose); positive "misses"
  expose DoCA fuzzy-match noise.
- **Gold set** (`src/validation/build_cca_coding_template.py`): `validation/cca_coding_template.parquet`
  (2,553 rows, score-band stratified, prefix-stratified) + `validation/cca_coding_first500.csv`
  (the MVP floor, ready to code). Coding columns: `cca_event`, `event_type`, `us_event`, `event_location`.
- **Eval** (`src/validation/cca_slice_eval.py`): `apply_cca_model` + `evaluate_cca_slice`.
- **DoCA form flags** on positives (`cca_doca_positives.parquet`): `any_street` (10,632), conventional
  (3,502), lawsuit (2,672), boycott (752), `no_form` (338). NO `strike` code exists in DoCA here.

638 tests green. ~16 commits on `cca-doca-retrain`.

## Day 2 progress (2026-06-16)

Gold set coded (`cca_coding_first500_coded.csv`, 500 rows: 169 pos / 331 neg) and the
eval/experiment harness built. Part 1 of the full embed finished overnight (1.83M rows,
1960–1975, 8 shards in `cca_doca/embed_cache/full/`); Part 2 (1976–1995 `--append`) still
pending and non-blocking.

**Leakage guard.** `create_cca_doca_data(holdout_ids=)` + `run_cca_doca --holdout-ids <template>`
drop the full 2,553-row coding template from the training unlabeled pool (all unlabeled; zero
DoCA positives), so gold ids are never trained as noisy negatives. `run_cca_doca --out` writes
per-experiment weights + sidecar. The honest base model is actually slightly BETTER separated
than the day-1 leaky one (training true-positives-as-negatives was mildly hurting).

**Eval harness** (`src/validation/run_cca_eval.py`). Re-scores gold ids with the current weights
(the template `cca_score` is stale — from a model that trained on them), then reports per logit
threshold: RAW precision/recall (cross-experiment comparable); IPW-REWEIGHTED precision/recall
(corpus operating point; weight = `corpus_band / gold_band` by `sample_stratum`) with unweighted
support counts so high-variance cells are visible; DoCA test-recall (over held-out test-split
DoCA positives — the trustworthy recall); and per-event-type recall. IPW core (`band_ipw_weights`,
`evaluate_cca_slice_weighted`, `recall_at_thresholds`) + `assign_score_band` (single-source band
boundaries) live in `cca_slice_eval.py` / `build_cca_coding_template.py`. Records land in
`cca_doca/experiments/eval_*.json` (identify by `prior` field) — the seed of `compare_experiments`.

**Honest baseline (leakage-holdout, π=0.02).** At logit ≥1.0: precision ≈0.80 (raw 0.79 ≈
reweighted 0.82 — they converge in the densely-sampled high band, so it's trustworthy),
DoCA-recall 0.35. At ≥2.0: P≈0.86, recall 0.19. At logit 0: P≈0.55–0.73, DoCA-recall 0.61.
Operate at a HIGH threshold (≥1.0): the corpus is ~87% low-band, so a low cut floods with false
positives (reweighted P collapses to 0.55 @0, 0.26 @−1).

**π sweep finding: π is an operating-point knob, not a quality knob.** Across π ∈
{0.01,0.02,0.03,0.05}, DoCA-recall at matched precision is invariant (~0.51 at P=0.75 for all
four). π only rescales the logit so prob-0.5 lands elsewhere on a FIXED precision–recall frontier.
So: π=0.02 is fine (don't fuss 0.02 vs 0.03); pick the deployment threshold from the gold-set PR
curve, not the prior's prob-0.5; and frontier-MOVING experiments (labels/forms/hyperparams), not
π, are where quality gains live.

**Per-event-type recall** (base, @logit 0): street 0.84, conventional 0.78, boycott 1.00 (n=3),
lawsuit 0.50 (→0.20 @logit 1.0). Lawsuit/conventional score lower — motivates the street-restricted
retrain (next).

645 tests green, lint clean.

### Street-restricted experiment (CCA = street only) — corrected, hypothesis SUPPORTED

`run_cca_doca --form-filter any_street` redefines CCA as street events only: positives = street
DoCA ids (10,632); non-street DoCA (4,982) become UNLABELED presumed-negatives (NOT dropped —
they are informative hard negatives for the street/not-street boundary). The eval
(`run_cca_eval --form-filter any_street`) redefines the gold positive to `cca_event AND
event_type=='street'` (144 of 500; the 25 non-street CCA and 15 street-but-not-CCA rows become
negatives). `filter_positives_by_form` + `restrict_label_to_form` are the pure cores.

**Head-to-head on the street task** (same 144 gold positives; both models hold out the template,
so both leakage-clean): the street-trained model beats the all-forms model as a *street* detector
by +0.025–0.086 raw precision at matched recall (0.55–0.80), and on reweighted precision 0.770 vs
0.737 @logit 1.5, 0.868 vs 0.778 @logit 2.0. It does so with FEWER positives (9,277 vs 13,742) —
label cleanliness + hard negatives, not data volume. Supports the hypothesis that the non-street
forms carry the label noise.

> A first pass got this wrong: it DROPPED non-street DoCA from the table and evaluated against the
> 169-positive all-forms label, yielding a spurious "frontier-neutral, hypothesis not supported".
> Corrected to non-street→unlabeled (hard negatives) + gold label→street-only. The wrong version
> never reached git. Lesson: a form-restricted retrain is a CCA *definition* change — it must change
> both the training pool (non-form → presumed-negative) AND the eval positive set.

**Decision: carry BOTH definitions forward in parallel** (the street-only scoping is a collaborator
call, not yet made). Maintain the all-forms model (`cca_doca.weights.h5`) and the street model
(`cca_doca_street.weights.h5`) as parallel tracks so the full range of options can be presented.
Future experiments (threshold lock, frontier-movers) should run for both.

## Overnight embed (split to free the morning GPU)

- **Part 1 RUNNING**: `--years 1960-1975` (1,831,300 articles, ~9.8h) → `cca_doca/embed_cache/full/`,
  shards 000+. Log: `/tmp/embed_full_part1.log`.
- **Part 2 (tomorrow evening)**: append the rest:
  ```
  PYTHONUNBUFFERED=1 uv run python -m src.embed_corpus --full --years 1976-1995 \
      --stamp 20260616 --out-suffix full --append
  ```
  `--append` continues shard numbering so the two parts form one canonical cache; `load_cache` reads all.
- The full cache is for the STRETCH (full-corpus discovered events) and a fuller gold set. Tomorrow's
  tuning runs on the existing `train250k` cache, so they do NOT block on this.

## Tomorrow, step 1: code the gold set (human)

Code `validation/cca_coding_first500.csv` (≥500 rows): `cca_event` (true/false), `event_type`
(street/strike/boycott/conventional/lawsuit/other), and ideally `us_event`/`event_location`.
Prefix-stratified, so the first 500 is already a valid stratified set.

## Tomorrow, step 2: the eval + experiment harness (design to build together)

The cached embeddings make each retrain ~minutes, so we can run a real sweep. We need a harness that
keeps results **comparable and recorded**. Proposed design (refine in the morning):

**North-star metric = gold-set precision** (hand-coded), with two cheap proxies for runs we don't
re-code: **DoCA-recall** (`doca_recall` over known positives) and the held-out **spot-check**
separation. Report all three per run.

**Leakage guard (important).** The gold set was drawn from the unlabeled training pool, so those
ids are trained on as (noisy) negatives. For an honest precision estimate, **exclude the gold-set ids
from the training unlabeled pool** in every experiment. Add a `--holdout-ids <template>` option to
`create_cca_doca_data`/`run_cca_doca` that drops those ids before splitting.

**Experiment = knobs → train → metrics record.** Extend `run_cca_doca.py` to accept the tuning knobs
and write a JSON record per run to a registry `cca_doca/experiments/<run_id>.json`:
- knobs: `form_filter` (e.g. street-only via `any_street`), `prior`, `lr`, `warmup/decay`,
  `ratio_batch.train_pos`, `epochs`, `dropout`, `focal_gamma`, `weight_decay`, US `threshold`.
- metrics: final train/val loss + PR-AUC/precision/recall, held-out spot-check stats, DoCA-recall,
  and (when the gold set is coded) `evaluate_cca_slice` precision/recall/F1 + a per-`event_type` and
  per-threshold breakdown.
- provenance: cache suffix, weights path, git commit, timestamp (stamp passed in).
A `src/validation/compare_experiments.py` loads all records → a sorted table (by gold-set precision,
then DoCA-recall). This is the "recording and comparing" surface.

**Eval runner.** `src/validation/run_cca_eval.py`: ingest the coded CSV (parse `cca_event`, join
`cca_logit` from the template by id), run `evaluate_cca_slice` across a threshold grid + per
`event_type`, print + write to the run record. Build this first thing (it's small and unblocks numbers).

## Tomorrow, step 3: the experiments to run

1. **Street-restricted retrain** — `form_filter=any_street` (10,632 positives). Tests the hypothesis
   that `conventional`/`no_form`/`lawsuit` positives are the label noise. Compare gold-set precision +
   the positive "misses" list against the all-forms model.
2. **π sensitivity sweep** — train at π ∈ {0.01, 0.02, 0.03, 0.05}; confirm 0.02 is robust; see how
   precision/recall trade off (π mostly shifts the operating point).
3. **Hyperparameter search** — LR, ratio-batch (1:9 vs 1:5 vs 1:19), epochs/early-stop, dropout,
   focal γ. Cheap on cached features; grid or a few rounds.
4. **Threshold tuning** — pick the operating threshold from the gold-set PR curve (the recipe in
   `docs/notes/us-filter-threshold-recipe.md` is the analogue).
5. **US threshold revisit** — the foreign-protest leak suggests a higher US cutoff or a better US
   gate for the US-CCA intersection (relevant to "discovered events", not CCA detection per se).

## Open decisions for the morning
- Form filter for the "main" model: all-forms vs street-only vs street+boycott? (Decide after the
  street-restricted comparison + the coded `event_type` distribution.)
- How to weight gold-set precision vs DoCA-recall when ranking experiments.
- Whether to re-draw the gold set from the FULL cache (more high-band predicted-positives) once part 2
  finishes, or keep the 2,553-row train250k draw (the first 500 are coded either way; "draw once" holds
  unless we want >2,553 or more high-band).

## Day 2 (cont.): model characterization + out-of-sample prep

Wrote `docs/notes/cca-model-characterization.md` — what the model does, the eval methodology, and
**performance vs baselines in corpus space**: random-at-prior P≈0.02 (1x), a protest-keyword lexicon
P≈0.19 (~9x), the model P≈0.82 at logit ≥1.0 (~41x). The keyword baseline's *raw* gold precision
(0.83) looks like it beats the model, but that's a stratification artifact — reweighted to corpus
space it's 0.19, and the model is ~4x more precise. (Methodological catch: never compare a model's
reweighted precision to a baseline's raw gold precision.)

**Yield projection** (`scripts/cca_corpus_scores.py` → `cca_doca/experiments/corpus_scores_*.json`):
on part-1 (1960–1975, 1.61M US-restricted), all-forms @logit 1.0 flags 9,722 → ~8,000 true CCA
events at ~0.35 DoCA recall; lower the threshold for ~17,300 true events at 0.55 precision.

**Out-of-sample prep**: `embed_corpus` gained `--corpus` / `--year-column` so it can target the LDC
corpus (Hive-partitioned on `publication_year`). LDC 1995–2007 = 681,470 articles (~3.6h to embed),
with `cca_descriptor` = 8,981 positives as the noisy out-of-period reference (DoCA matches are gone
post-1995). Plan: embed overnight, then score both models and evaluate vs `cca_descriptor` for
temporal + cross-source generalization.

## Resume checklist
- `git checkout cca-doca-retrain`. Harness + holdout guard + honest baseline + π sweep + the corrected
  street experiment + the characterization doc are committed.
- **Overnight LDC embed** (instead of API part-2): see the command in this doc's day-2 section / the
  characterization doc's out-of-sample section. After it lands, score both models + eval vs `cca_descriptor`.
- **Two parallel model tracks** (collaborator call on street-only scoping pending):
  - all-forms: `cca_doca.weights.h5` (169-positive gold task), π=0.02.
  - street-only: `cca_doca_street.weights.h5` (`--form-filter any_street`, 144-positive gold task), π=0.02.
  - Run future experiments + the threshold lock for BOTH; report side by side.
- **Candidate next steps** (frontier-movers — π and form filter are settled operating-point knobs):
  (1) lock the deployment threshold per track from the gold PR curve; (2) encoder unfreeze (LayerLRModel
  `freeze_encoder=False`) — biggest representational lever, heavier (token-mode) run; (3) refine label
  definitions; (4) broaden/ form-stratify the gold set (only 500 of 2,553 coded; street-dominated).
- Part 2 full embed (1976–1995 `--append`) still pending (evening; non-blocking for tuning).
- π models: `cca_doca/cca_doca_pi{0.01,0.03,0.05}.weights.h5`. Eval records: `cca_doca/experiments/eval_*.json`
  (identify by `prior` + `form_filter` + `weights_path`). A `compare_experiments.py` (load records → frontier
  table) is the obvious next harness piece.

## Day 3 (2026-06-18): US-filter hardening, reconcile, calibration, memo

The day's goal was the MVP collaborator deliverable, gated on hardening the US filter (which turned
out to be a 200-step smoke test, not a real run). All of it is done.

**US filter — properly retrained + calibrated + validated.** The filter trains token-mode (full
backbone over ~1.16M LDC rows), so we made it cheap by embedding its training set once and training
the head features-mode:
- `src/embed_corpus.py` gained `--source-pattern` / `--lead-column` / `--label-column` / `--no-year`
  / `--limit` to embed `us_filter/ldc_labeled.parquet` (stripped_text channel, us_label carried) →
  cache `us_train_ldc/` (630,663 rows). This also fixed a latent `data_from_parquet` bug (greedy
  `us_filter/**/*.parquet` glob pulled in `audit/api_ldc_matched.parquet`; added a `pattern` override).
  **`run_us_classification.py` still has this bug — apply the same `pattern` fix when next touched.**
- `src/run_us_features.py` (new): features-mode BCE US trainer → `us_classifier_full.weights.h5`,
  held-out **F1 0.97**. `src/calibrate_us_filter.py` (new): Platt on natural-balance val (A=1.03,
  B=−0.22, ECE 0.007→0.004). Old smoke `us_classifier.weights.h5` kept for comparison only.
- Validated: DoCA-recall sweep (0.96 @ calib 0.5; recipe thr 0.25 → 0.98 recall, the deployment
  operating point); gold-set `us_event` agreement 0.934 (new) vs 0.912 (old) — +11 foreign rejected
  at no US-recall cost.

**CCA reconciled against the new filter.** `run_cca_doca.py` gained `--us-weights` (re-scores
`us_logit` by applying the new calibrated US head to cached CLS — valid, shared frozen backbone — no
re-embed). Both tracks re-fit at calibrated US thr 0.5. Result: **frontier unchanged within noise**
(raw precision at matched DoCA recall differs ≤0.01; training curves near-twins). The apparent
reweighted-precision "drop" at fixed logit was IPW variance + nnPU logit-scale non-identifiability
(same as the π-sweep). US-hardened models PROMOTED to canonical (`cca_doca.weights.h5`,
`cca_doca_street.weights.h5`); pre-hardening kept as `*_oldus`.

**CCA calibration.** `src/calibrate_cca.py` (new) + `platt_fit` gained `sample_weight`: IPW-weighted
Platt so probabilities map to corpus base rate, not the gold's 34% (all-forms A=1.32/B=−1.02; street
A=1.81/B=−1.05; IPW-weighted mean calibrated prob == positive rate, aggregate-calibrated). Artifact
triples now complete for all three models.

**Out-of-sample** (`src/validation/cca_oos_eval.py`, new): scored LDC 1995–2007 (out-of-period +
cross-source) vs `cca_descriptor`. ROC-AUC 0.89 (all-forms) / 0.90 (street). US-restricted pipeline
over 681k LDC articles: ~4,524 US events @ score 1.0, ~3,100 untagged by the NYT descriptor — the
dataset-expansion evidence, including events the tag missed (e.g. Buffalo 1999 abortion protests).

**Deliverable:** `docs/reports/cca-collaborator-memo.md` finalized (operate at score ≥ 1.5 / calib
P ≥ 0.73 → ~0.79 precision / ~0.33 DoCA recall / ~40× base rate). Operator ships it Friday.

648 tests green, my files ruff-clean.

### Next session — pickup points (in priority order)
1. **Immigration labels (the crux of the multi-head).** Trustworthy immigration *training* positives
   are the open problem — descriptor/keyword tags are over-generous (the same disease DoCA-matching
   cured for CCA), and there's no DoCA-equivalent gold source. The 500 gold rows carry `immig` codes
   for *eval* only. Timeboxed go/no-go scoping is the first task (task board #9): assess refining NYT
   descriptors, NYT-API subject keywords, and whether a small hand-coded positive set is needed.
2. **Multi-head ICA** (#10, gated on #1): if labels crack, the immigration head trains features-mode
   on cached embeddings cheaply (like CCA); then assemble US → CCA + immigration → ICA.
3. **Saturday tier** (#11/#12): full CLAUDE.md/README reconciliation (this handoff + the
   `project-state-and-data-map.md` note are the interim source of truth); full-corpus apply
   (`api_us_scores` + `api_cca_scores`); broaden the gold set; the `run_us_classification.py` glob fix.
