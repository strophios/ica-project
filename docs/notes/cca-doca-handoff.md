# CCA/DoCA Retrain — Handoff & Tuning Plan

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

## Resume checklist
- `git checkout cca-doca-retrain`; `tail /tmp/embed_full_part1.log` (part 1 status).
- If part 1 done: run part 2 (command above) in the evening.
- Build `run_cca_eval.py`, then the registry + `compare_experiments.py`, then run the experiments.
