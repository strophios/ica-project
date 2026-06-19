# Project state and data map

*Last updated: 2026-06-19. The **data/artifact MAP** — where things live and what
state they're in. For **what's next / deferred**, see `docs/notes/roadmap.md` (the
live roadmap + index). The top-level `CLAUDE.md` (2026-06-10) and `README.md`
(April) predate the entire `cca-doca-retrain` arc and are stale; full reconciliation
is deferred (a roadmap item). This note is a snapshot, not a contract — lead facts
are verified against the filesystem on the date above. Updated 06-19 after the
relevance head + smarter US gate landed (see below).*

## The one-paragraph version

The CCA classifier was retrained on the NYT Archive API corpus (1960–1995) with DoCA-matched
articles as positives, replacing the old over-generous NYT-descriptor labels. It trains in
minutes because it runs on cached frozen-DAPT CLS embeddings (768-d), not live forward passes.
Honest gold-set metrics exist (500 hand-coded articles, leakage-held-out). The US pre-filter
that gates the CCA training background is **not** finished — its current weights are a 200-step
smoke test, uncalibrated and unapplied. A real US filter is being retrained now (overnight embed
of its training set running as of this writing). All large data and weights live *outside* this
repo, in sibling and grandparent directories.

## Out-of-repo directory map

The repo holds code; the data and trained artifacts do not. Three levels matter.

**Grandparent — `/Users/strophios/immigration_project/00_ML_data_expansion/`** (`../../` from repo root):
- `LDC2008T19/data/` — raw LDC corpus (NYT XML, 1987–2007) plus its R processing pipeline
  (`scripts/`, XML → RDS → CSV → parquet) and the produced artifacts (`parsed_to_rds/`, etc.).
  - `cca_matches_good.rds` (top level, ~1.89 MB, dated May 8) — **the authoritative DoCA→article
    match.** ~15,627 unique `article_id`, ~19,568 rows `match_quality == "succeeded"`. Matched to
    the API corpus across the whole span with LDC as backup. This is what feeds the CCA positives.
  - `cca_matches.rds` (Apr 29) and `cca_matches_v1/*.rds` (Feb) are **superseded** — older split
    strategies. Don't read them by accident; size/date alone makes `cca_matches.rds` look current.
- `nyt_headlines.R` — downloads + processes the NYT Archive API corpus. It does **not** extract a
  dateline; the API payload has none. So no datelines exist on the API side, by source, not by loss.
- `nyt_archive_by_year/` — the processed API corpus split per year as `.rds` (keywords as a
  list-column).
- `tmp.R` — the current DoCA→NYT matcher (fuzzy headline match + exact `pub_date` block, ~80% hit).
  Modified today (Jun 17); see the freshness flag below.
- `nyt_location_checking.R` — the original US/not-US heuristic, since vendored to `r/vendored/us_assign.R`.

**Parent — `/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/`** (`../` from repo root):
- `api_corpus/` — NYT API corpus as parquet, 1960–1995, ~3.7M rows, 36 per-year files. Columns
  include `id, headline, lead_paragraph, abstract, keywords, year, news_desk, section_name`. **No
  dateline column.**
- `ldc_corpus/` — LDC corpus as hive parquet (partitioned on `publication_year`), 1987–2007, ~1.16M
  rows. Optimized for the original CCA run, so it's **missing columns** versus the full data; only
  `id, headline, lead_paragraph` overlap with `api_corpus`, plus it carries `full_text` and the
  `cca`/`immig` descriptor labels.
- `cca_doca/` — retrained CCA artifacts: weights + config sidecars, per-run metrics CSVs,
  `experiments/` (eval + corpus-score JSON), `prior_estimate.json`, `embed_cache/`, plus
  `scored_candidates.parquet` and the `face_validity_*.csv` dumps.
- `us_filter/` — US classifier: `us_classifier.weights.h5` (the smoke-test weights, see below),
  `ldc_labeled.parquet` (the training source: `us_label` + `stripped_text`), `audit/`, `logs/`,
  `classifier/`, `us_set/`.
- `validation/` — the gold set: `cca_coding_first500_coded.csv` (500 hand-coded rows),
  `cca_coding_template.parquet` (2,553-row score-stratified template), and the uncoded sample.
- `doca.csv` — DoCA event data as CSV (~23.6k events, 1960–1995). Modified today (Jun 17).
- `cca_set/`, `cca_logs/` — **stale.** Artifacts from the *original* keyword-label CCA run,
  superseded by `cca_doca/`. Don't wire new work to these.

**Repo — `ica_project/`** — the git repo, branch `cca-doca-retrain`.

## Data lineage

Two NYT corpora plus the DoCA event set, joined to produce labels.

- **LDC corpus:** XML → R (`LDC2008T19/data/scripts/01_all_to_rds.R` and on) → `parsed_to_rds/` →
  `ldc_corpus/` parquet. Datelines survive here (extracted from the NITF XML).
- **NYT API corpus:** API → `nyt_headlines.R` → `nyt_archive_by_year/` → `api_corpus/` parquet.
  No datelines.
- **CCA positives:** `doca.csv` → matched by `tmp.R` → `cca_matches_good.rds` →
  `r/doca/export_cca_positives.R` → `cca_doca/cca_doca_positives.parquet` → the `cca_label`.
- **US labels:** `r/dateline/build_labels.R` over the LDC corpus → `us_filter/ldc_labeled.parquet`,
  where `us_label` comes from datelines fused with desk/section signals, and `stripped_text` is the
  leakage-guarded text channel (dateline removed so the model can't cheat).

**Freshness flag:** `doca.csv` and `tmp.R` were both touched today (Jun 17), but
`cca_matches_good.rds` is dated May 8 and `cca_doca_positives.parquet` Jun 15. So today's DoCA edit
has **not** propagated to the match or the positives. If the edit was substantive, the match needs a
re-run; if incidental, ignore. Worth a one-line confirmation before the final retrain.

## Current model state (`cca-doca-retrain`)

**CCA classifier — retrained, evaluated, two tracks.**
- Trained features-mode on cached CLS embeddings (`cca_doca/embed_cache/train250k/`) via
  `run_cca_doca.py`. Positives = DoCA matches (~13,742 in the training table after US-restriction);
  unlabeled background = US-restricted articles (`us_logit ≥ 0`, raw — see the US caveat).
- Prior π = 0.02, DEDPUL-re-estimated for this population (`prior_estimate.json`); swept over
  {0.01, 0.02, 0.03, 0.05} and found to be an operating-point knob, not a quality knob.
- Loss: focal nnPU (FLPU). Pipeline scripts: `embed_corpus.py` → `build_cca_doca_table.py` →
  `run_cca_doca_prior.py` → `run_cca_doca.py`.
- Two parallel definitions carried forward (a collaborator scoping call): **all-forms**
  (`cca_doca.weights.h5`) and **street-only** (`cca_doca_street.weights.h5`).
- Honest eval: 500 hand-coded gold rows (169 positive for all-forms, 144 for street),
  leakage-held-out, IPW-reweighted to corpus base rate. All-forms at logit ≥ 1.0: reweighted
  precision ≈ 0.82, DoCA recall ≈ 0.35; versus a protest-keyword lexicon at 0.19 and random at 0.02
  (~41× the base rate). Numbers live in `docs/notes/cca-model-characterization.md` and
  `cca_doca/experiments/eval_*.json`.

**Embedding caches** (`cca_doca/embed_cache/`):
- `train250k/` — the CCA training cache. Done.
- `full/` — corpus-wide API embed; part 1 (1960–1975) done, part 2 pending, non-blocking.
- `ldc_9507/` — LDC 1995–2007, for the out-of-sample generalization check. Done; not yet scored.
- `us_train_ldc/` — the 630,663 labeled rows of `ldc_labeled.parquet` (stripped_text channel,
  us_label carried), the reusable cache that retrains the US head features-mode in minutes.

**US filter — retrained, calibrated, validated (2026-06-18).** `us_classifier_full.weights.h5`
(+ `.config.json` + `.calibration.json`) is the real filter: features-mode BCE head trained on the
`us_train_ldc` cache (`src/run_us_features.py`), held-out test **F1 0.97**, Platt-calibrated on the
natural-balance LDC val (`src/calibrate_us_filter.py`; A=1.03, B=−0.22, ECE 0.007→0.004). The old
`us_classifier.weights.h5` (200-step smoke test) is kept for comparison only — do not use it.
Validated: DoCA-recall sweep (0.96 @ calib 0.5; recipe threshold 0.25 for 0.98 — the deployment
operating point), and the gold-set `us_event` comparison (agreement 0.934 vs the old 0.912). Not yet
applied to the full API corpus (`us_filter/api_us_scores/` absent — that's the Saturday apply step).

## Verified solid vs. open

**Verified this session:**
- The retrain used the correct match file (`cca_matches_good.rds`) and the right corpus
  (`api_corpus`, 1960–1995). The "wrong match file" scare was a stale-copy misread.
- Prior re-estimated (π = 0.02) and swept.
- Honest, leakage-held-out, reweighted gold-set metrics exist.
- One shared `ClassificationHead` (MLP: dropout → dense 768→768 relu → dropout → dense 768→1) serves
  both US and CCA — no bespoke US head.

**Done 2026-06-18:**
- US filter retrained (F1 0.97) + calibrated + validated (see above).
- CCA models reconciled against the new US filter and re-fit (both tracks) — frontier unchanged
  within noise (raw precision at matched DoCA recall differs ≤0.01; the scale shift is the known
  nnPU logit non-identifiability, frontier-invariant). The US-hardened models are now canonical
  (`cca_doca.weights.h5`, `cca_doca_street.weights.h5`); the pre-hardening ones are `*_oldus` backups.
- Both CCA tracks Platt-calibrated, IPW-weighted to corpus base rate (`src/calibrate_cca.py`;
  all-forms A=1.32/B=−1.02, street A=1.81/B=−1.05). Artifact triples complete for all three models.
- Out-of-sample eval (`src/validation/cca_oos_eval.py`): LDC 1995–2007 ROC-AUC 0.89 (all-forms) /
  0.90 (street) vs `cca_descriptor`; US-restricted yield ~4,524 events @ score 1.0 (~3,100 untagged).
- Collaborator memo finalized: `docs/reports/cca-collaborator-memo.md`.

**Done 2026-06-19 (relevance head + smarter US gate):**
- **Immigrant-relevance head — built** (`relevance.weights.h5`, η=0, fused-gated).
  Positives = curated immigration-content descriptors ∪ 466 ICA anchors, US-restricted,
  FLPU features-mode; gold AUC ≈ 0.94. The nnPNU reliable-negative experiment was a
  clean negative result (η=0 canonical). See `relevance-head-handoff.md`.
- **Smarter US gate** — `src/preproc/us_location.py` fuses the dateline ML filter with
  a location signal (`any_us`/`any_not_us`); the fused gate halves the foreign-event
  leak. Threaded into `run_relevance`. Deferred gate options B/C in `roadmap.md`.
- New data products (gitignored): `relevance/{candidates,ica_anchors,reliable_negatives}.parquet`,
  `cca_doca/embed_cache/relevance_{pos,train}/`, `relevance/relevance*.weights.h5`.

**Open gaps (full live list in `roadmap.md`):**
- Multi-head ICA model (US gate → CCA + relevance → ICA): not assembled — the next major piece.
- No full-corpus apply outputs (`api_cca_scores/`, `api_us_scores/` absent) — the Saturday apply step.
- Larger hand-coded gold set (only 500 of 2,553 drawn; street-dominated) to tighten precision SEs.
- Full CLAUDE.md/README reconciliation (banners added 06-19; full rewrite deferred).

**Latent bug fixed this session:** `data_from_parquet` globbed `us_filter/**/*.parquet` greedily,
pulling in `audit/api_ldc_matched.parquet` (no `id` column → crash). Fixed with an additive `pattern`
override. `run_us_classification.py` still reads `us_filter` via the greedy glob and will hit the same
crash on its next run — apply the same fix when that path is next touched.
