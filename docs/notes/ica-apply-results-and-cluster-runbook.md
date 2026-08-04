# ICA apply — local results + cluster runbook

*Created 2026-06-26. Branch `cca-doca-retrain`. Closes Phase 6 of the multi-head
ICA assembly. Records what ran locally, the channel correction that made it
possible, and the cluster work still outstanding.*

## What the assembled model produces

`src/apply_ica.py` runs the assembled `IcaModel` (frozen DAPT encoder + calibrated
`{us, cca, rel}` heads + the `fusion.json` product combiner with composed Platt)
over a cache of CLS features and writes ranked `ica_candidates`. Two runs landed
locally on existing caches.

### LDC 1996–2007 (out-of-DoCA expansion test) — done locally

- Cache: `ldc_9507` (raw `headline_with_lead`), 624,842 rows in 1996–2007.
- Gate: gold-first — dateline `us_label` decides 56.5%, ML US head (on raw CLS)
  the other 43.5%. 86.8% gated in at τ_us=0.02.
- Output: `cca_doca/ica_candidates/ldc_1996_2007.parquet`
  (`id, year, us_score, cca_score, rel_score, ica_score, gated, gate_source`).
- Face validity (top candidates): "Asylum-Seekers Are Confined To Dormitories
  After Protest" (0.95), "Latinos Protest in California In Latest Immigration
  March" (0.93), "Demanding Parole, Immigrants Held in Queens Stage Protest",
  "Hunger Strike by 6 Immigrants Enters 2nd Week", "Across the U.S., Protests for
  Immigrants Draw Thousands". Clean US immigrant collective action. Gold-first
  visibly rescues low-ML-US rows ("Hispanic March Draws Crowd to Capital",
  ML us=0.34, gated in by gold).

### API 1960–1975 — done locally (partial; 1976–1995 deferred)

- Cache: `full` (raw), 1,831,300 rows — this is API part-1 only (1960–1975).
- Gate: ML US head on raw CLS, no gold labels for the API corpus. τ_us=0.02 is
  very lenient here (97.4% pass), so the gate is nearly a no-op and the cca×rel
  product does the ranking.
- Output: `cca_doca/ica_candidates/api_1960_1995.parquet` + per-year
  `us_filter/api_us_scores/` and `cca_doca/api_cca_scores/`.
- Face validity (top candidates): real diaspora ICA — "100,000 IN MARCH FOR
  SOVIET JEWS", "Cuban Refugees Protest Plan for Detente With Castro", "March
  Backs Greek Cypriote Refugees", "10 Vietnamese Here Arrested At Sit-In". These
  are exactly the diaspora category the US head under-scores (many have ML
  us<0.1), surfaced because the lenient gate doesn't drop them.
- Caveat: the lenient gate also lets through some genuinely foreign events
  ("Police Hold Moscow Jews Protesting", "Immigrants Protest in London"). With no
  gold labels for the API corpus, the US-head diaspora ceiling is unmitigated
  here. The US-head retrain (`us-head-retrain-plan.md`) is what improves API
  precision.

## The channel correction (why the plan changed)

Phase 6's plan called for re-embedding LDC with `stripped_text` for *all* heads.
That's wrong for CCA and relevance: both were trained on **raw**
`headline_with_lead` (`train250k` and `relevance_train` caches, `lead_column=None`),
so the existing raw `ldc_9507` cache is the *correct* channel for them. Datelines
aren't a leakage risk for CCA/rel — their labels (collective-action,
immigrant-relevance) don't correlate with the dateline; only the US head's label
is dateline-derived. So:

- CCA / rel scoring: raw `ldc_9507` CLS — correct, no re-embed needed.
- US head: trained on stripped text. For the 56% of LDC rows with a gold dateline
  `us_label`, gold-first bypasses the ML head entirely. For the 44% without, the
  ML head scores raw CLS — a channel mismatch, the one place a stripped re-embed
  would help. That re-embed is an optional cluster refinement, not a blocker.

This let both apply paths run locally on existing caches.

## Outstanding cluster work

These are full-backbone forward passes (GPU/cluster-scale), deferred from Phase 6
Task 2. Run on Explorer.

1. **API 1976–1995 embed** — finish the `full` cache:
   ```
   uv run python -m src.embed_corpus --full --years 1976-1995 --append \
       --corpus api_corpus --out-suffix full --stamp <YYYYMMDD>
   ```
   Then re-run `uv run python -m src.apply_ica --corpus api` to extend the API
   candidates to the full 1960–1995 in-period range.

2. **Stripped LDC 1996–2007 re-embed** (optional US-fallback refinement) — improves
   the US ML head on the 44% of LDC rows without a gold label:
   ```
   uv run python -m scripts.build_stripped_ldc_source   # join → stripped source parquet (cheap, can run locally)
   uv run python -m src.embed_corpus --full \
       --source-pattern <stripped_source.parquet> --lead-column stripped_text \
       --no-year --out-suffix ldc_9607_stripped --stamp <YYYYMMDD>   # smoke with --limit first
   ```
   Then `uv run python -m src.apply_ica --corpus ldc --cache-suffix ldc_9607_stripped`.
   Note: only the US head benefits; CCA/rel should still use raw `ldc_9507` (a
   future refinement could score US from the stripped cache and CCA/rel from raw —
   `apply_ica` currently scores all three from one cache, so the stripped re-embed
   trades a small CCA/rel channel mismatch for a US gain on the non-gold subset;
   measure before adopting).

3. **Bigger follow-up: US-head retrain** — see `us-head-retrain-plan.md`. Addresses
   the diaspora recall ceiling at the root (DoCA + section labels, nnPNU,
   validate-before-swap). After it lands, re-run calibration → τ_us → fusion
   (Phase 4 scripts all three) and re-apply.

## Artifact map delta

New/changed data products (gitignored):
- `cca_doca/ica_candidates/ldc_1996_2007.parquet` — LDC expansion candidates (done).
- `cca_doca/ica_candidates/api_1960_1995.parquet` — API candidates (1960–1975 done;
  extend after the cluster embed).
- `cca_doca/ica_fusion.fusion.json` + `ica_fusion_metrics.json` — fusion config.
- `relevance/relevance.{weights.h5,config.json,calibration.json}` — retrained `rel`
  head + calibrator.
- `cca_doca/cca_doca*.weights.h5` — retrained CCA heads (all-forms + any_street).

---

## 2026-07-31 forward apply — finish 1960-1995 + embed/apply 1996-2025

*Added 2026-07-31. Goal: topline numbers for the Aug 6 meeting — complete the
1960-1995 candidates (the `full` cache was 1960-1975 only) and produce forward
candidates through 2025, both with the DEPLOYED frozen stack (mixed-stack
productionization deliberately deferred to avoid extra encoder passes; see
roadmap §A1). Local engineering landed in commit d0c0898: `--lead-fallback-column`
(coalesce channel), `--dedupe-ids`, `apply_ica --out-name/--years`.*

### Channel + hygiene decisions (do not re-litigate at the terminal)

- **1976-1995 embed: raw channel, NO coalesce** — pre-1996 stays exactly on the
  trained channel; the lead-free eras (1960-63, 65-69, 80) remain headline-only.
  Historical coalesce is gated behind the pre-registered 1970s experiment.
- **1996-2025 embed: `headline</s>coalesce(lead_paragraph, abstrct)`** — 2025
  has zero leads (contemporaneous abstracts, sound); pre-2025 the fallback only
  fills gaps (2005 is the known bad patch, ~15% both-missing).
- **`--dedupe-ids` on the 9625 embed only**: resolves the 911 pull-overlap dup
  ids (prefer non-empty lead, then earliest year — runs before the year filter
  so the two parts agree on cross-year dups) and drops 13 empty-id 2025 junk
  rows. Expected log line: `6366847 -> 6365855 rows (992 duplicate rows dropped)`.

### Operator sequence

1. **Sync up**: the 1996-2025 `api_corpus/*.parquet` files (1960-1995 files are
   unchanged since Jun 10; end state = 66 files incl. `2025.parquet` — the
   preflight asserts this), and `git pull` the repo on the cluster.
2. **Smokes first** (minutes each; inspect stats + provenance, then rm the
   throwaway suffix):
   `sbatch scripts/embed_apply_9625.sbatch smoke`
   `sbatch scripts/embed_finish_full.sbatch smoke`
3. **Jobs** (independent — can run concurrently):
   `sbatch scripts/embed_finish_full.sbatch all`   (~1.9M rows + apply)
   `sbatch scripts/embed_apply_9625.sbatch all`    (~2.7M rows in 2 parts + apply)
   Each stage is individually resubmittable (`embed|apply` /
   `part1|part2|apply`); the scripts carry preflight guards and the
   **resume rule** (a part that died mid-run left no `provenance.<offset>.json`
   — delete that part's partial shards before resubmitting it).
4. **Sync back** (small files only; leave the ~20GB of CLS shards on-cluster):
   - `cca_doca/ica_candidates/api_1960_1995.parquet` (now truly 1960-1995)
   - `cca_doca/ica_candidates/api_1996_2025.parquet`
   - `cca_doca/embed_cache/{full,api_9625}/provenance.*.json` (for the record)
   - optionally `us_filter/api_us_scores/` + `cca_doca/api_cca_scores/` per-year files
5. **Topline eval** (local; scripts under construction — roadmap §A1 item 3):
   CCA/composed recall vs DoCA on full 1960-1995; ranks of the 214 hand-coded
   ICA positives; per-year candidate rates 1996-2025 **with 2025 sliced
   separately** (abstract-register shift); top-K face-validity CSVs. The
   1996-2007 overlap with the existing LDC gold-first candidates doubles as a
   cross-corpus consistency check (API ML-gate vs LDC gold-first).

### Cluster submission gotchas (learned 2026-08-03/04)

- **The `gpu` partition's submit filter rejects gres-less or malformed gres
  requests** with the unhelpful `Access/permission denied` — it is not a
  privileges problem. `--gres=gpu:1` and `--gres=gpu:h200` both submit fine;
  bare `--gres=gpu` does not. The scripts now default to `--gres=gpu:1`
  (any GPU; a100s queue much faster than h200s) with an 8h limit sized for
  a100 throughput; pin h200 via CLI override if the queue is kind.
- **Sidecar backbone paths are machine-absolute** — fixed in code
  (`resolve_backbone_path`, commit c155612): a synced sidecar recording the
  writing machine's path now resolves to the platform-canonical
  `DAPT_BACKBONE_WEIGHTS` with a loud log line. Expect
  `backbone path /Users/... absent on this platform; resolved to canonical
  /projects/ahd/dapt_backbone.weights.h5` in every cluster embed log — its
  absence on a default-branch run would itself be suspicious.
- **Diagnostic pattern that worked:** a `--wrap` probe echoing per-boundary
  markers (log-write / module / cd / `bash -n` syntax) discriminates
  environment-layer failures from script failures without burning GPU queue
  time. An instant job death (~1s) with a created-but-empty log means the
  failure predates the script's first line — check the submission layer, not
  the script.

### 2026-08-04 smoke findings — corpus growth, fp32 pin, population contract

The first on-cluster smokes surfaced three findings; scripts updated accordingly.

1. **The corpus grew underneath the arc (and that's fine, once pinned).** The
   operator's backward transform ran 2026-07-31: `api_corpus/` now holds
   **156 top-level per-year parquets, 1870-2025, 14,991,212 rows** locally
   (~8.6M backward rows; the 1870-1959 skeleton is no longer raw-only). The
   earlier "66 files" preflight assert is retired — **the year filter inside
   `embed_corpus` is the population pin**, and it held (2025 = 47,798 rows on
   both machines; the extra dedupe drops, 1072 vs 992, are within-backward-set
   rotating-month overlaps that cannot cross the 1959/1996 divide). Preflights
   now check year *bookends* (1976+1995 / 1996+2025). Embeds pass
   `--source-pattern 'api_corpus/*.parquet'` (top-level only) for explicitness.
   NOTE: root `CLAUDE.md` + `project-state-and-data-map.md` still describe the
   66-file corpus — reconcile after this arc; the 1870-1959 era still owes the
   pre-registered schema/missingness/register audit before anything trains or
   applies on it.
2. **Precision pinned to fp32.** The production caches (`full` part-1,
   `ldc_9507`, `train250k`) were produced LOCALLY in float32 (verified via
   their provenance paths); cluster default is `mixed_float16`. Both scripts
   now `export ICA_DTYPE_POLICY=float32` (new env override in `src/config.py`)
   so the appended/new shards are precision-uniform with what the heads and
   calibrators were fit on — and so no fp16/fp32 seam lands on the 1975/76
   era boundary of an era-comparison cache.
3. **us_logit vintage check (open until re-smoke).** Cluster smoke produced
   identical CLS stats (cls_std 0.3806) but shifted us_logit
   (mean 1.99 vs 2.91 local) on the same 200 rows. fp32 re-smoke
   discriminates: if it matches local exactly, it was fp16 numerics; if not,
   the cluster's `us_classifier.weights.h5` is a different vintage (local:
   503,758,392 bytes) — re-sync it + its `.config.json`. Either way the
   apply path is insulated (IcaModel rescores US from CLS via
   `us_classifier_full`; cached `us_logit` is inert for apply), but the
   cached `us_logit` matters for future table builds, so pin it.

**Re-smoke acceptance criteria (cluster, after git pull):**
`(6366847, 6)` corpus load if the backward files are nested on the cluster
(or `(14991212, 6)` if top-level — either is fine, the year filter pins);
`dedupe` line consistent with the loaded set; `year filter 2025-2025: 47798`;
us_logit `min/mean/max = -4.91/2.91/7.59` matching local; provenance carries
`lead_fallback_column=abstrct`, `dedupe_ids=true`. Then submit both `all` jobs.

### 2026-08-04 (later): RESOLVED — the "us_logit mystery" was a LOCAL tensorflow-metal bug

The smoke-discrepancy investigation concluded, and it inverts the earlier
framing: **the cluster numbers were correct all along; the local (MPS)
numbers are wrong.** Supersedes the "re-smoke acceptance criteria" above —
the correct 2025-smoke us_logit stats are the CLUSTER's
`min/mean/max = -4.53/1.99/6.05`, NOT the local -4.91/2.91/7.59.

**Finding.** tensorflow-metal mis-executes the `ClassificationHead` dropout
sub-path inside compiled `model.predict` graphs: deterministic per process,
but wrong versus true math. Chain of evidence: cluster/local CLS bit-identical
(7e-6) with us_logit shifted on all rows → head weights verified
loaded-correctly on BOTH machines (h5 dataset comparison) → cluster cache
self-consistent (head(CLS) == cached logits exactly) while the LOCAL cache is
not → local predict-vs-direct disagrees (deterministically) on MPS and is
EXACT on local CPU. Magnitudes on real CLS features: us mean +1.6 (maxabs
5.5), cca -1.4, rel -0.7 logits.

**Blast radius.** Every score product computed on local MPS predict paths:
the cached `us_logit` in all locally-embedded caches (train250k spot check:
mean -0.84 shift, ~2.9% of rows flip sign at the us_logit≥0 gate used for
US-restriction of training tables), the June ica_candidates (both files),
the three Platt calibration fits, the fusion fit, and the eval-set score
products behind the memo numbers (ROC 0.80/0.82 etc. — measured through a
consistently-distorted pipeline, so internally coherent but not the true
model's numbers). CLS features themselves are CORRECT everywhere (the bug is
head-path-only); trained weight artifacts are unaffected as artifacts.

**Guard added.** `apply_ica.assert_scoring_integrity` (commit with this note)
runs predict-vs-direct on a 256-row sample before writing any candidates and
fails loudly on a distorted stack — local MPS apply now refuses to run, by
design. Cluster (CUDA) and CPU pass.

**Consequences for this arc.** The two cluster jobs are cleared to launch —
cluster embeds + cluster applies produce TRUE scores end to end. Follow-ups
(operator decision pending): re-apply LDC on the cluster (its CLS cache is
valid; only scores need regenerating), refit calibrations/fusion on true
logits (CPU or cluster), re-run the eval suite for true topline numbers, and
adjust memo language if it has not gone out. Upstream: file a
tensorflow-metal issue with the minimal repro (predict-vs-direct on a
dropout-bearing MLP).

### 2026-08-04 (final): damage quantification — reported eval numbers hold to ~±0.01

`scripts/compare_scoring_paths.py` scores the 1,131-row eval set through the
exact own-terms/fusion recipe twice — default device (MPS = the distorted path
that produced the memo numbers) vs CPU (exact) — same weights, calibrators,
fusion. MPS column reproduces the production numbers (sanity anchor). Deltas
(true − distorted): composed ICA ROC +0.003 (0.793→0.795), composed PR −0.005;
us own-terms ROC −0.013 (0.925→0.912), rel −0.010 (0.829→0.819), cca +0.002;
gate passes +3.8pp of eval rows under true scores at τ_us=0.02. Conclusion:
the MPS bug's effect on reported metrics is within quoting noise; memo numbers
stand with a one-line correction/footnote, and the calibration/fusion refit is
a post-meeting hygiene item, not a blocker. (Known pre-existing wart observed
en route, both paths equally: apply_us_model's text-mode reload emits a large
skip_mismatch warning block — the production own-terms eval always ran through
it; investigate on next touch of the token-mode eval path.)

### 2026-08-04 (later still): the silent job deaths were the SCRIPT, not the nodes

Correction to the "submission gotchas" and "diagnostic pattern" notes above:
the empty-log instant deaths (jobs 8910960, 8935575, 8935696 — h200 AND a100
nodes) were caused by a bash footgun in the sbatch scripts themselves, not by
the submission layer or node images (d4054/d4055 exonerated). Under
`set -euo pipefail`, the stage guards counted files with
`ls "$CACHE"/glob 2>/dev/null | wc -l`: when the glob doesn't match (the
NORMAL part1 precondition — the cache doesn't exist yet), GNU ls exits 2 with
its error suppressed by the 2>/dev/null, pipefail propagates the 2 through
the pipeline, and set -e kills the script with nothing written. The
interactive smokes never hit the line (the smoke stage has no shard count);
the --wrap probes had no such construct; `bash -n` passes (runtime
semantics, not syntax). Repro: `set -euo pipefail; n=$(ls /nope/x_* 2>/dev/null | wc -l)`
— dies silently, exit 2 (GNU) / 1 (BSD — which is also why the bug never
showed locally).

Fix (committed with this note): pure-bash `count_glob` with `shopt -s
nullglob` in both scripts — no subprocess, no pipefail interaction — plus an
unconditional first-line echo so no run can ever die log-silent again.
**Convention for all future sbatch scripts: never `ls | wc -l` under
pipefail; count with nullglob arrays.** The h200-vs-a100 node pattern in the
earlier note was coincidence (the working probe was a --wrap with no
counting; the failures were all `all`-stage runs).

### 2026-08-04 addendum: the apply stage needs the model artifacts synced up

The 9625 embed parts completed on an h200 in ~20 min (2,666,424 rows —
exactly the expected 1996-2025 population), then the apply stage failed:
`IcaModel()` loads the three June-trained artifact triples + fusion sidecar,
which were produced LOCALLY and had never been synced (the embed stage only
needs the DAPT backbone + the old us_classifier smoke, so the gap surfaces
only at first cluster apply). **Add to the operator sync-up list** (10 files,
~21MB): `cca_doca/cca_doca.{weights.h5,config.json,calibration.json}`,
`cca_doca/ica_fusion.fusion.json`,
`relevance/relevance.{weights.h5,config.json,calibration.json}`,
`us_filter/us_classifier_full.{weights.h5,config.json,calibration.json}`.
Then resubmit the failed stage only (`... apply` — a CPU partition works;
the stage is features-mode). Expect the log line `scoring integrity check
passed` before candidates are written.

### 2026-08-04 addendum 2: `full` cache rebuilt fresh on-cluster (1960-1995)

The finish-full job's preflight fired correctly: the June part-1 (1960-1975)
cache exists only LOCALLY — never synced, same gap class as the model
artifacts. Decision: **rebuild 1960-1995 fresh on the cluster instead of
syncing part-1 up.** At the measured throughput (2.67M rows / ~20 min on
h200) the full 3.7M-row rebuild is ~30 min — faster than uploading ~5.7GB —
and it retires the part-1 metadata contamination (MPS-distorted us_logit,
~3% sign flips at the us_logit>=0 gate) from the production cache. CLS
changes only at kernel-noise level (7e-6 measured local-fp32 vs cluster).
`embed_finish_full.sbatch` is now a fresh 1960-1995 build (empty-cache
preflight, stamp 20260804, no append machinery). The LOCAL `full` cache
(part-1, distorted us_logit) is superseded — do not build new tables from
its us_logit column; the cluster cache is canonical going forward.
