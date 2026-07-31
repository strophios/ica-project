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
