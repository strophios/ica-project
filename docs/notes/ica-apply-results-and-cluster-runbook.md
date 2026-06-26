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
