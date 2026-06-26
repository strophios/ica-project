# Roadmap & index — current live next-steps

*Created 2026-06-19. Branch `cca-doca-retrain`. THIS is the single live list of
what's next and what's deferred. The root `CLAUDE.md` and `README.md` are stale
(pre-`cca-doca-retrain`) — full reconciliation is itself a deferred item below.*

## Where to look (index)

| For… | See |
|---|---|
| **What's next / deferred (live)** | this doc |
| **Data layout, artifacts, model state** | `project-state-and-data-map.md` |
| CCA/DoCA retrain arc (detail/reasoning) | `cca-doca-handoff.md`, `cca-doca-retrain-design.md`, `cca-model-characterization.md` |
| Relevance-head arc (detail/reasoning) | `relevance-head-handoff.md` |
| US filter / dateline pipeline | `us-filter-*.md`, `r/CLAUDE.md` |
| **US head retrain (diaspora recall)** | `us-head-retrain-plan.md` |
| Tier 1–5 audit/refactor history | `tiers-and-checkpoints.md`, `tier{2,3,4,5}-design.md` |
| Deferred substantive questions | `pinned-questions.md` |
| Process / engineering patterns | `process-patterns.md`, `engineering-patterns.md` |

The per-arc docs retain narrative and reasoning; their forward-looking lists are
consolidated **here** so nothing drifts out of sight.

## Status snapshot

CCA head (DoCA-matched, calibrated, two tracks) — done. US filter (F1 0.97,
calibrated) — done. **Relevance head (η=0, fused-gated)** — done this session.
**Smarter US gate** (location-fused) — done. Next major piece: assemble the
multi-head ICA model.

## A. Active thread — multi-head ICA assembly (next major piece)

**DESIGN DECIDED (2026-06-19): `docs/design-plans/2026-06-19-multi-head-ica-assembly.md`**
(6 phases + pre-flight checklist; implementation plan next). Pieces all exist:
fused US gate, CCA head, relevance head. Assembly =
`fused US gate → {CCA head, relevance head} → calibrated composition`. The three
open questions are now resolved in the design doc:
- **Fusion-head training labels** → no heavy fusion head: per-head-calibrated
  AND baseline vs. a ≤3-param LR challenger (EPV-capped at 466 anchors), chosen
  empirically; the anchors are spent on a clean held-out eval, not on training a
  combiner.
- **Shared vs. separate encoder** → shared *frozen* encoder (already de-facto, via
  the one CLS cache). Frozen kills negative transfer; top-N joint fine-tune is a
  tracked, separately-measured ceiling-lift experiment (deferred, not in scope).
- **Binary vs. typed relevance** → assemble with the binary head now; typed
  multi-label heads (B4) folded in later.

Key design move: a **harmonized retrain** of CCA + relevance (same fused gate,
same held-out joint-ICA eval ids) to kill anchor contamination by construction;
US is a recall-tuned **gate**, not a fusion term (the heads are conditional-on-US
estimators). Apply targets: `api_corpus` 1960–1995 + LDC 1996–2007 (the
out-of-DoCA expansion test).

## A2. US head retrain — deferred follow-up to the assembly (2026-06-26)

The Phase-4 fusion fit surfaced a hard recall ceiling: the US gate misses
**diaspora/solidarity protests** (US-soil collective action about foreign topics —
e.g. Haitian exiles marching in NYC), which are the highest-value ICA category. It
drops 27% of held-out ICA positives; 19% of the DoCA anchors score `p_us < 0.2`.
Ruled out era/dateline/sparsity — the cause is content (foreign-topic text swamps
the US-location signal), rooted in dateline labels encoding *filing* location, not
*event* location. The fused gate can't rescue ML-negatives.

The assembly ships on the current head with **gold-first gating** (trust DoCA /
dateline `us_label` over the ML head) — fine for known positives, leaves a
documented ceiling for novel diaspora events. The retrain (DoCA + section labels,
nnPNU, strip-vs-no-strip experiment, validate-before-swap) is a substantial
follow-up: full findings + design in **`us-head-retrain-plan.md`**.

## B. Relevance head — deferred (from `relevance-head-handoff.md`)

1. **Operating-threshold re-tuning** for the fused-gated head — the retrain
   shifted the logit scale; pick the threshold from the gold PR curve
   (`us-filter-threshold-recipe.md` is the analogue). Frontier improved; only the
   operating point is stale.
2. **Richer location signal (gate option C)** — the heuristic gazetteer is thin
   (gates ~47% of foreign alone); fuller country/city lists or light lede NER
   would lift the fused gate. Bare US cities w/o a state parenthetical ("Syracuse")
   are the known miss.
3. **Feature-fusion US retrain (gate option B)** — add the `us_assign` signals
   (`any_us`/`any_not_us`/desk/section) as FEATURES to the US classifier and
   retrain (learns the weighting vs. the hand-rule). Fold in when the multi-head
   US head is (re)built.
4. **Typed multi-label relevance heads** — the η per-type result (foreign-negative
   pressure helps exclusionary/access, hurts diaspora) says the four types want
   separate sigmoid heads; also yields the typology output. Typed anchors exist
   (`event_type4`).
5. **Honest precision eval** — gold immig is thin (17 pos) + CCA-stratified; add
   IPW reweighting + a relevance-stratified gold draw.
6. **DEDPUL π for relevance** — pass-1 used π=0.05 by analogy; estimate + sweep
   (frontier-invariant, but record it).

NOT to revive without a different negative set: **nnPNU reliable negatives** — the
η-sweep was a clean negative result (foreign-news and diaspora relevance are the
same content signal). Machinery retained; η=0 canonical.

## C. Saturday Tier — apply + project hygiene (from `cca-doca-handoff.md`)

- **Full-corpus apply** — produce `cca_doca/api_cca_scores/` and
  `us_filter/api_us_scores/` over `api_corpus` (absent today). Use the fused gate.
- **Broaden the gold set** — only 500 of 2,553 template rows coded; street-/
  CCA-stratified. Re-draw stratified for relevance + a larger US/immig slice.
- **`run_us_classification.py` greedy-glob bug** — still reads `us_filter/**/*.parquet`
  greedily (pulls in `audit/api_ldc_matched.parquet`, no `id` → crash). Apply the
  additive `pattern=` fix (as done for `data_from_parquet`) when next touched.
- **CLAUDE.md / README.md reconciliation** — full rewrite to current state
  (deferred until a working multi-head lands; banners added 2026-06-19).

## D. Older deferred (indexed, not duplicated)

- **Tier 5 empirical runs** (real-data short/full + cluster `mixed_float16` runs;
  π=0.03-vs-0.02 research handoff) — runbooks in
  `docs/notes/tier5-implementation-plan/phase_07.md`/`phase_08.md`, status in
  `tiers-and-checkpoints.md`. HUMAN-OPERATED.
- **US filter operator-gated items** (cluster shakedown, gold-set hand-coding →
  slice eval / DoCA recall / escalation) — `us-filter-design-handoff.md`,
  `docs/implementation-plans/2026-06-06-us-filter/phase_*.md`.
- **Substantive deferred questions** (ALUM/VAT, nnPU+α+γ composition, multi-class
  heads, preprocessor train/predict split) — `pinned-questions.md`.
