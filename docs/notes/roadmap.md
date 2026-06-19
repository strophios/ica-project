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

Pieces all exist now: fused US gate, CCA head, relevance head. Assembly =
`fused US gate → {CCA head, relevance head} → final fusion head` ("is this a CCA
event of special relevance to immigrants?"). Open design questions to work
through (the deliberate pause is here):
- **Fusion-head training labels.** The 466 ICA anchors are the only joint
  US∩CCA∩relevance gold — thin. How to train the fusion head on that.
- **Shared vs. separate encoder.** Each head is currently a standalone
  frozen-DAPT backbone + single head; whether to unify onto a shared encoder.
- **Binary vs. typed relevance.** Assemble with the binary relevance head now, or
  wait for the typed multi-label heads (B4)?

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
