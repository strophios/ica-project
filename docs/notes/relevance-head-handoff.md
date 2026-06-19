# Immigrant-relevance head — handoff & deferred work

*Created 2026-06-19. Branch `cca-doca-retrain`. Captures the relevance-head arc
(pass-1 → nnPNU negative result → smarter US gate) and the explicit deferred
items, so the multi-head assembly can be picked up cleanly. Read with
`docs/notes/project-state-and-data-map.md` (data layout) and the root `CLAUDE.md`
("Planned Architecture").*

## The target (architecture recap)

The **immigrant-relevance head** detects, at the *article* level, "something of
special and specific relevance to immigrants" — an **orthogonal, whole-corpus**
axis, NOT gated on collective action. The final ICA head fuses **relevance +
CCA** scores ("is this a CCA event of special relevance to immigrants?"). The
**US filter is a raw gate, not a fusion input** — it decides which articles enter
training/scoring at all.

## What landed this session

**1. Pass-1 relevance head (`relevance.weights.h5`, η=0).** Positives = curated
immigration-content NYT descriptors (`scripts/build_relevance_candidates.py`,
lexicon documented there) ∪ the 466 hand-coded ICA anchors
(`r/doca/export_ica_anchors.R`); US-restricted; FLPU features-mode on cached CLS
(`src/run_relevance.py`). Gold AUC ≈ 0.91; recovers relevant articles the lexicon
misses (the embedding-generalization win). Pipeline mirrors the CCA/DoCA recipe.

**2. nnPNU reliable-negatives — NEGATIVE RESULT (do not revive without a
different negative set).** Hypothesis: feed confidently-foreign articles as
reliable negatives so the head stops over-firing on US-datelined foreign news.
Built the nnPNU loss extension (`FLPULoss.nnpnu_eta`, tested, η=0 bit-identical),
the reliable-N selector (`scripts/build_reliable_negatives.py`), the 3-way split
(`data.create_relevance_data`), and swept η ∈ {0,.3,.5,.7}. **η HURTS**: gold AUC
0.908 → 0.756 monotonically; diaspora recall falls. **Why:** foreign-news and
diaspora *relevance* are the same content signal — a "foreign = negative" set
can't separate them, so suppressing foreign drags diaspora down. The machinery is
retained as correct/reusable; η=0 is canonical. (Per-type aside: η *helped*
exclusionary/access while hurting diaspora — evidence the types want separate
heads; see Deferred.)

**3. Smarter US gate (the over-firing fix that works).** The fix belongs in the
gate (location-based), not the content head. `src/preproc/us_location.py` ports
the vendored `r/vendored/us_assign.R` location heuristic: per article
`(any_us, any_not_us)` from glocations + desk/section. **Clearly foreign =
`any_not_us AND NOT any_us`** — diaspora carries a US location (Miami/Brooklyn) so
it is NOT clearly foreign, which is exactly the separation content couldn't make.
**Fused gate = ML filter passes AND not clearly-foreign**, threaded into
`run_relevance` (gates positives + background). Retrained η=0 behind it:
- gold `us_event` gating of foreign events 0.886 → 0.936 (≈ halves the leak),
  small US-recall cost (0.954 → 0.931);
- end-to-end relevance: gold AUC 0.931 → 0.944, foreign-tagged FPs 1 → 0;
- removed 6,406 foreign background + 1,110 foreign positives from training.

## Current canonical state

- `relevance.weights.h5` — η=0, **fused-gated** (the current best relevance head).
- `relevance_eta0.0.weights.h5` — η=0, dateline-gated (pass-1 reference).
- `relevance_eta{0.3,0.5,0.7}.weights.h5` — sweep models (negative result; keep
  for the record, not for use).
- Data products (gitignored): `relevance/candidates.parquet` (15,200 positives),
  `relevance/ica_anchors.parquet` (466), `relevance/reliable_negatives.parquet`
  (4,764), `cca_doca/embed_cache/relevance_{pos,train}/`.

## Deferred improvements (DO NOT LOSE — explicitly parked)

1. **Operating-threshold re-tuning for the fused-gated head.** The retrain shifted
   the logit scale, so logit≥1 is no longer the right cut. Pick the deployment
   threshold from the gold PR curve (the CCA recipe in
   `docs/notes/us-filter-threshold-recipe.md` is the analogue). Frontier improved;
   only the operating point is stale.
2. **Richer location signal (gate option C).** The heuristic gazetteer is thin
   (gates only ~47% of foreign *alone*; it's a complement to the ML filter, not a
   replacement). A fuller country/city gazetteer or light NER over the lede would
   strengthen the location channel and lift the fused gate. Bare US city names
   without a state parenthetical ("Syracuse") are the known miss.
3. **Feature-fusion US retrain (gate option B).** Instead of the hand-rule fusion,
   add the `us_assign` signals (`any_us`, `any_not_us`, desk/section) as FEATURES
   to the US classifier and retrain, letting it learn the optimal weighting.
   Likely beats the rule; fold in when the multi-head US head is (re)built.
4. **Typed multi-label relevance heads.** The η per-type result (foreign-negative
   pressure helps exclusionary/access, hurts diaspora) says the four types
   (documentation/access/diaspora/exclusionary) want separate sigmoid heads —
   diaspora especially can't share a global foreign signal. This also yields the
   typology output. Typed anchors already exist (`event_type4` on the anchors).
5. **Honest precision eval.** Gold immig is thin (17 positives) and CCA-stratified;
   add IPW reweighting + a relevance-stratified gold draw for a trustworthy
   precision number (the current headline signal is AUC + face validity).
6. **DEDPUL π for relevance + sweep.** Pass-1 used π=0.05 by analogy; estimate it
   properly and sweep (frontier-invariant per the CCA finding, but records it).
7. **Typology is provisional.** It was derived on CCA *events*, may evolve; the
   multi-label design (binary head primary + non-exhaustive type heads) respects
   this.

## Next major piece: multi-head ICA assembly (to design together)

Pieces now exist: fused US gate, CCA head, relevance head. Assembly =
`fused US gate → {CCA head, relevance head} → final fusion head`. Open design
questions to think through: how the fusion head is trained (the 466 ICA anchors
are the only joint CCA∩relevance∩US labels — thin); whether to share the frozen
encoder across heads (currently each is a standalone frozen-DAPT + single head);
and whether to assemble with the binary relevance head or wait for the typed
heads (#4). This is the deliberate pause point.

## Commits this session (on `cca-doca-retrain`)

`cefe705` pass-1 relevance pipeline · `f037db3` nnPNU loss · `589b7b2` nnPNU
wiring + η-sweep (negative result) · `ee78f98` us_location fused gate ·
`02e8242` fused gate threaded into training + retrain.
