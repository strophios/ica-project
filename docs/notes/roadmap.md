# Roadmap & index — current live next-steps

*Created 2026-06-19, last touched 2026-07-23. THIS is the single live list of
what's next and what's deferred. Reconciled 2026-07-23 with the 2026-07-10 team
meeting (`docs/meetings/20260710_notes.md`) and the external task plan
(`~/tasks/projects/ica-project.md`); the active thread below is the pre-Aug-6
refine-and-apply arc. The root `CLAUDE.md`/`README.md` rewrite is in progress
this session (was deferred until a working multi-head landed — condition met).*

## Where to look (index)

| For… | See |
|---|---|
| **What's next / deferred (live)** | this doc |
| **Data layout, artifacts, model state** | `project-state-and-data-map.md` |
| **Team meeting notes (2026-07-10)** | `docs/meetings/20260710_notes.md` |
| Model-state memo for collaborators | `ml_memo/ica_model_state_2026-06.md` |
| CCA/DoCA retrain arc (detail/reasoning) | `cca-doca-handoff.md`, `cca-doca-retrain-design.md`, `cca-model-characterization.md` |
| Relevance-head arc (detail/reasoning) | `relevance-head-handoff.md` |
| US filter / dateline pipeline | `us-filter-*.md`, `r/CLAUDE.md` |
| **US head retrain (diaspora recall)** | `us-head-retrain-plan.md` |
| **ICA apply results + cluster runbook** | `ica-apply-results-and-cluster-runbook.md` |
| Per-head own-terms eval (2026-07-10) | `ml_memo/ica_model_state_2026-06.md` ("The heads on their own terms"); `scripts/eval_heads_own_terms.py` |
| Multi-head assembly design (landed) | `docs/design-plans/2026-06-19-multi-head-ica-assembly.md` |
| Tier 1–5 audit/refactor history | `tiers-and-checkpoints.md`, `tier{2,3,4,5}-design.md` |
| Deferred substantive questions | `pinned-questions.md` |
| Process / engineering patterns | `process-patterns.md`, `engineering-patterns.md` |

The per-arc docs retain narrative and reasoning; their forward-looking lists are
consolidated **here** so nothing drifts out of sight.

## Status snapshot

Multi-head `IcaModel` — assembled, calibrated, applied (candidates exist for
API 1960–1995 and LDC 1996–2007). Per-head own-terms eval done (US 0.925 /
CCA 0.927 / rel 0.829 ROC — rel is the weak head at its own job; the US head's
0.86 event-location recall is the diaspora ceiling). Team meeting 2026-07-10 set
the direction: **refine the model, generate post-1995 ICA candidates, and check
recall against DoCA + the team's ICA dataset — write-up to the team by Aug 1,
meeting Aug 6.**

## A. Active thread — pre-Aug-6 refine-and-apply arc

Milestone: **write-up in the team's hands by 2026-08-01** (meeting 08-06).
Reconciles the meeting outcomes with the external task plan. Two scope tiers
with a pre-registered promotion rule (below).

### Committed scope (in order; API pull runs in parallel throughout)

1. **API headline pull** — running (started 2026-07-24; `r/api_ingest/pull_archive.R`,
   resumable). Forward first (1996 → mid-2026), then backward (`--skeleton`
   rotating-month sampling for temporal coverage, then densify).
   - **Data-drift finding (audited 2026-07-27, resolving the 07-24 watchout):**
     per-year audit over the converted 1996–2025 parquets. `lead_paragraph` is
     healthy 1996–2024 (0.1–2% empty; one bad patch: 2005, 15.5% missing BOTH
     lead and abstract) — but **2025 is a hard cutover: 100% of rows have no
     lead_paragraph** (abstract 97% present). The hand-checked stories were
     2025 rows. Abstract is NOT simply the lead (only 8.9% identical in 2024
     when both present) but covers 91.7% of lead-empty rows.
     **POLICY: text channel for new-year embedding = `headline + "</s>" +
     coalesce(lead_paragraph, abstrct)`** — implement as an additive
     `lead_fallback_column` on the loader/embed path when the post-1995 embed
     is built. Pre-2025 this fills only the small gaps; 2025+ rows ride the
     abstract register (a mild channel shift — slice any 2025+ eval by era).
     Corpus state after assembly: `api_corpus/` parquet now spans 1960–2025
     (66 files); the 1870–1959 skeleton is raw-checkpoint-only by design.
   - **General principle (same date): expect the data to shift under us across
     the 175-year corpus.** Known instances so far: the API desk/section signal
     cutover (~1981 — `news_desk` is `"None"` and `section_name` is
     `"Archives"` before it), datelines existing only in LDC text, and the
     lead/abstract shift above. Every new era we ingest gets a schema +
     missingness + convention audit before training or applying on it.
2. **US-head retrain (diaspora recall)** — ~~scoped design in
   `us-head-retrain-plan.md`~~ **DONE 2026-07-24, decision: NO SWAP** (results
   addendum in that doc). v1 (stripped, nnPNU, dateline-only negatives) trained
   and evaluated; at the deployed gate regime (recall-matched ≥0.98) the
   current head equals or beats the candidate on both foreign rejection and
   diaspora recovery — the "27% drop" was a τ=0.3 artifact; at deployed τ=0.02
   the ceiling is ~5/26 diaspora anchors. Real deployed weakness = foreign
   leak at high recall (~27–31% rejection), a separation problem that
   relabeling alone doesn't fix (→ gate options B/C, encoder unfreeze).
   Keeper findings: nnPU collapses at π≈0.83 (PN term load-bearing); the PNU
   corpus + dual-id-space holdout machinery is permanent infrastructure.
3. **Encoder unfreeze + discriminative LR** — machinery exists (`LayerLRModel`,
   `top_n_group_fn`; the escalation branch in `run_us_classification.py:154-179`
   is the proven template). **Training-strategy decision MADE 2026-07-27**
   (`encoder-unfreeze-strategy.md`, literature-grounded): **rel-first
   sequential** — text-mode rel training w/ top-N unfreeze → re-embed →
   features-retrain US/CCA on the new cache (built-in negative-transfer check)
   → recalibrate → refit fusion. Pre-registered escalation: **joint CCA+rel**
   (same population/channel/loss family → same batches carry both labels, one
   λ; no PU-interleaving risk); three-head joint stays out. Build list (7
   items, from the 2026-07-27 readiness inventory): RunConfig escalation
   knobs; text-bearing rel population table; RELEVANCE_SET_DIR; 3-stream text
   Ratio-Batch; `run_relevance_text.py`; smoke script + sbatch variant; cluster
   time/mem budgeting (full-encoder backprop regardless of N).
   - ~~**Pre-flight correctness check:** `LayerLRModel.train_step` loss-weighting
     vs stock Keras~~ — **RESOLVED 2026-07-24**: the flatten discrepancy was
     already fixed (verified vs installed Keras 3.12 stock trainer); batch-size
     weighting is correct-by-design (partial-batch proportionality). Real drift
     found + fixed in the check: the tracked loss now mirrors stock's replica
     unscaling under tf.distribute (identity single-device, display-only).
4. **Train + tune the refined model** — hyperparameter search (LRs, unfreeze
   depth, etc.) on the cluster; the meeting endorsed doing the deeper training
   there.
5. **Apply + evaluate** — (a) generate ranked post-1995 ICA candidates (LDC
   1996–2007 + whatever API-forward data has landed); (b) re-run 1960–1995 and
   report CCA recall vs DoCA and ICA recall vs the team's coded ICA events
   specifically (this comparison is a potential methods piece).
6. **Write-up** — findings memo to the team by **08-01**.

### Stretch scope + promotion rule (pre-registered 2026-07-23)

Stretch items are attempted **only** if the committed scope is ahead of
schedule; conditions are checkable, not vibes:

- **VAT/ALUM**: start only if by EOD **07-27** (1) the US-head retrain is
  validated-and-swapped (ICA-positive gate recall improves, US own-terms
  precision does not collapse) AND (2) the encoder-unfreeze training path is
  running on the cluster.
- **Temporal signal**: start only if VAT lands by EOD **07-29** (it forces a
  DAPT re-run — the most expensive single item; also note the API-forward data
  makes train/apply eras diverge, which is the argument *for* it).
- **Write-up freeze: EOD 07-30.** The memo reports whatever model state exists
  then. Stretch work continuing past the freeze targets the post-meeting arc,
  not the memo.

## A2. Post-meeting arc (from the 2026-07-10 meeting — not before Aug 6)

- **1950s backward test** — apply the (contemporary-trained) model to 1950s
  pulls; compare candidates against the hand-coders' output and the dictionary
  search; iterate decade-by-decade backward. Part of the same candidate methods
  piece as the recall comparison above.
- **Team's hand-coded events as data** — ~800 coded events, 1870–1960 (most
  ICA, all ICA-plausible). Candidate eval anchors and eventually training
  positives for the backward expansion.
- **Definitional alignment** — the team's draft coding doc defines ICA as
  **CA + IMM + CLAIM** (three levels for CA and IMM; allies=1 only for some
  claim types; "diasporic" requires a call on the US, not merely a foreign
  gov't; labor rules). This is *close to but not identical to* the heads'
  current label semantics — before training on their coded data, map their
  scheme onto the `us`/`cca`/`rel` decomposition and decide whether CLAIM needs
  its own head or lives in `rel`.
- **Domains/claims identification** — can LLMs code domains? Compare zero-shot
  vs. an orchestrated agent workflow (accuracy and cost).
- **Active-learning efficiency study** — starting from ~1000 good examples, how
  many 100-article hand-checked batches until performance saturates, vs. the
  cost of coding a random sample up front?
- **Full-article texts** for high-ranked candidates (via the team).

## B. Relevance head — deferred (from `relevance-head-handoff.md`)

*Priority note (2026-07-10): the per-head own-terms eval shows rel is the
weakest head at its own dimension (ROC 0.829, PR-AUC 0.52 @ 20% base on the
hand-coded set) while CCA is strong on its own terms (0.927) — so the
encoder-unfreeze work in thread A attaches here first.*

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
   US head is (re)built — natural to consider inside thread A item 2.
4. **Typed multi-label relevance heads** — the η per-type result (foreign-negative
   pressure helps exclusionary/access, hurts diaspora) says the four types want
   separate sigmoid heads; also yields the typology output. Typed anchors exist
   (`event_type4`). Relevant to the CLAIM-dimension question in A2.
5. **Honest precision eval** — gold immig is thin (17 pos) + CCA-stratified; add
   IPW reweighting + a relevance-stratified gold draw. (The 2026-07-10 own-terms
   eval is unweighted on the ICA-stratified set — corpus-anchored per-head
   precision needs this IPW work; the ICA strata weren't designed per-head.)
6. **DEDPUL π for relevance** — pass-1 used π=0.05 by analogy; estimate + sweep
   (frontier-invariant, but record it).

NOT to revive without a different negative set: **nnPNU reliable negatives** — the
η-sweep was a clean negative result (foreign-news and diaspora relevance are the
same content signal). Machinery retained; η=0 canonical.

## C. Apply + project hygiene (from `cca-doca-handoff.md`)

- **Full-corpus apply** — `us_filter/api_us_scores/` still absent; the
  assembled-model candidates exist (`cca_doca/ica_candidates/`) but the
  standalone US-score product was never produced. Fold into thread A item 5 or
  drop if the assembled apply supersedes it.
- **Broaden the gold set** — only 500 of 2,553 template rows coded; street-/
  CCA-stratified. Re-draw stratified for relevance + a larger US/immig slice.
- **`run_us_classification.py` greedy-glob bug** — still reads `us_filter/**/*.parquet`
  greedily (pulls in `audit/api_ldc_matched.parquet`, no `id` → crash). Apply the
  additive `pattern=` fix (as done for `data_from_parquet`) when next touched —
  the US-head retrain (thread A item 2) is the likely next touch.
- **CLAUDE.md / README.md reconciliation** — in progress 2026-07-23.

## D. Older deferred (indexed, not duplicated)

- **Tier 5 empirical runs** (real-data short/full + cluster `mixed_float16` runs;
  π=0.03-vs-0.02 research handoff) — runbooks in
  `docs/notes/tier5-implementation-plan/phase_07.md`/`phase_08.md`, status in
  `tiers-and-checkpoints.md`. HUMAN-OPERATED. The thread-A cluster tuning runs
  will largely supersede the original intent here; close these out explicitly
  when they do.
- **US filter operator-gated items** (cluster shakedown, gold-set hand-coding →
  slice eval / DoCA recall / escalation) — `us-filter-design-handoff.md`,
  `docs/implementation-plans/2026-06-06-us-filter/phase_*.md`. Partially
  superseded by the 06-18 features-mode retrain + validation; reconcile when the
  US head is next touched.
- **VAT/ALUM + temporal signal** — now stretch items in thread A (promotion rule
  above), not free-floating deferred items.
- **Substantive deferred questions** (nnPU+α+γ composition, multi-class
  heads, preprocessor train/predict split) — `pinned-questions.md`.

## History (landed major arcs)

- **Multi-head ICA assembly (2026-06-26)** — design
  `docs/design-plans/2026-06-19-multi-head-ica-assembly.md`, all 6 phases landed:
  harmonized CCA+rel retrain, three Platt calibrators, empirically-chosen fusion
  (product-AND vs ≤3-param LR, 1-SE rule), assembled `IcaModel`, apply →
  `cca_doca/ica_candidates/`. Detail in `project-state-and-data-map.md`.
- **CCA/DoCA retrain, US filter retrain + calibration, relevance head, fused US
  gate (2026-06)** — see the per-arc docs in the index.
- **Tiers 1–5 audit/refactor + diagnostics (2026-04/05)** —
  `tiers-and-checkpoints.md`.
