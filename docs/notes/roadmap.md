# Roadmap & index — current live next-steps

*Created 2026-06-19, last touched 2026-08-12. THIS is the single live list of
what's next and what's deferred. The pre-Aug-6 refine-and-apply arc (§A) is
CLOSED at the 2026-07-30 write-up freeze; the **active thread is §A1, the
post-meeting model arc**, reordered 2026-08-11 by the branched-encoder decision
(`branched-encoder-strategy.md`): the experiment ladder replaces "productionize
the mixed stack, then joint" as items 1–2. The 2026-08-06 meeting produced no
project feedback (consumed by an R&R on the earlier pre-ML article) — internal
priorities stand. Updated 2026-08-12: ladder stage 1 executed and passed; the
**metal-execution finding** (`metal-execution-findings.md`) corrected the July
tuned-head numbers; §A1 item 3 reconciled with the post-07-30 ops commits
(d0c0898…acabb3b) this doc had missed. Previously reconciled 2026-07-30 with
the encoder-unfreeze arc outcome (`encoder-unfreeze-strategy.md`,
`tuned-retrain-runbook.md`) and the July memo
(`ml_memo/ica_model_update_2026-07.md`). Root `CLAUDE.md` reconciled
2026-08-12; `README.md` still predates `cca-doca-retrain`.*

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
| **Branched-encoder decision + experiment ladder (live)** | `branched-encoder-strategy.md` |
| **Metal-execution finding + deployment rules (2026-08-12)** | `metal-execution-findings.md` |
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
API 1960–1995 and LDC 1996–2007); **deployed model unchanged this arc.** The
pre-Aug-6 arc closed 2026-07-30: US-head retrain executed → no swap; encoder
unfreeze (rel-first sequential) executed → **rel wins big, CCA/US negative
transfer caught by the pre-registered check; a validated but un-deployed
mixed-stack candidate lifts composed ICA 0.80→0.82 and diaspora recall@0.10
0.221→0.250** (`encoder-unfreeze-strategy.md`). Corpus expanded to 1960–2025.
Memo sent 2026-08-01 (`ml_memo/ica_model_update_2026-07.md`); the Aug 6 meeting
did not reach this project (R&R on the earlier article), so no external
re-prioritization. **Next: the branched-encoder experiment ladder
(`branched-encoder-strategy.md` — graft test → head-capacity control → rel
depth sweep → joint CCA+rel), with `fit_fusion.py` parameterization first;
then productionize the winner; then apply to the expanded corpus** (§A1).

## A. Pre-Aug-6 refine-and-apply arc — CLOSED 2026-07-30 (kept for the record)

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
   - **Full-span audit (2026-07-28) + the provenance caveat.** Extended the
     audit backward: `lead_paragraph` is ~100% EMPTY for 1960–63, 1965–69,
     1980 (1964 is an island of coverage), healthy 1970–79 and 1981–2024 —
     i.e. the training corpus has three lead-free eras the models silently
     trained through (headline-only text). Pre-1960 skeleton decades:
     lead ~0% everywhere; abstract 48–80% (rising to ~80% by the 1930s).
     **Provenance caveat (operator, 2026-07-28): modern abstracts are
     contemporaneous editorial text (2025 coalesce is sound); HISTORICAL
     Archive-API abstracts are NYT-Index-register entries** — verified by
     sampling: telegraphic, abbreviated ("Apptd Consul Gen in Miami", "MTA
     repts…(S)") vs full-prose ledes. Coalescing historical rows is a real
     channel change, to be adopted deliberately or not at all — NOT bundled
     silently with any re-embed (esp. not the encoder-tune re-embed; two
     simultaneous channel changes destroy attribution).
     **Pre-registered validation experiment** (before any historical
     coalesce): on 1970s rows where BOTH exist, score identical articles
     under headline+lead vs headline+abstract channels with the current
     heads; compare score distributions + gold-slice metrics. If the
     Index register materially shifts scores, historical expansion needs
     either register-normalization, abstract-inclusive training, or the
     temporal-signal work — a strategy call for the backward arc.
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

### Arc outcome (2026-07-30, at the write-up freeze)

Items 2–3 done with findings (US retrain: no-swap; encoder unfreeze: rel-first
executed — rel wins big, CCA/US negative transfer caught by the pre-registered
check, mixed stack lifts composed ICA 0.80→0.82 and diaspora recall 0.38→0.66;
full record in `encoder-unfreeze-strategy.md` execution findings). Item 4
(cluster tuning) subsumed into the unfreeze runs. Item 5 (apply to expanded
corpus) and mixed-stack productionization + joint CCA+rel escalation move to
the post-meeting arc. Item 6: write-up drafted
(`ml_memo/ica_model_update_2026-07.md`). Stretch items (VAT, temporal) were
not promoted — the collapse investigation consumed the stretch window and was
worth more.

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

## A1. Active thread — post-meeting model arc (priority order)

The engineering follow-through from the §A arc outcome. In priority order; each
1–2 lines, pointers to the runbook/notes for detail. (§A2 below is the broader
research agenda from the 2026-07-10 meeting; this list is the immediate model
work.)

1. **Parameterize `src/fit_fusion.py`** — needed by every downstream path
   (branched, joint, any retrain). It is fully hardcoded — cca/rel/us weights +
   cache; `output_dir` silently defaults to the *production* fusion path; exact
   call sites in `tuned-retrain-runbook.md` §"Step 6" / "Gaps".
2. **Branched-encoder experiment ladder** (`branched-encoder-strategy.md` —
   the 2026-08-11 decision record; supersedes "mixed-stack productionization
   then joint" as separate items). **Stage 1 (graft test) PASSED 2026-08-12**:
   graft == full-tuned within 4e-4 on all metrics; the mixed stack IS a K=1
   branched model at ~1.08× apply (`graft_test_v2.json`). En route, the
   **metal-execution finding** (`metal-execution-findings.md`): the July
   tuned-head artifacts are execution-bound (trained on tensorflow-metal);
   correct-math retrains confirm the rel gain (vs-ICA 0.852–0.855, diaspora
   0.662) and correct the transfer verdict magnitudes (CCA 0.928→0.795, US
   F1 0.97→0.951 — direction unchanged, branched still evidence-favored).
   **Stage 2 (head-capacity control) DONE 2026-08-12** — expected outcome:
   depth on frozen features buys nothing (deep 0.779 vs shallow-control 0.788
   vs-ICA); representation bottleneck affirmed; single-run noise measured at
   ~±0.015 vs-ICA (stage-3 rule: treat smaller single-run deltas as ties).
   **Stage 3 (depth sweep) DONE 2026-08-13/14 — verdict: N=1 for solo
   tuning** (N=2 buys nothing, N=3 unstable; flat is the scheme — the
   "graded wins at N=1" read was retracted 2026-08-18, graded≡flat at N=1
   so those four cells are same-config replicates pinning text-mode noise
   at ~±0.01/draw; hard-freeze training ~−0.01 and moot — deploy rule is
   multiplier-freeze-train + graft-at-deploy; the July artifact stays the
   deployable layer-12; the July-gap (~+3σ) is parked with excluded factors
   recorded — strategy note execution record). **Next: (4) joint CCA+rel,
   design revised 2026-08-18**: N ∈ {1,2} × λ 3-point × 2 seeds, flat;
   selection on the **composed proxy** (calibrated CCA·rel product ROC vs
   ica_event on gold), not head-solo vs-ICA; three-sided success rule incl.
   US surviving as a passenger; build includes the `fit_fusion.py`
   parameterization (item 1). Detail: strategy note execution record. Joint wins ⇒ single-encoder swap; otherwise
   branched is the production architecture. Then productionize the winner:
   fusion refit (re-measures the composed mixed-stack number, currently
   metal-pending) + per-head-features `IcaModel` support + CPU-portable
   replacement artifacts w/ calibrations + swap decision. All ladder runs
   CPU-forced or cluster-side; new heads get the CPU-vs-GPU rank-consistency
   acceptance check. US gets no tuned branch; VAT unbundled (post-ladder
   A/B); temporal evidence-gated — per the note's companion decisions.
3. **Apply to the expanded corpus** — post-1995 candidates through 2025.
   *Reconciled 2026-08-12 with the post-07-30 ops commits this doc had missed:*
   the coalesce `lead_fallback_column` embed knob, `--dedupe-ids` (resolves the
   1996–2025 pull-overlap duplicates + drops the 13 empty-id 2025 rows), and
   `apply_ica` `--out-name`/`--years` parameterization all **landed 2026-07-31**
   (d0c0898, 1193 tests); cluster sbatch jobs for the 1976–1995 `full`-cache
   completion and the 1996–2025 forward embed exist (492671e…acabb3b — check
   cluster state for what actually ran). **Still wait for the ladder's encoder
   decision before paying the 1996–2025 embed** (encoder-dependent; embedding
   twice is the waste); **era-slice any 2025+ eval** (abstract-register shift).
   Then re-report CCA recall vs DoCA and ICA recall vs the team's coded events
   (the potential methods piece).
4. **Paired 1970s lead-vs-abstract channel experiment** — pre-registered
   (`roadmap.md` §A item 1 detail); run BEFORE any *historical* coalesce, since
   historical abstracts are NYT-Index register not article text. Score identical
   1970s articles (both fields present) under headline+lead vs headline+abstract;
   compare distributions + gold-slice metrics.
5. **Backward densify pull** — operator-side, ongoing (`r/api_ingest/pull_archive.R`
   `--skeleton` → densify).
6. **Deep metal/numerics investigation** — operator-requested 2026-08-11
   ("public and private good"). Minimal reproduction of the metal-vs-CPU
   divergence (single Dense at ~100-logit scale; accumulation-order vs kernel
   bug), whether metal *training* diverges from CPU training same-seed, then
   file upstream. Scope + starting points: `metal-execution-findings.md`
   §"Queued". Related: the US-head full-model load failure
   (`diag_us_head_load.py` thread) that currently blocks local token-mode US
   gold evals.
7. **Small gaps:**
   - `src/run_relevance.py` has no `--us-weights` rescore knob (unlike
     `run_cca_doca.py`) — its US-restriction is whatever `us_logit` the cache was
     built with (`tuned-retrain-runbook.md` §"Step 3" / "Gaps" #1).
   - **991 duplicate `id`s** in `api_corpus/` (1960–2025) — the embed path now
     has `--dedupe-ids` (d0c0898); still outstanding: dedupe inside
     `load_location_signals` (`unique(subset="id", keep="first",
     maintain_order=True)` + logged count) — one dup id
     (`nyt://article/2f64313c-…`) fans out the training-table join and trips
     the id-uniqueness assert; the 2026-08-12 scratch retrains monkeypatched it
     (`metal-execution-findings.md` §"Incidental").
   - **Calibration execution-path rule** (2026-08-12): Platt fits do not
     transfer CPU↔GPU for large-logit heads; fit on the apply-time path and
     record the path in the sidecar when the calibrators are next touched
     (`metal-execution-findings.md` deployment rule 3).
   - **CPU-force / acceptance-check the features-mode trainers** when next
     touched (`metal-execution-findings.md` deployment rules 1–2).
   - **`eval_heads_own_terms` step-order:** calibration is required BEFORE the eval;
     the runbook's Step 4 (eval) currently precedes Step 5 (calibrate) — swap them
     or note the dependency when next running the tuned compare.
   - γ-on-positive-risk-only FLPU experiment — pinned (`pinned-questions.md`).
   - rel DEDPUL robustness re-run (kde_mode / seed sweep) — π̂=0.02 was a single
     estimate; §B item 6.

## A2. Post-meeting arc (from the 2026-07-10 meeting)

*The 2026-08-06 meeting did not reach these (consumed by the R&R on the earlier
pre-ML article) — still pending team discussion.*

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
- **README.md reconciliation** — still predates `cca-doca-retrain`. (Root
  `CLAUDE.md` reconciled 2026-06-26, again 2026-07-30 for the encoder-unfreeze arc.)

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
- **VAT/ALUM + temporal signal** — dispositioned 2026-08-11
  (`branched-encoder-strategy.md` companion decisions): VAT is a controlled
  A/B on top of the ladder's stage-4 baseline, never bundled; temporal is
  gated on era-sliced eval evidence from the expanded-corpus apply (it forces
  a DAPT re-run — no measurement yet shows era shift hurts).
- **Substantive deferred questions** (nnPU+α+γ composition, multi-class
  heads, preprocessor train/predict split) — `pinned-questions.md`.

## History (landed major arcs)

- **Encoder-unfreeze / tuned-cache arc (2026-07-23→30)** — US-head v1 retrain
  (no swap), rel-first sequential encoder unfreeze (rel wins, CCA/US negative
  transfer, mixed-stack +0.023 composed ICA), corpus expansion to 1960–2025,
  the PU-collapse methods finding. Detail: `encoder-unfreeze-strategy.md`,
  `tuned-retrain-runbook.md`, `us-head-retrain-plan.md` addendum,
  `ml_memo/ica_model_update_2026-07.md`; artifact delta in
  `project-state-and-data-map.md` §"Delta 2026-07-30".
- **Multi-head ICA assembly (2026-06-26)** — design
  `docs/design-plans/2026-06-19-multi-head-ica-assembly.md`, all 6 phases landed:
  harmonized CCA+rel retrain, three Platt calibrators, empirically-chosen fusion
  (product-AND vs ≤3-param LR, 1-SE rule), assembled `IcaModel`, apply →
  `cca_doca/ica_candidates/`. Detail in `project-state-and-data-map.md`.
- **CCA/DoCA retrain, US filter retrain + calibration, relevance head, fused US
  gate (2026-06)** — see the per-arc docs in the index.
- **Tiers 1–5 audit/refactor + diagnostics (2026-04/05)** —
  `tiers-and-checkpoints.md`.
