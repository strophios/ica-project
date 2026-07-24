# US head retrain — diagnostic findings and design

*Created 2026-06-26. Branch `cca-doca-retrain`. A deferred follow-up to the
multi-head ICA assembly (Phases 1–6). The assembly ships on the current US head
with gold-first gating; this doc records why a retrain is worth doing and how to
do it.*

## Why this exists

The US head (`us_classifier_full`) is a hard pre-filter: the assembled `IcaModel`
gates on it before CCA/relevance/ICA scoring. So the US head's recall on genuine
US events is an upper bound on the whole system's ICA recall — anything it drops
is never recovered downstream. During the Phase 4 fusion fit we found that ceiling
is both real and badly placed.

## The finding: the US head misses diaspora collective action

On the hand-coded ICA eval, the US gate at τ_us=0.3 dropped 57 of 214 ICA
positives (27%). Chasing why, I scored the 139 held-out anchors (all DoCA-matched
US events, so all genuine US positives) with `apply_us_model`. 26 of them (19%)
score `p_us < 0.2`. They are not random.

I ruled out the three obvious causes:

- **Pre-1987 out-of-distribution.** No. The low-scoring anchors span 1960–1994,
  and pre-1987 anchors score *higher* on average than post-1987 ones (mean p_us
  0.839 vs 0.737). The failure is a tail, not a distribution shift.
- **Dateline / text-channel mismatch.** No. The head trains on dateline-stripped
  text, but only 12 of 139 anchor leads carry a dateline at all — there's nothing
  to strip on 91% of them, so feeding raw text can't be the main driver.
- **Text sparsity (empty leads).** No. Empty-lead fraction is 27% in the
  low-p_us group vs 29% overall — no difference.

What the low-p_us anchors actually share is content. Examples (year, p_us):

- "Visiting Greek Aide Stirs Civil Rights Protest" — 75 demonstrators incl.
  Melina Mercouri (1973, 0.003)
- "Haiti Exiles Delay Soccer Here" — Haitian exiles marching in downtown NYC
  (1973, 0.017)
- "Dominican Pickets at U.N." (1965, 0.019)
- "250 Irish Here Greet Lynch With Boos" (1973, 0.029)
- "Crackdown in Beijing; For Students in U.S." — Chinese students reacting
  (1989, 0.010)
- "Rights Rally in City" for a Soviet dissident (1977, 0.034)

These are **diaspora and solidarity protests**: collective action on US soil
("Here", "in City", "at U.N.") about foreign matters (Greece, Haiti, the
Dominican Republic, Ireland, China, the USSR). They are unambiguously US events.
They are also, by definition, immigrant collective action — the single
highest-value ICA category the project exists to surface. The head keys on
topic/location words, and the foreign-topic content swamps the weak US-location
signal, so it scores them as not-US.

The fused gate doesn't rescue them. `us = us_ml & ~(any_not_us & ~any_us)` can
only make the ML gate *more* restrictive (drop clearly-foreign from ML-positives);
it cannot recover an ML-negative. So the recall ceiling lands exactly on diaspora
ICA, and the gate's location logic can't lift it.

### Consequence for the current assembly

This is why the assembly uses **gold-first gating**: where a row has an
authoritative US label (a DoCA match, or an LDC dateline `us_label`), trust it
over the ML head. That rescues known positives like the anchors so the fusion fit
isn't biased by gated-out positives. The unfixed part is **novel** diaspora ICA at
apply time — events with no DoCA/dateline label, governed by the ML head, where
the ceiling still applies. That is the documented limitation until this retrain
lands.

## The retrain design

The label source is the root cause. The current head learns from datelines, which
encode where a story was *filed*, not where the *event* happened — a protest in
NYC about Greece may be filed from New York yet read as foreign. DoCA labels
encode event location, which is what we actually want.

### Label sources

- **Reliable positives:** DoCA-matched articles (US collective action by
  construction, and rich in exactly the diaspora cases the current head misses) +
  high-confidence section signal (National/`Nation` desk).
- **Reliable negatives:** clear foreign datelines + `World` desk.
- **Unlabeled:** everything else.

Section labels are noisier than datelines — a `World` story can be about the US
and a `National` story about foreign policy — so treat only the high-confidence
ones as reliable and leave the rest unlabeled rather than forcing a clean P/N
split.

### Loss: lean nnPNU

The project already has nnPNU machinery (the relevance head uses it). It fits here
better than the alternatives: unlike the ICA needle-in-haystack, US/not-US has a
milder balance and genuine reliable negatives (foreign/`World`), so the PN term
gets real signal that pure PU would throw away — while the unlabeled stream still
absorbs the section-label noise.

### Strip datelines or not

Lean toward **not stripping**, but only if the eval is guarded. Keeping datelines
gives the head a real US signal for the 27–40% of articles that have one. The
original reason to strip was leakage: dateline-derived labels + datelines-in-text
let the model memorize the dateline, which is circular and useless for the
no-dateline majority. Diversifying the label source (DoCA + sections, not just
datelines) dilutes that circularity but doesn't kill it.

The guard that makes unstripped training safe: **evaluate on labels that don't
come from datelines** (DoCA matches, hand-coding, or section), and include a
no-dateline slice in the eval. Then a high score reflects genuine generalization,
not dateline-reading. Without that guard an unstripped head looks great on
dateline-bearing validation and fails on the no-dateline majority at apply. Worth
training both (strip vs no-strip) and comparing on that dateline-independent eval.

### Leakage management

The ICA eval anchors are DoCA articles. If DoCA becomes US-head training
positives, the eval anchors must be held out of the US-head retrain too. The
holdout then spans four consumers: US head, CCA head, relevance head, and the
clean ICA eval. This is the main place to get it wrong — extend
`assert_holdout_excluded` coverage to the US-head retrain path.

## Risks

- **Section noise** lowers precision if "reliable" is drawn too generously. nnPNU
  (only high-confidence rows as reliable P/N) is the mitigation.
- **Memorization** if unstripped and the eval isn't dateline-independent. Covered
  by the eval-design guard above.
- **Scope.** A new head ripples through the whole downstream: recalibrate →
  re-pick τ_us → re-fit fusion → re-embed if the apply text channel changes. It's
  a sub-project, not a tweak.
- **It might just be worse.** Section labels and unstripped text could degrade the
  bulk performance the current head already gets right (p_us separates `us_event`
  cleanly: 0.835 vs 0.206). Hence validate-before-swap below.

## Sequencing: validate before swap

Do this as a follow-up to the assembly, not a precondition (operator decision,
2026-06-26). The current head's failure mode is documented and gold-first contains
it for the eval and known positives. The risk that the retrain "winds up worse" is
exactly the case where the critical path shouldn't depend on the new artifact
until it's proven.

Concrete next steps when picking this up:

1. Assemble the training corpus: DoCA matches + section-derived reliable P/N +
   unlabeled rest, with the ICA-eval anchors held out.
2. Train two heads (strip vs no-strip), nnPNU.
3. Evaluate both on a dateline-independent eval with an explicit no-dateline
   slice; measure diaspora recall specifically (the metric that motivated this).
4. If a variant clearly beats the current head, swap it in and re-run calibration
   → τ_us → fusion. Phase 4 scripts all three, so the swap is a re-run, not a
   rebuild.
5. Update `project-state-and-data-map.md` and this doc with results.

---

## v1 OUTCOME (2026-07-24): retrain executed — decision: NO SWAP

The v1 retrain (stripped channel, nnPNU, dateline-only reliable negatives) was
built, trained, and evaluated per the design above (corpus commit `b66da55`,
training/eval commit `b49da45`; artifacts `us_filter/us_pnu.weights.h5` +
sidecar, `cca_doca/experiments/{us_pnu_eta_grid,eval_us_retrain}.json`).
Validate-before-swap concluded **keep the current head**. This section records
why — the numbers reframe the problem this doc was written to solve.

### What the retrain got right

- **η=1.0 selected** (val PR-AUC 0.9968). **Pure nnPU (η=0) collapses
  structurally at this prior** (π̂=0.8253, estimated from the current head's
  calibrated scores over the unlabeled sample): the nnPU negative-risk clip
  fires on essentially every batch and the loss minimizes by predicting
  everything positive. The reliable-negative PN term is load-bearing — the
  design's nnPNU choice is validated a posteriori. Prior sensitivity (±0.1):
  flat; η is the first-order lever, not π.
- At mid-curve thresholds the candidate recovers 17/26 diaspora anchors
  (logit>0) and has much higher dateline-negative specificity (0.973 vs 0.883).

### Why no swap: the gate regime is what matters, and there the current head wins

The US head is deployed as a lenient recall-tuned **gate** (τ_us=0.02 calibrated,
~94% pass on the hard eval set, ~97% on the API corpus), not a mid-curve
classifier. Recall-matched comparison on gold `us_event` (n=1,129):

| matched US recall | head | foreign rejection | diaspora recovered |
|---|---|---|---|
| 0.98 | current | **0.312** | **21/26** |
| 0.98 | v1 candidate | 0.260 | 19/26 |
| 0.99 | current | **0.167** | **24/26** |
| 0.99 | v1 candidate | 0.135 | 23/26 |

A rank-average ensemble of the two heads also fails to beat the current head
in this regime (0.292 rejection @0.98). The candidate's wins live at operating
points the gate never uses; swapping would trade away bulk dateline recall
(0.98 → 0.74 at logit>0) for nothing the gate can cash in.

### The reframed ceiling

The headline "drops 27% of ICA positives" figure was measured at τ_us=0.3. At
the **deployed** τ_us=0.02 the gate passes 21/26 diaspora anchors — the
recall ceiling is real but modest (~5/26 diaspora anchors ≈ 4% of anchors).
The gate's actual weakness at deployment is the **foreign leak at high recall**
(only ~27–31% of known-foreign rejected at ≥0.98 recall), which is a
*precision* problem on the API side (no gold labels there), and neither the
label-source swap nor the ensemble improves it. What plausibly would: richer
location signals (gate options B/C in `roadmap.md`), the feature-fusion US
retrain (option B), or encoder unfreezing — i.e., better separation, not
relabeling alone.

### Kept artifacts + incidental findings

- The candidate weights/sidecar and both experiment JSONs are kept for
  comparison; nothing was wired into `IcaModel`.
- The PNU corpus + both-id-space holdout machinery (`build_us_pnu_table.py`,
  `build_holdout_ids_ldc.py`, `create_us_pnu_data`) are permanent
  infrastructure — any future US-head experiment starts from them.
- **FLPULoss shape footgun** (found writing diagnostics): calling
  `FLPULoss.call` with BOTH y_true and y_pred flattened 1-D silently yields an
  η-invariant loss; the production path always passes (batch,1) logits. Out-of-
  graph diagnostic code must reshape y_pred to (-1,1) (regression-tested in
  `tests/test_run_us_pnu.py`).
- The no-strip variant was never needed for the swap decision; the LDC 1987–94
  raw embed (if run) serves the encoder-unfreeze work instead.
