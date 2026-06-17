# Calibration notes

*Last updated: 2026-06-16. Reasoning groundwork for the US-filter's Platt calibration seam — the project's first calibration work. Written dialogically (Phase 8 of the us-filter plan) to support calibration decisions now, to be picked up cold in six months, and to feed an eventual methods section by way of the author rather than by direct extraction. Companion to the code in `src/calibration/` and the threshold recipe in `docs/notes/us-filter-threshold-recipe.md`.*

The empirical section (fitted `A`, `B`, ECE before/after, reliability bins) is a **marked slot** at the bottom — the full training + calibration run hasn't happened yet, so the numbers don't exist. The concepts below are what Phase 8 deferred; the numbers get their own short addendum when the run lands.

## Concept 1 — Platt scaling: what `A` and `B` actually do

The probe emits a logit `z`. Applying `σ(z)` — the sigmoid, `1/(1 + e^{-z})`, which maps the unbounded logit to a `(0,1)` probability — implicitly claims that `z` is *already* a calibrated probability in disguise. That claim is usually wrong. Platt scaling inserts a two-parameter repair before the squash:

```
p_calibrated = σ(A·z + B)
```

The two knobs do different jobs, and keeping them separate is most of the intuition:

- **`A` is the slope — it sets the steepness of the logit→probability transfer.** It answers "how much is one unit of logit worth?" Neural nets trained with cross-entropy are systematically overconfident (Guo et al. 2017): their logits spread wider than the accuracy they deliver justifies. The fitted `A` comes out below 1 and softens that spread. An underconfident model would get `A > 1`.
- **`B` is the intercept — it sets the location of the 0.5 crossover.** It slides the whole curve along the logit axis, moving the logit value that maps to "coin flip." Equivalently, it repairs the base rate the model has internalized.

### Why Platt over temperature scaling

Temperature scaling is `σ(z/T)` — which is exactly Platt with `A = 1/T` and `B` pinned to zero. Pinning `B = 0` is an *assumption that the model's implied base rate is already correct*. This project can't make that assumption: the labeled data runs 77/23 positive, and the deployment corpus (the dateline-less API archive) has a prior we don't know. Platt keeps the intercept free precisely so something can absorb base-rate skew. Temperature scaling throws away the one knob we most need.

### Why not isotonic regression

Isotonic regression fits an arbitrary monotonic map instead of an affine one, so it can correct non-affine miscalibration that Platt can only approximate. The cost is data: it needs roughly 1000+ examples per class to fit stably, and it's prone to overfitting and step-function artifacts under class imbalance. With a calibration split drawn from a 77/23 distribution, the minority class is thin enough that isotonic's flexibility turns into variance. Platt's two parameters are the right amount of structure for the data we have.

### What `B` corrects here — and why a large `B` is a signal

In this project the base-rate component of `B` should be near zero, and the reason is structural: BCE training on the natural-balance (77/23) stream drives the model to internalize that base rate, and the calibration split shares it. There's no train→val prior shift for `B` to absorb. (This is the quiet payoff of fitting the calibrator on natural-balance val and never on rebalanced training batches — see the fit-population rule below. Had we fit on rebalanced 50/50 batches, the model would have learned a 50% base rate and `B` would need a large correction to drag predictions back to the real prior.)

`B` won't necessarily land at exactly zero, though. With base rates matched, what's left for `B` to pick up is *residual asymmetric miscalibration*: nothing in BCE training guarantees the empirical 50% point sits exactly at `z = 0`. When the residual misfit is symmetric around `z = 0`, `A` handles it alone and `B → 0`; when the misfit is lopsided — the logit distribution sharp on one side of the boundary and diffuse on the other — a single affine map can't fix it, and `B` slides to give the best affine compromise.

That makes `B` a diagnostic, not just a parameter. Since we expect it small, a `B` that comes out large after the run is a signal to investigate, with at least three candidate causes:

1. **Label error** breaking the assumed base rate (the labeled split isn't actually 77/23).
2. **A systematically biased probe** (training or data issue pushing logits high or low).
3. **Strongly asymmetric, non-affine miscalibration** that the affine map can only partly absorb.

Record `B` and look at it; don't just file it.

### The base rate that actually threatens us: val vs deployment

The reassurance above — `B` idle because train and val share a base rate — is real *for the validation split* and silent about deployment. `B` is fit to correct miscalibration relative to the 77/23 **labeled** distribution. We then apply the calibrated model to the dateline-less API corpus, whose true US/not-US prior we don't know and can't fit against, because it's unlabeled. So the calibration is calibrated to the labeled distribution used as a *proxy* for deployment, and the gap between the two is the irreducible calibration risk.

This is why `fit_population` is a recorded field on the calibrator rather than an afterthought: the scores are calibrated *to a stated distribution*, and naming it keeps the proxy honest. It's also the bridge to the Phase-6 proxy-gap diagnostic, which measures the dateline-vs-event-location disagreement as a stand-in for how far the proxy is from the truth. Developed in full below.

## Concept 2 — what ECE measures, and what it can't see

Expected Calibration Error bins predictions by confidence (`calibration_report` uses 15 equal-width bins over `[0,1]`), and in each bin compares two numbers: the mean predicted confidence and the observed accuracy. ECE is the population-weighted average of `|accuracy − confidence|` across bins. Zero means that when the model says 0.8, it's right 80% of the time — bin by bin.

The sharp limit shows up in one example. Take a degenerate model that ignores the article and outputs 0.77 (the base rate) for everything. Every prediction lands in one bin; the accuracy in that bin is 0.77, because 77% of articles are positive; the confidence is 0.77. The gap is zero, so the ECE is zero. **A model that learned nothing but the base rate is perfectly calibrated.** ECE certifies that probabilities are numerically honest in aggregate; it says nothing about whether the model can tell two articles apart.

The property the constant predictor lacks has a name: **resolution** (or sharpness) — the ability to push predictions away from the base rate toward 0 and 1, separating the classes. ECE is blind to resolution by construction, because it only ever compares confidence to accuracy *within* a bin and never asks whether the bins differ from each other.

### Why Brier sits alongside ECE

Precision/recall measure discrimination; ECE measures calibration. Those are the two orthogonal axes, so "a PR curve I'm happy with plus an ECE I'm happy with" does in principle pin down both — the argument for ECE-and-PR-are-enough isn't wrong, it just leaves attribution on the table. Brier score is what recovers it. Binary Brier is exactly mean squared error, `(1/N)·Σ(p_i − y_i)²`, and it decomposes (Murphy 1973) into three terms:

```
Brier  =  Reliability  −  Resolution  +  Uncertainty
```

- **Reliability** — the weighted mean *squared* gap between confidence and accuracy per bin. ECE's twin: ECE is the L1 version (`|conf − acc|`), reliability the L2. Lower is better. This is the part Platt scaling can fix.
- **Resolution** — how far the bin accuracies sit from the overall base rate, i.e. how much the model actually separates classes. Higher is better, and it's *subtracted*. Platt can't touch it — calibration is monotonic and never changes the ranking.
- **Uncertainty** — `base_rate × (1 − base_rate)`, here ≈ 0.77 × 0.23 ≈ 0.177. Irreducible; a property of the data, not the model.

Run the constant predictor through it: reliability 0, resolution 0, uncertainty 0.177, so Brier = 0.177. ECE called it perfect; Brier pinned it at the uncertainty floor and refused to go lower. The resolution term is exactly what ECE couldn't see.

So Brier is a single proper scoring rule that can't be gamed by either failure mode — the constant predictor is pinned at the floor by missing resolution, an overconfident-but-discriminating model is punished by the reliability term — and its decomposition gives *attribution*. Faced with a disappointing Brier, the decomposition says whether the loss is miscalibration (Platt's job, fixable here) or a resolution deficit (not Platt's job — you need a better probe). ECE and PR each flag that something's off without telling you whether calibration is the lever.

### How much this matters for the current consumer

The immediate consumer — the DoCA-recall threshold recipe — barely needs calibration at all. It picks the largest threshold meeting a recall target, and because Platt is monotonic, the achievable precision/recall operating points are identical whether you threshold on raw logits or calibrated probabilities. Calibration buys the recipe an interpretable threshold (0.5 means 50%) and a threshold that's comparable across retrains, but not new capability.

Calibration earns its keep downstream: when the US score is combined probabilistically with the CCA / immigration / ICA head scores in the multi-head future, "0.8" has to actually mean 80% for the combination to be valid. That's the real answer to "why calibrate at all," and it's worth being explicit that the current consumer doesn't force it — so calibration quality is an investment in the multi-head future, not a blocker for the threshold-recipe present.

## Concept 3 — the distribution interaction (load-bearing)

Calibrated probabilities are calibrated *to a distribution*. The val split's base rate (77/23) matches training, so `B` is idle and the calibration looks clean — but that cleanliness is a statement about the validation distribution, and it's silent about the dateline-less API corpus we actually deploy on, whose true US rate we don't know and can't measure, because it's unlabeled. That's not a footnote; it's the central limit on what this calibration can promise.

### Why a base-rate gap shows up as a constant logit offset

Any posterior logit splits into two pieces:

```
logit P(y=US | x)  =  logit P(y=US)        +  log [ p(x | US) / p(x | foreign) ]
                      └── prior log-odds ──┘    └──── log-likelihood-ratio ────┘
                          (the base rate)        (what the features discriminate)
```

The second term is feature evidence — it does the discriminating. The first is the base rate in logit units. Training on 77/23 bakes in `logit(0.77) = log(0.77/0.23) ≈ +1.21` as the prior term. If the true deployment rate were, say, 50/50, the correct prior term is `logit(0.50) = 0`, so every logit comes out `+1.21` too high — a constant offset, identical for every input.

For the frozen-probe architecture this is concrete: the probe is a frozen backbone plus one dense output neuron, and that neuron's **bias weight is the prior log-odds**. `B` recalibrates exactly that one number.

### Three consequences

1. **Scores skew too high (when deployment is less positive than val).** The baked-in prior exceeds the deployment prior and adds to every score.
2. **It's a pure-`B` problem.** A constant offset in logit space is exactly what `B` removes; the wanted correction is `B_deploy − B_val = logit(prior_deploy) − logit(prior_val)`. `A` is irrelevant to it. And `B`'s idleness — reassuring on val — is now precisely the defect: we'd want `B` pulling the base rate down, and it isn't, because we fit it where there was nothing to pull.
3. **Perfect classification can coexist with broken calibration.** A constant added to every logit preserves their order, so ranking — and therefore discrimination and any threshold-based classification — is untouched, while the probability *values* all shift. This is the cleanest demonstration of the Concept-2 point that discrimination and calibration are orthogonal: a transformation that wrecks one and leaves the other intact.

### The boundary: label shift vs covariate shift

The clean "`B` could fix it" story assumes **label shift** — only `P(y)` changes between val and deployment; the class-conditional feature distributions `p(x | US)` and `p(x | foreign)` stay fixed. Then the gap is a pure constant offset and affine calibration can address it in principle. Our shift is also temporal (LDC training era 1987–2007 → API deployment 1960–1995), and what a foreign or US article looks like plausibly changed across decades — covariate shift *within* class. That component is not a constant logit offset, so no single `(A, B)` repairs it even with a perfect deployment sample. "`B` would fix it if we could fit against deployment" is the optimistic case; the real gap may carry a component no affine calibration touches.

### What to do about it — the decision logic

`fit_population` exists to keep this honest: the calibrator records the distribution its scores are calibrated *to*, so the proxy is never silent. The operational sequence, and the reason the gold set matters:

1. **Default:** fit Platt on natural-balance val, record `fit_population`, apply to the API corpus. This is what ships absent evidence the gap is harmful.
2. **Measure the gap with the pre-1986 gold set.** Once that hand-coded sample exists it does four jobs: it reveals the true deployment base rate (the hypothetical `−1.21` becomes a measured number); it supports real ECE/Brier on deployment rather than on the val proxy; it can itself become a recalibration set; and the Phase-6 `proxy_gap` diagnostic on it measures the related question of whether the dateline label still tracks true event location in the older era.
3. **If the gap is harmful, respond in ranked order:** refit `B` on the hand-labeled deployment sample (cheapest — addresses the label-shift component directly); escalate the frozen probe to fine-tuning via the Phase-6 escalation machinery (addresses what affine calibration can't — the covariate-shift component); or accept the gap and document the bound (when the sample is too small or the deployment distribution too unstable to fit against reliably).

The through-line: the calibration is honest but bounded. It's correct for the distribution it names, the gold set is the instrument that measures how far that distribution sits from deployment, and the escalation ladder is what we do once we know.

## Where this lives in the code

The concepts above map onto `src/calibration/` (FCIS-split: pure math in the core, I/O in the shell):

- **`calibrator.py`** (Functional Core) — `platt_fit(logits, labels) -> (A, B)` is the 1-D logistic regression; `platt_transform(logits, A, B)` applies `σ(A·z + B)`. `PlattCalibrator` is a frozen dataclass carrying `A`, `B`, `fit_population`, `n`, `method`. Its `fit(logits, labels, *, fit_population)` classmethod takes `fit_population` as a **keyword-only required argument** — there's no way to fit a calibrator without naming the distribution it's calibrated to, which is the Concept-3 honesty rule enforced at the type level.
- **`report.py`** (Functional Core) — `calibration_report(probs, labels, n_bins=15) -> {"ece", "brier", "reliability"}`. ECE and Brier per the definitions above; `reliability` is the per-bin list backing a reliability diagram. Binning is 15 equal-width bins with `p == 1.0` clipped into the last bin.
- **`sidecar.py`** (Imperative Shell) — `calibration_path_for_weights`, `save_calibration`, `load_calibration` persist the calibrator to `*.calibration.json`. This is the third leg of the artifact triple (`*.weights.h5` + `*.config.json` + `*.calibration.json`); `src/validation/artifact_check.py:reload_and_score` proves the triple alone reproduces scores cross-process.

The fit-population rule has a code-level consequence worth stating once: **fit the calibrator on the natural-balance validation split, never on rebalanced training batches.** Rebalanced batches would teach the calibrator a 50/50 prior and force `B` into a large correction back toward the real rate — defeating the point. The natural-balance fit is what keeps `B` near zero on val (Concept 1) and makes a large `B` a meaningful diagnostic signal rather than an expected artifact.

## Empirical results — MARKED SLOT (pending the full run)

*Not yet populated. The full training + calibration run hasn't happened (it's operator-gated; see `docs/test-plans/2026-06-06-us-filter.md`). When it lands, record here:*

- *Fitted `A` and `B` on the natural-balance val split, with `fit_population`. Flag whether `B` is near zero (expected) or large (the Concept-1 diagnostic — investigate label error / probe bias / asymmetric miscalibration).*
- *ECE and Brier before vs after Platt, on the val split — confirm post-calibration ECE ≤ pre-calibration ECE (AC4.5).*
- *Reliability diagram bins, or a pointer to the saved figure.*
- *Once the gold set exists: the measured deployment base rate, ECE/Brier on the gold set (real deployment calibration, not the val proxy), the `proxy_gap` figure, and — if a recalibration or escalation was triggered — the decision and its rationale per the Concept-3 ladder.*
