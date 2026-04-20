# Pinned Questions

A running record of substantive questions that have been deliberately deferred,
not forgotten. Each entry should explain what we know, what we don't, and what
deferring means for the work we're doing in the meantime.

When picking up a pinned question for real engagement, consider whether the
deferred-thinking decisions made in the interim still hold up under deeper
analysis.

---

## 1. Composing nnPU + class balancing + focal modulation + (eventually) ALUM

**Pinned:** 2026-04-19, during Phase 1/Phase 3 audit and FLPU verification.

### The observation

Our PU classification loss currently combines three (eventually four)
mechanisms that operate at different conceptual levels and were each designed
to solve different problems:

- **nnPU's class-prior weighting (π_p):** a *mathematically load-bearing*
  ingredient. The unbiased risk decomposition that lets us substitute
  `loss(unlabeled, neg) - π_p · loss(positives, neg)` for a true negative-class
  risk only works because of the π_p multiplier. It is not optional class
  balancing — it's part of the proof.

- **Vanilla class-balance α (e.g., from focal loss):** a *practical heuristic*
  for "the common class swamps the loss." Operates at the level of per-sample
  loss contribution, with a constant per-class weight.

- **Focal modulation (1−p)^γ:** a *practical heuristic* for "easy examples
  swamp the loss." Operates per-sample, per-step, with weight depending on
  current prediction confidence.

- **(Eventually) ALUM / VAT:** a *practical heuristic* for "model is brittle
  / generalizes poorly," operating on the embedding space, orthogonal to the
  loss decomposition.

So we have **three (eventually four) different kinds of solution operating at
different levels, applied to two distinct underlying problems** (PU-ness and
class-imbalance / easy-example dominance).

### Why this matters

- Stacking these naïvely creates interactions that aren't analyzed in any
  single source paper. Ji 2023 demonstrates empirically that their combination
  (nnPU + α + γ + ALUM) works on their benchmarks but does not argue
  *principled* reasons for the combination. We may be inheriting choices that
  are domain- or dataset-specific.
- Calibration: the model's raw output probabilities will be distorted in
  hard-to-predict ways by the combination. Decision threshold selection and
  output calibration both need to take this into account.
- Interpretation during training: when training looks weird (loss not
  decreasing, oscillating, exploding) it's hard to know which mechanism to
  suspect first.
- Hyperparameter tuning: each mechanism brings its own knob (π_p, α, γ, plus
  ALUM's ε, η, etc.), and the knobs are not independent.

### What we're doing in the meantime

- Defaulting `apply_class_balancing=False` in FLPU. Lets π_p do all the class
  balancing. This is the cleanest reading and matches "nnPU as proof, focal
  γ as practical add-on."
- Keeping `focal_alpha` as an opt-in knob with a *uniform-application*
  interpretation (matching Ji 2023's Eq. 14 as literally written, not the
  Keras default of asymmetric α / 1-α weighting).
- Documenting this as a deferred-thinking item rather than treating Ji's
  combination as gospel.

### What deeper engagement would look like

Suggested elements for when we come back to this:

- Trace the nnPU derivation explicitly (Kiryo 2017, Plessis 2014/2015) and
  note exactly where π_p is mathematically required vs. where it could in
  principle be replaced.
- Decide whether the focal modulation (γ) and the nnPU prior weighting (π_p)
  are doing redundant work in the positive-as-negative bias correction term.
- Propose one or more *principled* combination strategies, ideally with
  testable predictions about loss curves or calibration behavior.
- Run small empirical comparisons (synthetic data with known prior) to
  separate the contributions of each mechanism.
- Decide what "calibrated output" means under our combination, and whether
  standard calibration techniques (temperature scaling, Platt scaling, isotonic
  regression) are applicable as-is or need adjustment.

### Related code touchpoints

- `src/loss_functions/loss.py` — FLPU implementation
- `src/run_cca_classification.py` — where the loss is parameterized
  (currently `prior=0.03`, `kiryo_clawback=False`)
- `src/run_prior_estimate.py` — where π_p comes from (DEDPUL pipeline)

---
