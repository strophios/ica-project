# Pinned Questions

A running record of substantive questions that have been deliberately deferred,
not forgotten. Each entry should explain what we know, what we don't, and what
deferring means for the work we're doing in the meantime.

When picking up a pinned question for real engagement, consider whether the
deferred-thinking decisions made in the interim still hold up under deeper
analysis.

---

## 1. Composing the mechanisms of the PU classification loss

**Pinned:** 2026-04-19 (audit and FLPU verification).
**Revised:** 2026-04-20 (four-layer framing + α correction, after adversarial
review and more careful engagement with the composition question).

### The question

Our PU classification loss combines multiple mechanisms that were designed
to solve different problems. How do they compose? Specifically: how do we
add practical class-balance/signal-shaping machinery on top of nnPU without
breaking the properties that make nnPU work in the first place?

Naïvely the stack contains:

- **nnPU** (Kiryo 2017): a distributional-identity-based risk decomposition
  that lets us estimate the supervised risk from positive + unlabeled data.
- **Focal γ modulation** (Lin 2020): a per-sample loss-function reweighting
  that down-weights easy-to-classify examples.
- **Focal α class balancing** (Lin 2020): a per-class constant weighting.
- **Ratio Batch sampling** (Ji 2023): up-sampling positives to guarantee
  per-batch signal despite class imbalance.
- **ALUM / VAT** (Liu 2020, Miyato 2018, to be implemented): an adversarial
  regularizer that encourages smoothness around the input embedding.

These do different things, operate at different levels, and in principle
compose in non-obvious ways. An earlier version of this note tried to sort
them by "mathematically load-bearing vs. practical heuristic," which was a
category error — it confused π_p's role as a mixture coefficient in a
density identity with its functional appearance at the loss level. A much
cleaner framing is below.

### The nnPU identity is loss-agnostic

The nnPU decomposition rests on a distributional identity:

$$p_u(x) = \pi_p p_p(x) + \pi_n p_n(x)
\quad \Longrightarrow \quad
\pi_n \mathbb{E}_{p_n}[\ell] = \mathbb{E}_{p_u}[\ell] - \pi_p \mathbb{E}_{p_p}[\ell]$$

This is a statement about **distributions**, not about loss functions.
Whatever per-sample loss ℓ you plug in — cross-entropy, focal, squared
error, anything — the identity holds. The nnPU estimator therefore gives
an unbiased estimate of *whatever risk you have chosen* as your target,
modulo:

- The max(0, ·) non-negative clip, which introduces bias (biased but
  consistent — Kiryo's trade-off).
- The requirement that the identity be respected when you write the
  estimator. In particular: the two "negative-side" terms (unlabeled
  treated as negative, and positives treated as negative) come from the
  *same* expectation $\pi_n \mathbb{E}_{p_n}[\ell]$ and must carry the
  same coefficient. Otherwise the identity-based substitution no longer
  gives back a supervised risk.

So the right question is not "does adding focal loss break unbiasedness?"
It is "what risk do we want to minimize, and do we have an unbiased
estimator of *that* risk?" If you accept focal loss as the per-sample
loss that defines your target risk, nnPU gives you unbiased (or
consistent, under clipping) estimation of it.

### A four-layer composition structure

Viewing the stack this way lets us separate concerns:

**Layer 1 — Ground-truth risk definition.**
*What are you trying to minimize?* Pick a per-sample loss function
(cross-entropy, focal with γ, focal with γ and per-class cost weights,
etc.). Optionally specify class costs $(\alpha_+, \alpha_-)$ for
cost-sensitive learning. This layer selects a specific target risk;
everything downstream estimates or optimizes *that* risk.

**Layer 2 — PU estimation machinery.**
*How do you estimate the risk from PU data?* The nnPU identity
decomposes the chosen risk into PU-observable terms. The identity is
loss-agnostic at Layer 1, but imposes constraints on Layer 2: if your
Layer-1 risk has asymmetric class weights $(\alpha_+, \alpha_-)$, then
Layer 2 must apply $\alpha_+$ to the positive term, $\alpha_-$ to *both*
unlabeled-as-negative *and* positives-as-negative terms. The max(0, ·)
clip lives here and introduces Kiryo's bias–stability trade-off.

**Layer 3 — Sample allocation and gradient signal.**
*How do you feed data to the estimator?* Batch composition (Ratio Batch
and how aggressively to rebalance), shuffle buffers, prefetching,
per-epoch step counts. These affect the *variance* of per-batch gradient
estimates, not the underlying risk. In expectation over batches, an
up-sampled batch is still an unbiased estimate of Layer-2's estimator.

**Layer 4 — Optimization-level regularization.**
*What do you add to help training converge well?* ALUM/VAT, weight
decay, dropout, learning-rate schedules. These add terms to the total
objective without modifying the nnPU risk estimator itself.

The key property: **these layers compose independently** as long as
each layer's internal consistency rules are respected. You pick ℓ at
Layer 1, wrap it in nnPU at Layer 2, choose a sampling strategy at
Layer 3, and add any regularization at Layer 4.

What would *break* the composition:

- Using different coefficients on the two negative-side terms (breaks
  Layer 2's identity).
- Over-sampling positives at Layer 3 and *then* using the up-sampled
  ratio in place of the true π_p at Layer 2 (double-counting).
- Applying per-sample re-weighting at Layer 1 that depends on the
  sampling ratio at Layer 3 (cross-layer coupling).

### Placing our current mechanisms in the layers

| Mechanism | Layer | Notes |
|---|---|---|
| Cross-entropy vs focal cross-entropy | 1 | Pick ℓ. We currently use focal with γ=2. |
| Focal γ (easy-example down-weighting) | 1 | Part of the per-sample loss definition. Does not "break" nnPU; it chooses a different target risk. |
| Class-balance α (Lin 2020 style) | 1 | When applied consistently across $T_2$ and $T_3$, it is exactly cost-sensitive nnPU with cost ratio $\alpha_+ : \alpha_-$. Preserves the Layer-2 identity. |
| Positive class prior π_p | 2 | Required mixture coefficient; the identity does not work without it. |
| Non-negative clip (max(0, ·)) | 2 | Kiryo's bias–stability trade-off; always on in our implementation. |
| Kiryo clawback (nn_beta, nn_gamma) | 2 | Optional stronger overfitting-recovery variant. Currently `kiryo_clawback=False`. |
| Ratio Batch sampling (1:10 pos:unl) | 3 | Up-samples positives beyond their natural ratio (~1:55). Compatible with Layers 1 and 2. |
| Shuffle / prefetch / seed | 3 | Variance, reproducibility. |
| ALUM / VAT (to be implemented) | 4 | Adds a KL-divergence term to the total objective; orthogonal to Layers 1 and 2. |
| Weight decay in AdamW | 4 | L2 regularization on parameters; orthogonal. |

### Revisiting the α decision

The FLPU docstring originally justified removing `focal_alpha` by claiming
the Keras asymmetric α formulation had "no principled justification in the
PU setting." On closer analysis, this was wrong: Keras's
`BinaryFocalCrossentropy(apply_class_balancing=True, alpha=α)` plugged into
FLPU exactly as the original code did gives a clean **cost-sensitive
nnPU** estimator with cost ratio $\alpha : (1-\alpha)$. Both negative-side
terms automatically receive the same $(1-\alpha)$ coefficient (because
Keras applies the weight by label, and both are y=0), which is the exact
consistency property Layer 2 requires.

So the Keras α formulation is principled, and our original "drop α because
it's unprincipled" argument was itself the unprincipled part.

The *real* reason to keep α=off as the default for the CCA head is:

1. We don't have a deliberate cost-sensitivity preference for CCA
   (both false positives and false negatives are roughly equally costly
   for the overall research goal of building a filterable hay-to-needle
   dataset).
2. Lin 2020's α=0.25 specifically *down-weights* the positive class.
   For a rare-positive problem where we are trying to identify the
   minority class, the Lin 2020 default is almost certainly the wrong
   sign of adjustment — it would reduce the signal we already have too
   little of.
3. If we ever do want cost-sensitive PU, we should parameterize
   $(\alpha_+, \alpha_-)$ directly as a deliberate knob, not hack it
   through Keras's α parameter (whose $\alpha, 1-\alpha$ constraint is
   borrowed from a different context).

This is a cleaner argument than the docstring originally made. The
default behavior is the same (no class balancing), the rationale is
more honest, and we have a clear path for bringing cost-sensitive
learning back later if any head needs it.

### Multi-head lookahead

When we build the planned multi-head classifier (CCA head, immigrant-
involvement head, combined ICA head), the four-layer framing still
applies, but with a cross-cutting distinction between **per-head**
choices and **model-level** choices:

- **Per-head** (each head can differ):
  - Layer 1: per-sample loss, γ, cost weights.
  - Layer 2: class prior π_p (each head has its own prior).
  - Layer 4: per-head regularization weight.

- **Model-level** (must be the same across heads during training):
  - Layer 3: batch composition, shuffle buffer, prefetching. A single
    batch of data is presented to all heads simultaneously, so you can
    only have one sampling strategy per batch. If two heads want different
    positive ratios, only one of them can get it — unless you do something
    architecturally more complex (e.g., per-head batching).
  - Layer 4 mostly: ALUM / VAT operates on the shared encoder, so is
    applied once across all heads.

A new question multi-head raises: **how do the per-head losses combine
into the overall training loss?** Sum, weighted sum, or something more
structured? Does the combined-ICA head's loss couple to the individual
heads' losses through shared gradients into the encoder? These are
Layer-1-ish questions at the *model-level* rather than per-head level,
and they deserve their own careful analysis when we get to the multi-
head implementation.

### Related: DEDPUL attribution, a worked example of the epistemic trap

During the initial FLPU verification, a bug was fixed in `run_prior_estimate.py`
(the DEDPUL pipeline was receiving raw logits where it expected
probabilities). The commit attributed the empirical effect — π_pos moving
from ~0.04 to ~0.02 on our cached data — to the sigmoid fix.

Adversarial review caught that this was wrong. A four-variant re-run
(`scripts/compare_dedpul_logit_vs_prob.py`) showed that almost the entire
shift was driven by DEDPUL's bandwidth-tuning grid being calibrated for
[0, 1]-valued inputs but being applied to logit-scale inputs in the buggy
variant. The sigmoid fix is correct on semantic grounds (DEDPUL expects
probabilities) and happens to also resolve the bandwidth-scale issue (as
a free side effect), but attributing the effect to "the fix" was
overstating what we had actually demonstrated.

This is a useful cautionary example of exactly the failure mode this
pinned-question is trying to prevent for the classification loss:
small unargued choices can have large unintended effects, and writing
up conclusions before isolating the effects of each choice risks
locking in wrong attributions.

### What we are doing in the meantime (current defaults)

- **Layer 1**: per-sample focal cross-entropy with γ=2, no class-balance α.
- **Layer 2**: standard nnPU with clipping (`kiryo_clawback=False`),
  π_p ≈ 0.02 (post-DEDPUL-fix estimate; current code still uses 0.03 for
  continuity with already-trained models, to be updated on next retrain).
- **Layer 3**: Ratio Batch with 1:10 positive:unlabeled, shuffle buffer
  100000, seed 200.
- **Layer 4**: AdamW with weight_decay=5e-3; no ALUM yet.

### What deeper engagement would look like

Open threads worth working through when this comes off the pin:

- **Cost-sensitivity per head.** Do any of the planned heads have
  asymmetric misclassification costs? If yes, parameterize
  $(\alpha_+, \alpha_-)$ deliberately.
- **Calibration under our Layer-1 choice.** Focal loss produces
  uncalibrated probabilities. nnPU layered on top produces outputs that
  are shifted in a combined way that standard calibration techniques
  (temperature scaling, Platt scaling, isotonic regression) may or may
  not apply to cleanly.
- **Ratio Batch aggressiveness.** Ji 2023's Eq. 8 prescribes $n_p =
  \max(1, \lceil N_p/(N_p+N_u) \times N_b \rceil)$ (≈ 1:55 for us).
  Our 1:10 is ~5× more aggressive. Unargued; worth a sensitivity sweep
  alongside other hyperparameter work.
- **Multi-head loss composition.** Structured total loss, per-head
  weights, gradient interaction via the shared encoder. Needs its own
  worked analysis when multi-head work starts.
- **ALUM composition.** Adversarial training interacts with the output
  probabilities FLPU produces. How should the KL regularizer be
  parameterized to not fight the nnPU structure?

### Related code touchpoints

- `src/loss_functions/loss.py` — FLPU implementation (Layers 1+2).
- `src/run_cca_classification.py` — Layer 3 sampling, Layer 4 optimizer.
- `src/run_prior_estimate.py` — π_p source (Layer 2 parameter).
- `scripts/compare_dedpul_logit_vs_prob.py` — the attribution worked
  example.

---
