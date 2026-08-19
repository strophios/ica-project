# Branched-encoder strategy — the joint-fine-tune design space and the experiment ladder

*Created 2026-08-11. Decision record for the post-meeting model arc: how to get
the validated rel encoder gain into production, and whether the route is a
joint CCA+rel fine-tune, a deeper unfreeze, or a branched (per-head top-K)
encoder. Successor to `encoder-unfreeze-strategy.md`, which records the
rel-first-vs-joint decision and the rel-first execution findings this note
builds on. Written before execution; decision rules below are pre-registered.*

## Context

The rel-first sequential unfreeze (2026-07-28/30, `encoder-unfreeze-strategy.md`
execution findings) produced a two-sided result: rel improved substantially
(vs-ICA 0.783 → 0.853, own-terms 0.783 → 0.836, diaspora recall 0.382 → 0.662
@ 0.30 review rate) while the tuned representation damaged the sibling heads
(CCA own-terms 0.927 → 0.739, US 0.925 → 0.830). The validated-but-undeployed
**mixed stack** (tuned rel head on tuned CLS, production CCA/US on production
CLS) lifts composed ICA ROC 0.797 → 0.820 at the cost of two encoder passes
per corpus at apply.

Roadmap §A1 ordered the follow-through as (1) mixed-stack productionization,
(2) joint CCA+rel fine-tune. A 2026-08-11 strategy review revisited that
ordering. Two inputs:

- **The 2026-08-06 team meeting produced no project feedback** — the session
  was consumed by an R&R on an earlier (pre-ML) article from the project. The
  internal priority order stands on its own reasoning.
- The mixed stack as originally framed (two full encoder passes, two-cache
  `IcaModel`) is machinery that exists only to serve a configuration the joint
  fine-tune is meant to obsolete. Building it first risks dead code; but the
  winning rel run's actual shape (below) collapses most of that cost.

The winning rel run was `--unfreeze-top-n 1` (job 8823087; η=0, π̂=0.02):
only transformer layer 12 received a real learning rate. Zero-multiplier
"frozen" layers drifted only by AdamW weight decay — measured ~2.3e-3 max-abs
over 5 epochs vs 1.19e-1 for the tuned layer, a ~40× separation
(`extract_tuned_backbone.py` layer-diff verification). So the explored region
is one point (N=1, rel-only) in a larger design space.

## The design space

Four candidate shapes, all answers to one question: **where does per-head
adaptation capacity live, and what does it cost the one-encoder/one-cache
economy?**

1. **Joint CCA+rel, top-1 unfreeze.** Both losses negotiate one shared layer
   via a scalarization weight λ. Preserves the single-cache economy fully.
   The pre-registered escalation from `encoder-unfreeze-strategy.md`.
2. **Joint, deeper unfreeze + per-layer discriminative LR.** More shared
   capacity; hyperparameters multiply (N × decay × λ), all tuned against the
   1,131-row hand-coded eval that is already double-booked (fusion fitting +
   swap decisions). LP-FT (Kumar 2022) and the representational-collapse line
   (Aghajanyan 2020/21) both say deeper fine-tuning distorts more — and
   negative transfer was already observed at N=1.
3. **Branched top-K.** Shared frozen trunk (embeddings + layers 1..12−K);
   per-head *copies* of the top K layers, warm-started from DAPT, each tuned
   on its own head's loss. Negative transfer is impossible by construction
   (no gradient crosses branches), and the branch trainings are fully
   independent — the population/channel disjointness that ruled out
   three-head joint (US on stripped LDC; CCA/rel on raw API) is irrelevant
   here. A head that wants no tuning keeps the *original* DAPT layers as its
   branch: bit-identical to production, zero risk. Apply cost for B tuned
   branches at depth K is ~(12 + K·B)/12 of one encoder pass, emitting one
   CLS vector per branch in a single pass (the slim per-head CLS caches
   survive; only the embed job's structure changes).
4. **Deeper heads after the encoder.** Keep the shared encoder; add capacity
   inside the heads. Not equivalent to (3): branch layers are warm-started
   and see pre-pooling sequence state; appended head layers are random-init,
   and if they're transformer layers the CLS pooling has to move after them
   (sequence-level plumbing). Random-init transformer layers against rel's
   ~17k positives is a data-hunger mismatch — dominated by (3) in
   expectation. The degenerate cheap variant (a second Dense on the pooled
   CLS, features-mode) survives as a control experiment, not a candidate
   architecture (stage 2 below). Current head: Dense(768, relu) with dropout
   → Dense(1) on the pooled CLS only (`src/model_setup/heads.py`).

**The reframe that reorganizes the sequencing: the mixed stack is option 3
with K=1, pending one verification.** If the ~2.3e-3 sub-branch drift is
negligible, the tuned encoder differs from production DAPT only in layer 12 —
so "two full passes + two caches" is really "shared pristine trunk + two
layer-12 variants": ~13/12 ≈ 1.08× of one pass, not 2×. Mixed-stack
productionization and the architecture exploration stop being rivals; the
branched build *is* the productionization, and it generalizes to any future
per-head branch.

## What joint still offers, and its third success criterion

Branching gives up two things only a shared tuned encoder provides: 1× apply
with a single cache, and cross-task regularization (CCA's ~15k DoCA positives
as a sibling signal against rel overfitting its ~17k). Joint CCA+rel remains
worth running for that prize. But the rel-first result exposes a criterion
the original pre-registration didn't state: **US isn't at the joint table**,
and a rel-tuned encoder cost US 0.925 → 0.830 own-terms as a passenger. If
the joint-tuned representation degrades US similarly, US needs a separate
representation — two caches again, forfeiting joint's main prize. Joint's
success test is therefore three-sided: rel gain holds, CCA holds, *and US
survives features-retrained on the shared tuned cache*.

## DECISION (2026-08-11): staged experiment ladder, cheap → expensive

Each stage is pre-registered with its decision rule; each informs the next.
No stage requires the 1996–2025 embed — the expanded-corpus embed waits for
the encoder decision so it's paid once (roadmap §A1 item 4).

1. **Graft test** (local, near-free). Compose pristine DAPT embeddings +
   layers 1–11 with the tuned layer 12 from `relevance_tuned`; score the rel
   evals (own-terms, vs-ICA, diaspora recall) against the full tuned encoder.
   **Pass:** vs-ICA ROC within ~0.01 and diaspora recall within ~2 anchors of
   0.853 / 0.662. Pass ⇒ the branched frame is real and the K=1 branched
   mixed stack is the productionization path. **Fail ⇒** the sub-branch drift
   is load-bearing; re-run the rel tune with hard freezing
   (`layer.trainable=False` below the branch — the multiplier-freezing ≠
   trainable=False finding, `encoder-unfreeze-strategy.md` 2026-07-29), one
   known-recipe cluster job, after which the graft is exact by construction.
2. **Head-capacity control** (features-mode, minutes). Add a second hidden
   Dense to the rel head on the *production frozen* cache. This measures the
   assumption underneath the whole encoder-tuning program — that the frozen
   CLS representation, not head capacity, is rel's bottleneck.
   **Interpretation rule:** if extra head depth recovers a substantial
   fraction (≥ ~a third) of the tuned-encoder vs-ICA gain (0.783 → 0.853),
   the representation-bottleneck story weakens and cheap head capacity
   becomes the first-line lever; expected outcome is little gain, which
   affirmatively licenses the encoder work.
3. **Rel depth sweep** (cluster, one job per grid point). N ∈ {1, 2, 3},
   flat vs graded decay (machinery: `--unfreeze-top-n`, `--graded-decay`),
   hard freezing below the unfrozen block. **Pre-registered selection
   metric:** vs-ICA ROC on gold (primary), diaspora recall @ 0.30 review
   rate (secondary), rel own-terms as guardrail (no degradation vs N=1).
   This fills in the depth axis rel-first never explored and fixes the K any
   branch or joint run uses. Keep the grid at ≤6 points — every comparison
   spends the double-booked eval set.
4. **Joint CCA+rel at the chosen depth** (cluster, λ 3-point grid, same
   harmonized population/channel/loss family as pre-registered in
   `encoder-unfreeze-strategy.md`). Now a fair fight against a known branched
   alternative at the same depth. **Joint wins only if:** rel matches its
   branched counterpart within noise, AND CCA matches its branched-or-
   production counterpart within noise, AND US features-retrained on the
   joint cache holds own-terms (survives as a passenger; reference 0.925,
   the rel-first passenger outcome 0.830 is a fail). Win ⇒ single-encoder
   swap (re-embed, features-retrain, recalibrate, refit fusion, gold
   re-eval — the `tuned-retrain-runbook.md` sequence). Lose on any side ⇒
   **branched is the production architecture** and joint retires with a
   clean negative result.

### Companion decisions

- **US gets no tuned branch.** US sits at 0.925 own-terms; its known
  weaknesses (diaspora labels, foreign-leak separation —
  `us-head-retrain-plan.md`) are label- and gate-problems, not
  representation problems. US keeps the original DAPT top layers as its
  branch (bit-identical) in any branched outcome.
- **VAT/ALUM stays unbundled.** Its natural home is a Layer-4 addition to
  whichever fine-tune wins (`pinned-questions.md` §1), run as a controlled
  A/B on top of the stage-4 baseline — never bundled into the same run
  (one channel change at a time, or attribution dies).
- **Temporal signal stays evidence-gated.** It forces a DAPT re-run (the
  most expensive single item) and no measurement yet shows era shift hurts.
  The gate: era-sliced eval on the expanded-corpus apply + the 1970s
  lead-vs-abstract channel experiment (roadmap §A1). Flat era slices ⇒ no
  temporal work.

## Engineering consequences

- **`fit_fusion.py` parameterization stays first regardless** — every
  outcome (branched, joint, any retrain) needs a fusion refit on new scores,
  and the script is fully hardcoded with `output_dir` silently defaulting to
  the production fusion path (`tuned-retrain-runbook.md` gaps).
- **The `IcaModel` per-head-features change survives the reframe.** Branched
  means rel reads a different CLS vector than CCA/US — exactly what
  "two-cache support" meant. What the reframe cheapens is the *embed* side:
  one branched pass emitting per-head CLS caches instead of two full passes.
- **Hard freezing is now a requirement, not a nicety.** The branched frame's
  correctness argument (shared trunk bit-identical across branches) needs
  `trainable=False` sub-branch layers, not zero-multiplier layers that
  drift under AdamW weight decay.
- Small gaps carried from the runbook: `run_relevance.py` lacks a
  `--us-weights` rescore knob; `eval_heads_own_terms` requires calibration
  before eval (runbook step-order); both unchanged by this note.

## Execution record

**Stage 1 — PASS (2026-08-12).** Run via `scripts/graft_test.py`
(`cca_doca/experiments/graft_test_v2.json`): graft (pristine trunk + tuned
`transformer_layer_11`) matches the full tuned encoder within **4e-4** on every
metric (vs-ICA 0.8555 vs 0.8550; diaspora @0.30 0.662 == 0.662, @0.10 0.250 ==
0.250; own-terms 0.8365 vs 0.8362), judged by a correct-math reference head.
The sub-branch drift (max 3.3e-3, layers 7–9) carries nothing. **The branched
frame is real: the mixed stack is a K=1 branched model at ~1.08× apply.**

En route, stage 1 v1 exposed that the production `relevance_tuned` artifact is
**tensorflow-metal-execution-bound** (vs-ICA 0.853 under metal, 0.386 under
correct math — trained on metal, so the weights fit metal's distorted forward).
Full investigation, corrected numbers, and deployment rules:
`metal-execution-findings.md`. Consequences for this ladder:

- The correct-math reference head is a fresh CPU-trained π=0.02 probe
  (`relevance/scratch_diag/relevance_tuned_p02_cpu.weights.h5`); the rel gain
  reproduces under correct math (so does a π=0.05 probe — the July artifact's
  collapse was metal-specific, not prior-specific).
- **Negative-transfer verdict re-measured under correct math: survives
  qualitatively, magnitudes corrected** — CCA own-terms 0.928 → 0.795 (metal
  said 0.739); US features test F1 0.97 → 0.951 (metal-era eval said 0.938
  own-terms-equivalent). Branched remains evidence-favored; stage 4's
  three-sided rule unchanged.
- All ladder training/eval runs are CPU-forced or cluster-side from here; every
  new head gets the CPU-vs-GPU rank-consistency acceptance check
  (`metal-execution-findings.md` deployment rules).

**Stage 2 — head-capacity control: expected outcome (2026-08-12).** Same-session
pair on the frozen `relevance_train` cache, prior 0.05, CPU: shallow control
own/vs-ICA/diaspora 0.833/0.788/0.397 vs deep head (intermediate Dense →
Dense-Dropout-Dense) 0.820/0.779/0.412. Depth on frozen features buys nothing
(slightly negative); far below the pre-registered 0.800 weakening threshold.
Representation-bottleneck premise affirmed; stages 3–4 licensed. **Calibration
by-product:** the fresh shallow control (0.788) vs the production head (0.773,
CPU) puts single-run training noise at ~±0.015 vs-ICA on this eval —
stage 3's selection rule should treat single-run deltas under ~0.015 as ties
(2 seeds per grid point, or coarse reads only). Scratch artifacts:
`relevance/scratch_diag/rel_{shallow_ctrl,deep}_cpu.weights.h5` (the deep
artifact does not reload through `apply_relevance_model` — nonstandard head
structure, loud failure by design).

**Stage 3 — prepared 2026-08-12, awaiting cluster submission.** Build:
`hard_freeze` knob (`RunConfig` field, back-compat `False`; CLI
`--hard-freeze` default ON in `run_relevance_text.py`; `trainable=False` on
embeddings + sub-branch transformer layers via
`escalation.frozen_sublayer_names` → `build_endpoint_model`) and `--seed`
(both trainers; `RunConfig.seed`, back-compat 200; split seed untouched).
Per-point scorer `scripts/eval_rel_sweep_point.py` (CPU-forced, compact JSON
— validated against the job8823087 artifact: CPU 0.8346/0.8531 vs cluster
0.833/0.854, confirming cluster-trained artifacts are execution-portable).
Submission: `sbatch scripts/rel_depth_sweep.sbatch` (array 0–11 = N ∈ {1,2,3}
× {flat, graded 2.6} × seeds {200,201}; each job trains + self-scores; rsync
only `*.eval.json` home). Selection per the pre-registered metric, treating
single-run deltas < ~0.015 vs-ICA as ties (stage-2 noise floor); the winner
gets: backbone extraction (`extract_tuned_backbone`, hard-frozen ⇒ 0.0 drift
expected), re-embed, probe retrain (CPU), CPU-vs-GPU acceptance check,
calibration on the apply path.

**Stage 3 — depth sweep COMPLETE (2026-08-13/14). Verdict: N=1; the July
artifact stays the deployable layer-12.** Full grid (12 points + a 2-job
hard-freeze A/B; eval JSONs in `relevance/sweep/`, logs
`cca_logs/cluster/slurm-91*.out`), all scored CPU-forced on gold:

- **Depth**: N=2 ≈ N=1 flat (0.820–0.823 vs-ICA); N=3 unstable across seeds
  (0.657/0.759 flat; 0.742/0.818 graded) — deeper never wins. **N=1 is the
  depth** for any branch and for stage 4.
- **CORRECTION (2026-08-18): at N=1, "graded" ≡ "flat"** —
  `graded_multipliers(1)` assigns layer 11 exactly `base·decay⁰ = 0.1`,
  identical to flat's `encoder_top: 0.1` (only the group *name* differs).
  The four hard-freeze N=1 cells are therefore four same-config replicates:
  {0.8142, 0.8174, 0.8303, 0.8359} vs-ICA — **text-mode per-draw noise is
  ~±0.01 (spread 0.022)**, and the earlier "graded is the best cell" read
  is retracted. Graded's only real test is N≥2, where it LOST (N=2 graded
  0.743/0.783 vs flat 0.820/0.823) — flat is the scheme going forward.
- **Hard-freeze A/B** (N=1 flat): −0.01 for hard-freeze training (nhf
  0.823/0.827 vs hf 0.814/0.817) — at the noise floor, and moot for
  deployment: stage 1 proved graft-onto-pristine-trunk is lossless, so the
  deploy rule is **train with multiplier freezing, graft at deploy** —
  trunk exactness comes from the graft, not from training-time freezing.
- **The July-gap open question**: the four same-config N=1 replicates put
  the modern mean at 0.8245 (sd ~0.0096); the July run's 0.853 is ~+3σ.
  Excluded by measurement: config (sidecar diff — identical except
  `hard_freeze`), hard freezing (A/B ≈ 0.01), GPU type (V100/A100/H200 all
  present in both good and bad cells), epochs/early stopping (all runs 6–8
  epochs, healthy val trajectories). Remaining candidates: single-draw
  variance (July was the *only* healthy run of its debug arc, not a
  selected-best — but +3σ is a stretch), cluster env drift since 07-29
  (CUDA/cudnn modules; uv.lock unchanged), or a rebuilt text table on the
  cluster (unchecked — steps_per_epoch identical at 234, so n_pos matches).
  NOT decision-relevant: the July layer-12 is in hand, verified (graft
  0.855, probe retrains 0.852–0.855), and better than every sweep artifact —
  reproducing its training is not required to deploy it. Park unless a
  future retrain needs it.

**Stage 4 (joint CCA+rel) — design revised 2026-08-18 (operator review):**

- **Depth is swept, not inherited**: N ∈ {1, 2} × λ ∈ {0.25, 0.5, 0.75} ×
  seeds {200, 201} = 12 jobs, flat multipliers. The solo N=1 optimum need
  not transfer — joint roughly doubles the positive signal (~15k DoCA +
  ~17k rel) and diversifies gradients, exactly what the solo sweep's
  N-depth failures were starved of. N=3 only if N=2-joint > N=1-joint by
  more than noise (pre-registered contingency).
- **Selection metric is composed, not head-solo** (operator's
  complementarity point: λ explicitly trades the heads off; a rel head
  that is slightly worse solo but less redundant with CCA composes
  better). Per cell: Platt per head on the natural-balance val stream (no
  gold contact) → product of calibrated CCA·rel probs (production's
  default fusion shape; composed Platt is monotone ⇒ ROC-invariant; US
  gate + product-vs-LR fusion selection deferred to the winner) →
  **composed ROC vs `ica_event` + diaspora recall on gold** primary.
  Guardrails: per-head own-terms (CCA ~0.93, rel ~baseline); US-passenger
  check on the winner only. Rel-solo vs-ICA demoted to diagnostic. The
  λ=1 endpoint (rel-only) is already measured by the stage-3 replicates;
  the winner gets the full production composition — which requires the
  `fit_fusion.py` parameterization (roadmap §A1 item 1) as part of the
  stage-4 build.
- Reference bar: the mixed-stack decomposed numbers (rel 0.853 solo; the
  composed-proxy of July-rel × production-CCA, computed once as the
  branched baseline the joint must beat to justify a single encoder).

**Stage 4 — joint CCA+rel sweep COMPLETE (2026-08-19). VERDICT: joint does
not win; THE BRANCHED ARCHITECTURE IS THE PRODUCTION DESIGN.** 11/12 cells
scored (N2_λ.50_s201's eval JSON missing — its region sits ~0.15 below the
bar, verdict-irrelevant); results `relevance/joint_sweep/*.eval.json`,
baseline `cca_doca/experiments/branched_baseline_proxy.json`.

- **The rule's outcome**: best region N1/λ=0.25, seed-pair composed proxy
  {0.7996, 0.8173}, mean 0.808 ≈ the branched bar 0.8064 — a tie at best —
  and both draws fail both guardrails (CCA own 0.90 < 0.91; rel own
  0.75–0.79 vs ~0.83 baseline). Every other cell is 0.05–0.18 below the
  bar. Pre-registered tie rule ⇒ branched.
- **Keeper finding 1 — complementarity is real but expensive.** λ=0.25
  composition ADDS over its own rel-solo (+0.04 to +0.10; e.g. 0.700→0.800)
  where branched composition SUBTRACTS (0.852→0.806): joint training does
  reshape the heads toward complementary — the operator's hypothesis
  confirmed — but at ruinous per-head cost (rel-solo 0.70–0.78 vs the
  branched 0.853). The complementarity lever exists; this λ-scalarized
  implementation can't cash it.
- **Keeper finding 2 — negative transfer through a shared layer is
  zero-sum at these data scales.** Rel-first tuning wrecked CCA
  (0.93→0.74); joint tuning wrecks rel (0.85→0.5–0.78 solo) while CCA
  mostly holds. Whoever's gradients dominate the shared layer wins it.
  N=2 does not rescue joint (mostly worse than N=1).
- **Recorded limitations**: early stopping monitored the λ-mixed val_loss,
  so cross-λ comparisons entangle λ's representation effect with its
  stopping effect (operator-flagged 2026-08-19; magnitudes ~0.01-scale
  per the plateau shapes, far below the 0.05+ gaps); the 12th cell is
  unscored; text-mode draw noise ±0.01 applies throughout.
- **Riders for any future tuning run** (not needed for the branched
  deploy, whose rel artifact — the July layer-12 — already exists):
  `restore_best_weights=True`; λ-independent rank-based val monitor;
  the one-shot schedule A/B (current vs stretched-decay 12-epoch),
  selected on val-stream metrics, never gold.

**Productionization (the ladder's output, now unblocked):** branched
`IcaModel` — shared pristine DAPT trunk + the July tuned layer-12 as rel's
branch (graft-at-deploy, stage-1-proven lossless), original layer-12 for
CCA/US; per-head CLS caches from one ~1.08× embed pass; CPU-portable
replacement rel artifact (the p02 probe + its calibration sidecar already
exist in scratch_diag); fusion refit via the parameterized `fit_fusion.py`;
gold re-eval; swap decision. Roadmap §A1.

## Pointers

`encoder-unfreeze-strategy.md` (predecessor decision + rel-first findings and
literature survey); `tuned-retrain-runbook.md` (the retrain-and-compare
command sequence and its gaps); `docs/notes/roadmap.md` §A1 (the live list
this note reorders); `pinned-questions.md` §1 (VAT/nnPU composition);
`us-head-retrain-plan.md` (why US's ceiling isn't representational).
