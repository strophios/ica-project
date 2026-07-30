# Encoder-unfreeze strategy — sequential vs. joint, with the loss-balancing literature

*Created 2026-07-27, during the pre-Aug-6 arc (roadmap thread A item 3). Decision
record for HOW to fine-tune the shared frozen DAPT encoder: which head(s')
loss(es) tune it, and what restores the shared-encoder architecture afterward.
Grounded in a web-literature survey (agent-run, 2026-07-27) plus this project's
own empirical findings. Written before the decision was executed.*

## The problem, precisely

The assembled `IcaModel`'s economy rests on ONE shared encoder and one CLS cache
(one forward pass serves all three heads over 3.7M+ articles). Unfreezing
requires text-mode training (features-mode is definitionally frozen), and after
the encoder moves, every head must be re-fit or re-validated on the new
representation. The fork:

- **(A) Sequential (rel-first):** tune the encoder with the rel head's loss only
  (top-N unfreeze, discriminative LR), then re-embed the corpora once and retrain
  US + CCA features-mode on the new cache (minutes each), recalibrate, refit
  fusion. Zero new hyperparameters beyond the unfreeze knobs (N, per-group LR
  multipliers, epochs).
- **(B) Joint:** one training run where multiple heads' losses tune the shared
  encoder. Requires (1) a loss-aggregation scheme `L = Σ w_k L_k` and (2) a
  data-interleaving scheme — the heads do NOT share a training population
  (US: stripped-channel LDC, 630k; CCA/rel: raw-channel API, 250k/266k tables),
  so joint batches must be per-task and round-robined/sampled, making the
  sampling ratio a second, partially-redundant balancing knob.
- **(C) Per-head encoders:** ruled out — triples apply cost, kills the shared
  cache.

## What the literature settles (survey 2026-07-27; citations at bottom)

1. **Loss-balancing has no winner.** Kendall/Gal/Cipolla 2018 (uncertainty
   weighting), GradNorm 2018, DWA 2019, MGDA 2018, PCGrad 2020, CAGrad 2021,
   FAMO 2023 all exist; but Kurin et al. NeurIPS 2022 ("In Defense of the
   Unitary Scalarization") and Xin et al. NeurIPS 2022 ("Do Current Multi-Task
   Optimization Methods Even Help?") showed that under fair tuning budgets none
   reliably beat plain (even equal-weight) scalarization with standard
   regularization. Shi et al. NeurIPS 2023 adds a theoretical caveat
   (scalarization can't trace the full Pareto front in under-parametrized
   regimes) that mostly doesn't bind for over-parametrized encoders. LibMTL
   benchmarking (JMLR 2023 + later releases) reports no dominant method. **The
   standing default is tuned scalarization; DWA/uncertainty weighting are the
   only cheap adaptive upgrades; gradient-surgery methods need empirical
   justification.**
2. **NLP joint-vs-single evidence favors joint — in homogeneous supervised
   setups.** MT-DNN (ACL 2019) and instruction-era results (ICLR 2025
   task-aware sampling work) show joint fine-tuning of a shared encoder beating
   single-task, *when tasks share loss family and input distribution and the
   goal is average performance across tasks*. Counter-evidence at BERT scale:
   representational collapse under multi-task fine-tuning (Aghajanyan et al.
   2021; "Better Fine-Tuning by Reducing Representational Collapse" 2020).
3. **Sequential's footing.** ULMFiT (ACL 2018) is the source of our
   discriminative-LR + gradual-unfreeze machinery. LP-FT (Kumar et al. 2022):
   probe-then-fine-tune protects features; fine-tuning distorts representations
   (esp. middle/late layers), and small-LR + top-N-only is the standard
   mitigation. Direct evidence on "fine-tune task A, then probe siblings B/C on
   the tuned encoder" is thin — drift is real but measurable, which is exactly
   what our re-embed → features-retrain → gold-re-eval step measures.
4. **PU-in-MTL is a literature gap.** No published work trains nnPU/focal-PU
   losses jointly with other losses on a shared encoder (searched 2017–2025;
   nearest is disease-prediction multi-task PU with separate training). The
   nnPU batch-level risk correction interacting with task interleaving is
   untested territory.

## Project-specific factors the generic literature can't weigh

- **We have direct empirical evidence of nnPU batch-composition sensitivity:**
  the US-retrain η=0 collapse (2026-07-24, `us-head-retrain-plan.md` addendum) —
  at high prior the nnPU clip fired every batch and the loss degenerated. Joint
  interleaving changes effective batch composition per loss; this is the exact
  axis our loss family is fragile on.
- **The goal is lifting ONE weak head** (rel, own-terms ROC 0.829 vs US 0.925 /
  CCA 0.927), not maximizing an average across tasks — the setting where MT-DNN
  style joint wins are demonstrated is the latter.
- **The eval set is double-booked.** The ~1,131-row hand-coded set already
  serves fusion fitting and swap decisions; it is score-stratified, not a
  general tuning resource. Every balancing hyperparameter tunes against it.
  (The survey's suggestion that 1,100 rows is "sufficient budget" for joint
  validation ignores this booking.)
- **Channel/population disjointness:** a jointly-tuned encoder would see a
  mixture of text channels (stripped LDC + raw API) that no single head sees at
  inference.
- Note: the survey's own decision framework ("choose sequential if any head's
  loss is PU; if loss families are heterogeneous") points at sequential for this
  project, even though its top-line synthesis said joint-first — the top-line
  leaned on the homogeneous-supervised NLP evidence (point 2) without carrying
  its own PU caveat (point 4) through. Recorded here because it's a nice
  example of why we cross-examine research output.

## DECISION (2026-07-27): (A) rel-first sequential, with a pre-registered escalation

1. **Now:** text-mode rel-head training with top-N unfreeze + discriminative LR
   (ULMFiT recipe, machinery proven in `run_us_classification.py:154-179`),
   FLPU loss unchanged. Then re-embed → features-retrain US + CCA on the new
   cache → recalibrate → refit fusion → gold re-eval. The US/CCA re-evals are
   the built-in negative-transfer check (validate-before-swap again: if the
   tuned encoder degrades the strong heads more than rel gains, don't ship it).
2. **Pre-registered escalation — joint CCA+rel (NOT three-head):** if rel-first
   underdelivers or the transfer check fails. CCA and rel share population,
   channel, and loss family (both FLPU on the harmonized table), so the joint
   problem collapses: same batches can carry both labels (no interleaving), one
   scalarization weight λ (grid over ~3 values), no channel conflict. This
   captures most of the joint signal-sharing upside while dodging the
   balancing-method swamp and the PU-interleaving unknown. Three-head joint
   (US included) stays out unless CCA+rel joint shows clear gains AND the US
   head is being retrained anyway.
3. **If joint is ever run:** equal-weight or 3-point-grid scalarization; no
   GradNorm/MGDA/PCGrad without a diagnosed gradient pathology (per-group grad
   norms from the Tier-5 diagnostics are the instrument that would diagnose it).

**What would change this decision:** published or observed evidence that nnPU
losses interleave stably; a much larger hand-coded eval (removing the tuning
bottleneck); or the rel-first run showing the encoder needs more signal than
one head provides (e.g., rel gains but immediately overfits its 17k positives —
the case where CCA's 15k DoCA positives as a sibling signal would regularize).

## Execution findings (added as the arc ran)

- **2026-07-28/29, the η=0 collapse and its resolution:** text-mode unfreezing
  at the canonical rel config (η=0, π=0.05) collapsed into the nnPU
  all-positive basin (job 8808071); η=0.25 stabilized but taxed ranking
  (echoing the historical η-sweep); DEDPUL re-estimation gave π̂=0.02, and
  **η=0 at π̂ trains cleanly and wins on gold** (job 8823087: own-terms 0.833,
  vs-ICA 0.854 vs frozen 0.782, diaspora recall 0.662 vs 0.382 @ 0.30 review
  rate). The wrong prior was the dominant collapse cause; unfrozen capacity
  was the amplifier. Full mechanism + solution-space taxonomy: session notes
  2026-07-29 / `us-head-retrain-plan.md` addendum's sibling finding.
- **Multiplier-freezing is not `trainable=False` (2026-07-29):** AdamW's
  decoupled weight decay applies to every trainable variable *regardless of
  the gradient multiplier*, so zero-multiplier "frozen" layers drift by
  lr·wd·var each step — measured ~2.3e-3 max-abs over 5 epochs vs 1.19e-1 for
  the deliberately-tuned layer (~40× separation; benign here, compounds on
  longer runs). If exact freezing ever matters (e.g., cache-reuse arguments
  that assume lower layers are bit-identical), either set
  `layer.trainable=False` for permanently-frozen blocks or exclude
  zero-multiplier groups from weight decay. Recorded empirically by
  `src/extract_tuned_backbone.py`'s layer-diff verification.

- **The backbone-clobber bug (2026-07-29, found via the transfer check):**
  `embed_corpus._build_embed_model` loaded the `us_weights` file over the full
  inference model AFTER the `--backbone-weights` override — and the default
  us-weights file (`us_classifier.weights.h5`, the old 200-step smoke) is a
  FULL-model save, so the load silently restored the smoke's frozen-DAPT
  backbone. Harmless for every production cache (smoke backbone == exact DAPT)
  but it voided the first tuned re-embed: the 2026-07-29 "tuned" caches were
  DAPT embeds (their fresh-probe evals reproducing production numbers was the
  tell; co-trained-head mismatch at ROC 0.67 and local-vs-cache CLS cosine 0.6
  were the smoking guns). Fixed by re-applying the override after the us-load;
  proof: co-trained rel head on fixed-path local tuned CLS recovers
  0.833/0.851 (vs the artifact's own 0.833/0.854). Corollary finding:
  **the tuned gain TRANSPORTS through features-mode** — the co-trained head +
  genuine tuned cache preserves the vs-ICA improvement, so the deployed
  features-mode pattern works; deploy the co-trained head rather than a fresh
  probe. Void caches/artifacts renamed `*.VOID*`; cluster re-embed required.

- **The transfer verdict (2026-07-29/30, genuine tuned caches):** rel-first
  sequential produced exactly the two-sided outcome the pre-registration
  anticipated. **rel: clear win** (fresh probe own-terms 0.836, vs-ICA 0.853
  vs production 0.783; diaspora recall 0.662 vs 0.382 @0.30) — and the fresh
  probe MATCHES the co-trained head on true tuned features (0.8526 vs 0.8508),
  so co-training carries nothing beyond the representation; probe deploys, for
  pipeline uniformity. **CCA: severe negative transfer** (own-terms 0.927 →
  0.739); **US: real, milder** (0.925 → 0.830; dateline test F1 0.97 → 0.938).
  **Per the pre-registration: NO wholesale encoder swap.**
  **Hand-composed mixed stack** (tuned rel on tuned CLS; production CCA/US on
  production CLS; production fusion/gate/calibrations unchanged — conservative):
  composed ICA ROC **0.797 → 0.820**, diaspora recall@0.10 0.221 → 0.250.
  A genuine system gain from the rel swap alone, at the cost of TWO embed
  passes per corpus at apply (the known two-encoder trade-off). The joint
  CCA+rel escalation (pre-registered above) is now evidence-motivated as the
  route back to a single shared encoder carrying both gains. Cross-platform
  embed consistency (cluster vs local) verified exactly post-clobber-fix.

## Sources

Kurin et al. 2022 arXiv:2201.04122; Xin et al. 2022 arXiv:2209.11379; Shi et
al. 2023 arXiv:2308.13985; Kendall/Gal/Cipolla CVPR 2018; Chen et al. (GradNorm)
ICML 2018; Liu et al. (DWA) CVPR 2019; Sener & Koltun NeurIPS 2018; Yu et al.
(PCGrad) NeurIPS 2020; Liu et al. (CAGrad) NeurIPS 2021; Liu et al. (FAMO)
arXiv:2306.03792; Liu et al. (MT-DNN) ACL 2019 arXiv:1901.11504; Kumar et al.
(LP-FT) arXiv:2202.10054; Howard & Ruder (ULMFiT) ACL 2018; Aghajanyan et al.
2020 arXiv:2008.03156; Kiryo et al. 2017 arXiv:1703.00593; LibMTL JMLR 2023.
Survey run 2026-07-27 by web-research agent; claims cross-checked against the
papers' abstracts/venues, not re-derived.
