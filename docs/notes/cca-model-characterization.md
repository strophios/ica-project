# CCA model characterization

*Created 2026-06-16. Branch `cca-doca-retrain`. What the current CCA/DoCA classifier
does, how we measure it, how it compares to baselines, and what we'd get if we ran it
on the corpus. Read with `cca-doca-handoff.md` (build log) and
`cca-doca-retrain-design.md` (design). The numbers here come from the leakage-honest
models trained this round and the 500-row hand-coded gold set.*

## What the model is

A binary classifier that flags news articles reporting a collective-action event. It
runs on cached embeddings, so it's cheap to train and apply:

- **Encoder (frozen):** a RoBERTa-base masked-LM, domain-adapted on ~1.16M NYT
  headline+lede pairs (DAPT). We run each article through it once and cache the CLS
  vector (768-d). The encoder is *not* trained during classification.
- **Head (trained):** a small MLP on the cached CLS vector —
  `Dropout → Dense(768, relu) → Dropout → Dense(1)` — producing a logit.
- **Loss:** focal non-negative PU (FLPU). We have DoCA-confirmed positives and no
  confirmed negatives, so the unlabeled pool is treated as a positive/negative mix at
  the estimated class prior (π ≈ 0.02). The focal term handles the heavy class
  imbalance.
- **Population:** the unlabeled background is US-restricted (the US filter's logit ≥ 0).
  DoCA events are US by construction, so confirmed positives are kept regardless.

Two CCA definitions are live, carried as parallel tracks (the scoping choice is a
collaborator call):

- **all-forms** — any DoCA collective-action event is positive (`cca_doca.weights.h5`).
- **street** — only street events (demonstrations, marches, rallies, pickets, riots)
  are positive; conventional/lawsuit/boycott protests are presumed-negative
  (`cca_doca_street.weights.h5`). The street model is the cleaner detector for street
  events — see "Definition comparison" below.

## How we measure it

The hard part isn't computing precision, it's computing a precision that means
something. Three instruments, each correcting a specific bias:

**Gold set.** 500 articles hand-coded for `cca_event` (and `event_type`, `us_event`).
Drawn from the unlabeled background and **stratified by model score** — the high-score
band is oversampled so there are enough predicted-positives to estimate precision (a
uniform draw would be ~2% positive). The gold ids are held out of training, so these
numbers aren't memorized.

**Raw vs. reweighted precision.** Because the gold set oversamples high scores, its raw
positive rate (~34%) is far above the corpus rate (~2%). So *raw* precision on the gold
set overstates corpus precision — for any classifier. We correct it with
inverse-probability weights: each coded row is weighted by `corpus_band / gold_band` for
its sampling stratum, which recovers the corpus operating point. Raw precision stays
useful for *comparing* models (the bias is constant across them); reweighted precision
is the number you'd actually see in deployment. The two converge at high thresholds,
where the densely-sampled high band dominates — so the high-threshold reweighted number
is both honest and stable.

**DoCA recall.** Recall on a score-stratified set is biased, so we report recall over a
clean held-out set instead: the test-split DoCA positives (confirmed events the model
never trained on). This is the trustworthy "of real events, how many do we catch."

## How good it is: vs. baselines

A precision number means nothing without a floor. Two baselines, both measured in
corpus space (reweighted the same way as the model):

| detector (all-forms) | corpus precision | recall | lift over base rate |
|---|---|---|---|
| random at the class prior | 0.020 | — | 1× |
| protest-keyword lexicon | 0.186 | 0.26 | ~9× |
| **CCA model @ logit ≥1.0** | **0.822** | 0.35 (DoCA) | **~41×** |

The keyword baseline matches a fixed protest lexicon (`protest`, `demonstrat`, `march`,
`strike`, `boycott`, `picket`, `riot`, …) against the headline+lede. It's the "can ML
beat grep" comparator. Two things stand out:

- Keywords beat chance by ~9×. Lexical signal is real — protest words do carry
  information.
- The model beats keywords by another ~4×. At matched corpus recall (~0.26) the model
  is roughly 3× more precise, and it can push to 0.82 precision — an operating point the
  keyword detector can't reach at all, because in the real corpus (87% low-score
  articles) most keyword hits are false positives: `strike` in sports, `march` the
  month, `rally` in markets.

A warning worth recording: the keyword baseline's *raw* gold-set precision looks like
0.83, which would say keywords beat the model. That's a stratification artifact. Raw
gold precision isn't corpus precision for *any* classifier, and the comparison is only
honest once both sides are reweighted. The reweighting machinery exists for exactly this.

## Operating points

Pick the threshold from this curve, not from the prior's implied 0.5 probability. Raw P
is for cross-model comparison; reweighted P and DoCA recall are the real operating point.

**all-forms** (`cca_doca.weights.h5`, π=0.02):

| logit ≥ | raw P / R | reweighted P | DoCA recall |
|---|---|---|---|
| 0.0 | 0.726 / 0.817 | 0.545 | 0.608 |
| 0.5 | 0.765 / 0.751 | 0.626 | 0.469 |
| **1.0** | 0.794 / 0.728 | **0.822** | 0.351 |
| 1.5 | 0.805 / 0.562 | 0.805 | 0.257 |
| 2.0 | 0.857 / 0.320 | 0.857 | 0.185 |

**street** (`cca_doca_street.weights.h5`, π=0.02; 144 street gold positives):

| logit ≥ | raw P / R | reweighted P | DoCA recall |
|---|---|---|---|
| 0.0 | 0.628 / 0.903 | 0.414 | 0.738 |
| 0.5 | 0.696 / 0.812 | 0.510 | 0.606 |
| 1.0 | 0.732 / 0.799 | 0.616 | 0.464 |
| 1.5 | 0.781 / 0.743 | 0.770 | 0.348 |
| 2.0 | 0.841 / 0.514 | 0.868 | 0.247 |

Precision climbs and recall falls as you raise the threshold — the usual trade. The
prior (π) and the form filter only slide *where* you sit on this curve; they don't move
the curve itself. logit ≥1.0 is a reasonable default: ~0.82 precision at ~0.35 recall
for all-forms.

## What we'd get if we ran it

Applied to the part-1 embedding (1960–1975, 1,831,300 articles; 1,611,948 pass the US
filter at 88%). "~true events" = flagged × reweighted precision; coverage is DoCA recall.

**all-forms, US-restricted 1960–1975:**

| logit ≥ | flagged | reweighted P | ~true events | DoCA recall |
|---|---|---|---|---|
| 0.0 | 31,707 | 0.545 | ~17,300 | 0.608 |
| 0.5 | 17,262 | 0.626 | ~10,800 | 0.469 |
| **1.0** | 9,722 | 0.822 | **~8,000** | 0.351 |
| 1.5 | 5,519 | 0.805 | ~4,400 | 0.257 |
| 2.0 | 3,162 | 0.857 | ~2,700 | 0.185 |

So at logit ≥1.0 over 16 years of US articles, we'd surface ~9,700 candidates, ~8,000 of
them real, catching about a third of the DoCA-style events present. Lower the threshold
to ~17,300 true events at the cost of precision (0.55). The internal arithmetic is
consistent: ~8,000 true at 0.35 recall implies ~22,800 events in this slice, a ~1.4%
base rate — close to the π≈0.02 prior.

The street model flags somewhat more (11,522 at logit ≥1.0) because its scores run
higher, but its "true events" count means true *street* events, a different (smaller)
target — the two yield columns aren't directly comparable.

## Definition comparison: street-only is the cleaner street detector

Evaluated on the same street task (144 street gold positives, both models leakage-clean),
the street-trained model beats the all-forms model as a street detector: +0.025–0.086 raw
precision at matched recall, and 0.770 vs 0.737 / 0.868 vs 0.778 reweighted precision at
logit 1.5 / 2.0. It does this with *fewer* training positives (9,277 vs 13,742), so it's
label cleanliness plus hard negatives (the non-street DoCA events as presumed-negatives),
not data volume. If street events are the target, restricting the definition helps. If
broader collective action is the target, the all-forms model is the one — at a precision
cost the street comparison quantifies.

## Limitations

These bound how far to trust the numbers above:

- **Gold-set resolution.** 144–169 positives means precision has a standard error of
  ~0.03–0.05. Differences smaller than that are noise. This caps how finely we can tune
  or compare, and it's the strongest argument for coding more of the gold set (2,553
  drawn, 500 coded, and the coded set is street-dominated).
- **Precision transfer.** The yield projection multiplies 1960–1975 flagged counts by
  precision estimated on the gold set, which isn't drawn exclusively from those years. It
  assumes precision is stable across the corpus. The high-threshold estimates are the
  most robust (raw and reweighted precision agree there); the low-threshold ones are
  softer.
- **US-filter coupling.** The pipeline runs the US filter first, then CCA. Face-validity
  review found the US gate is too loose (foreign protests leak through), so the
  full-corpus yield will tighten once the US filter is hardened. Treat the projection as
  provisional on that.
- **Frozen-encoder ceiling.** Everything here is a head trained on fixed DAPT-CLS
  features. The biggest quality gains likely need encoder fine-tuning, which is out of
  scope for the current cached-features regime.

## Out-of-sample test (planned)

The strongest evidence is still missing: does the model hold up out of its training
period and source? The plan is to embed the **LDC corpus, 1995–2007** — out of the
training period (the DoCA overlap is ~1987–1995) and a different source than the API
corpus — and evaluate against the NYT indexer's CCA descriptors as a noisy reference.
That tests temporal and cross-source generalization. It's the one piece that needs new
compute (a frozen-encoder embedding pass over LDC 1995–2007), scheduled to run overnight
in place of the API part-2 embed. The indexer tags are noisy — they're what we're trying
to improve on — so read it as a "does it fall apart out of period" check, not ground
truth.
