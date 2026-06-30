# ICA classifier — current state and what it gets us

*2026-06-26. A diagnostic snapshot of the multi-head ICA classifier as it stands,
for collaborators. Companion to the full methods memo (`ml_memo.qmd`); this one is
just the numbers and what they mean for using the model now. The model is useful
but not finished — see "what's next" for the ceilings we're hitting.*

## What the model is

A multi-head classifier over NYT headline+lead text. A frozen domain-adapted
RoBERTa encoder feeds three independently-trained, separately-calibrated heads:

- **US gate** — is this a US event? (a pre-filter, not a ranking signal)
- **CCA** — is this collective/contentious action? (DoCA-trained)
- **Relevance** — is this immigrant-relevant?

A row is scored ICA by gating on US, then combining the calibrated CCA and
relevance probabilities (the product, AND, beat a learned logistic combiner on a
held-out comparison), with a final calibration step. One composed `ica_score` in
[0,1] comes out the back.

We evaluate on a hand-coded set of 1,128 articles (214 hand-judged ICA events),
deliberately drawn near the model's decision boundary plus confirmed DoCA anchors —
so it is **enriched** for hard cases (19% positive), not a random corpus sample.
That matters for reading the numbers below.

## How well it discriminates

On the hand-coded eval, the composed `ica_score` separates ICA from non-ICA at:

- **ROC-AUC 0.80** (this is base-rate-independent, so it's the most portable
  single number — the model genuinely tells ICA from non-ICA).
- **PR-AUC 0.61** (on the 19%-enriched eval).

Decomposed by head (ROC-AUC vs the ICA label):

| Head | ROC-AUC | Reading |
|------|---------|---------|
| Relevance | **0.78** | the workhorse — does most of the ICA discrimination |
| CCA | 0.62 | adds signal; the weaker ranker (likely a frozen-encoder ceiling) |
| US | 0.38 | *below chance as a ranker* — by design |

The US head scoring below 0.5 as a *ranker* is not a defect: ICA is not monotone in
US-ness. Many non-ICA articles are strongly US (US protests that aren't about
immigrants), and some true ICA scores low on US (diaspora protests — more below).
That's exactly why US is used as a gate, not a fusion feature.

## Operating points (on the enriched eval)

Reading the precision/recall trade-off off the composed score:

| If you want… | you get… |
|---|---|
| precision 0.90 | recall 0.14 |
| precision 0.80 | recall 0.29 |
| precision 0.70 | recall 0.42 |
| precision 0.60 | recall 0.54 |
| recall 0.50 | precision 0.63 |
| recall 0.70 | precision 0.41 |

By false-positive rate: TPR (recall) is **0.45 at FPR 0.05**, **0.58 at FPR 0.10**,
0.68 at FPR 0.20.

**Important caveat for these precision numbers:** the eval is boundary-enriched, so
its precision *overstates* what you'd see on the raw corpus, where true ICA is far
rarer and each candidate competes against many more true negatives. The recall and
ROC numbers travel better than the precision numbers. A random-sample gold set
(none exists yet) is what we'd need to pin absolute corpus precision/recall — that's
a known open item.

## Calibration

The composed score is well-calibrated **to the eval population** (ECE 0.039, Brier
0.113). Reliability:

| predicted ica_score | n | empirical ICA rate |
|---|---|---|
| 0.0–0.2 | 693 | 0.08 |
| 0.2–0.4 | 259 | 0.18 |
| 0.4–0.6 | 113 | 0.51 |
| 0.6–0.8 | 52 | 0.81 |
| 0.8–1.0 | 11 | 1.00 |

Monotone and close to the diagonal (slightly over-confident in the 0.2–0.4 bin).
Caveat again: this is calibrated to the eval's ~19% base rate, not the corpus's much
lower one, so on the full corpus `ica_score` behaves as a strong *ranking* score
rather than an absolute event-probability.

## What it gets us on new data (LDC 1996–2007)

This is the payoff — applying the model to 624,842 NYT articles from 1996–2007, a
period *outside* the original DoCA window (the expansion test). The model ranks them;
candidate counts by threshold:

| ica_score ≥ | candidates | % of corpus |
|---|---|---|
| 0.3 | 4,320 | 0.69% |
| 0.5 | 860 | 0.14% |
| 0.7 | 137 | 0.02% |

So from ~625k articles, the model surfaces a few hundred to a few thousand ranked
candidates to review — a tractable haystack reduction for human coding. US gating
was gold-first where a dateline label exists (56.5% of rows) and fell back to the ML
head otherwise (43.5%).

Face validity at the top is strong. The highest-scoring 1996–2007 candidates:

- "Asylum-Seekers Are Confined To Dormitories After Protest" (0.95)
- "Latinos Protest in California In Latest Immigration March" (0.93)
- "Demanding Parole, Immigrants Held in Queens Stage Protest" (0.89)
- "Detained Immigrants Still Not Eating in Protest"
- "Across the U.S., Protests for Immigrants Draw Thousands" (the 2006 wave)
- "Hunger Strike by 6 Immigrants Enters 2nd Week"

On the 1960–1975 API slice (also scored) the top candidates are the harder diaspora
cases — Soviet Jewry, Cuban exile, and Greek Cypriot marches — which is encouraging
because those are the events the field most often misses.

## Where the quality is bounded, and what's next

The model is useful now, but it's hitting a few ceilings we know how to lift:

1. **US-head diaspora recall.** The US head misses diaspora/solidarity protests
   (US-soil action about foreign topics) — the highest-value ICA category — because
   their text is foreign-topic-heavy. Gold dateline labels rescue these where they
   exist, but novel ones slip the gate. The fix is a **US-head retrain** on event-
   location labels (DoCA matches + section heuristics) rather than dateline labels.
   This is the single biggest near-term quality lever.

2. **Frozen encoder.** Every head sits on a frozen encoder. CCA's modest 0.62 ROC is
   the likely symptom. **Unfreezing the top encoder layers** during head training
   should lift discrimination, CCA especially.

3. **No temporal signal.** The encoder has no notion of *when* an article was
   written. We plan to fold **time tokens into the domain-adaptive pretraining**,
   then thread them through the heads and retrain (with a partially-unfrozen
   backbone). That's the larger build that lets us apply confidently both forward
   (post-2007) and backward.

A rough ceiling guess: relevance already reaches 0.78 ROC frozen, and CCA is the
laggard. With the US retrain (recall) + an unfrozen top + time tokens, lifting the
composed ROC from 0.80 toward the mid-to-high 0.80s seems plausible — enough that the
binding constraint shifts from the model to the label definitions and the absence of
a random-sample gold set. That's a guess, not a promise.

## One-line summary

A calibrated multi-head classifier that separates ICA from non-ICA at ROC-AUC 0.80,
turns ~625k out-of-period articles into a few hundred high-quality ranked candidates
with strong face validity, and has three clear, already-scoped levers (US retrain,
unfrozen encoder, time tokens) before we'd call it done.
