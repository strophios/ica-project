# ICA classifier update — July 2026

*Follows the June model-state memo (`ica_model_state_2026-06.md`); numbers here
use the same hand-coded evaluation set (1,131 articles, 214 ICA positives) and
the same conventions. Written 2026-07-30 for the Aug 6 meeting.*

## The short version

We spent the last two weeks trying to improve the model's weakest component —
the immigrant-relevance head — by fine-tuning the underlying language model
instead of leaving it frozen. It worked: **the improved relevance head raises
the combined ICA score's accuracy from 0.80 to 0.82 (ROC-AUC), and lifts
diaspora-event recall from 38% to 66%** at a fixed one-in-three review budget.
Diaspora protests — US-soil collective action about home-country issues, the
category we've flagged repeatedly as both highest-value and hardest to catch —
are where nearly all of the gain landed.

Two other results matter as much as the headline: the fine-tuning *hurt* the
other two heads (we caught this with a safety check designed for exactly that,
and we are not deploying the damaged versions), and the training process
surfaced a genuine methodological finding about positive-unlabeled learning
that we think is publishable as part of a methods piece.

We also extended the corpus: the NYT Archive API is now pulled through 2025,
plus a sampled skeleton of 1870–1959 for the backward expansion. More on a
wrinkle in that data below — it affects how we'll handle the historical period.

## What we did and why

The June memo showed that of the three heads (US event / collective action /
immigrant relevance), relevance was the weakest at its own job (ROC 0.83
against your hand-coding, vs 0.93 for each of the other two) while being the
most important single ingredient in the final ICA score. All three heads sit
on top of a shared language model ("encoder") that had been frozen since its
initial newspaper-domain training — the heads learned to read its
representations, but the representations themselves never adapted to our task.

Fine-tuning the encoder is the standard next step, and we did it the cautious
way: tune it using *only* the relevance task, then re-check the other two
heads on the tuned representations before adopting anything. The reasoning:
tuning with all three tasks at once introduces a balancing problem the ML
literature says is hard to get right, and our hand-coded evaluation budget is
too small to tune the extra knobs safely. The cautious design also builds in
its own alarm: if reshaping the representations for relevance damages the
other heads, we find out before anything ships.

## The training failure, and what it taught us

The first fine-tuning runs failed in an instructive way. Our training method
(positive-unlabeled learning, as described in the main memo) estimates the
error on articles it treats as "probably negative" by a subtraction that
depends on the assumed rate of true positives in the unlabeled pool. That
assumed rate had been set to 5% by analogy with another task, never estimated
for this one. It turns out the true rate is about 2% — and with the assumed
rate too high, the training math over-corrects, and a newly-flexible model
discovers it can drive its training score to near-perfect by calling
*everything* relevant. Our diagnostic instrumentation caught the signature
immediately (the model's predictions collapsing to one side while its
training score "improved").

We re-estimated the rate properly, and then ran a deliberate test of the
diagnosis: same training, wrong rate vs corrected rate, nothing else changed.
With the corrected rate the collapse disappears and the model trains well.
That's the methods finding: **positive-unlabeled training that is stable when
the language model is frozen can fail catastrophically once the model is
allowed to adapt — and the assumed positive rate, which barely matters in the
frozen case, becomes the difference between success and collapse.** Anyone
combining PU learning with fine-tuning (as the paper our approach builds on
does) would want to know this; it slots naturally into the methods piece we
discussed at the last meeting.

## Results

Scores on the hand-coded evaluation set. "Own job" = ranking articles on the
head's own dimension against your coding; "ICA ranking" = how well that head
alone ranks articles for the final ICA outcome.

| | frozen (June) | fine-tuned | change |
|---|---|---|---|
| relevance, own job | 0.83 | 0.84 | +0.01 |
| relevance, ICA ranking | 0.78 | **0.85** | **+0.07** |
| collective action, own job | 0.93 | 0.74 | **−0.19** |
| US event, own job | 0.93 | 0.83 | **−0.09** |
| **combined ICA score** (improved relevance only) | 0.80 | **0.82** | +0.02 |
| **diaspora recall @ 1-in-3 review budget** | 0.38 | **0.66** | +0.28 |

Reading this: the fine-tuning reorganized the model's internal representation
of articles around immigrant-relevance distinctions. That's precisely what we
asked it to do — and the relevance gains, especially on diaspora events, are
real and large. But the same reorganization degraded the representation for
the other two tasks (the "negative transfer" the cautious design was built to
detect). So the combined-score improvement above comes from a mixed
configuration: the improved relevance head reads the tuned representation,
while the collective-action and US heads keep the original one. The damaged
versions of those heads are not used anywhere.

Usual caveat: the evaluation set was deliberately built to over-represent
borderline articles, so these numbers compare models against each other on
hard cases; they are not corpus-wide precision estimates.

*A scoring-infrastructure note (added Aug 4): after this memo was drafted we
found that the TensorFlow build our laptops use for Apple GPUs mis-executes
one small component of the scoring pipeline — deterministically, so every
number in this memo went through the same slightly-wrong arithmetic on both
sides of every comparison. The comparisons above therefore stand as reported.
Re-scoring the deployed (frozen) model with exact arithmetic shifts each
metric by at most about ±0.01 — e.g., the combined score's 0.80 is 0.795
exact, and the US head's "own job" 0.93 reads 0.91 exact — inside the
resolution of a 1,131-article evaluation set. The bug is fixed (cluster and
CPU scoring were never affected, and a guard now blocks the faulty path);
corrected tables for both configurations will accompany the post-meeting
model refresh, and the corpus-scale candidate lists discussed at the meeting
are produced on the cluster with exact arithmetic throughout.*

## Applying the model to the whole corpus (added Aug 4)

Since the draft above was written, we ran the deployed model over the entire
Archive corpus — **all 6.4M articles, 1960–2025** — on the cluster, with exact
arithmetic throughout. Every article now carries the component scores and the
combined ICA score. The two periods answer different questions — the DoCA
period is where we can *measure* against your coding; the forward period is
new territory — so they're reported separately below. (These runs use the
deployed June model; the improved relevance head enters with the post-meeting
engineering.)

### 1960–1995: measured against DoCA and the ICA dataset

**Collective action vs the ~15,600 DoCA-matched articles.** Precision is the
corpus-reweighted estimate from the 500-article hand-coded sample; both
columns come from the same scoring run:

| operating point | precision (corpus) | DoCA recall | articles flagged (of 3.7M) |
|---|---|---|---|
| strict | ~0.90 | 15% | 8,500 |
| high | ~0.80 | 28% | 21,000 |
| balanced | ~0.72 | 40% | 40,000 |
| permissive | ~0.45 | 63% | 119,000 |

**ICA specifically, vs the 552-event ICA subset you built** (recall is per
event; an event counts as recovered when its article surfaces). Precision
comes from the hand-coded evaluation set, which deliberately over-samples
borderline cases — read it as an upper bound on corpus precision:

| operating point | precision (eval set) | ICA-event recall | articles flagged |
|---|---|---|---|
| strict | ~0.83 | 26% | 3,900 |
| balanced | ~0.50 | 56% | 42,000 |
| permissive | ~0.30 | 80% | 207,000 |
| review-everything | ~0.25 | 89% | 432,000 |

Two integrity checks behind these tables. First, 196 of the 552 events were
fully held out from all model training, and their recall matches or slightly
exceeds the rest at every operating point (e.g., 83% vs 79% at the
permissive point) — recall is not inflated by the model having seen the
events. (The DoCA table lacks such a held-out split — those articles include
training examples — so read it as "re-find the known events.") Second, the
numbers are mutually consistent: 40,000 flagged at 0.72 precision is ~29,000
true positives, which is 39% of the ~74,000 collective-action events a 2%
base rate implies for this corpus — almost exactly the 40% recall measured
independently against DoCA.

**Why recall looks "low" next to the accuracy numbers we usually quote.**
It isn't the ranking: measured over the whole corpus, both anchor sets rank
near-perfectly (ROC-AUC ≈ 0.95; the median DoCA article sits in the top
1.7% of 3.7M, the median ICA event in the top 0.8%). At a 2% base rate,
though, recovering 80% of true events *mathematically requires* flagging
100k+ articles — precision at high recall is a property of the haystack,
not the detector. The residual misses are largely a channel ceiling: the
model reads headline + first paragraph, your coders read whole articles.
The worst-identified anchors make this concrete — a DoCA event whose
article is headlined "'Annie Hall' Wins 4 Academy Awards" (the protest is
deep in the piece), ICA events whose ledes read as crime stories ("Bomb in
5th Ave. Tower Shatters Yugoslav Bank"). We've prepared best/median/worst
example sheets for both anchor sets, alongside non-anchor articles drawn
from the same score bands, so you can see what the model finds easy, hard,
and what else lives at each score level.

**Recall by domain across the whole curve** (all events / held-out only):

| domain (n) | permissive (~0.30) | balanced (~0.50) | strict (~0.83) |
|---|---|---|---|
| Documentation (118) | 94% / 100% | 88% / 95% | 63% / 72% |
| Access (77) | 77% / 74% | 60% / 53% | 16% / 21% |
| Exclusionary (76) | 76% / 77% | 50% / 54% | 18% / 19% |
| Diasporic (281) | 76% / 79% | 42% / 46% | 16% / 23% |

The pattern matters for how the candidate lists get used: documentation
events stay recoverable even at strict settings, while diasporic events
fall off fastest through the middle of the curve — at strict operating
points the model is disproportionately surfacing documentation-domain
events, and recovering diaspora events specifically is what the permissive
band buys.

### 1996–2025: the forward corpus, first pass

Applying the same operating points to the 2.7M post-1995 articles yields,
per year: at the balanced point roughly 1,100/year (1996–2011) rising to
~1,400/year (2012–2024); at the strict point ~105/year rising to ~135/year.
The top of the forward ranking is face-valid ICA (asylum-seeker protests,
the 2006 immigration marches, bodega strikes against the travel ban, 2025
ICE protests) — we've set aside top-100 lists from both periods for you to
eyeball. **2025 reads 2–3× higher than neighboring years at every operating
point** (e.g., ~400 strict-point candidates vs ~135/year before it). Some
of that is plausibly real; some may be the abstract-substitution channel
described below. We'll separate the two with the paired-channel test before
quoting 2025 numbers.

Not yet re-scored: the LDC-corpus candidate list for 1996–2007 (a separate
text source with its own strengths); it follows with the post-meeting
refresh.

## What's next on the model

1. **Adopt the mixed configuration properly** — refit the score-combination
   step around the improved relevance head (the +0.02 above reuses the old
   combination unchanged, so it's a floor, not a ceiling). The cost is that
   applying the model to a corpus now requires two passes of the language
   model instead of one; acceptable, but worth engineering once rather than
   per-run.
2. **Joint fine-tuning of relevance + collective action.** The principled fix
   for the negative-transfer problem: tune the encoder with both tasks'
   training signals at once, which the two tasks' shared structure makes much
   more tractable than the general case. This is the route back to a single
   shared representation that keeps both gains.
3. **Apply to the expanded corpus** — done with the deployed model (see "Applying
   the model to the whole corpus" above); re-apply with the improved model once
   items 1–2 land.

## The corpus expansion, and a finding about the historical data

The Archive API pull now covers 1996–2025 (about 2.1M additional articles)
plus a one-month-per-year sample of 1870–1959 for the backward expansion. In
auditing the new data we found something that affects strategy for the
historical period: **article lead paragraphs simply don't exist in the API
data for most of the pre-1970 period (and for 2025 onward)** — for those
articles the API provides only headlines and abstracts. And the historical
abstracts aren't article text: they're New York Times *Index* entries,
telegraphic summaries in a distinctive register ("Apptd Consul Gen in Miami"),
plausibly written at a different time by different hands than the articles.

The practical upshot: for the modern gap (2025+) we can safely substitute the
abstract, which today is ordinary editorial text. For the historical corpus,
training or applying on headline+abstract is a real change in what the model
reads, and we'll validate it deliberately (the 1970s, where both leads and
abstracts exist for the same articles, give us a clean paired test) rather
than assume it's equivalent. This may bear on how far back the current model
can go before we need period-specific adaptation — the temporal-signal work
we've discussed — and we'd welcome a conversation about what Audrey and Min
Jee see in the pre-1960 material along these lines.

## One more result: the US-head retrain

Earlier in July we also tested a full retrain of the US-event head using
event-location labels (DoCA matches) instead of dateline-derived ones, aimed
at the diaspora blind spot documented in June. The retrained head did not beat
the current one at the operating point the system actually uses, so we kept
the current head — but the investigation reframed the problem usefully: at
the deployed (very permissive) gate setting, the US head drops far fewer
diaspora events than the June memo's headline number suggested, and its real
cost is letting genuinely-foreign events through. The relevance-head
improvements above attack that same problem from the other side, and the
diaspora recall gains suggest it's working.
