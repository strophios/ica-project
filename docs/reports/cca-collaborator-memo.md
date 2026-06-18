# Detecting collective-action events in the NYT, 1960–1995: preliminary results

*2026-06-18. A classifier that flags New York Times articles reporting a collective-action
event, trained and evaluated on the DoCA period. This covers the
contentious-collective-action (CCA) detector only — one component of the planned
immigrant-collective-action (ICA) system. It is preliminary in that narrow sense: it is a
complete, evaluated CCA classifier, not a teaser, but it does not yet do the immigration
half of the eventual task.*

The point of this memo is not to sell the model. It's to lay out plainly what it does, how
we measure it, and — most of all — the conditions under which its numbers mean what they
appear to mean. Where you can trust it, and where you can't, both follow from the
measurement design, so that's where most of the space goes.

## What it is

The model reads an article's headline and lead paragraph and outputs a score for "this
article reports a collective-action event." It runs on the full NYT Archive API corpus for
1960–1995 — the DoCA period — at headline+lead granularity, which is what the NYT API
provides freely and what makes eventual extension to the full archive feasible.

Two design facts shape everything downstream:

- **Positives come from DoCA, not from keywords.** Earlier versions labeled articles using
  the NYT's own subject tags ("demonstrations and riots," "strikes," and so on). Those tags
  are over-generous — they catch a lot that isn't a reported event. This version instead
  treats an article as a confirmed positive only when it matches a DoCA event (1960–1995),
  via headline-and-date matching, at roughly an 80% match rate. So the positive labels are
  events you already trust.
- **We have confirmed positives but no confirmed negatives.** An article we haven't matched
  to a DoCA event might still report one — DoCA isn't exhaustive, and the match isn't
  perfect. So "unlabeled" is not "negative." We train with positive-unlabeled (PU) learning,
  which treats the unlabeled pool as a mix of hidden positives and true negatives at an
  estimated rate, rather than assuming everything unmatched is a non-event. This matters for
  honesty: it's the difference between "the model is wrong" and "the model found something
  DoCA missed," and we don't want to punish the second as if it were the first.

The encoder is a RoBERTa language model adapted to NYT headline/lead text; the trainable
part on top is a small classifier head. The background pool is restricted to US-reporting
articles by a separate US/not-US filter, because DoCA events are US events and we don't want
the model learning "US-ness" as a proxy for "protest." That filter is now properly trained
(F1 0.97 held-out) and calibrated; an earlier draft of these numbers used a weaker version,
and hardening it changed the results negligibly (see Limitations).

## How we measure it, and why the numbers mean what they mean

This is the part worth reading slowly, because a precision number is only as trustworthy as
the set it's computed on.

**The gold set.** We hand-coded 500 articles for whether they actually report a
collective-action event. These are the ground truth the metrics below are computed against —
not the DoCA tags, which are themselves what we're trying to improve on.

**Why raw precision on that set would mislead.** We didn't draw the 500 uniformly. A uniform
draw from the corpus would be ~2% events, so almost every coded article would be a negative
and we'd learn almost nothing about precision. Instead we oversampled high-scoring articles,
so the coded set runs ~34% positive. That oversampling means raw precision on the gold set
is inflated relative to what you'd see across the whole corpus — for any classifier.

**Reweighting recovers the real operating point.** We correct the oversampling with
inverse-probability weights: each coded article is weighted by how much we over- or
under-sampled its score band, which reconstructs corpus-level precision. So in the tables
below, **reweighted precision is the number you'd actually see in deployment**; raw precision
is kept only because it's a fair way to *compare* models (the inflation is constant across
them). The two converge at high score thresholds, which is why the high-threshold numbers are
both the most useful and the most stable.

**Recall is measured separately, on clean data.** Recall on a score-stratified set is biased,
so we don't compute it there. Instead we report recall over held-out DoCA positives — known
events the model never trained on. That's the trustworthy "of real events, how many does it
catch."

**No leakage.** The 500 coded articles, and the larger pool they were drawn from, are held
out of training, so the model is never scored on text it learned from.

The short version: trust the **reweighted precision** and the **DoCA recall**; read raw
precision as a model-to-model comparison only; and trust the high-threshold numbers most.

## Results

Operating points for the main (all-forms) model, from the gold set. Pick a threshold by
whether you want precision or coverage; they trade off the usual way. The model outputs a
score; we also calibrate it to a probability, so each operating point is given both ways.

| operating point | reweighted precision | DoCA recall | as calibrated prob |
|---|---|---|---|
| broad (score ≥ 1.0) | 0.68 | 0.44 | P ≥ 0.58 |
| **default (score ≥ 1.5)** | **0.79** | **0.33** | **P ≥ 0.73** |
| strict (score ≥ 2.0) | 0.83 | 0.24 | P ≥ 0.84 |

At the default operating point: ~79% of the articles it flags genuinely report a
collective-action event, and it catches ~33% of DoCA-style events. Lower the threshold to
catch more at lower precision; raise it for the reverse.

**Against baselines**, both measured in corpus terms:

| detector | corpus precision | lift over base rate |
|---|---|---|
| random (the base rate itself) | 0.02 | 1× |
| protest-keyword lexicon | 0.19 | ~9× |
| this model (default) | 0.79 | ~40× |

A keyword lexicon (`protest`, `strike`, `march`, `boycott`, `riot`, …) beats chance by ~9× —
lexical signal is real. The model beats the lexicon by another ~4×, and reaches a precision
the lexicon can't, because in the real corpus most keyword hits are false: `strike` in
sports, `march` the month, `rally` in markets.

## Out-of-sample: does it hold up on new data?

The strongest test is whether the model works outside the period and source it trained on.
We scored it on the **LDC corpus, 1995–2007** — a later period *and* a different NYT data
source than the 1960–1995 API corpus it trained on — and checked it against the NYT's own
`cca_descriptor` tag. (The tag is the noisy thing we're improving on, so read this as a
"does it fall apart out of period" check, not ground truth.)

It holds up well. The model's ranking achieves **ROC-AUC 0.89** against the tag (0.90 for the
street model) — the signal transfers cleanly across both the decade gap and the source change.

Run as the real pipeline (US filter → CCA) over all 681,470 LDC articles from 1995–2007:

- 568,000 pass the US filter (83%).
- At a **discovery threshold** (broad, score ≥ 1.0), the model flags **4,524 US
  collective-action events**, ~3,100 of which carry **no** NYT CCA tag — candidate events the
  tagging missed. At the high-precision default it flags 2,438, ~1,373 untagged.

That is the dataset-expansion case made concrete: thousands of candidate US events beyond
what the NYT's own tagging captured, in a single 12-year window, at a known precision. A
sample of what it surfaces, all real events spanning the window:

- *46,000 March on South Carolina Capitol to Bring Down Confederate Flag* (2000)
- *Jesse Jackson and 18 Others Are Arrested in Yale Protest* (2003)
- *In Downtown New Orleans, Thousands March Against Killings* (2007)
- *Violence at Demonstrations on Immigration* (1996)
- *Sit-In at Princeton in President's Office* (1995)

And two it scored highly that the NYT tag *missed* — the clearest evidence it adds coverage
rather than just reproducing the tags: *A Week of Abortion Protests in Buffalo Begins Loudly
but Peacefully* (1999) and *Protesters at King March Oppose Air Force Flyover* (2006).

## A note on event types: all forms vs. street only

Everything above uses the **all-forms** definition: any DoCA collective-action event counts —
demonstrations, strikes, boycotts, conventional protest, and lawsuits alike. This is the main
model, and the one to read as the result.

There's a second option worth putting on the table for discussion. If we restrict the
definition to **street events only** (demonstrations, marches, rallies, pickets, riots) and
treat the other forms as non-targets, the resulting model is a meaningfully cleaner detector
*of street events* — higher precision at matched recall, achieved with fewer training
positives, because the non-street forms carry more of the labeling noise. By event type, the
all-forms model already catches street events best (recall ~0.90) and is weakest on lawsuits.

I'm raising this as a question, not a recommendation. The all-forms model is the right
default, and non-street forms were part of what mattered when we built the ICA subset of DoCA
for 1960–1995. But if street events are the analytic priority, scoping the detector to them
buys a cleaner instrument, and it's worth deciding deliberately rather than by default.

## How far to trust it (limitations)

- **Probabilities are conditional on the labeling.** Because of the PU setup, a calibrated
  probability should be read as "evidence of an event relative to the DoCA-confirmed
  positives, in corpus proportion," not a standalone physical probability. The score
  thresholds are the safer thing to operate on; the probabilities are there for combining
  scores in the eventual multi-head model.
- **Gold-set resolution.** With 169 coded positives, precision has a margin of roughly
  ±0.03–0.05. Differences smaller than that are noise, and coding more of the gold set is the
  main way to tighten it.
- **Frozen encoder.** The language model underneath is fixed; only the small head is trained.
  This is what makes training and tuning cheap, but it also caps quality — the largest future
  gains likely require fine-tuning the encoder, which is planned, not done.
- **The US filter is now hardened, not provisional.** The earlier version of these numbers
  used a lightly-trained US filter; the properly-trained one (F1 0.97, calibrated) is now in
  place. It tightened the gate slightly — dropping foreign protests that had leaked into the
  background — and the CCA results were unchanged within noise, so the figures above already
  reflect it.

## What's next

1. The immigration detector, then the combined immigrant-collective-action model — the full
   ICA system this CCA detector is the first head of.
2. Encoder fine-tuning, the main lever left for quality.
3. A larger hand-coded gold set, to tighten the precision estimates and let us compare the
   all-forms and street definitions more finely.
