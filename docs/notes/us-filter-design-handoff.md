# US/not-US Pre-Filter — Design Handoff

*Created: 2026-06-04. Status: mid-design, paused in Phase 1 (Context Gathering) of the `starting-a-design-plan` process. This is a warm handoff to resume in a fresh session.*

## What this is

We're designing a **US/not-US pre-filter** for NYT headline+lede articles. Its job is to filter the corpus down to articles reporting on events happening in the US, as a preprocessing step before the main model. This unblocks two downstream things and serves a third:

1. The **corrected-prior (π≈0.02) single-head CCA retrain**.
2. **Immigration label-definition refinement** (immigration is a global topic, so filtering to US first removes a lot of foreign-immigration noise before keyword ID).
3. Eventually, the **durable production filter** as the corpus expands forward (post-1995/2007) and especially *backward* into the deep NYT archive (toward 1851).

This started as a label-definition-refinement task but became its own thing. Note the broader change in the project: the user used the NYT API to download all headlines+ledes 1960–1995 (full DoCA coverage) and matched ~85% of DoCA events into NYT data, yielding **~15k CCA positives directly from DoCA**. That obviates index-term CCA labeling. Immigration still needs keyword ID. US/not-US filtering is the current bottleneck.

## The decision we worked through

Three options were on the table:

1. **Ship the metadata heuristic** as-is (accept its error rates).
2. **US/not-US as a classification head** in the main multi-head model (no hard pre-filter).
3. **Train a separate model** as the filter.

### Where we converged

**Split by consumer, because the two downstream consumers differ in error-sensitivity and urgency:**

- **CCA retrain → ship the conservative heuristic now.** CCA is *filter-insensitive*: its positives are DoCA-matched and therefore US by construction, so the filter only thins the ~1M-article unlabeled pool, where dropping some true-US articles is near-costless under SCAR. So a conservatively-thresholded heuristic (assume-US default) unblocks the retrain immediately at low risk.
- **Immigration + production → build the probe-first text classifier.** This is the *real, durable* filter.

### Why option 2 (head) was rejected

- **Sequencing/dependency:** the filter is a *precondition* for assembling the main model's training data. A head inside that model can't be a precondition for building its own training data — circular.
- **It doesn't solve the actual problem,** which is *label scarcity*, not where the prediction lives. A head inherits all the heuristic's label noise plus entangles US-classification with the FLPU/PU CCA training dynamics.
- The user clarified option 2's live form is really the question "should we hard-filter at all, vs. let the model see everything and learn US-ness as a soft signal?" We judged **hard pre-filter is right**, because: (a) the irreversibility concern (a hard gate's false-negatives are unrecoverable) is muted for CCA since positives are US by construction and the filter only touches the unlabeled pool; (b) filtering removes foreign-protest **hard negatives** that otherwise risk teaching the PU model "protest → positive" and lighting up on foreign demonstrations at inference; (c) it aligns train/inference distributions (deployment target is US-only). Use a *conservative* threshold to protect recall.

### Why the text model (option 3) is not a detour

- **The heuristic's failure mode is missing metadata, but the text still carries the signal** — the heuristic only reads keywords/section/desk, not the headline+lede. A text model accesses a different, richer source, which is why it breaks through the diminishing-returns wall.
- **Production expansion is the decisive argument (user-raised).** Filter quality is correlated with metadata density, which *falls off exactly where the user wants to expand* (backward into the archive — more articles listed only as "Archives", no news desk). So a metadata heuristic is structurally a training-time crutch, not a production filter. The text model is needed eventually regardless.
- **Cheap to build:** reuse the existing `src/prior_estimation/lu_classifier.py` scaffolding (frozen DAPT backbone + single Dense layer) — same architecture, different labels. US/not-US from headline+lede is far more separable than the PU task.
- **Even pre/post-1986 split** of the ~15k positives means the metadata-poor zone holds ~half the positives, so the model earns its keep at *training* time too, not just production.

### The probe-first decision (locked)

Start with a **linear probe on frozen DAPT features**; escalate to light fine-tuning (unfreeze top layers) **only on evidence** from the validation audit — specifically if the probe underperforms in the pre-1986 zone. Escalate on evidence, not anticipation.

## Key technical pieces of the converged approach

- **Data sources (two, no shared ID):**
  - **NYT Archive API**: 1960–1995, full DoCA coverage. Has bylines but **no consistent datelines** (the Archive API spec doesn't mention them). Has its own keyword/section/news-desk metadata.
  - **LDC NYT Annotated Corpus**: 1987–2007. **Has datelines.** Separate keyword system (created by a different group, postdates API system); keywords often don't fully overlap with the API's.
  - **No shared article ID** between them. Join must be **fuzzy headline + date matching**. Date brackets the search hard (within a day → dozens of candidates, not millions), so high-precision matching is feasible. The user expects API↔LDC headline matching to be *better* than the DoCA↔newspaper matching (which was painful because the "correct" headline wasn't always obvious from the newspaper page), but still imperfect. **We only need partial, high-precision join recall** (~40–60% is plenty) for training labels + audit — ambiguous matches can be discarded.

- **Datelines as a LABEL source, not an input feature.** Derive labels from 1987–1995 datelines (via the join); train the text model on headline+lede with any inline dateline **stripped** so the input distribution matches the dateline-less pre-1986 API data; apply the model backward to 1960–1986. Bet: US-vs-not-US text cues (Washington, Congress, foreign capitals, agency names) are era-stable enough to transfer; DAPT backbone helps. *Note:* for 1987–1995 you can often just use the LDC dateline directly via the join — the model is really needed for **1960–1986** (the metadata/dateline-poor zone).

- **Dateline → US/not-US conversion:**
  - **Exploit NYT dateline style convention first:** smaller US cities get a state abbrev ("VANCOUVER, Wash."); foreign cities get a country; major US cities and world capitals stand bare ("WASHINGTON", "LONDON"). Rule: *has-US-state-abbrev → US; has-country-name → foreign; bare → look up in major-US-cities + world-capitals lists.* This resolves a large fraction before population-ranking is needed, and often pre-solves the Vancouver-WA-vs-Vancouver-BC problem via the ", Wash." already in the dateline.
  - **GeoNames gazetteer** (`cities500`/`cities15000` dumps: name, country code, population, alternate names) for the genuinely ambiguous bare cases; population-ranking disambiguates (Vancouver, Paris TX vs France, London ON vs UK). Collapse to country==US for the binary. **Reusable** for immigration location work and for parsing the thin location keywords — raises its ROI.

- **Validation gets cheaper than feared:**
  - The LDC↔API join gives a **free audit** of the current heuristic over 1987–1995 (real error rates, no hand-labeling). Caveat: it's "error rate on cleanly-joinable articles," so mildly biased by whatever makes articles joinable (headline quirks, era) — note, don't block on it.
  - The only thing it can't validate is the **era-transfer into 1960–1986**, so a **small, pre-1986-focused hand-labeled slice** (a few hundred, stratified by section/era) is the one piece of ground truth that can't be obtained otherwise. Piggyback it on the CCA validation set the user is building anyway.

## Nuances the user raised (don't lose these)

- **Sports/arts aren't uniformly irrelevant.** Players' union strikes, sports integration fights, arts free-speech/censorship contention are real collective action. BUT under assume-US the error runs in the **benign direction**: assume-US *keeps* ambiguous sports/arts articles in the pool (protective for the relevant tail), and since CCA positives are DoCA-US by construction, this tail can't cost positives — it can only add a little foreign noise (a foreign footballers' strike wrongly kept as US, a hard negative). So the tail doesn't threaten recall; it argues for keeping assume-US rather than getting clever and dropping low-metadata sports/arts.
- **Archive-zone error flips direction.** Less metadata → fewer affirmative not-US labels → more articles fall into assume-US → *kept*. So metadata-poverty causes **under-filtering** (foreign noise retained), not over-filtering (US articles dropped). The user's "removing too many relevant negatives this round" worry is therefore the *less* likely failure for this training round; the real exposure is production-time foreign-noise retention — which is exactly what the text model fixes.
- **Heuristic mechanics:** the current heuristic labels US and not-US **independently** (an article can be US, not-US, both, or neither). It treats as not-US (i.e., filters) only those labeled **not-US AND not US**. The "~1/15 wrong" figure = of the only-not-US set, ~1/15 are actually US. ~50% of articles get a useful affirmative label; the unlabelable ~50% are assume-US and are mostly sports/arts/business.
- **DoCA match quality:** ~85% of DoCA events matched into NYT; match rate is **consistent across covariates** (publication year, collective-action type) — no large selection bias in the matched positives. Pre/post-1986 split is roughly even, slightly post.

## Where we are in the design process

Using the `ed3d-plan-and-execute:starting-a-design-plan` skill (six phases: Context Gathering → Clarification → Definition of Done → Brainstorming → Design Documentation → Planning Handoff).

**Currently:** Phase 1 (Context Gathering), in progress. We did most of the conceptual context in conversation; what remains is the **concrete state of code and data**. The open Phase 1 questions for the user (asked, not yet answered):

1. **Where does the data physically live, and in what format?** Especially the new API 1960–1995 download — already a parquet/dataframe with keywords/section/news_desk/headline/lead_paragraph columns? And the LDC corpus on-disk form / what `LDC_CORPUS` config path points at.
2. **Does the current heuristic already exist as code** (and where), or is it ad hoc / in a notebook? Determines whether "conservative heuristic" is a refactor or a fresh write.
3. **Where does the DoCA↔NYT matching code/output live?** The API↔LDC fuzzy join should reuse its headline-normalization machinery rather than reinvent it.
4. **Scope of this design doc** — whole pipeline (gazetteer + dateline parser → fuzzy join → label construction → probe model → validation), or is some of it already done / handled separately?
5. **Final artifact shape** — boolean `us` column on the dataframe(s)? Saved model artifact? Both?

Offer made: an investigator subagent can answer 1–3 from the repo instead of the user typing.

## How to resume

1. Read this doc.
2. Re-invoke `ed3d-plan-and-execute:starting-a-design-plan` (or just continue from Phase 1).
3. Either get the user's answers to the 5 questions above, or dispatch a `codebase-investigator` / `Explore` subagent to answer 1–3 from the repo, then confirm 4–5 with the user.
4. Proceed: Phase 2 (asking-clarifying-questions) → Phase 3 (Definition of Done + create `docs/design-plans/YYYY-MM-DD-us-filter.md`) → Phase 4 (brainstorming) → Phase 5 (writing-design-plans) → Phase 6 (handoff to start-implementation-plan in fresh context).

### Likely shape of the implementation phases (to validate in Phase 4, not yet decided)

- Gazetteer + dateline parser (NYT-style rules + GeoNames; reusable utility).
- API↔LDC fuzzy join (headline-normalization reuse; date-bucketed; high-precision/partial-recall).
- Label construction (dateline-derived US/not-US; datelines stripped from model input text).
- Conservative heuristic productionization (assume-US default) — ships to unblock CCA retrain.
- Probe model (reuse `lu_classifier` frozen-DAPT + linear head); escalation path to fine-tuning gated on validation.
- Validation protocol (free LDC audit on 1987–1995 + small pre-1986 hand-labeled slice).

## Relevant existing code / paths

- `src/prior_estimation/lu_classifier.py` — frozen-DAPT-backbone + Dense-layer classifier; **scaffolding to reuse** for the probe.
- `src/config.py` — platform-conditional paths incl. `LDC_CORPUS`, `CCA_SET_DIR`, `DAPT_BACKBONE_WEIGHTS`; `IS_CLUSTER`, `DTYPE_POLICY`.
- `src/data_setup/data.py` — data loading / split logic (`seed=200`).
- DoCA↔NYT matching code — *location TBD (Phase 1 question 3)*.
