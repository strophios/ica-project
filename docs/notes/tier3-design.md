# Tier 3 Design

*Living document — appended piece-by-piece as each design discussion lands.*
*Started: 2026-05-08.*

This captures the design decisions made during Tier 3 of the audit/refactor.
Like the Tier 2 design doc, it is not a spec drafted up-front; each
section is the outcome of a dialogic design discussion with the user.
Record the decision plus enough reasoning that someone reading it later
knows *why* we landed where we did.

See `docs/notes/tiers-and-checkpoints.md` for the overall frame,
`docs/notes/tier2-design.md` for the prior tier (with a "Post-review
corrections" section pointing at the Tier 2 review findings inherited
into Tier 3), and `docs/notes/pinned-questions.md` for deferred
substantive questions.

---

## Overall framing

**Intent:** harden the boundaries the multi-head future will lean on.
Tier 2 reshaped the code; Tier 3 enforces the contracts that reshape
created. The multi-head migration will move faster and break fewer
things if the contracts are checked rather than trusted.

**The work splits into two flavors that look similar but aren't:**

1. **Boundary enforcement.** The Tier 2 review's three Important
   findings (I3, I4, I5) are each a place where two pieces of code
   rely on agreement that no mechanism currently enforces. These are
   *foundation* work — they don't add features, they prevent silent
   corruption.
2. **Test coverage expansion.** Original Tier 3 scope: tests for
   label construction in `create_classifier_data`, missing-value
   handling in `data_from_parquet`, possibly fill-in shape contracts
   the Piece-3 preprocessor tests didn't cover. These are *grooming*
   work — mechanical, valuable, less architecturally interesting.

The boundary work is the primary spine; the test-coverage work is
secondary and trails it.

**Boundary inventory as the unifying frame for the boundary work.**
The house-style `defense-in-depth` skill's *animating insight* —
validate at every layer; each layer catches what others miss; turn
bugs from "vigilance keeps us safe" into "the structure prevents
them" — transfers cleanly. The skill's specific four-layer carving
(entry-point → business-logic → environment-guards → debug
instrumentation) is web-app-flavored and only partially applies:
Layer 1 (entry-point) and Layer 2 (business-logic) map well; Layer 3
(environment guards for destructive ops) doesn't fit our context;
Layer 4 (debug instrumentation) maps to "informative error
messages" in a degraded form.

So we use the principle, not the procrustean carving. Each finding
is approached the same way: **inventory the boundaries data or
config crosses, then decide what each boundary should check.** The
taxonomy is system-specific; the principle is generic.

The three findings, with their boundaries:

  - **I3 — `ClassifierPreprocessor` boundaries.** Two: (a) `__init__`
    (configuration in, prepared object out) and (b) `__call__` (data
    dict in, model-shaped tensors out). Each catches a different
    drift mode — `__init__` catches *internally inconsistent
    configuration* (e.g., `endpoint_model=False` with empty
    `label_keys` is nonsense regardless of data; `target_dtype`
    that isn't a valid Keras dtype string would fail mid-graph
    today); `__call__` catches *configuration vs. actual-data
    mismatch* (the configured `text_key` isn't a column in this
    batch). Validate at both.
  - **I4 — train/eval coupling boundaries.** Two `__init__`-class
    boundaries (one in the training script, one in the eval script,
    each constructing its own preprocessor + head config) plus a
    cross-process gap between them. The cross-process gap is the
    dangerous one — there's no shared object holding the contract,
    so drift is silent. Bridge with a config object that's either
    shared statically (Option B) or serialized at train and loaded
    at eval (Option C, anticipated for the user's multi-model
    workflow). Validate the loaded config matches expectations.
  - **I5 — Pattern-2 serialization boundary.** One: the on-disk
    weights file. The contract — "training-graph variable hierarchy
    matches a fresh inference-graph variable hierarchy by name" —
    is enforceable as a round-trip test (Pattern A model fits,
    saves; fresh inference model loads; predictions match Pattern
    A predictions bitwise) and tightenable by passing
    `skip_mismatch=False` so partial loads raise. Validate at load
    by failing loud rather than soft.

Each finding's boundaries are different, the checks are different,
and what each protects against is different. I3's entry-point
checks won't catch a config drift between scripts (I4's territory);
I4's config-coupling won't catch a weight-naming refactor (I5's
territory). The findings are not redundant; they cover structurally
different failure modes that happen to share "the contract is
unstated" as their root.

## Pieces (anticipated)

The intended order, with reasoning for the ordering:

1. **Piece 1: I3 — preprocessor input validation.** Smallest and most
   localized. Establishes the validation idiom (boundary-by-boundary
   checks at both `__init__` and `__call__`, with construction-time
   checks for internal-config-validity and call-time checks for
   config-vs-data mismatch) that Pieces 2 and 3 will lean on. Pulls
   in M2 from the Tier 2 review's Tier-4 deferred list (`target_dtype`
   construction validation), since it lives naturally in the
   construction-validation block.
2. **Piece 2: I5 — Pattern-2 serialization round-trip test.**
   Mechanically close to existing assembly tests. Adds a regression
   test that the round-trip works, plus a small `eval_cca_classifier.py`
   change to call `load_weights(..., skip_mismatch=False)` so future
   variable-name drift fails loudly instead of silently.
3. **Piece 3: I4 — train/eval config coupling.** The most
   architectural piece. Two flavors of coupling are live: a static
   shared config module (catches static drift) and run-time
   serialized config saved alongside weights (catches per-run drift).
   The user's research workflow expects multiple models living
   side-by-side during hyperparameter search and during the
   single-head → multi-head migration, so the per-run-config flavor
   is in scope. Whether it's serialized config alone (Option B from
   the design discussion) or static module + serialized run config
   (Option C) is the open decision for Piece 3's design section.

   **Forward pressure from pinned question #3.** The config object
   should *not* encode the train/predict distinction as part of its
   serialized state. Train/predict is a call-site choice; only the
   task configuration (`text_key`, `label_keys` non-empty,
   `target_dtype`, head names, prior, seq_length, etc.) is config.
   This anticipates the eventual `ClassifierPreprocessor` API
   refactor (pinned question #3) where the empty-`label_keys`
   workaround disappears. Designing Piece 3's config object around
   "task config, period" rather than "task config plus
   train-or-predict-flag" keeps the config stable across that
   refactor.
4. **Piece 4: Original-scope test coverage.** Label-construction
   tests for the cca/immig/descriptor boolean combinations in
   `create_classifier_data`; missing-value (`fill_null("")`) coverage;
   any preprocessor shape-contract gaps surfaced by Piece 1's
   validation work. TDD pattern; less conceptually loaded than the
   boundary pieces. Implementation can lean on the user-does-less-
   implementation flavor of pedagogical mode that Tier 3 generally
   moves toward.

**Deferred (not Tier 3): evaluation harness with calibration.**
Originally listed in the Tier 3 plan, but it's a research deliverable
(threshold selection on a hand-labeled PN test set, possibly Platt
scaling or isotonic regression), not foundation work. Punt to a
separate piece of work after Tier 4 hygiene.

**Deferred Tier-2-review Minor findings (Tier 4 hygiene scope, not
this tier):** M1 (`test_script.py` raise placement), M3
(`_default_group_fn` separator), M4 (default `ClassificationHead`
name collision risk).

**M2 promoted into Piece 1.** M2 (`target_dtype` validation in
`ClassifierPreprocessor.__init__`) was originally listed as Tier 4
hygiene, but Piece 1's scope expanded during design discussion to
cover construction-time validation alongside call-time validation —
making M2 a natural fit (same file, same `__init__`, same
validation block). Folding it in saves a round-trip and keeps the
construction-validation logic in one place. `tiers-and-checkpoints.md`
and `CLAUDE.md` will need their Tier 4 inheritance lists updated to
remove M2 when Piece 1 closes.

## Process commitments (continuing from Tier 2)

The Tier 2 process commitments stay in effect:

- **Design pass first.** This document, appended per piece, before
  code lands.
- **Staged implementation.** Each piece a focused commit with its own
  status entry; no giant Tier 3 dump.
- **Tests follow the refactor.** Existing tests should keep passing;
  new tests pin the new invariants.

**Pedagogical mode stays on, with one expected shift.** The user
flagged that Tier 3's content (validation, coupling, serialization
contracts) is more "ML-engineering good practice" than the
ML-conceptual material Tier 2 was built around — and that it's
relatively less internalized than the conceptual material. So
explanatory comments in code stay dense, named patterns get
introduced explicitly, but the user expects to do less hand-
implementation than in Tier 2. The "you implement it" mode is offered
where the implementation itself carries the learning; otherwise
Claude implements with detailed comments.

---

## Piece 1: I3 — preprocessor input validation

**Status:** Implemented in commit `79ab31c` (2026-05-08).
`ClassifierPreprocessor` at `src/preproc/preprocessor.py`; tests at
`tests/test_preprocessor.py` (14 new tests across two new test
classes, file at 26 total). Suite: 87 → 101 tests passing. Smoke
test re-run unchanged (Pattern A vs. Pattern 2 max-diff 0.00e+00).

### Decision

`ClassifierPreprocessor` gains validation at **both** of its
boundaries:

**Boundary 1 — `__init__` (configuration in, prepared object
out).** Construction-time checks for *internally inconsistent
configuration*, independent of any data:

  - `text_key` is a non-empty string.
  - `label_keys` is a `dict` (catches the easy
    `[("k", "v"), ...]` typo where the caller passed a list of
    tuples instead of a dict).
  - `target_dtype` is a Keras-recognized dtype string (this is
    M2 from the Tier 2 review, folded in here — see overall
    framing).
  - **Business rule**: if `endpoint_model=False` (standard mode),
    `label_keys` must be non-empty. Standard mode emits
    `(features, targets_dict)`; an empty `targets_dict` has nothing
    to route via `compile(loss={...})` and is structurally
    nonsensical. Empty `label_keys` is *only* valid in endpoint
    mode (predict-only configurations like the eval script's
    `label_keys={}` pattern).

**Boundary 2 — `__call__` (data dict in, model-shaped tensors
out).** Call-time check for *configuration vs. actual-data
mismatch*:

  - `inputs` (dict-valued batch from `tf.data.Dataset`) must
    contain `text_key` as a key.
  - `inputs` must contain every value of `label_keys` (i.e., every
    source column named in the `output_dict_key → source_column`
    mapping).

If either check fails, the preprocessor raises an exception with a
message that names the violation and the configuration that drove
the expectation. Construction-time checks raise `ValueError` (the
configuration itself is malformed); call-time checks raise
`KeyError` (consistent with the existing `inputs[key]`-raises-
`KeyError` failure mode, just earlier and informative).

The two boundaries catch structurally different bugs. Construction
checks fire at preprocessor build time (eagerly, before any data
flows). Call checks fire at `tf.data.Dataset.map` tracing time (the
first time the preprocessor is invoked on a batch). Both fire well
before model fit/predict actually starts.

**Schema-aware construction validation is deferred to Piece 3.**
The current construction checks are schema-*independent* (they don't
know what columns the dataset has). Piece 3 (I4) will introduce a
config object that's a natural carrier for `expected_columns: set[str]`;
when that lands, the preprocessor can optionally accept the config
and validate `text_key` and `label_keys.values()` against the
schema at construction. That's a strict tightening of Boundary 1
that doesn't require any redesign of Piece 1's check block.

### Reasoning

**Why both boundaries, not just one.** The two boundaries catch
*structurally different* drift modes; neither subsumes the other.

A `__call__`-only check would catch "the configuration says
`text_key='headline_with_lead'` but this batch has no such column"
— but would silently accept `endpoint_model=False, label_keys={}`,
producing a preprocessor that emits `(features, {})` tuples that
Keras's `compile(loss={...})` routing has nothing to do with. It
would also accept `text_key=None`, which would propagate to a
confusing `TypeError` from `inputs[None]` rather than a clean
"text_key must be a non-empty string" message.

A `__init__`-only check would catch the internal-consistency bugs
above, but would silently accept a perfectly-shaped configuration
that points at columns the actual dataset doesn't have — the bug
would surface deep inside `tf.data.map`'s tracing context with a
stack pointing at TF internals rather than the user's
misconfiguration.

Each boundary's check is doing work the other can't do. This is
exactly the defense-in-depth principle: validate at every layer;
each catches what others miss; the bug becomes structurally
impossible (rather than caught by one well-placed check that erodes
over time).

**Why these specific construction-time checks.** Each addresses a
real failure mode visible in the current codebase:

- *`text_key` non-empty string*: catches `text_key=None` (default-
  forgotten parameter), `text_key=""` (string typo), `text_key=42`
  (wrong type). Today these produce confusing errors deep in the
  tokenizer call.
- *`label_keys` is a dict*: catches the most plausible typo —
  `label_keys=[("cca_targets", "cca_label")]` (list of tuples,
  hand-rolled when the user remembers there are two strings per
  entry but forgets the dict shape). Today this produces a confusing
  `AttributeError` on `.items()`.
- *`target_dtype` valid Keras dtype*: catches `target_dtype="flot32"`
  (typo). Today the cast call fails mid-graph with a less readable
  message. This is M2 from the Tier 2 review; the construction-
  validation block is the natural home.
- *`endpoint_model=False` ⇒ non-empty `label_keys`*: catches a
  structurally-nonsensical configuration (standard mode emits
  targets dict for `compile(loss={...})` routing; empty targets
  dict has nothing to route). Today this would silently produce a
  preprocessor that emits `(features, {})` tuples; the model would
  fit with no actual loss attached to anything and the user would
  see suspicious training behavior much later.

The empty-`label_keys`-in-endpoint-mode case is *deliberately
allowed* — that's the eval script's predict-only pattern, an
established (if smelly — see pinned question #3) caller. The
construction check is precisely conditional on `endpoint_model`.

**Why `ValueError` for construction-time, `KeyError` for call-time.**
Different exception classes signal different categories of bug:

- `ValueError` at construction means "the configuration you passed
  is malformed." The fix is in the caller's preprocessor
  construction.
- `KeyError` at call means "the data you fed in is missing a
  column the configuration expected." The fix is in the dataset
  pipeline, not the preprocessor configuration.

`KeyError` for call-time stays consistent with the existing failure
mode (`inputs[key]` raises `KeyError`), so a caller catching
`KeyError` for missing-column handling continues to work, just with
a useful message and an earlier stack.

**Why enumerate-all-missing rather than fail-fast.** When a caller
makes multiple mistakes (e.g., wrong `text_key` *and* wrong source
column for a label), listing all of them at once gives a complete
diagnostic rather than forcing the caller through one round of
fix-and-rerun per mistake. The cost is a set construction and a
set-difference per traced map call (so once per dataset
construction in normal `tf.data` use), which is negligible.

**Why no runtime business-logic / dtype checks.** Possible
candidates — verifying the text column is string-typed, verifying
the label columns can be cast to `target_dtype` at runtime — would
be useful but are runtime-tensor-introspection territory, awkward
inside a `tf.data` map function (tensors are KerasTensors at
tracing time; dtypes are known but values are not), and the
existing `keras.ops.cast` call already fails loudly on incompatible
dtypes. Skipped for this piece.

**Cross-tier framing — what I3 *can't* catch.** Even with both
boundaries validated, I3 won't catch:

- A *config drift* between train and eval scripts (e.g., training
  uses `text_key="headline_with_lead"`, eval typos to
  `text_key="headline+lead"`). Both preprocessors construct
  successfully and validate their own inputs successfully against
  their own dicts; the drift is at the cross-script level. **That's
  I4's territory.**
- A head-name change that breaks Pattern-2 weight loading. The
  preprocessor is fine; the variable hierarchy is what shifted.
  **That's I5's territory.**

I3 is doing exactly the work assigned to it; no more, no less.

### Layout

- `src/preproc/preprocessor.py` — `ClassifierPreprocessor.__init__`
  gains a construction-time validation block; `__call__` gains a
  call-time validation block at the top. No other changes to the
  file.
- `tests/test_preprocessor.py` — two new test classes:
  `TestConstructionValidation` (covers `__init__` checks) and
  `TestCallTimeInputValidation` (covers `__call__` checks).

No new files; no architectural changes.

### Contracts

**`ClassifierPreprocessor.__init__(...)`** — additional preconditions:

- `text_key` must be a non-empty string. Otherwise raise
  `ValueError("text_key must be a non-empty string; got <repr>")`.
- `label_keys` must be a `dict`. Otherwise raise
  `ValueError("label_keys must be a dict[str, str]; got <type>")`.
  (Per pinned question #3, the long-term shape may further require
  non-empty `label_keys` regardless of mode; for now the
  endpoint-mode predict-only pattern keeps empty `label_keys` valid
  in that one case.)
- `target_dtype` must be a Keras-recognized dtype string. Validated
  via `keras.backend.standardize_dtype(target_dtype)` (which raises
  on invalid input); preprocessor wraps the raise with a message
  naming the bad value.
- If `endpoint_model is False`, `label_keys` must be non-empty.
  Otherwise raise `ValueError("standard mode (endpoint_model=False)
  requires non-empty label_keys; empty label_keys is only valid in
  endpoint mode (predict-only configuration)")`.

All construction-time checks fire eagerly during `__init__` —
preprocessor construction is the right time for "is this
configuration even coherent."

**`ClassifierPreprocessor.__call__(inputs)`** — preconditions:

- `inputs` (a dict-valued batch from `tf.data.Dataset`) must contain
  the key `self.text_key`.
- `inputs` must contain every value of `self.label_keys` as a key
  (i.e., every source column named in the `output_dict_key →
  source_column` mapping must be present in the batch).

If either precondition fails, the preprocessor raises `KeyError`
with a message that includes:

- The set of missing column names.
- The configured `text_key` and the configured `label_keys` source
  columns (so the user can see what configuration drove the
  expectation).
- The set of keys actually present in the input batch (so the user
  can see what they got).

The call-time check runs on every `__call__` invocation. Cost is
one set construction and one set-difference per traced map call (so
once per dataset construction in normal `tf.data` use), which is
negligible. Multiple missing columns are *enumerated* in a single
error rather than failing fast on the first one.

The behavior on success is exactly as before — the new checks are
purely additive.

### Test coverage anticipated

`tests/test_preprocessor.py`, two new test classes — one per
boundary.

**`TestConstructionValidation`** (covers `__init__` checks):

- **TestTextKeyNonEmptyString**: `text_key=None`, `text_key=""`, and
  `text_key=42` each raise `ValueError` with a message naming the
  bad value.
- **TestLabelKeysIsDict**: `label_keys=[("k", "v")]` (list of
  tuples) and `label_keys=None` each raise `ValueError`. Empty
  `dict()` is *not* rejected here — that's a separate business-rule
  check covered below.
- **TestTargetDtypeValid**: `target_dtype="float32"` (default)
  accepted; `target_dtype="flot32"` (typo) raises `ValueError`
  naming the bad value. (Retires M2 from the Tier 4 deferred list.)
- **TestStandardModeRequiresNonEmptyLabelKeys**:
  `endpoint_model=False, label_keys={}` raises `ValueError` with a
  message explaining that empty `label_keys` is only valid in
  endpoint mode.
- **TestEndpointModeAllowsEmptyLabelKeys**:
  `endpoint_model=True, label_keys={}` is *accepted* (no
  exception). Pins the contract for the eval script's predict-only
  configuration. **This is the most important test in this class**
  — without it, a future "tighten validation" rewrite could regress
  the eval script. (When pinned question #3 lands, this test will
  flip to a rejection test; until then, it pins the current
  contract.)
- **TestEndpointModeWithLabelKeysAccepted**: the standard training
  configuration (`endpoint_model=True, label_keys={...non-empty...}`)
  is accepted. Sanity-check that the new construction validation
  doesn't accidentally reject the primary use case.

**`TestCallTimeInputValidation`** (covers `__call__` checks):

- **TestMissingTextKey**: dict missing the configured `text_key`
  raises `KeyError`; the message names the missing key, the
  configured `text_key`, and includes the available keys.
- **TestMissingLabelSourceColumn**: dict missing one of the
  `label_keys` source columns raises `KeyError`; the message names
  the missing source column.
- **TestMultipleMissingColumns**: dict missing both `text_key` and a
  source column raises `KeyError` enumerating both. Pins the
  enumerate-all-missing contract — matters for diagnostic quality
  when a caller makes multiple mistakes.
- **TestPredictModeWithoutLabelSourceColumns**: a predict-only
  preprocessor (`endpoint_model=True, label_keys={}`) successfully
  processes an input dict that only contains `text_key` (no label
  source columns). Pins the predict-mode call-time contract; pairs
  with `TestEndpointModeAllowsEmptyLabelKeys` from the construction
  class to fully cover the predict-only flow.

Total at landing: 10 construction + 4 call-time = 14 new tests
(the construction class ended up with 10 rather than the originally-
sketched 6 because each `text_key`-rejected case got its own test
for diagnostic clarity, plus a separate "default `target_dtype`
accepted" sanity test). Combined with the existing 12 in
`test_preprocessor.py`, the file lands at 26 tests after Piece 1.

### Patterns introduced

- **Validate at every boundary; each catches what others miss.**
  The unifying defense-in-depth principle, applied locally:
  `__init__` checks for *internal-config-validity* bugs;
  `__call__` checks for *config-vs-data-mismatch* bugs. Each
  boundary's check is necessary; neither is sufficient on its
  own. The shape this takes in code is two distinct check blocks
  in two distinct methods, not one big check at one location.

- **Construction-time `ValueError` vs. call-time `KeyError` as
  bug-category signaling.** Different exception classes communicate
  *where the bug is*. `ValueError` at `__init__` says "your
  preprocessor configuration is malformed — fix the construction
  call." `KeyError` at `__call__` says "the data you fed in is
  missing a column the configuration expected — fix the dataset
  pipeline." The exception class is part of the error message.

- **Enumerate-all-missing rather than fail-fast.** When validating
  multiple required keys, list every missing one at once rather
  than failing on the first. Diagnostic quality matters more than
  microseconds in a one-time-per-dataset check.

- **Informative error messages name the configuration that drove
  the expectation.** Don't just say "missing column X"; say
  "missing column X; configuration expected source columns
  {X, Y, Z} from `label_keys` and `text_key=...`; got {Y, Z, W}."
  The user needs to see both halves of the contract to debug it.

- **Mode-conditional business rules at construction.** The
  `endpoint_model=False` ⇒ non-empty `label_keys` check is a
  business rule that fires only in one mode. Encoding mode-
  conditional invariants as construction-time checks (rather than
  trying to enforce them via type system or runtime) keeps the
  validation block readable and the rule explicit at the point it
  applies.

### Open / deferred

- **Schema-aware construction validation.** Currently the
  construction-time checks are schema-*independent* — they don't
  know what columns the dataset will provide. Piece 3 (I4) will
  introduce a config object that's a natural carrier for
  `expected_columns: set[str]`; once that lands, the preprocessor
  can optionally accept the config and validate `text_key` and
  `label_keys.values()` against the actual schema. This is a
  *strict tightening* of Boundary 1 that doesn't require any
  redesign of Piece 1's check block — just an additional check
  conditional on whether the config was provided.
- **Runtime business-logic / dtype checks.** Verifying text column
  is string-typed, label columns cast cleanly to `target_dtype`.
  Defer until a real bug motivates it; `keras.ops.cast` already
  fails loudly on incompatible dtypes today, and runtime tensor
  introspection inside `tf.data.map`'s tracing context is awkward.
- **`label_keys` always-non-empty (per pinned question #3).** When
  the preprocessor API refactor in pinned question #3 lands, the
  construction-time business rule tightens to "label_keys must be
  non-empty in *all* modes" (no special endpoint-mode exception),
  and `TestEndpointModeAllowsEmptyLabelKeys` flips to a rejection
  test. No structural rework — one-line validation change, one-line
  test change.

---
