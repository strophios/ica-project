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

## Piece 2: I5 — Pattern-2 serialization round-trip test

**Status:** Implemented in commit `4243c63` (2026-05-08). Tests at
`tests/test_assembly.py::TestPatternTwoSerialization` (2 tests).
Production-path `load_weights` calls tightened in
`src/eval_cca_classifier.py`, `src/model_setup/backbone.py`, and
`scripts/smoke_test_integrated_stack.py`. Suite: 101 → 103.
Reframed during implementation after the empirical finding below
(name-mismatch → shape-mismatch).

### Empirical finding (2026-05-08, during Piece 2 implementation)

The original design framed Piece 2 around "Pattern-2 weight-
loading-**by-name**" — the assumption that variable names in the
training graph and the inference graph must match for load to
succeed. Investigation during implementation showed this
assumption is wrong.

`Model.save_weights(path)` writes a `.weights.h5` file using paths
like:

  - `layers/classification_head/dense/vars/0`
  - `layers/classification_head/logits/vars/0`
  - `layers/functional/layers/embedding/vars/0`

These are *layer-class-name + positional variable index* paths,
**not** the variables' user-given `.path` attribute. The user-
given head name (`name="cca"`) appears nowhere in the file
structure.

`Model.load_weights(path, skip_mismatch=False)` matches variables
by this structural path. So:

- Renaming a head from `"cca"` to `"ccaa"` (one-letter divergence
  the original mismatch test used) **does not break load** — both
  are `ClassificationHead` layers in the same structural position;
  weights load correctly into the renamed head.
- The fail-loud mode that `skip_mismatch=False` actually catches
  is **shape mismatch** (e.g., a head built with `hidden_dim=16`
  trying to load weights from a `hidden_dim=8` save) and **count
  mismatch** (different number of weight tensors).

**Implication for the contract being pinned.** Pattern 2 is
load-by-*structure*, not load-by-*name*. The invariant for
preserving Pattern 2 across train/eval is: **the
architectural shape (layer types, layer ordering, weight shapes)
must match between train and eval scripts**. Head names, being
purely user-facing labels, are *not* load-bearing for weight
loading. They remain load-bearing for *call-site* contracts
(`compile(loss={head_name: ...})` routing, dict-output key
matching) — which are Python-string equality at call time, a
separate concern.

**Implication for the test.** The original mismatch test
(rename head, expect raise) was testing a failure mode that
doesn't exist. The test reframes as a *shape* mismatch (build
inference head with different `hidden_dim`, expect raise) — which
is the failure mode `skip_mismatch=False` is designed to catch
and a realistic bug to defend against (e.g., someone tweaks
`hidden_dim` in the eval script and tries to load weights from a
mismatched-shape trained model).

**Implication for I4 (Piece 3).** The relaxation is useful for
Piece 3's design: the I4 config object doesn't need to enforce
head-name agreement between train and eval as a *load-bearing*
contract. It still should enforce it (drift in head names breaks
call-site routing — Keras would fail loud on `compile(loss=...)`
key not matching output key), but that's a different invariant
than weight loading. Keep both contracts in mind when designing
the config object.

### Decision

Pin the Pattern-2 weight-loading contract as an **executable
invariant** in `tests/test_assembly.py`, plus tighten the
production load sites to explicit `skip_mismatch=False` discipline.

The contract being protected: when a fresh inference model is
built with matching architectural configuration to a previously-
saved training model and weights are loaded, the resulting model
produces **bitwise-identical** predictions to the in-process
Pattern A inference model that shared head Layer instances with
the training model. Any breakage of this invariant — silently-
skipped weights on a shape-changed variable, partial loads —
surfaces as either a load-time exception (with the discipline
change) or a test failure (with the round-trip test).

### Decision

Pin the Pattern-2 weight-loading contract as an **executable
invariant** in `tests/test_assembly.py`, plus tighten the
production load sites to explicit `skip_mismatch=False` discipline.

The contract being protected: when a fresh inference model is
built with matching configuration to a previously-saved training
model and weights are loaded by name, the resulting model produces
**bitwise-identical** predictions to the in-process Pattern A
inference model that shared head Layer instances with the training
model. Any breakage of this invariant — silently-skipped weights
on a renamed variable, partial loads, dtype mismatches at the load
boundary — surfaces as either a load-time exception (with the
discipline change) or a test failure (with the round-trip test).

Two changes:

**1. `tests/test_assembly.py` gains a `TestPatternTwoSerialization`
class** with two tests:

  - `test_round_trip_predictions_match_bitwise`: build Pattern A
    train + inference models sharing head + backbone instances;
    save weights to `tmp_path`; build a *fresh* backbone + head +
    inference model (Pattern 2); confirm Pattern 2's predictions
    *differ* from Pattern A's *before* load (sanity-check that the
    test isn't accidentally passing because both models share the
    same fresh init); load weights with `skip_mismatch=False`;
    confirm Pattern 2's predictions now match Pattern A's bitwise
    (max-diff 0.0). The smoke test does an end-to-end version of
    this on synthetic data; this test pins the same property as a
    fast unit test using the existing fake backbone.
  - `test_load_weights_raises_on_shape_mismatch`: build + save a
    Pattern A model with `hidden_dim=8`; build a fresh inference
    model with `hidden_dim=16` (different head shape, same head
    name and class); attempt `load_weights(..., skip_mismatch=False)`;
    assert the call raises rather than silently accepting partial-
    or-empty load. This is the *tightening* — pins the contract
    that architectural-shape drift between train and eval fails
    loud at load time. (Originally drafted as a name-mismatch test
    but reframed after the empirical finding above; rename
    doesn't actually break load, shape change does.)

**2. Production-path `load_weights` calls explicitly pass
`skip_mismatch=False`** in:

  - `src/eval_cca_classifier.py:81` — the original I5 site.
    Loading head + backbone weights for Pattern 2 inference.
  - `src/model_setup/backbone.py:46` — DAPT backbone weight load
    inside `load_dapt_backbone`. Same defensive intent, same
    one-line tightening. Different invariant (keras_hub backbone
    variable names rather than our head names) but the same shape
    of bug if it ever drifts.

If `skip_mismatch=False` is already Keras 3's default, the
explicit pass documents intent (load-bearing contract; future
readers shouldn't have to know the default to know it's
load-bearing). If not, this is a behavior change. Either way, the
production code becomes self-documenting.

### Reasoning

**Why both a round-trip test *and* a shape-mismatch test, not
just one.** They pin different invariants:

- The round-trip test pins *the happy path works* — predictions
  match bitwise when architectures align. Without it, a future
  change that silently breaks load (e.g., a save-format change,
  a dtype-on-disk shift, an unrelated layer-structure tweak)
  would surface only when the eval script run gives wrong
  predictions, possibly long after the change.
- The shape-mismatch test pins *the unhappy path fails loud* —
  silent partial loads don't ship. Without it, `skip_mismatch=False`
  discipline could regress (e.g., someone passes `skip_mismatch=True`
  when debugging and forgets to revert, or Keras's default flips
  in a future version) and the system would silently accept
  partial loads.

Together they cover both directions of the contract.

**Why the round-trip test must verify "predictions differ before
load."** If both Pattern A and Pattern 2 models are built with the
same seed (or with no explicit seed under deterministic-init
conditions), their fresh weights are identical, and predictions
match bitwise *without* any load happening. A test that just
asserts post-load equality could pass even if `load_weights` were
a no-op. The pre-load-difference assertion breaks that ambiguity:
it asserts the test is exercising the load path, not just
coincident initialization.

**Why bitwise (max-diff 0.0), not "approximately equal."**
Pattern 2 weight loading should be *exact*: variable names match,
dtypes match, shapes match, on-disk bytes round-trip cleanly.
Any non-bitwise mismatch is a real bug (precision loss in save/
load, dtype drift, missing weights). Allowing tolerance lets such
bugs slip through. The smoke test already verifies max-diff
0.00e+00 on the integrated stack; this test pins the same bar
as a fast unit test.

**Why include `backbone.py`'s `load_weights` call alongside the
eval script.** Same pattern, same defensive intent, parallel
one-line tightening. The DAPT backbone load is a Pattern-2-shaped
operation: a fresh keras_hub backbone gets weights loaded by
variable name from disk. If keras_hub renames internal variables
between versions (or if our DAPT save format ever drifts), silent
partial-load is the same shape of bug as on the head side.
Pulling both into the same piece keeps the discipline change
coherent.

**What I5 *can't* catch on its own.** I5 is about
*mechanical-load correctness*: the bytes on disk match the
variables in memory after load. It doesn't catch *semantic*
configuration drift between train and eval (e.g., training used
`prior=0.02` but eval reconstructs the head with `prior=0.05`).
The head's `loss_fn` is reconstructed from script-side config,
not from saved weights — variables align, predictions don't
depend on `loss_fn` at inference time, so I5's invariant holds
even with semantic drift. **Catching semantic drift is I4's
territory** (Piece 3); I5 explicitly stays in its lane.

### Layout

- `tests/test_assembly.py` — new `TestPatternTwoSerialization`
  class with two tests, plus a `_build_pattern_a_and_save` helper
  shared between them. Uses the existing `fresh_backbone` /
  `fresh_head` fixtures plus pytest's `tmp_path` fixture for
  weight-file scratch space.
- `src/eval_cca_classifier.py` — `load_weights` call gains
  explicit `skip_mismatch=False`, with a comment explaining the
  contract and noting that Keras's matching is structural.
- `src/model_setup/backbone.py` — `load_weights` call gains
  explicit `skip_mismatch=False`, with a comment.
- `scripts/smoke_test_integrated_stack.py` — Pattern-2
  `load_weights` call also gains explicit `skip_mismatch=False`
  to match the production-side discipline.

No new files. No architectural changes.

### Contracts

**Round-trip invariant (pinned by
`test_round_trip_predictions_match_bitwise`):**

  Given a Pattern A `(train_model, inf_model)` pair sharing head
  and backbone Layer instances, with weights `W` (post-fit or
  any non-default state):

    save_weights(inf_model, path)
    fresh_inf_model = build_inference_model(
        backbone=fresh_backbone (matching config),
        heads={head_name: ClassificationHead(matching config, name=head_name)},
        seq_length=seq_length,
    )
    fresh_inf_model.load_weights(path, skip_mismatch=False)

    For any input X:
        inf_model.predict(X) == fresh_inf_model.predict(X)  exactly (bitwise)

**Fail-loud invariant (pinned by
`test_load_weights_raises_on_shape_mismatch`):**

  Given a saved weights file from a model with head
  `hidden_dim=8`, and a fresh inference model with `hidden_dim=16`
  (or any architectural-shape mismatch — different layer count,
  different layer types, different weight shapes):

    fresh_inf_model.load_weights(path, skip_mismatch=False)
    must raise an exception (rather than silently completing with
    partial or zero loaded weights).

  Note: head *name* mismatches do NOT trigger this — Keras's
  `.weights.h5` save format keys variables by layer-class-name +
  positional index, not by user-given name. See "Empirical
  finding" subsection above.

### Test coverage anticipated

`tests/test_assembly.py`, new `TestPatternTwoSerialization` class,
2 tests:

- **test_round_trip_predictions_match_bitwise**: described above.
  Test flow:
  1. Build Pattern A train + inference models on `fresh_backbone` /
     `fresh_head`.
  2. `train_model.compile(optimizer="adam"); train_model.fit(...)`
     for one step to get non-default weights.
  3. `inf_model.save_weights(tmp_path / "test.weights.h5")`.
  4. Build a fresh `fake_backbone_2` and fresh
     `ClassificationHead(name="cca", ...)`; build
     `inf_model_2 = build_inference_model(fresh_backbone_2, {"cca": head_2}, ...)`.
  5. Run `inf_model_2.predict(X)` *before* load; assert it differs
     from `inf_model.predict(X)` (the test isn't accidentally
     passing on init coincidence).
  6. `inf_model_2.load_weights(tmp_path / "test.weights.h5",
     skip_mismatch=False)`.
  7. Run `inf_model_2.predict(X)` *after* load; assert
     `np.array_equal` to `inf_model.predict(X)` (bitwise match).
- **test_load_weights_raises_on_shape_mismatch**: Test flow:
  1. Build + fit + save Pattern A model (head with `hidden_dim=8`).
  2. Build fresh inference model with head `hidden_dim=16`.
  3. `with pytest.raises(<some exception>):
        inf_model_wrong.load_weights(path, skip_mismatch=False)`.

  The exact exception class depends on Keras's load_weights
  implementation; likely `ValueError`. The test uses bare
  `pytest.raises(Exception)` first to characterize, then tightens
  to the specific class once the implementation is observed.

Total: 2 new tests in `tests/test_assembly.py`. File goes from 13
→ 15 tests. Suite: 101 → 103.

### Patterns introduced

- **Round-trip test as serialization invariant.** The general
  shape: "save state in mode X; rebuild + load in mode Y;
  assert behavior matches X bitwise." Applies anywhere
  serialization is a load-bearing contract — extends naturally
  to other on-disk artifacts later (preprocessor configs,
  optimizer state if we ever serialize it).

- **Pre-load difference assertion.** When testing a load path,
  break symmetry between the pre-load and post-load states so
  the test cannot pass on coincidental initialization. This is
  the standard discipline for testing any "side-effecting load"
  (read-then-mutate) function — without it, a no-op
  implementation could test green.

- **Fail-loud-on-mismatch as a separate test.** Happy-path
  invariants and unhappy-path invariants need separate tests.
  A single "predictions match" test doesn't catch silent
  partial loads; a single "raises on mismatch" test doesn't
  catch broken load logic when names match. Pair them.

- **Explicit `skip_mismatch=False` as documentation of intent.**
  Even when it's the framework's default, passing it explicitly
  marks the call as load-bearing. Future readers don't have to
  know what the default is to know that the contract matters here.

### Open / deferred

- **Strict-load wrapper for fail-loud-on-name-mismatch.** Keras's
  `.weights.h5` save format keys by structure (layer class +
  positional index), not by user-given name. Renaming a head
  silently loads weights into the renamed head's same-shape
  slots. If we ever want a *name*-mismatch fail-loud invariant,
  it would require a custom wrapper that reads the model's
  variable paths, inspects the h5 file's contents, and verifies
  agreement before calling `load_weights`. Not warranted now —
  the head-name contract is enforced at *call sites* by Keras's
  `compile(loss={head_name: ...})` routing (which fails loud on
  key mismatch) and dict-output structure. Pinning that contract
  is I4's territory (Piece 3) via the config object's
  cross-script invariant.
- **Save-format choice.** Currently using `.weights.h5` (legacy
  HDF5 weights-only format). Keras 3 also supports
  `.keras` (full-model archive) and may add other formats. If we
  ever migrate, the round-trip test needs to be re-checked
  against the new format's contract. Not warranted now; the
  current format works for our needs (weights-only, Pattern 2
  rebuilds the architecture from code).
- **Backbone-loader test parity.** The backbone-loader change is
  a one-line tightening with no test added — the round-trip test
  exercises the head's load path, not the backbone's. Adding a
  parallel test would pin the backbone discipline too.
  Considered; deferred unless a concrete failure mode appears,
  since the backbone variable structure is owned by keras_hub
  (not us) and harder to drift in ways our test could catch
  usefully.
- **`load_weights` exception class.** Characterized at
  implementation time: Keras raises `ValueError` on shape
  mismatch with `skip_mismatch=False`, with a message naming
  the target variable and the shape mismatch. The test pins
  `pytest.raises(ValueError)`. If Keras's behavior changes in a
  future version, the test will fire on either side: a *missing*
  raise (regression) or a *changed* exception class (a tighten-
  the-test signal).

---
