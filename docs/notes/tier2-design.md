# Tier 2 Design

*Living document — appended piece-by-piece as each design discussion lands.*
*Started: 2026-04-20.*

This captures the design decisions made during Tier 2 of the audit/refactor.
It is not a spec drafted up-front; each section is the outcome of a
dialogic design discussion with the user. Record the decision plus enough
reasoning that someone reading it later knows *why* we landed where we did.

See `docs/notes/tiers-and-checkpoints.md` for the overall frame and
`docs/notes/pinned-questions.md` for deferred substantive questions.

---

## Piece 1: Model setup and head composition

**Status:** Implemented in commit `789d88c`. `ClassificationHead` at
`src/model_setup/heads.py`; tests at `tests/test_heads.py`.
`CombinedClassificationHead` design is captured below but not yet
implemented — the gradient-flow "Open decision" remains unresolved.

### Decision

Heads are implemented as `keras.layers.Layer` subclasses (Option B in the
discussion), supporting both **endpoint mode** (loss computed internally
via `add_loss`, used when the loss depends on internal state — FLPU,
eventual ALUM) and **standard mode** (`loss_fn=None`, loss handled by
`compile(loss={...})`, used for any simpler-case head).

A single `ClassificationHead` class covers the simple heads (CCA, immigrant
involvement, possibly US). A separate `CombinedClassificationHead` covers
the combined ICA head, which takes both the shared backbone features *and*
the component heads' logits as inputs.

### Reasoning

The multi-head future will need endpoint-layer heads (FLPU today, ALUM
later). Option A (heads as plain functions) would force two different
abstractions — one for simple heads, one for endpoint heads. Option B
unifies both: a head with `loss_fn=None` behaves like Option A would
have; a head with `loss_fn=<something>` activates endpoint mode.

Why endpoint-layer pattern, conceptually: Keras's functional API treats
models as pure functions, which works until you need losses that depend
on internal state (intermediate activations, input embeddings, conditional
behavior on labels at train time). ALUM explicitly needs embedding-level
access. The endpoint-layer pattern is the Keras-documented escape hatch
for this class of problems, so using it here is aligned with Keras intent
rather than fighting the framework.

### Layout

- `src/model_setup/heads.py` — `ClassificationHead` and
  `CombinedClassificationHead` live here.
- `src/model_setup/backbone.py` — backbone-loading utilities
  (to be designed in a later piece, split from the existing
  `classifier_from_dapt_checkpoint`).
- `src/model_setup/assembly.py` — the classifier assembly function
  that wires backbone + heads into a full `keras.Model`
  (to be designed in a later piece).
- `src/model_setup/classification_setup.py` — the existing code,
  retained until the new structure is complete; removed afterwards.

### Contracts

**`ClassificationHead.__init__(hidden_dim, dropout=0.1, loss_fn=None, name=None)`**

- `hidden_dim`: width of the intermediate dense layer. Typically matches
  the backbone's `hidden_dim` (= 768 for RoBERTa-base).
- `dropout`: applied before and after the intermediate dense.
- `loss_fn`: a `keras.losses.Loss` instance (e.g., `FLPULoss(prior=...)`)
  or `None`. If provided, the head operates in endpoint mode.
- `name`: passed to Keras. Shows up in summaries and serialization.

**`ClassificationHead.call(features, targets=None)`**

- `features`: shared backbone output, typically the [CLS]-token
  representation. Shape `(batch, hidden_dim)`.
- `targets`: optional, shape `(batch,)` or `(batch, 1)`. If provided
  *and* `loss_fn is not None`, the head calls `self.add_loss(loss_fn(targets, logits))`
  and the outer model's `compile()` call does *not* specify a loss for
  this output.
- **Returns**: per-sample logits, shape `(batch, 1)`.

**`CombinedClassificationHead.__init__(hidden_dim, dropout=0.1, loss_fn=None, name=None)`**

- Same parameters as `ClassificationHead`.
- `hidden_dim` refers to the intermediate layer inside the combined head.
  Its input is actually the concatenation of backbone features and
  component-head logits, so the first Dense's input dim is inferred by
  Keras at build time.

**`CombinedClassificationHead.call(features, component_logits, targets=None)`**

- `features`: shared backbone output, as above.
- `component_logits`: **list** of tensors, each shape `(batch, 1)`, from
  the component heads. (Using a list rather than a dict to keep the
  Layer signature simple; if we later want to name them for clarity,
  a dict is fine.)
- `targets`: same semantics as in `ClassificationHead`.
- **Returns**: per-sample logits, shape `(batch, 1)`.

### Open decision, to resolve before writing `CombinedClassificationHead` internals

**Should the combined head's loss backpropagate into the component heads'
final Dense layers?**

- Default (Yes): gradient flows from combined loss through component
  logits into component heads. Joint multi-task learning — component
  heads are pressured to produce logits useful for both their own task
  and the combined prediction.
- Alternative (No, via `tf.stop_gradient` on `component_logits` inside
  the combined head): component heads are trained purely by their own
  losses; the combined head is a stacked-ensemble predictor on top of
  their (for-its-purposes-frozen) outputs.

Both are defensible and lead to different training dynamics. Decision
pending.

### Patterns introduced (for the mental vocabulary)

- **Endpoint-layer pattern**: a Layer whose `call` conditionally calls
  `self.add_loss(...)` when targets are provided. Exists because
  Keras's functional API otherwise treats models as pure functions;
  endpoint layers are the escape hatch for losses that depend on
  internal state.
- **Head-as-Layer**: each head is a first-class `keras.layers.Layer`
  with its own `__init__`, `call`, and weights scope. Subtly better
  than inline head construction: heads are reusable, nameable,
  serializable, and can be composed into multi-head models cleanly.
- **Dict-valued model outputs**: `keras.Model(inputs, outputs={'cca': ..., 'immig': ...})`
  gives each output a name. When paired with dict-valued targets in
  `.fit()` and dict-valued losses in `compile(loss={...})`, Keras
  routes each target/loss to the matching output by name. This is
  how multi-head training is expressed idiomatically.

---

## Piece 2: Per-layer learning rates and selective unfreezing

**Status:** Implemented in commit `ad0f94b`. `LayerLRModel` at
`src/model_setup/layer_lr_model.py`; tests at
`tests/test_layer_lr_model.py`. A companion callback module for
time-varying multipliers (`src/model_setup/lr_scheduling.py`) was
scoped in the design but deferred until a use case arises; the
current `set_multiplier` API is sufficient for hand-constructed
schedules.

### Decision

A custom `keras.Model` subclass (`LayerLRModel`) that overrides
`train_step` to apply per-variable **learning-rate multipliers** before
invoking the base optimizer. A single base optimizer + single LR
schedule + per-variable multipliers handles both patterns we care about:

- **Discriminative fine-tuning**: set multipliers geometrically by
  layer depth (head = 1.0, encoder layer 11 = 0.95, layer 10 = 0.9,
  …, embeddings = small-or-zero). Stays fixed during training.
- **Gradual / selective unfreezing**: set some multipliers to 0
  initially and update them via a callback at epoch boundaries.
  Zero-multiplier is equivalent to "this group contributes nothing
  to gradient updates right now" without disrupting optimizer state.

### Reasoning

Considered three approaches: multiple optimizers (Option A), single
optimizer with per-variable multipliers (Option B, chosen), and
callback-based `trainable` flag flipping (Option C, ruled out because
it doesn't cover discriminative fine-tuning).

Option B is chosen over Option A because:

- All groups naturally share the same schedule *shape* (warmup →
  cosine decay), just scaled by their multiplier. We don't need
  per-group schedules with different shapes.
- Fewer moving parts: one optimizer's state to maintain, one schedule
  to configure, one place the LR actually lives.
- Multiplier=0 gives clean semantics for freezing without
  optimizer-state surgery; Adam's moments keep updating symmetrically
  (multiplied by the decay factors), and unfreezing by flipping the
  multiplier is discontinuity-free.
- If we ever need totally independent per-group schedule shapes, the
  migration from B to A is not a big refactor.

### The `trainable=False` vs. `multiplier=0` composition

These are not competing mechanisms; they cover different cases and
compose cleanly:

- **`trainable=False`** (permanently frozen): the variable is not
  watched by `GradientTape`, so the backward pass doesn't flow
  through those layers. Real compute savings (substantial for a
  frozen 12-layer transformer). Use for parameters that will never
  train in this run.
- **`multiplier=0`** (temporarily frozen): variable is in the
  trainable set and gradients are computed, but scaled to zero before
  application. Costs the forward + backward compute, but lets us
  toggle freeze status mid-training via callback without optimizer
  surgery. Use for gradual-unfreezing scenarios.

`LayerLRModel.train_step` iterates over `self.trainable_variables` (so
`trainable=False` ones are naturally excluded) and applies multipliers
to what remains.

### Contracts

**`LayerLRModel(inputs, outputs, group_fn, multipliers, **kwargs)`**

- `group_fn: Callable[[tf.Variable], str]` — maps each trainable
  variable to a group name. For a transformer backbone plus named
  heads, this is typically implemented by inspecting `variable.name` (NOTE: check whether this should be `variable.path`)
  (e.g., `"cca_head/intermediate_dense/kernel"` → `"cca_head"`).
- `multipliers: dict[str, float]` — maps group names to scalar
  multipliers in [0, ∞). Missing entries default to 1.0.
- `**kwargs` forwarded to `keras.Model.__init__` (so the usual
  functional API `inputs=..., outputs=...` works, plus `name`, etc.).

**`LayerLRModel.get_multiplier(variable) -> float`**

Look up the scalar multiplier for a given variable by calling
`self.group_fn(variable)` and looking up the result in
`self.multipliers`. Returns 1.0 for variables whose group isn't in
the dict — so "no configuration" means "train normally."

**`LayerLRModel.set_multiplier(group_name, value)`**

Update a single group's multiplier. Called by callbacks at epoch
boundaries for gradual unfreezing.

**`LayerLRModel.train_step(data)`**

Overrides `keras.Model.train_step`. Computes gradients via
`tf.GradientTape`, scales each by its variable's multiplier, applies
via `self.optimizer.apply_gradients`. Updates metrics and returns the
standard log dict.

### Layout

- `src/model_setup/layer_lr_model.py` — `LayerLRModel` class.
- `src/model_setup/lr_scheduling.py` — callback(s) for time-varying
  multipliers (gradual unfreezing schedules). To be added in a
  follow-on commit once the base class works — kept separate so the
  base abstraction can be tested without schedule machinery.
- `tests/test_layer_lr_model.py` — tests for the Model subclass.

### Patterns introduced

- **`Model.train_step` override**: Keras's documented way to
  customize per-step training behavior while keeping `fit()` and its
  infrastructure (callbacks, checkpoints, metrics, distribution
  strategies). This is *the* pattern for custom training logic in
  Keras — when ALUM lands, it will reach for the same pattern.
- **Per-variable gradient scaling**: mathematically equivalent to
  per-variable LR. `var -= lr * m * grad == var -= lr * (m*grad)`.
  Cheaper and simpler than a custom optimizer when the optimizer's
  update rule itself is standard.
- **Dynamic multiplier updates via callback**: standard Keras
  callback API. `on_epoch_begin` mutates `model.multipliers`, next
  epoch trains with new multipliers. No retraining-from-scratch, no
  optimizer reinitialization.
- **`GradientTape` and `compute_loss`**: the low-level pieces. `tape`
  records ops, `tape.gradient(loss, vars)` extracts gradients,
  `model.compute_loss(x, y, y_pred)` aggregates compile-time loss
  with `add_loss` contributions from endpoint layers. Understanding
  these is fundamental for any Level-2+ custom training.

### Backend-agnosticism note

Using `tf.GradientTape` in `train_step` commits this particular code
path to the TensorFlow backend. This is a deliberate choice: our data
pipeline (`tf.data.Dataset`) already commits us to TF, so the
training-loop commitment is free. Writing a backend-agnostic
train_step via Keras's stateless API would add complexity we don't
benefit from, since we have no plans to migrate backends.

The Layers (heads, FLPU) remain backend-agnostic via `keras.ops`; only
the training-step machinery is TF-specific.

---

## Piece 3: Preprocessor refactor for multi-head targets

**Status:** Implemented in commit `e3dda6a`.
`ClassifierPreprocessor` at `src/preproc/preprocessor.py`; tests at
`tests/test_preprocessor.py` (12 tests covering construction, single-
and multi-head shape contracts in both modes, source→output routing,
and dtype casting).

### Decision

`ClassifierPreprocessor` takes a **dict** of label keys mapping
output-dict-key → source-column-name, rather than a single
`label_key`. It supports both endpoint and standard modes (preserving
parity with `ClassificationHead`), and drops the positional-input
branch (datasets are required to be dict-valued, which is already the
case in practice). Dtype handling for targets stays at the loss layer
(FLPU casts to `float32` internally); the preprocessor is dtype-agnostic
on the target side.

A single-head preprocessor is the degenerate case of a one-entry dict,
not a separate code path.

### Reasoning

**Dict over list for `label_keys`.** The output-dict key has to match
the corresponding `keras.Input` name on the model side (in endpoint
mode) or the model output name (in standard mode). It is *semantically
distinct* from the source column name — e.g., source column `cca_label`
maps to model input `cca_targets`. Encoding this mapping as a dict
makes the contract explicit at the call site. A list-with-naming-
convention would hide the coupling behind a derivation rule; that's
exactly the kind of implicit contract that bites later when the
naming convention starts to feel arbitrary.

**Keep both modes.** Piece 1's `ClassificationHead` supports both
endpoint and standard modes, and the symmetry is cheap. FLPU and
eventual ALUM require endpoint mode, so endpoint is the primary path,
but standard mode lets us run a simple-loss baseline (e.g.,
`BinaryCrossentropy`) without a separate preprocessor. The two modes
differ in *which side* the targets-dict serves: in endpoint mode it's
folded into the inputs (consumed by `keras.Input`s wired to head
`targets` arguments); in standard mode it's the second tuple element
(consumed by Keras's `compile(loss={output_name: ...})` routing).

**Drop the positional branch.** The current `text_key=None,
label_key=None` path handled `(text, label)` tuple-valued datasets.
For multi-head this generalizes to `(text, l1, l2, ...)` where
positional order becomes load-bearing — exactly the hidden contract
we just argued against on the label-key side. `run_cca_classification.py`
already produces dict-valued datasets, so there are no live callers to
migrate; any future positional caller can `.map()` to dict form on
construction.

**Targets are model inputs in endpoint mode.** This is the inherent
weirdness of the endpoint-layer pattern, not something the preprocessor
introduces. The preprocessor just produces the dict-shape that Keras's
`.fit()` will route to named `keras.Input`s. Worth a comment in the
class docstring so a reader doesn't have to reverse-engineer the
contract from the head implementation.

**Dtype handling is layered: preprocessor and loss each own a
distinct invariant.** The preprocessor casts targets to a stable,
predictable training dtype (`float32` by default). FLPU still casts
`y_true` to match `y_pred.dtype` at the loss boundary. These look
like the same decision but address different sources of dtype variance:

- The *preprocessor cast* handles **task variance**. Tokenized text
  could in principle feed many tasks with different label dtypes
  (binary classification, regression, generative). Keras-framework
  preprocessors (e.g., `keras_hub.models.RobertaPreprocessor`) leave
  labels untyped because they're general-purpose and don't know the
  task. Our preprocessor *does* know — we're committing to binary
  classification heads — so we can usefully commit to a label dtype
  here.
- The *loss cast* handles **runtime variance** in `y_pred.dtype`
  under `mixed_float16`. `y_pred` may arrive at the loss as `float16`
  or `float32` depending on Keras's loss-scaling machinery; the loss
  has to align `y_true` to whatever `y_pred` actually is. This
  invariant exists regardless of how specific the preprocessor is.

Concretely, the preprocessor cast secures three benefits: (a) the
cached dataset (`cca_set/`, ~minutes to recompute) carries known dtype
rather than re-casting per batch; (b) inference data has the same
shape/dtype contract as training data, with one place to look; (c)
"what does the model see?" has a single clean answer at the
preprocessor.

The loss cast secures mixed-precision robustness — it's a no-op when
dtypes already align, but load-bearing when they don't.

This is not duplication. It is two layers handling two structurally
different problems that happen to share the `ops.cast` primitive.
Removing either cast leaves a real invariant unprotected.

### Layout

- `src/preproc/preprocessor.py` — `ClassifierPreprocessor` rewritten
  in place. `CustomPreprocessor` (the MLM/DAPT preprocessor) is
  unrelated and unchanged.
- `tests/test_preprocessor.py` — new test module covering output-shape
  contracts (single-head, multi-head; standard mode, endpoint mode),
  dict-vs-position rejection, and key-routing correctness. Added
  alongside the implementation.

### Contracts

**`ClassifierPreprocessor.__init__(SEQ_LENGTH, text_key, label_keys, tokenizer=None, endpoint_model=False, target_dtype="float32")`**

- `SEQ_LENGTH`: int, sequence length used for padding/truncation.
- `text_key`: str, dict key of the text column in the input batch.
  Required (no positional fallback).
- `label_keys`: `dict[str, str]` mapping output-dict key → source-column
  name. Order is preserved (Python ≥ 3.7 dict semantics) but not
  load-bearing — Keras routes by name.
- `tokenizer`: optional preconstructed `RobertaTokenizer`. Defaults to
  the `roberta_base_en` preset.
- `endpoint_model`: bool, selects output mode. Default `False`.
- `target_dtype`: dtype string for all targets, default `"float32"`.
  All targets in the output dict are cast to this dtype. Per-output
  dtype overrides are not supported in this version — see Open /
  deferred for the multi-class extension path.

**`ClassifierPreprocessor.__call__(inputs)`** — `inputs` is a dict
yielded by a `tf.data.Dataset`. Tokenizes `inputs[text_key]`, packs
to `SEQ_LENGTH`, casts each target to `target_dtype`, then assembles
output:

- **Endpoint mode** (`endpoint_model=True`): returns a single dict
  `{"token_ids": ..., "padding_mask": ..., **{out_key: inputs[src_col] for out_key, src_col in label_keys.items()}}`.
  All targets folded into the inputs side.
- **Standard mode** (`endpoint_model=False`): returns
  `(features_dict, targets_dict)` where
  `features_dict = {"token_ids": ..., "padding_mask": ...}` and
  `targets_dict = {out_key: inputs[src_col] for out_key, src_col in label_keys.items()}`.
  Targets-dict keys must match the *model output* names so
  `compile(loss={...})` routes correctly.

### Open / deferred

**Asymmetric label availability across heads.** If we ever pull in
data sources where some articles have a CCA label but no immigration
label (or vice versa), the preprocessor will need a masking scheme:
either per-head sample weights (zero out the missing-label samples'
contribution to that head's loss) or a sentinel value combined with
masked-loss handling. Not a live concern — current data has both
columns defined for all rows (null → 0 from Tier 1's data-handling
fixes). Flag here so it isn't lost; promote to a pinned question if it
becomes pressing before Tier 3.

**Per-output target dtypes.** The current spec uses a single
`target_dtype` for all outputs because every head we currently plan is
binary (float32 targets). If a future head needs a different target
dtype — most likely a multi-class head wanting integer class indices
(`int32` or `int64`) for `SparseCategoricalCrossentropy`, or one-hot
floats for `CategoricalCrossentropy` — the natural extension is to
broaden `label_keys` from `dict[str, str]` to
`dict[str, tuple[str, str]]` (where the tuple is `(source_column,
target_dtype)`), or to add a parallel `label_dtypes: dict[str, str]`.
Either keeps the single-dtype shorthand as the default. No need to
build for this now; the extension is local to the preprocessor and
won't disturb callers that don't need it.

### Patterns introduced

- **Explicit output-key → source-column mapping**: when a transformer
  has to produce output names that downstream code routes by name,
  the mapping should be data, not convention. Saves "what was the
  rule again?" lookups every time you add a head.
- **Endpoint-mode targets-as-inputs**: the dict-output shape encodes
  the endpoint-layer contract — targets are inputs because the head's
  `add_loss` consumes them inside the model. This is the
  preprocessor-side counterpart to Piece 1's endpoint-layer pattern.
- **Mode-aware output shape**: a single class produces tuple-shaped
  output for `compile(loss=...)` routing or dict-shaped output for
  `add_loss`-routing. The choice belongs to the caller (because it
  belongs to the model architecture); the preprocessor just honors
  it.

---

## Piece 4: Paths/config consolidation, data-pipeline rename, and integration

**Status:** In progress (design 2026-04-26). Subdivided into three
commits: 4a (config.py + dapt_data rename), 4b (backbone.py +
assembly.py), 4c (training/eval script integration + retirement of
`classifier_from_dapt_checkpoint`).

### Overall framing

Piece 4 is where the abstractions from Pieces 1–3 (`ClassificationHead`,
`LayerLRModel`, multi-head `ClassifierPreprocessor`) actually meet the
training and eval scripts. No new model abstractions are introduced;
the work is plumbing — connecting things that already exist.

The piece is large enough to subdivide. Three sub-commits:

- **4a — Config consolidation + dapt_data rename.** Mechanical:
  introduce `src/config.py` with platform detection + paths +
  dtype-policy default; rename `src/data_setup/dapt_data.py` to
  `src/data_setup/data.py`; update all callers. No semantic changes;
  test suite stays green by construction.
- **4b — Backbone + assembly abstractions.** Add `src/model_setup/backbone.py`
  (DAPT-checkpoint loading split out from `classifier_from_dapt_checkpoint`)
  and `src/model_setup/assembly.py` (wires backbone + heads into a
  full model, returning a `LayerLRModel` for forward-compatibility
  with discriminative LR / unfreezing). Adds an integration test that
  exercises the assembled stack end-to-end on dummy data. Doesn't
  yet touch training/eval scripts.
- **4c — Wiring + retirement.** Rewrite `run_cca_classification.py`
  and `eval_cca_classifier.py` to use the new abstractions; delete
  `classifier_from_dapt_checkpoint` and the now-empty
  `src/model_setup/classification_setup.py`.

### Why `config.py`, not `paths.py`

The platform-detection bit (`IS_CLUSTER`) is the same regardless of
what we use it for. Several values become platform-conditional once
we have it:

- **Paths** (`/projects/ahd/...` vs. local `~/...`).
- **Dtype policy** (`mixed_float16` on cluster CUDA; `float32` on
  local MPS — mixed precision support on MPS has historically been
  patchy and the speedup motivation evaporates anyway).
- **`jit_compile`**, eventually (XLA on Linux+CUDA is reliable; MPS
  XLA less so). Currently `"auto"` works fine; deferred until it
  becomes friction.
- Possibly later: per-platform parallelism, batch-size fallbacks.

If we put detection in `paths.py`, then add `precision.py` later,
either the detection duplicates (drift risk) or `paths.py` exports
an `IS_CLUSTER` symbol that other modules import — which is awkward
because the symbol has nothing to do with paths *qua* paths. A single
`config.py` cleanly absorbs all the platform-conditional defaults
under one name. Adding new platform-conditional values later is a
one-file edit, not a new architectural decision.

### Decisions made (overall)

1. **Detection mechanism: file-existence.** `Path("/projects/ahd").exists()`
   decides cluster vs. local. Simple and robust; the failure mode
   (a laptop that happens to have `/projects/ahd` mounted) is rare
   and survivable via an `ICA_ENV=local` env-var override. Hostname
   detection was rejected because cluster compute nodes have varied
   names. Env-var-only detection was rejected because it requires
   discipline that's easy to forget.
2. **Granular paths.** `config.PROJECT_ROOT` plus named constants
   (`CCA_SET_DIR`, `DAPT_BACKBONE_WEIGHTS`, `LDC_CORPUS`,
   `CCA_CLASSIFIER_DIR`, etc.). Becomes a glossary of "where things
   live in this project," with documentation value beyond convenience.
3. **Single rename, single neutral name.** `src/data_setup/dapt_data.py`
   → `src/data_setup/data.py`. Splitting into `loading.py` +
   `pipeline.py` was considered; deferred to Tier 4 hygiene if the
   need persists. Tier 2 is about shape, not perfection.
4. **`LayerLRModel` wired in 4b/4c, not deferred.** Default multipliers
   are no-op (1.0); current freeze-encoder behavior is preserved.
   Establishing the wiring now closes the question; cost is near-zero.
5. **Delete `classifier_from_dapt_checkpoint` and `classification_setup.py`
   outright in 4c.** Once nothing calls them, keeping them is dead
   code. Git history preserves them if ever needed.
6. **Explicit dtype-policy application at call sites.**
   `config.DTYPE_POLICY` is a value; scripts call
   `keras.config.set_dtype_policy(config.DTYPE_POLICY)` themselves.
   Import-time side effects were considered and rejected for
   surprise reasons — tests want to override; future readers
   shouldn't have to know that importing `config` mutates global
   Keras state. The repetition cost is two lines per script in fewer
   than five scripts.

### Open decisions, to resolve before writing 4b

These belong to 4b's implementation; flagging here so they aren't
forgotten during 4a.

**Train-vs-inference model split.** The training model has
`cca_targets` as a `keras.Input` (endpoint-layer pattern); the
inference model can't, because predict-time data has no targets.
Two patterns:
- Build two models that share head-layer instances (weights shared
  by Python identity within one process).
- Build two separate models and load weights by name across them
  (weights shared by serialization).

The first is cleaner inside `run_cca_classification.py` (one process,
fit + predict); the second is what `eval_cca_classifier.py` needs
(separate process, weights from disk). Probably both patterns get
used. Decision deferred to 4b design discussion.

**`assemble_classifier` signature.** Sketch:

```python
def assemble_classifier(
    backbone,                                    # already-loaded
    heads: dict[str, ClassificationHead],        # output_name -> head
    seq_length: int,
    target_dtype: str = "float32",
    freeze_encoder: bool = False,
    layer_multipliers: dict[str, float] | None = None,
    mode: Literal["train", "inference"] = "train",
) -> LayerLRModel:
    ...
```

Open: should `mode` be a param of one function, or two functions?
Should `target_dtype` be inferred from heads vs. specified here?
Should `freeze_encoder` and `layer_multipliers` be exclusive?
Decisions deferred to 4b.

### Layout (anticipated)

- `src/config.py` — platform detection, paths, dtype policy. New in 4a.
- `src/data_setup/data.py` — renamed from `dapt_data.py`. 4a.
- `src/model_setup/backbone.py` — backbone-loading utilities split
  out of `classifier_from_dapt_checkpoint`. New in 4b.
- `src/model_setup/assembly.py` — assembly function. New in 4b.
- `src/model_setup/classification_setup.py` — deleted in 4c.
- `tests/test_assembly.py` — integration test. New in 4b.

---

## Piece 4a: Config consolidation + dapt_data rename

**Status:** In progress (about to be implemented).

### Decision

`src/config.py` exports:

- `IS_CLUSTER: bool` — `Path("/projects/ahd").exists()`, with
  `ICA_ENV` env var as override (`ICA_ENV=local` forces False;
  `ICA_ENV=cluster` forces True).
- `PROJECT_ROOT: pathlib.Path` — `/projects/ahd` if cluster else
  `~/immigration_project/00_ML_data_expansion/00_explorer`.
- Granular paths derived from `PROJECT_ROOT`: `LDC_CORPUS`,
  `CCA_SET_DIR`, `DAPT_BACKBONE_WEIGHTS`, `DAPT_LM_HEAD_WEIGHTS`,
  `CCA_CLASSIFIER_DIR`, `CCA_LOGS_DIR`, `DAPT_CHECKPOINTS_DIR`,
  `DAPT_LOGS_DIR`, `DAPT_TRAINING_SET`, `DAPT_VALIDATION_SET`. All
  `pathlib.Path` instances.
- `DTYPE_POLICY: str` — `"mixed_float16"` if `IS_CLUSTER` else
  `"float32"`.

`src/data_setup/dapt_data.py` is renamed to `src/data_setup/data.py`.
Imports in `run_cca_classification.py`, `eval_cca_classifier.py`,
`run_prior_estimate.py`, `lu_classifier.py` (and any other callers)
are updated. The file's contents are unchanged — only the filename
moves.

### Reasoning

**Why `pathlib.Path` for the granular paths.** `pathlib.Path`
supports `/` for joining, has methods like `.exists()` and `.mkdir()`,
plays well with both `os.path.isdir(str(path))` (current usage) and
modern `path.is_dir()`. Avoids the `f"{path_prefix}/cca_set"` string
concatenation pattern that's currently strewn across scripts.

**Why granular paths over a single `PROJECT_ROOT`.** Reading the
current scripts, the same path constants are reconstructed across
files (e.g., `f"{path_prefix}/cca_set"` appears in
`run_cca_classification.py`, `eval_cca_classifier.py`, and
`run_prior_estimate.py`). Naming them once in `config.py` makes
"where does the cached CCA set live?" answerable without grep, and
catches any drift between scripts at config-load time.

**Env-var override syntax.** `ICA_ENV=local`, `ICA_ENV=cluster`, or
unset (use the file-existence default). Any other value is treated
as unset (with a warning), so a typo doesn't silently change
behavior.

### Layout

- `src/config.py` — new file.
- `src/data_setup/data.py` — renamed from `dapt_data.py`.
- All callers updated: `run_cca_classification.py`,
  `eval_cca_classifier.py`, `dapt.py`, `run_prior_estimate.py`,
  `lu_classifier.py`, and any other modules importing from
  `data_setup.dapt_data` or referencing `path_prefix`.

### Contracts

**`config.py`** — module-level constants. No functions, no classes.
Importing the module is enough to use the values. Side effects:
none — `keras.config.set_dtype_policy(...)` is left to callers
(per the explicit-application decision above).

**`config.IS_CLUSTER`**: bool. `True` when `/projects/ahd` exists or
`ICA_ENV=cluster`; `False` when `ICA_ENV=local`; defaults to
`Path("/projects/ahd").exists()`.

**`config.PROJECT_ROOT`**: `pathlib.Path`. `Path("/projects/ahd")` if
cluster, else `Path.home() / "immigration_project" / "00_ML_data_expansion" / "00_explorer"`.

**Named granular paths**: all `pathlib.Path` derived from
`PROJECT_ROOT`. Names follow the existing folder structure:
`CCA_SET_DIR = PROJECT_ROOT / "cca_set"` etc.

**`config.DTYPE_POLICY`**: str. `"mixed_float16"` if cluster else
`"float32"`.

### Patterns introduced

- **Single-source-of-truth platform detection**: one bit
  (`IS_CLUSTER`) is computed once at config load; all
  platform-conditional values flow from it.
- **Env-var override for boolean detection**: simple, debuggable,
  and survives the rare case where automatic detection is wrong.
- **Module-level path constants**: paths as named values rather
  than reconstructed strings. Mostly an ergonomic + reliability
  win, but also turns `config.py` into a glossary of project
  artifact locations.

### Test impact

No new tests in 4a — there's no logic to test beyond the platform
detection itself, which depends on the runtime environment and is
brittle to mock without sacrificing the test's value. The test
suite verifies the rename mechanically: imports in test files
either still resolve (if `data_setup.data` is correctly created
and old name removed) or fail loudly. Existing 65 tests should
continue to pass.

---
