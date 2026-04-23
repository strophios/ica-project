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
