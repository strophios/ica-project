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
