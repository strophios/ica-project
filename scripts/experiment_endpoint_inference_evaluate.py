"""
Experiment documenting the empirical safety of Pattern A
(shared-head-instances) for endpoint-mode train/inference model
splits in Keras 3.

Context: Tier 2 Piece 4b. We chose Pattern A — sharing
`ClassificationHead` Layer instances between the training model
(targets-as-inputs, head's add_loss handles loss) and the inference
model (no targets, head called with targets=None) — over Pattern 2
(always rebuild heads + load weights by name). The original concern
was that the head's `_losses` list, populated when the head is
called with targets in the training graph, would contaminate
`evaluate()` on the inference model.

This script demonstrates that Keras 3 correctly filters losses by
graph reachability, so the contamination doesn't fire:

  - `predict()` on shared-instance inference model: works.
  - `evaluate()` without compile-time loss: clean ValueError
    ("No loss to compute. Provide a `loss` argument in `compile()`.").
  - `evaluate()` with compile-time loss + labels: returns the
    correct fresh loss; matches a direct BCE computation to 6
    decimals — no contamination from the head's stale training-graph
    add_loss tensor.
  - `inf_model.losses` returns 0 tensors (Keras filters losses
    whose dependencies aren't in the inference graph).

Kept as a permanent fixture so this finding stays verifiable —
re-run if Keras versions change behavior, or as a sanity check
during model-stack changes. The script is self-contained and
doesn't depend on real data or weights.

Run with: `PYTHONPATH=. uv run python scripts/experiment_endpoint_inference_evaluate.py`

See `docs/notes/tier2-design.md` Piece 4b for the design context
and reasoning.
"""

import numpy as np
import keras

from src.model_setup.heads import ClassificationHead
from src.loss_functions.loss import FLPULoss


# Tiny synthetic setup
VOCAB = 100
SEQ_LEN = 4
HIDDEN_DIM = 8
BATCH = 4

# Shared sublayers (these will appear in BOTH functional graphs)
embed = keras.layers.Embedding(VOCAB, HIDDEN_DIM, name="fake_embed")
dense = keras.layers.Dense(HIDDEN_DIM, activation="relu", name="fake_dense")
head = ClassificationHead(
    hidden_dim=HIDDEN_DIM,
    loss_fn=FLPULoss(prior=0.1),
    name="cca_head",
)


def fake_backbone(tok):
    embedded = embed(tok)
    pooled = keras.ops.mean(embedded, axis=1)
    return dense(pooled)


# --- Build training model (with target inputs, endpoint head) ---
# (We drop padding_mask for the experiment — fake_backbone doesn't use it.
#  In the real model it'd be wired through the backbone's forward pass.)
token_ids = keras.Input(shape=(SEQ_LEN,), dtype="int32", name="token_ids")
cca_targets = keras.Input(shape=(), dtype="float32", name="cca_targets")

features_train = fake_backbone(token_ids)
logits_train = head(features_train, targets=cca_targets)

train_model = keras.Model(
    inputs={"token_ids": token_ids, "cca_targets": cca_targets},
    outputs={"cca": logits_train},
)

# --- Build inference model (no target input, head called with targets=None) ---
# Share embed, dense, head Layer instances — that's the whole point of Pattern A.
token_ids_inf = keras.Input(shape=(SEQ_LEN,), dtype="int32", name="token_ids")

features_inf = fake_backbone(token_ids_inf)
logits_inf = head(features_inf)  # NO targets, no add_loss this call

inf_model = keras.Model(
    inputs={"token_ids": token_ids_inf},
    outputs={"cca": logits_inf},
)


# --- Pre-fit inspection ---
print("=" * 60)
print("BEFORE training:")
print(f"  head._losses count:        {len(head._losses)}")
print(f"  train_model.losses count:  {len(train_model.losses)}")
print(f"  inf_model.losses count:    {len(inf_model.losses)}")


# --- Train one step ---
train_model.compile(optimizer="adam")  # no compile-time loss — head's add_loss handles it
data = {
    "token_ids": np.random.randint(0, VOCAB, (BATCH, SEQ_LEN)).astype("int32"),
    "cca_targets": np.array([0.0, 1.0, 0.0, 1.0], dtype="float32"),
}
train_model.fit(data, epochs=1, batch_size=BATCH, verbose=0)

print()
print("AFTER training:")
print(f"  head._losses count:        {len(head._losses)}")
print(f"  train_model.losses count:  {len(train_model.losses)}")
print(f"  inf_model.losses count:    {len(inf_model.losses)}")


# --- Probe predict() (should always work) ---
print()
print("=" * 60)
print("Probing inf_model.predict()...")
inf_inputs = {"token_ids": data["token_ids"]}
preds = inf_model.predict(inf_inputs, verbose=0)
out = preds["cca"] if isinstance(preds, dict) else preds
print(f"  predict() succeeded; output shape = {out.shape}")


# --- Probe evaluate() (this is the question) ---
print()
print("=" * 60)
print("Probing inf_model.evaluate()...")
inf_model.compile(optimizer="adam")  # compile without loss; user might do this for metrics

try:
    result = inf_model.evaluate(inf_inputs, verbose=0, return_dict=True)
    print(f"  evaluate() SUCCEEDED")
    print(f"  result: {result}")
    print(f"  (note whether 'loss' is finite, garbage, or absent)")
except Exception as e:
    print(f"  evaluate() FAILED with {type(e).__name__}")
    print(f"  message: {e}")


# --- Probe what happens if we explicitly try to access inf_model.losses ---
print()
print("=" * 60)
print("Inspecting inf_model.losses tensors (after no-loss-compile)...")
try:
    losses = inf_model.losses
    print(f"  inf_model.losses returned {len(losses)} tensor(s)")
    for i, loss_tensor in enumerate(losses):
        print(f"    [{i}] type={type(loss_tensor).__name__}, "
              f"repr={repr(loss_tensor)[:120]}")
except Exception as e:
    print(f"  inf_model.losses access FAILED with {type(e).__name__}: {e}")


# --- Now the scenario I care about: user compiles inf_model with a real loss
#     and provides labels. Does the head's stale add_loss tensor contaminate
#     the computed loss?
print()
print("=" * 60)
print("Probing inf_model.evaluate() WITH compile-time loss...")
inf_model.compile(
    optimizer="adam",
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
)
inf_inputs_with_labels = {"token_ids": data["token_ids"]}
labels = data["cca_targets"]  # reuse training labels for the experiment

try:
    result = inf_model.evaluate(
        inf_inputs_with_labels, labels, verbose=0, return_dict=True
    )
    print(f"  evaluate() SUCCEEDED")
    print(f"  result: {result}")
    # Compare to what BCE alone should give on these inputs
    preds_logits = inf_model.predict(inf_inputs_with_labels, verbose=0)
    preds_logits = preds_logits["cca"] if isinstance(preds_logits, dict) else preds_logits
    bce_only = keras.losses.BinaryCrossentropy(from_logits=True)(
        labels, preds_logits.squeeze()
    )
    print(f"  Direct BCE(labels, preds): {float(bce_only):.6f}")
    print(f"  evaluate-reported loss:     {result.get('loss', 'absent'):.6f}")
    if abs(float(result.get("loss", 0)) - float(bce_only)) < 1e-4:
        print("  → MATCH: evaluate's loss == fresh BCE; no add_loss contamination")
    else:
        print("  → MISMATCH: evaluate's loss differs; add_loss contamination?")
except Exception as e:
    print(f"  evaluate() FAILED with {type(e).__name__}")
    print(f"  message: {e}")

print()
print(f"Final inf_model.losses count: {len(inf_model.losses)}")
print(f"Final head._losses count:     {len(head._losses)}")
