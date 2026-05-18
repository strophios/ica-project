"""
Tests for `src.model_setup.layer_lr_model.LayerLRModel`.

The tests fall into three groups:

  - TestConstruction / TestMultiplierLookup: the non-training mechanics.
    `group_fn`, `multipliers`, `get_multiplier`, `set_multiplier`. These
    pass as soon as `__init__` and the lookup methods are present (they
    are already implemented in the skeleton), and fail if `train_step`
    is broken enough to block instantiation.

  - TestTrainStepIntegration: actual training. These construct a tiny
    2-layer dense model wrapped in `LayerLRModel`, compile it, and call
    `fit()` for a couple of steps with carefully-chosen multipliers to
    verify that:
      - multiplier=1.0 matches an equivalent plain-Keras model,
      - multiplier=0 fully freezes a group (weights don't change),
      - multiplier=0.5 halves the weight update relative to 1.0
        (using SGD — Adam's first-step normalization would mask scalar
        multipliers, so we use SGD for deterministic scaling).
    `fit()` + SGD keeps the test fast and the math predictable.

  - TestWithEndpointLoss: verifies that a model using the endpoint-layer
    pattern (loss registered via `add_loss`) trains correctly under
    `LayerLRModel`. This is the interaction with Piece 1's
    ClassificationHead, so we use that class here.

All tests use small synthetic data and a CPU-friendly model.
"""

import numpy as np
import pytest

import keras  # noqa: F401

from src.model_setup.layer_lr_model import LayerLRModel
from src.model_setup.heads import ClassificationHead
from src.diagnostics.trackers import (
    PerGroupGradNormTracker,
    GradientFiniteTracker,
    LossComponentTracker,
)
from src.loss_functions.loss import FLPULoss


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

INPUT_DIM = 4
HIDDEN_DIM = 8
BATCH = 16


def _tiny_model(multipliers=None, group_fn=None) -> LayerLRModel:
    """A 2-dense-layer functional-API model wrapped in LayerLRModel.

    Two named Dense layers — `lower` and `upper` — so group_fn can sort
    their weights into two groups for multiplier tests.
    """
    inputs = keras.Input(shape=(INPUT_DIM,), name="x")
    h = keras.layers.Dense(HIDDEN_DIM, activation="relu", name="lower")(inputs)
    outputs = keras.layers.Dense(1, activation=None, name="upper")(h)

    if group_fn is None:
        # In Keras 3, var.name is the short name ("kernel", "bias");
        # var.path is the full hierarchical path ("lower/kernel",
        # "upper/bias"). Groupings based on layer identity should use path.
        def group_fn(var):
            if "upper" in var.path:
                return "upper"
            if "lower" in var.path:
                return "lower"
            return "default"

    return LayerLRModel(
        inputs=inputs,
        outputs=outputs,
        group_fn=group_fn,
        multipliers=multipliers,
    )


def _dummy_data(n=BATCH):
    rng = np.random.RandomState(0)
    x = rng.randn(n, INPUT_DIM).astype(np.float32)
    y = rng.randn(n, 1).astype(np.float32)
    return x, y


# -----------------------------------------------------------------------------
# Construction and multiplier bookkeeping
# -----------------------------------------------------------------------------


class TestConstruction:
    def test_constructs_with_multipliers(self):
        model = _tiny_model(multipliers={"upper": 1.0, "lower": 0.5})
        assert model.multipliers == {"upper": 1.0, "lower": 0.5}

    def test_constructs_without_multipliers(self):
        """Omitting multipliers should give an empty dict (all defaults)."""
        model = _tiny_model()
        assert model.multipliers == {}

    def test_group_fn_stored(self):
        """The supplied group_fn should be callable on the model's variables."""
        model = _tiny_model(multipliers={"upper": 0.9, "lower": 0.1})
        groups = {model.group_fn(v) for v in model.trainable_variables}
        assert groups == {"upper", "lower"}


class TestMultiplierLookup:
    def test_get_multiplier_returns_configured_value(self):
        model = _tiny_model(multipliers={"upper": 0.7, "lower": 0.3})
        for var in model.trainable_variables:
            if "upper" in var.path:
                assert model.get_multiplier(var) == 0.7
            elif "lower" in var.path:
                assert model.get_multiplier(var) == 0.3

    def test_get_multiplier_defaults_to_1_for_unknown_group(self):
        """Variables whose group isn't in the multipliers dict should
        default to 1.0 — this is the 'not configured = train normally'
        property that lets LayerLRModel drop in as a Model replacement."""
        model = _tiny_model(multipliers={"upper": 0.5})  # no "lower" entry
        for var in model.trainable_variables:
            if "lower" in var.path:
                assert model.get_multiplier(var) == 1.0

    def test_set_multiplier_updates_value(self):
        """set_multiplier should mutate the dict; get_multiplier reflects it."""
        model = _tiny_model(multipliers={"upper": 1.0, "lower": 1.0})
        model.set_multiplier("upper", 0.25)
        upper_var = next(v for v in model.trainable_variables if "upper" in v.path)
        assert model.get_multiplier(upper_var) == 0.25


# -----------------------------------------------------------------------------
# train_step with real training (SGD, so multipliers affect weights linearly)
# -----------------------------------------------------------------------------


class TestTrainStepIntegration:
    def test_uniform_multiplier_matches_plain_sgd(self):
        """With all multipliers = 1.0, LayerLRModel should train identically
        to a plain Model on the same data. Tests that the train_step doesn't
        introduce spurious scaling when multipliers are trivial."""
        x, y = _dummy_data()

        # Two models with identical initializations.
        keras.utils.set_random_seed(42)
        model_lr = _tiny_model(multipliers={"upper": 1.0, "lower": 1.0})
        model_lr.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1), loss="mse")

        keras.utils.set_random_seed(42)
        inputs = keras.Input(shape=(INPUT_DIM,), name="x")
        h = keras.layers.Dense(HIDDEN_DIM, activation="relu", name="lower")(inputs)
        outputs = keras.layers.Dense(1, activation=None, name="upper")(h)
        model_plain = keras.Model(inputs=inputs, outputs=outputs)
        model_plain.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.1), loss="mse"
        )

        model_lr.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        model_plain.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)

        for w_lr, w_plain in zip(
            model_lr.trainable_weights, model_plain.trainable_weights
        ):
            np.testing.assert_allclose(
                w_lr.numpy(),
                w_plain.numpy(),
                rtol=1e-5,
                atol=1e-6,
                err_msg=(
                    f"Uniform multiplier=1.0 gave different weights than "
                    f"plain SGD for variable {w_lr.name}"
                ),
            )

    def test_multiplier_zero_freezes_group(self):
        """A group with multiplier=0 should have its weights unchanged
        after any amount of training. This is the freeze contract."""
        x, y = _dummy_data()

        keras.utils.set_random_seed(42)
        model = _tiny_model(multipliers={"upper": 1.0, "lower": 0.0})
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1), loss="mse")

        # Snapshot the 'lower' weights before training.
        lower_before = [
            v.numpy().copy() for v in model.trainable_variables if "lower" in v.path
        ]
        upper_before = [
            v.numpy().copy() for v in model.trainable_variables if "upper" in v.path
        ]

        model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)

        lower_after = [
            v.numpy() for v in model.trainable_variables if "lower" in v.path
        ]
        upper_after = [
            v.numpy() for v in model.trainable_variables if "upper" in v.path
        ]

        # "lower" should be unchanged (multiplier=0)
        for before, after in zip(lower_before, lower_after):
            np.testing.assert_allclose(
                before,
                after,
                rtol=1e-7,
                err_msg="multiplier=0 did not freeze 'lower' group weights",
            )

        # "upper" should have changed (multiplier=1.0)
        upper_changed = any(
            not np.allclose(b, a) for b, a in zip(upper_before, upper_after)
        )
        assert upper_changed, (
            "'upper' weights did not change; the test setup may be broken "
            "(check that the loss is actually affecting upper weights)."
        )

    def test_multiplier_scales_sgd_update_linearly(self):
        """With SGD, a gradient scaled by m produces a weight change scaled
        by m. So multiplier=0.5 should produce exactly half the weight delta
        of multiplier=1.0 (same initial weights, same data, same batch order).
        This is the core correctness test for per-variable LR scaling."""
        x, y = _dummy_data()

        keras.utils.set_random_seed(42)
        model_full = _tiny_model(multipliers={"upper": 1.0, "lower": 1.0})
        model_full.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.1), loss="mse"
        )

        keras.utils.set_random_seed(42)
        model_half = _tiny_model(multipliers={"upper": 1.0, "lower": 0.5})
        model_half.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.1), loss="mse"
        )

        # Snapshot initial weights (should match across the two models).
        initial_lower_full = [
            v.numpy().copy()
            for v in model_full.trainable_variables
            if "lower" in v.path
        ]
        initial_lower_half = [
            v.numpy().copy()
            for v in model_half.trainable_variables
            if "lower" in v.path
        ]
        for a, b in zip(initial_lower_full, initial_lower_half):
            np.testing.assert_allclose(
                a, b, rtol=1e-7, err_msg="models did not initialize identically"
            )

        # Train one batch on each (same data, same order).
        model_full.fit(x, y, epochs=1, batch_size=BATCH, verbose=0, shuffle=False)
        model_half.fit(x, y, epochs=1, batch_size=BATCH, verbose=0, shuffle=False)

        # Compare the actual 'lower' weight deltas.
        for name, init, full_var, half_var in zip(
            [v.name for v in model_full.trainable_variables if "lower" in v.path],
            initial_lower_full,
            [v for v in model_full.trainable_variables if "lower" in v.path],
            [v for v in model_half.trainable_variables if "lower" in v.path],
        ):
            delta_full = full_var.numpy() - init
            delta_half = half_var.numpy() - init
            # Half-multiplier's delta should be exactly half of full's delta.
            np.testing.assert_allclose(
                delta_half,
                delta_full * 0.5,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"multiplier=0.5 did not produce half the delta for {name}",
            )


# -----------------------------------------------------------------------------
# Integration with endpoint-layer heads (FLPU-style losses inside the model)
# -----------------------------------------------------------------------------


class TestWithEndpointLoss:
    def test_trains_with_endpoint_head(self):
        """LayerLRModel should handle endpoint-layer losses (registered via
        add_loss inside a head) the same as compile-time losses. This is the
        interaction with Piece 1's ClassificationHead."""
        from src.loss_functions.loss import FLPULoss

        # Inputs: features tensor (what the head expects, standing in for
        # the backbone CLS output) plus targets tensor for the endpoint
        # loss routing.
        features_input = keras.Input(shape=(INPUT_DIM,), name="features")
        targets_input = keras.Input(shape=(1,), name="targets")

        head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=FLPULoss(prior=0.1),
            name="test_head",
        )
        logits = head(features_input, targets=targets_input)

        def group_fn(var):
            return "head"

        model = LayerLRModel(
            inputs={"features": features_input, "targets": targets_input},
            outputs=logits,
            group_fn=group_fn,
            multipliers={"head": 1.0},
        )
        # No compile-time loss — the head's add_loss is the only loss.
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1))

        # Snapshot weights before.
        before = [v.numpy().copy() for v in model.trainable_variables]

        # Train one epoch.
        rng = np.random.RandomState(0)
        features = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        targets = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)
        model.fit(
            x={"features": features, "targets": targets},
            epochs=1,
            batch_size=BATCH,
            verbose=0,
        )

        # Weights should have changed.
        after = [v.numpy() for v in model.trainable_variables]
        any_changed = any(not np.allclose(b, a) for b, a in zip(before, after))
        assert any_changed, (
            "Model trained on endpoint loss did not update any weights. "
            "Likely failure mode: compute_loss did not include self.losses "
            "contributions, or add_loss was not called from the head."
        )


# -----------------------------------------------------------------------------
# Sparse-gradient regression
# -----------------------------------------------------------------------------

class TestSparseGradients:
    """
    Regression test for an `IndexedSlices` handling bug discovered during
    Piece 4b. `Embedding` layers produce sparse gradients (only the
    looked-up rows are non-zero), represented as `tf.IndexedSlices`
    rather than dense tensors. The original `train_step` did
    `multiplier * grad`, which fails for `IndexedSlices` because
    Python's `float * IndexedSlices` has no defined operator. The fix
    uses `tf.math.scalar_mul`, which handles both dense tensors and
    `IndexedSlices`.

    This test exercises that path: a model with a trainable
    `Embedding`, wrapped in `LayerLRModel`, runs one fit step. Before
    the fix this would raise:

        TypeError: unsupported operand type(s) for *: 'float' and
                   'IndexedSlices'

    After the fix, the step succeeds and the embedding weights update.
    """

    def test_embedding_layer_trains_under_layer_lr_model(self):
        VOCAB = 50
        EMBED_DIM = 4
        SEQ_LEN = 3

        # Functional model: Embedding → mean-pool → Dense head.
        # The Embedding's gradient comes back as IndexedSlices on TF.
        token_ids = keras.Input(shape=(SEQ_LEN,), dtype="int32", name="token_ids")
        embedded = keras.layers.Embedding(VOCAB, EMBED_DIM, name="embed")(token_ids)
        pooled = keras.ops.mean(embedded, axis=1)
        logits = keras.layers.Dense(1, activation=None, name="head")(pooled)

        model = LayerLRModel(
            inputs=token_ids,
            outputs=logits,
            group_fn=lambda v: "default",
            multipliers={"default": 0.5},  # non-1.0 to actually exercise scaling
        )
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-2),
                      loss=keras.losses.BinaryCrossentropy(from_logits=True))

        rng = np.random.RandomState(0)
        x = rng.randint(0, VOCAB, size=(BATCH, SEQ_LEN)).astype("int32")
        y = rng.randint(0, 2, size=(BATCH, 1)).astype("float32")

        # Snapshot embedding weights, run one step, check they updated.
        # If the IndexedSlices bug is back, fit() will raise TypeError.
        embed_layer = model.get_layer("embed")
        before = embed_layer.embeddings.numpy().copy()
        model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        after = embed_layer.embeddings.numpy()

        assert not np.allclose(before, after), (
            "Embedding weights unchanged after one training step — "
            "fit() may have silently no-op'd despite reaching the "
            "scaled-gradient path."
        )


# -----------------------------------------------------------------------------
# Loss tracking (regression for Tier 2 review C1)
# -----------------------------------------------------------------------------


class TestLossTracking:
    """Regression for a Tier 2 review finding (C1): the original
    `LayerLRModel.train_step` did not update `self._loss_tracker`, so
    `history.history["loss"]` was always 0 even when training was
    actively happening. Stock `keras.Model.train_step` (TF backend)
    calls `self._loss_tracker.update_state(loss, sample_weight=batch_size)`
    immediately after computing the loss; the original LayerLRModel
    train_step omitted this entirely.

    Symptom in production: fit's progress bar shows
    `loss: 0.0000e+00`, `history.history["loss"]` is all zeros,
    TensorBoard's train/loss curve is flat, callbacks reading the
    training-side loss see only zeros. Validation-side `val_loss`
    still works (stock `test_step` is unaffected); training still
    happens (gradients are computed correctly); only the training-
    side loss tracking is silently broken.

    The fix in `train_step` adds an explicit
    `_loss_tracker.update_state(loss, ...)` call. See
    `docs/notes/tier2-design.md` Piece 2 (post-review update) for
    the full reasoning, including the related `optimizer.scale_loss`
    omission that breaks `LossScaleOptimizer` under `mixed_float16`."""

    def test_fit_history_records_nonzero_loss(self):
        """The simplest expression of the bug: train one epoch with a
        compile-time loss and confirm history.history['loss'] is a
        finite non-zero value, not 0.0."""
        model = _tiny_model()
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.01),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        # Use binary labels so BCE produces a meaningful (>0) loss.
        rng = np.random.RandomState(42)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)

        history = model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)

        assert "loss" in history.history, (
            "history.history is missing 'loss' key — Keras's automatic "
            "loss-metric registration may have been disrupted."
        )
        loss_values = history.history["loss"]
        assert len(loss_values) == 1, (
            f"Expected one loss value (one epoch), got {len(loss_values)}: "
            f"{loss_values}"
        )
        assert np.isfinite(loss_values[0]), (
            f"Loss is non-finite: {loss_values[0]}"
        )
        assert loss_values[0] > 0, (
            f"Expected non-zero loss in history but got {loss_values[0]}. "
            "Likely cause: LayerLRModel.train_step does not call "
            "self._loss_tracker.update_state(loss). See Tier 2 review C1."
        )

    def test_fit_history_loss_close_to_evaluate_loss(self):
        """Stronger check: after one epoch of fit, history's reported
        loss should be in the same ballpark as evaluate()'s reported
        loss on the same data. They won't be exactly equal — fit's
        is averaged over training batches while weights change, while
        evaluate() uses the final post-fit weights — but they should
        agree to within ~50% relative difference. (BCE on a randomly-
        initialized linear model on random labels lands around
        ln(2) ≈ 0.69; one epoch of SGD doesn't move it much.)

        This catches the case where _loss_tracker is updated with
        a wrong value (e.g., scale_loss-scaled loss instead of raw
        loss, or an unrelated tensor)."""
        model = _tiny_model()
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.01),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        rng = np.random.RandomState(42)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)

        history = model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        eval_results = model.evaluate(x, y, batch_size=BATCH, verbose=0,
                                      return_dict=True)

        fit_loss = history.history["loss"][0]
        eval_loss = eval_results["loss"]
        rel_diff = abs(fit_loss - eval_loss) / max(abs(eval_loss), 1e-9)
        assert rel_diff < 0.5, (
            f"fit loss ({fit_loss:.4f}) and evaluate loss ({eval_loss:.4f}) "
            f"differ by {rel_diff:.1%} — expected < 50%. The loss tracker "
            "may be receiving a transformed/wrong value rather than the "
            "raw compute_loss result."
        )


class TestDiagnosticsNoOpRegression:
    """With diagnostic_trackers=None (the default every existing call site
    uses), LayerLRModel must behave exactly as before Tier 5 Phase 4."""

    def test_default_init_has_none_trackers(self):
        model = _tiny_model()
        assert model._diagnostic_trackers is None
        assert model._head_refs_by_name == {}

    def test_metrics_property_unchanged_when_none(self):
        model = _tiny_model()
        model.compile(
            optimizer=keras.optimizers.SGD(0.01),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        # Build metrics by running one step.
        rng = np.random.RandomState(0)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)
        model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        # No extra metrics beyond what stock Keras exposes (loss tracker).
        names = [m.name for m in model.metrics]
        assert "loss" in names
        assert not any(n.startswith("grad_norm/") for n in names)
        assert not any(n.startswith("grad_overflow") for n in names)

    def test_fit_history_records_nonzero_loss_no_trackers(self):
        # Mirror of the Tier-2 regression, asserted explicitly under the
        # Phase-4 change with trackers absent.
        model = _tiny_model()
        model.compile(
            optimizer=keras.optimizers.SGD(0.01),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        rng = np.random.RandomState(42)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)
        history = model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        assert np.isfinite(history.history["loss"][0])
        assert history.history["loss"][0] > 0


class TestGradientDiagnosticDispatch:
    def _bundle(self, trackers):
        return {
            "per_step": {
                "gradient": trackers,
                "loss_component": [],
                "batch_target": [],
            },
            "periodic": [],
        }

    def test_grad_norm_tracker_sees_raw_unscaled_gradient(self):
        # The design's load-bearing invariant (lines 165-176): trackers
        # observe the COMPUTED gradient, BEFORE per-variable multiplier
        # scaling. Differential test (no fragile closed-form re-derivation):
        # two models with IDENTICAL initial weights and the SAME batch, one
        # with multiplier 1.0 and one with 10.0 on the tracked "lower"
        # group, each with its own grad-norm tracker. The gradient computed
        # by tape.gradient is identical in both (multiplier is applied
        # AFTER). So if trackers observe the raw gradient (correct), both
        # report the SAME norm. If they observed multiplier*grad (the bug
        # this guards), the 10.0-model's tracker would be ~10x the other.
        def group_fn(v):
            return v.path.split("/")[0]
        rng = np.random.RandomState(1)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)

        tracker_1 = PerGroupGradNormTracker(group_name="lower", aggregation="mean")
        tracker_10 = PerGroupGradNormTracker(group_name="lower", aggregation="mean")
        base_1 = _tiny_model(group_fn=group_fn)
        base_10 = _tiny_model(group_fn=group_fn)
        model_1 = LayerLRModel(
            inputs=base_1.inputs, outputs=base_1.outputs, group_fn=group_fn,
            multipliers={"lower": 1.0},
            diagnostic_trackers=self._bundle([tracker_1]),
        )
        model_10 = LayerLRModel(
            inputs=base_10.inputs, outputs=base_10.outputs, group_fn=group_fn,
            multipliers={"lower": 10.0},
            diagnostic_trackers=self._bundle([tracker_10]),
        )
        # Force identical initial weights so the computed gradient is
        # identical across both models for the same batch.
        model_10.set_weights(model_1.get_weights())
        for m in (model_1, model_10):
            m.compile(
                optimizer=keras.optimizers.SGD(0.0),  # no update; weights stay equal
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
            )
        model_1.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        model_10.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)

        v1 = float(tracker_1.result())
        v10 = float(tracker_10.result())
        assert v1 > 0.0  # sanity: a gradient was actually observed
        # Correct (pre-scaling) behavior: equal regardless of multiplier.
        assert v10 == pytest.approx(v1, rel=1e-4), (
            f"grad-norm tracker is multiplier-sensitive: v1={v1}, v10={v10} "
            "— it is observing multiplier*grad (post-scaling), violating the "
            "design's pre-scaling invariant."
        )
        # Explicitly reject the specific failure mode.
        assert v10 != pytest.approx(10.0 * v1, rel=1e-3)

    def test_overflow_tracker_zero_under_float32(self):
        tracker = GradientFiniteTracker()
        base = _tiny_model()
        model = LayerLRModel(
            inputs=base.inputs,
            outputs=base.outputs,
            diagnostic_trackers=self._bundle([tracker]),
        )
        model.compile(
            optimizer=keras.optimizers.SGD(0.01),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        rng = np.random.RandomState(2)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)
        model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        assert float(tracker.result()) == 0.0  # finite grads under float32


class TestLossComponentBatchTargetDispatch:
    def test_loss_component_trackers_populated_from_last_components(self):
        # Build an endpoint LayerLRModel with a single CCA head exposing
        # loss components (mirror TestWithEndpointLoss setup). Note: batch-target
        # tracker requires y to be a dict of form {head_name: targets}, which only
        # works in models where targets are fed separately (not packed into x).
        # Here we omit the batch_target tracker and focus on loss_component.
        head = ClassificationHead(
            hidden_dim=INPUT_DIM,
            loss_fn=FLPULoss(prior=0.1),
            name="cca",
            expose_loss_components=True,
        )
        # Inputs: features tensor (what the head expects) plus targets
        # tensor for the endpoint loss routing.
        features_input = keras.Input(shape=(INPUT_DIM,), name="features")
        cca_targets_input = keras.Input(shape=(1,), name="cca_targets")

        # Call head with targets to register add_loss.
        logits = head(features_input, targets=cca_targets_input)

        lc = LossComponentTracker("cca", "positive_risk", "mean")
        bundle = {
            "per_step": {
                "gradient": [],
                "loss_component": [lc],
                "batch_target": [],
            },
            "periodic": [],
        }
        model = LayerLRModel(
            inputs={"features": features_input, "cca_targets": cca_targets_input},
            outputs={"cca": logits},
            diagnostic_trackers=bundle,
            diagnostic_head_refs=[head],
        )
        model.compile(optimizer=keras.optimizers.SGD(0.01))

        # Train one epoch on synthetic data.
        rng = np.random.RandomState(42)
        features = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        cca_targets = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)
        model.fit(
            x={"features": features, "cca_targets": cca_targets},
            epochs=1,
            batch_size=BATCH,
            verbose=0,
        )

        # Tracker should be populated after one epoch with loss component.
        assert np.isfinite(float(lc.result()))

    def test_batch_target_tracker_works_in_endpoint_mode(self):
        """Regression test for commit 4e5cce1 (Tier 5 Phase 6).

        This test guards the fix for a critical integration bug:
        in endpoint mode, targets are packed into x as {head}_targets
        keys and Keras yields y=None. The original _dispatch_diagnostics
        call received y=None, causing BatchLabelBalanceTracker.update_state(None)
        to fail with TypeError (cannot check 'key in None').

        The fix (commit 4e5cce1) extracts {head}_targets from x into
        a targets_for_dispatch dict when y is None and x is a dict,
        then passes targets_for_dispatch to _dispatch_diagnostics.

        This test:
        1. Builds an endpoint LayerLRModel with a BatchLabelBalanceTracker.
        2. Fits on synthetic data with a known positive-class fraction.
        3. Asserts the tracker was populated with a sensible value.

        Regression to the pre-fix behavior (y=None → no extraction) is caught
        because the tracker.update_state(None) call will raise TypeError.
        """
        from src.diagnostics.trackers import BatchLabelBalanceTracker

        head = ClassificationHead(
            hidden_dim=INPUT_DIM,
            loss_fn=FLPULoss(prior=0.1),
            name="cca",
            expose_loss_components=False,
        )
        features_input = keras.Input(shape=(INPUT_DIM,), name="features")
        cca_targets_input = keras.Input(shape=(1,), name="cca_targets")

        logits = head(features_input, targets=cca_targets_input)

        # Use BatchLabelBalanceTracker — the tracker that was broken
        # in endpoint mode before the fix (it needs y as a dict).
        batch_tracker = BatchLabelBalanceTracker("cca")
        bundle = {
            "per_step": {
                "gradient": [],
                "loss_component": [],
                "batch_target": [batch_tracker],
            },
            "periodic": [],
        }
        model = LayerLRModel(
            inputs={"features": features_input, "cca_targets": cca_targets_input},
            outputs={"cca": logits},
            diagnostic_trackers=bundle,
            diagnostic_head_refs=[head],
        )
        model.compile(optimizer=keras.optimizers.SGD(0.01))

        # Synthetic data: 12 positives, 4 negatives → 0.75 positive fraction
        rng = np.random.RandomState(99)
        features = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        # Deliberately create a batch with known positive-class fraction:
        # 12 ones (positive) + 4 zeros (negative) = 16 total
        targets = np.zeros((BATCH, 1), dtype=np.float32)
        targets[:12] = 1.0  # First 12 are positive
        rng.shuffle(targets)  # Shuffle to avoid trivial patterns

        model.fit(
            x={"features": features, "cca_targets": targets},
            epochs=1,
            batch_size=BATCH,
            verbose=0,
        )

        # Tracker should be populated and should reflect the known
        # positive fraction (0.75 ± small tolerance for floating point).
        tracker_result = float(batch_tracker.result())
        assert np.isfinite(tracker_result), (
            "BatchLabelBalanceTracker.result() is non-finite — indicates "
            "tracker was never populated or encountered NaN. The y=None "
            "endpoint-mode extraction may be broken."
        )
        assert 0.0 <= tracker_result <= 1.0, (
            f"Tracker result {tracker_result} is outside [0, 1] — indicates "
            "an invalid positive-fraction computation."
        )
        assert tracker_result > 0.0, (
            "Tracker result is exactly 0.0 — indicates the tracker never "
            "received any positive samples, or the extraction failed silently."
        )
        # The expected fraction is 0.75 (12 positives out of 16). Allow ±0.05
        # tolerance to account for batching and randomness.
        expected = 0.75
        assert abs(tracker_result - expected) < 0.05, (
            f"Tracker result {tracker_result} deviates from expected "
            f"{expected} by more than 0.05. Either the targets weren't "
            "propagated correctly, or the positive-fraction computation is wrong."
        )

    def test_head_ref_lookup_by_name(self):
        head = ClassificationHead(
            hidden_dim=INPUT_DIM,
            loss_fn=FLPULoss(prior=0.1),
            name="cca",
            expose_loss_components=True,
        )
        features_input = keras.Input(shape=(INPUT_DIM,), name="features")
        cca_targets_input = keras.Input(shape=(1,), name="cca_targets")
        logits = head(features_input, targets=cca_targets_input)

        bundle = {
            "per_step": {
                "gradient": [],
                "loss_component": [],
                "batch_target": [],
            },
            "periodic": [],
        }
        model = LayerLRModel(
            inputs={"features": features_input, "cca_targets": cca_targets_input},
            outputs={"cca": logits},
            diagnostic_trackers=bundle,
            diagnostic_head_refs=[head],
        )
        assert model._head_refs_by_name == {"cca": head}


class TestDiagnosticsKerasIntegration:
    def _model_with_trackers(self):
        def group_fn(v):
            return v.path.split("/")[0]
        base = _tiny_model(group_fn=group_fn)
        trackers = [
            PerGroupGradNormTracker(group_name="lower", aggregation="max"),
            GradientFiniteTracker(),
        ]
        bundle = {"per_step": {"gradient": trackers, "loss_component": [],
                               "batch_target": []}, "periodic": []}
        return LayerLRModel(
            inputs=base.inputs, outputs=base.outputs, group_fn=group_fn,
            diagnostic_trackers=bundle,
        ), trackers

    def test_loss_tracking_intact_with_trackers(self):
        # THE regression: metrics override must not break history["loss"].
        model, _ = self._model_with_trackers()
        model.compile(
            optimizer=keras.optimizers.SGD(0.01),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        rng = np.random.RandomState(42)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)
        history = model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        assert np.isfinite(history.history["loss"][0])
        assert history.history["loss"][0] > 0

    def test_diagnostic_scalars_appear_in_history(self):
        model, _ = self._model_with_trackers()
        model.compile(
            optimizer=keras.optimizers.SGD(0.01),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        rng = np.random.RandomState(3)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)
        history = model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        assert "grad_norm/lower/max" in history.history
        assert "grad_overflow_rate" in history.history

    def test_trackers_reset_at_epoch_boundary(self):
        model, trackers = self._model_with_trackers()
        model.compile(
            optimizer=keras.optimizers.SGD(0.01),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        rng = np.random.RandomState(4)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)
        history = model.fit(x, y, epochs=3, batch_size=BATCH, verbose=0)
        # 3 separate per-epoch values recorded (not a single monotone
        # accumulation across the whole run) — Keras reset_state per epoch.
        assert len(history.history["grad_overflow_rate"]) == 3

    def test_no_crash_compute_metrics_with_custom_trackers(self):
        # Trackers have custom update_state signatures; ensure Keras's
        # compute_metrics/get_metrics_result path (which iterates
        # self.metrics) does not call them with (y, y_pred).
        model, _ = self._model_with_trackers()
        model.compile(
            optimizer=keras.optimizers.SGD(0.01),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy(name="acc")],
        )
        rng = np.random.RandomState(5)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)
        history = model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        assert "acc" in history.history          # compiled metric still works
        assert "grad_overflow_rate" in history.history
        assert np.isfinite(history.history["loss"][0])
