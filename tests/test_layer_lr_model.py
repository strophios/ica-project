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
import tensorflow as tf

from src.model_setup.layer_lr_model import LayerLRModel
from src.model_setup.heads import ClassificationHead


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
