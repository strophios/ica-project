"""
Tests for `src.model_setup.heads.ClassificationHead`.

Structured to fail until the head is implemented, and to act as an
executable spec for the head's behavior:

  - TestConstruction: the constructor accepts the expected parameters
    and stores them correctly.
  - TestForwardPass: the forward pass produces logits of the expected
    shape and dtype.
  - TestEndpointMode: the endpoint-mode contract — losses are added
    when (and only when) a loss_fn is configured *and* targets are
    supplied at call time.
  - TestTrainableWeights: the head has the expected number of
    trainable weight tensors, and separate instances have separate
    weights (guards against the accidentally-shared-sublayer pitfall).

These tests use small synthetic features and do not exercise the
backbone. The head is designed to operate on whatever 2-D feature
tensor it is given; shape-compatibility with the real backbone is
verified later in the assembly-level tests.
"""

import numpy as np
import pytest

# Import keras before our head module so the backend gets initialized.
import keras  # noqa: F401

from src.model_setup.heads import ClassificationHead
from src.loss_functions.loss import FLPULoss


HIDDEN_DIM = 32  # deliberately small for fast tests
BATCH_SIZE = 4


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _dummy_features(batch=BATCH_SIZE, dim=HIDDEN_DIM):
    """Random 2-D features tensor standing in for the backbone's CLS output."""
    rng = np.random.RandomState(0)
    return rng.randn(batch, dim).astype(np.float32)


def _dummy_targets(batch=BATCH_SIZE):
    """Shape (batch,) binary targets, alternating 1 and 0."""
    return np.array([1.0, 0.0] * ((batch + 1) // 2), dtype=np.float32)[:batch]


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------

class TestConstruction:
    def test_standard_mode_constructs_with_only_hidden_dim(self):
        """Minimum-required-args construction path (standard mode)."""
        head = ClassificationHead(hidden_dim=HIDDEN_DIM)
        assert head.loss_fn is None, "Default loss_fn should be None (standard mode)"

    def test_endpoint_mode_constructs_with_loss_fn(self):
        """Constructing with a loss_fn should store it and activate endpoint mode."""
        loss = FLPULoss(prior=0.1)
        head = ClassificationHead(hidden_dim=HIDDEN_DIM, loss_fn=loss)
        assert head.loss_fn is loss

    def test_custom_dropout_and_name_are_stored(self):
        """Named constructor args should be stored (dropout rate and layer name)."""
        head = ClassificationHead(hidden_dim=HIDDEN_DIM, dropout=0.3, name="cca_head")
        assert head.dropout_rate == 0.3
        assert head.name == "cca_head"


# -----------------------------------------------------------------------------
# Forward pass shape
# -----------------------------------------------------------------------------

class TestForwardPass:
    def test_output_shape_is_batch_by_one(self):
        """Binary-classification head emits one logit per sample."""
        head = ClassificationHead(hidden_dim=HIDDEN_DIM)
        out = head(_dummy_features())
        assert tuple(out.shape) == (BATCH_SIZE, 1), (
            f"Expected output shape ({BATCH_SIZE}, 1), got {tuple(out.shape)}"
        )

    def test_forward_pass_runs_without_targets_in_standard_mode(self):
        """In standard mode, call(features) should work without targets."""
        head = ClassificationHead(hidden_dim=HIDDEN_DIM)
        # Should not raise.
        _ = head(_dummy_features())

    def test_forward_pass_runs_without_targets_in_endpoint_mode(self):
        """In endpoint mode, call(features) without targets should also work.
        This is the inference-time path — loss is not computed."""
        head = ClassificationHead(hidden_dim=HIDDEN_DIM, loss_fn=FLPULoss(prior=0.1))
        # Should not raise. No loss is added (see TestEndpointMode for that).
        _ = head(_dummy_features())


# -----------------------------------------------------------------------------
# Endpoint-mode contract (when add_loss fires)
# -----------------------------------------------------------------------------

class TestEndpointMode:
    def test_endpoint_mode_adds_loss_when_targets_provided(self):
        """The core endpoint-layer contract: with a loss_fn and with targets
        supplied to call(), the head registers exactly one loss via add_loss.
        Keras's outer Model picks this up automatically during training."""
        head = ClassificationHead(hidden_dim=HIDDEN_DIM, loss_fn=FLPULoss(prior=0.1))
        _ = head(_dummy_features(), targets=_dummy_targets())
        assert len(head.losses) == 1, (
            f"Expected exactly 1 loss registered via add_loss, got {len(head.losses)}"
        )

    def test_endpoint_mode_no_loss_when_targets_omitted(self):
        """Inference-time path: targets=None should NOT register a loss,
        even if loss_fn is configured. Otherwise prediction would corrupt
        the aggregated model loss state."""
        head = ClassificationHead(hidden_dim=HIDDEN_DIM, loss_fn=FLPULoss(prior=0.1))
        _ = head(_dummy_features())
        assert len(head.losses) == 0

    def test_standard_mode_no_loss_even_with_targets(self):
        """In standard mode (loss_fn=None), the head should NOT add a loss
        even if targets happen to be passed. In standard mode the outer
        Model's compile-time loss handles things; a head that also
        registered a loss would cause double-counting."""
        head = ClassificationHead(hidden_dim=HIDDEN_DIM, loss_fn=None)
        _ = head(_dummy_features(), targets=_dummy_targets())
        assert len(head.losses) == 0


# -----------------------------------------------------------------------------
# Weight structure
# -----------------------------------------------------------------------------

class TestTrainableWeights:
    def test_expected_number_of_trainable_weights(self):
        """Two Dense sub-layers (intermediate + output), each contributing
        a kernel and a bias, gives 4 trainable weight tensors total.
        Dropouts have no weights. A count mismatch typically means the
        head is missing a sub-layer or has an extra one."""
        head = ClassificationHead(hidden_dim=HIDDEN_DIM)
        _ = head(_dummy_features())  # build sublayers via first forward pass
        assert len(head.trainable_weights) == 4, (
            f"Expected 4 trainable weight tensors (2 Dense × 2), "
            f"got {len(head.trainable_weights)}"
        )

    def test_separate_instances_have_independent_weights(self):
        """Guards against a common Layer-subclass pitfall: constructing
        sub-layers at class level rather than inside __init__. If that
        happens, all instances share the same sub-layers (and weights),
        which silently breaks multi-head training."""
        head_a = ClassificationHead(hidden_dim=HIDDEN_DIM, name="a")
        head_b = ClassificationHead(hidden_dim=HIDDEN_DIM, name="b")
        _ = head_a(_dummy_features())
        _ = head_b(_dummy_features())
        for w_a, w_b in zip(head_a.trainable_weights, head_b.trainable_weights):
            assert w_a is not w_b, (
                "Weight tensors appear to be shared between instances — "
                "likely cause: sub-layers were constructed at class level "
                "rather than in __init__."
            )


# -----------------------------------------------------------------------------
# Per-head metrics
# -----------------------------------------------------------------------------

class TestMetrics:
    """The `metrics` parameter on `ClassificationHead` carries per-head
    metric objects. Symmetric with `loss_fn`: both fire only when targets
    are provided, both are part of the endpoint-layer pattern. The head
    renames each metric to be prefixed with its name so multi-head models
    don't collide on metric names (e.g., `"binary_accuracy"` becomes
    `"cca_binary_accuracy"`)."""

    def test_metrics_renamed_with_head_name_prefix(self):
        head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            metrics=[
                keras.metrics.BinaryAccuracy(),
                keras.metrics.Precision(name="precision"),
            ],
            name="cca",
        )
        names = [m.name for m in head.metric_objs]
        assert "cca_binary_accuracy" in names
        assert "cca_precision" in names

    def test_metric_originals_not_mutated(self):
        """The head should clone metrics rather than mutating in place
        — protects callers who reuse a metric instance elsewhere."""
        original = keras.metrics.BinaryAccuracy(name="binary_accuracy")
        head = ClassificationHead(
            hidden_dim=HIDDEN_DIM, metrics=[original], name="cca"
        )
        # Original keeps its name; head holds a renamed clone.
        assert original.name == "binary_accuracy"
        assert head.metric_objs[0] is not original
        assert head.metric_objs[0].name == "cca_binary_accuracy"

    def test_metric_state_updates_when_targets_provided(self):
        """`call(features, targets=...)` should call `update_state` on
        each metric. With known features + targets, the metric's
        `result()` should reflect the prediction."""
        head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            metrics=[keras.metrics.BinaryAccuracy(threshold=0.0)],
            name="cca",
        )
        features = _dummy_features()
        targets = _dummy_targets()
        head.metric_objs[0].reset_state()
        _ = head(features, targets=targets)
        # BinaryAccuracy result is a scalar in [0, 1].
        result = float(head.metric_objs[0].result())
        assert 0.0 <= result <= 1.0

    def test_metric_state_unchanged_when_targets_none(self):
        """Without targets, no update_state call should fire — mirrors
        the loss path's guard. Result should remain at the post-reset
        default."""
        head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            metrics=[keras.metrics.BinaryAccuracy(threshold=0.0)],
            name="cca",
        )
        head.metric_objs[0].reset_state()
        result_before = float(head.metric_objs[0].result())
        _ = head(_dummy_features(), targets=None)
        result_after = float(head.metric_objs[0].result())
        assert result_before == result_after

    def test_metrics_appear_in_layer_metrics(self):
        """Keras 3's tracker should expose the head's metrics via
        `Layer.metrics`, so they propagate to `Model.metrics` for
        fit/evaluate logging."""
        head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            metrics=[
                keras.metrics.BinaryAccuracy(),
                keras.metrics.Precision(name="precision"),
            ],
            name="cca",
        )
        layer_metric_names = {m.name for m in head.metrics}
        assert "cca_binary_accuracy" in layer_metric_names
        assert "cca_precision" in layer_metric_names

    def test_default_no_metrics_is_empty_list(self):
        """Backward compatibility: heads without explicit `metrics=`
        should have an empty `metric_objs` list and behave as before."""
        head = ClassificationHead(hidden_dim=HIDDEN_DIM, name="cca")
        assert head.metric_objs == []
        # Calling without targets shouldn't error.
        _ = head(_dummy_features(), targets=None)
        # Calling with targets shouldn't error either.
        _ = head(_dummy_features(), targets=_dummy_targets())
