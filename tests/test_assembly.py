"""
Integration tests for `src.model_setup.assembly`.

Exercises the assembled stack (backbone + heads via
`build_endpoint_model` / `build_inference_model`) end-to-end on a
fake backbone. The fake backbone is an Embedding + multiply-by-
padding-mask, sized small for speed; the head/assembly behavior is
identical to what it'd be with the real RoBERTa backbone, just
without the 50M-parameter overhead per test.

  - TestBuildEndpointModel: structural — returns LayerLRModel,
    inputs include targets, outputs keyed by head name.
  - TestBuildInferenceModel: structural — returns keras.Model, no
    target inputs, outputs keyed by head name.
  - TestForwardPass: shapes correct on dummy inputs.
  - TestTrainingStep: `fit()` succeeds and updates head weights;
    backbone weights also update when `freeze_encoder=False`.
  - TestFreezeEncoder: with `freeze_encoder=True`, backbone weights
    are unchanged after a training step.
  - TestPatternAWeightSharing: shared head Layer instances across
    train and inference models means training the train model
    changes the inference model's predictions.

Note on the fake backbone: the real RoBERTa backbone uses
`padding_mask` for attention masking; the fake mimics this just
loosely (multiplying the embedding by the mask, broadcast). The
goal is to satisfy the assembly's wiring contract — accept
`{"token_ids", "padding_mask"}` dict input, return `(batch, seq, hidden)`
tensor — not to faithfully reproduce attention.
"""

import numpy as np
import pytest

import keras  # noqa: F401  (initializes backend)

from src.model_setup.heads import ClassificationHead
from src.model_setup.layer_lr_model import LayerLRModel
from src.model_setup.assembly import (
    build_endpoint_model,
    build_inference_model,
)
from src.loss_functions.loss import FLPULoss


SEQ_LEN = 4
HIDDEN_DIM = 8
VOCAB = 100
BATCH = 4


# -----------------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------------

def _make_fake_backbone(seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM, vocab=VOCAB,
                        name="fake_backbone"):
    """
    Build a tiny stand-in for a real backbone. Accepts dict input
    `{"token_ids", "padding_mask"}` and returns a
    `(batch, seq, hidden)` tensor — the contract assembly relies on.
    """
    token_ids = keras.Input(shape=(seq_len,), dtype="int32", name="token_ids")
    padding_mask = keras.Input(shape=(seq_len,), dtype="int32", name="padding_mask")
    embed = keras.layers.Embedding(vocab, hidden_dim, name="fake_embed")
    embedded = embed(token_ids)
    # Use padding_mask: broadcast multiply so the input is connected
    # to the output (otherwise keras.Model construction errors).
    mask_float = keras.ops.cast(padding_mask, "float32")
    mask_expanded = keras.ops.expand_dims(mask_float, axis=-1)
    masked = embedded * mask_expanded
    return keras.Model(
        inputs={"token_ids": token_ids, "padding_mask": padding_mask},
        outputs=masked,
        name=name,
    )


def _make_dummy_inputs(with_targets=False, batch=BATCH, seq_len=SEQ_LEN, vocab=VOCAB):
    rng = np.random.RandomState(0)
    inputs = {
        "token_ids": rng.randint(0, vocab, size=(batch, seq_len)).astype("int32"),
        "padding_mask": np.ones((batch, seq_len), dtype="int32"),
    }
    if with_targets:
        # Target Input is named "<head>_targets" to avoid colliding with
        # the head Layer's own name (Keras requires unique op names within
        # a Functional graph). The head is named "cca"; its target Input
        # is "cca_targets".
        inputs["cca_targets"] = np.array(
            [0.0, 1.0, 0.0, 1.0], dtype="float32"
        )[:batch]
    return inputs


@pytest.fixture
def fresh_backbone():
    """Build a fresh fake backbone. Each test gets its own."""
    return _make_fake_backbone()


@pytest.fixture
def fresh_head():
    """Build a fresh ClassificationHead with FLPU loss."""
    return ClassificationHead(
        hidden_dim=HIDDEN_DIM,
        loss_fn=FLPULoss(prior=0.1),
        name="cca",
    )


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------

class TestBuildEndpointModel:
    def test_returns_layer_lr_model(self, fresh_backbone, fresh_head):
        model = build_endpoint_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
        )
        assert isinstance(model, LayerLRModel)

    def test_inputs_include_token_ids_padding_mask_and_targets(
        self, fresh_backbone, fresh_head
    ):
        model = build_endpoint_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
        )
        input_names = {inp.name for inp in model.inputs}
        assert "token_ids" in input_names
        assert "padding_mask" in input_names
        # Target Input name is "<head>_targets" — distinct from the
        # head Layer's name to avoid Keras op-name collision.
        assert "cca_targets" in input_names
        # And the head's own output name is *not* a model input.
        assert "cca" not in input_names

    def test_outputs_keyed_by_head_name(self, fresh_backbone, fresh_head):
        model = build_endpoint_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
        )
        # Output structure: dict keyed by head name
        # Functional models return outputs in the order/structure
        # they were declared. With dict-valued outputs, output_names
        # should include "cca".
        assert "cca" in model.output_names

    def test_multi_head_inputs_and_outputs(self, fresh_backbone):
        head_a = ClassificationHead(
            hidden_dim=HIDDEN_DIM, loss_fn=FLPULoss(prior=0.1), name="cca",
        )
        head_b = ClassificationHead(
            hidden_dim=HIDDEN_DIM, loss_fn=FLPULoss(prior=0.05), name="immig",
        )
        model = build_endpoint_model(
            backbone=fresh_backbone,
            heads={"cca": head_a, "immig": head_b},
            seq_length=SEQ_LEN,
        )
        input_names = {inp.name for inp in model.inputs}
        assert {
            "token_ids", "padding_mask", "cca_targets", "immig_targets"
        } <= input_names
        # Outputs keyed by head name (not "<head>_targets")
        assert {"cca", "immig"} <= set(model.output_names)


class TestBuildInferenceModel:
    def test_returns_keras_model_not_layer_lr_model(
        self, fresh_backbone, fresh_head
    ):
        model = build_inference_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
        )
        assert isinstance(model, keras.Model)
        # Inference model is a plain keras.Model, not LayerLRModel —
        # there's no training step to customize.
        assert not isinstance(model, LayerLRModel)

    def test_inputs_have_no_targets(self, fresh_backbone, fresh_head):
        model = build_inference_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
        )
        input_names = {inp.name for inp in model.inputs}
        assert input_names == {"token_ids", "padding_mask"}
        # Neither the head name nor the suffixed-targets name should
        # appear as a model input on the inference side.
        assert "cca" not in input_names
        assert "cca_targets" not in input_names

    def test_outputs_keyed_by_head_name(self, fresh_backbone, fresh_head):
        model = build_inference_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
        )
        assert "cca" in model.output_names


# -----------------------------------------------------------------------------
# Forward pass
# -----------------------------------------------------------------------------

class TestForwardPass:
    def test_endpoint_model_forward_shape(self, fresh_backbone, fresh_head):
        model = build_endpoint_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
        )
        inputs = _make_dummy_inputs(with_targets=True)
        outputs = model(inputs)
        cca_out = outputs["cca"] if isinstance(outputs, dict) else outputs
        assert cca_out.shape == (BATCH, 1)

    def test_inference_model_forward_shape(self, fresh_backbone, fresh_head):
        model = build_inference_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
        )
        inputs = _make_dummy_inputs(with_targets=False)
        outputs = model(inputs)
        cca_out = outputs["cca"] if isinstance(outputs, dict) else outputs
        assert cca_out.shape == (BATCH, 1)


# -----------------------------------------------------------------------------
# Training step
# -----------------------------------------------------------------------------

class TestTrainingStep:
    def test_fit_step_succeeds_and_updates_head_weights(
        self, fresh_backbone, fresh_head
    ):
        model = build_endpoint_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
            freeze_encoder=True,  # focus on head update
        )
        model.compile(optimizer="adam")  # head's add_loss handles loss

        # Snapshot head weights before training
        head_weights_before = [w.numpy().copy() for w in fresh_head.weights]

        inputs = _make_dummy_inputs(with_targets=True)
        model.fit(inputs, epochs=1, batch_size=BATCH, verbose=0)

        head_weights_after = [w.numpy() for w in fresh_head.weights]
        # At least one head weight tensor should have changed
        any_changed = any(
            not np.allclose(b, a)
            for b, a in zip(head_weights_before, head_weights_after)
        )
        assert any_changed, "Head weights did not update during training"

    def test_fit_step_updates_backbone_when_unfrozen(
        self, fresh_backbone, fresh_head
    ):
        model = build_endpoint_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
            freeze_encoder=False,  # backbone should train
        )
        model.compile(optimizer="adam")

        backbone_weights_before = [w.numpy().copy() for w in fresh_backbone.weights]

        inputs = _make_dummy_inputs(with_targets=True)
        model.fit(inputs, epochs=1, batch_size=BATCH, verbose=0)

        backbone_weights_after = [w.numpy() for w in fresh_backbone.weights]
        any_changed = any(
            not np.allclose(b, a)
            for b, a in zip(backbone_weights_before, backbone_weights_after)
        )
        assert any_changed, "Backbone weights did not update with freeze_encoder=False"


# -----------------------------------------------------------------------------
# freeze_encoder
# -----------------------------------------------------------------------------

class TestFreezeEncoder:
    def test_freeze_encoder_keeps_backbone_weights_unchanged(
        self, fresh_backbone, fresh_head
    ):
        model = build_endpoint_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
            freeze_encoder=True,
        )
        model.compile(optimizer="adam")

        backbone_weights_before = [w.numpy().copy() for w in fresh_backbone.weights]

        inputs = _make_dummy_inputs(with_targets=True)
        model.fit(inputs, epochs=1, batch_size=BATCH, verbose=0)

        backbone_weights_after = [w.numpy() for w in fresh_backbone.weights]
        for b, a in zip(backbone_weights_before, backbone_weights_after):
            np.testing.assert_array_equal(
                b, a, err_msg="Backbone weights changed despite freeze_encoder=True"
            )


# -----------------------------------------------------------------------------
# Pattern A: shared head instances → shared weights
# -----------------------------------------------------------------------------

class TestPatternAWeightSharing:
    def test_training_train_model_changes_inf_model_predictions(
        self, fresh_backbone, fresh_head
    ):
        """
        Build train and inference models sharing the same head and
        backbone instances. Predict on the inference model before
        and after training the training model — predictions should
        differ, demonstrating that weights are physically shared.
        """
        train_model = build_endpoint_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
            freeze_encoder=True,  # focus on head
        )
        inf_model = build_inference_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
        )

        train_model.compile(optimizer="adam")

        inf_inputs = _make_dummy_inputs(with_targets=False)
        preds_before = inf_model.predict(inf_inputs, verbose=0)
        preds_before = (
            preds_before["cca"] if isinstance(preds_before, dict) else preds_before
        )

        train_inputs = _make_dummy_inputs(with_targets=True)
        train_model.fit(train_inputs, epochs=1, batch_size=BATCH, verbose=0)

        preds_after = inf_model.predict(inf_inputs, verbose=0)
        preds_after = (
            preds_after["cca"] if isinstance(preds_after, dict) else preds_after
        )

        # Predictions should have changed: training the train model
        # updated the shared head's weights, which the inf model
        # picks up by Python identity.
        assert not np.allclose(preds_before, preds_after), (
            "Inference model predictions unchanged — Pattern A weight "
            "sharing not working as expected."
        )
