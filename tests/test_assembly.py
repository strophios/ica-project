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
  - TestPatternTwoSerialization (Tier 3 Piece 2): Pattern 2 weight-
    loading-by-name round-trip produces bitwise-identical
    predictions to Pattern A; loading with mismatched variable
    names fails loud rather than silently accepting partial loads.

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

    def test_build_endpoint_model_rejects_duplicate_head_names(
        self, fresh_backbone
    ):
        """Forward-compat boundary-inventory check: same unique-names
        invariant enforced at call-site (here) and construction-site
        (ClassificationHead.__init__ requires explicit name). Dict
        structurally prevents duplicates today, but a future API
        change (heads as list of pairs) could allow them — the
        assert is the guard. Test triggers the assert via a
        mapping-like fake that reports duplicate keys."""

        class _DuplicateKeyHeads:
            """Test helper: mimics dict.keys() returning duplicates."""
            def keys(self):
                return ["x", "x"]
            def items(self):
                return []  # Not reached; assert fires first
            def __len__(self):
                return 2

        fake_heads = _DuplicateKeyHeads()
        with pytest.raises(ValueError, match=r"duplicates?.*\['x'\]"):
            build_endpoint_model(backbone=fresh_backbone, heads=fake_heads, seq_length=SEQ_LEN)


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


# -----------------------------------------------------------------------------
# Pattern 2: cross-process serialization (Tier 3 Piece 2)
# -----------------------------------------------------------------------------
# Pattern 2 is what the eval script does: training in one process
# saves weights; eval in a separate process rebuilds the architecture
# from code with matching head configuration and loads weights by
# name. The contract requires the variable-name hierarchies of the
# two graphs to match exactly. These tests pin that contract as a
# regression test (round-trip → bitwise-identical predictions) and
# as a fail-loud test (mismatched names raise rather than silently
# loading partial weights). See `docs/notes/tier3-design.md` Piece 2
# for the design framing.


class TestPatternTwoSerialization:
    """Pattern 2 cross-process serialization invariant."""

    def _build_pattern_a_and_save(self, fresh_backbone, fresh_head, weights_path):
        """
        Helper: build a Pattern A train + inference pair, train one
        step so weights are non-default, save the inference model's
        weights to `weights_path`. Returns the inference model
        (Pattern A side) for later prediction comparison.
        """
        train_model = build_endpoint_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
            freeze_encoder=False,  # both backbone and head update
        )
        inf_model_a = build_inference_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
        )
        train_model.compile(optimizer="adam")
        train_model.fit(
            _make_dummy_inputs(with_targets=True),
            epochs=1, batch_size=BATCH, verbose=0,
        )
        inf_model_a.save_weights(str(weights_path))
        return inf_model_a

    def test_round_trip_predictions_match_bitwise(
        self, fresh_backbone, fresh_head, tmp_path,
    ):
        """
        The core Pattern 2 invariant: save + rebuild + load
        produces bitwise-identical predictions to the original
        Pattern A inference model.

        Test flow:
          1. Build Pattern A (train + inf sharing head/backbone),
             fit one step so weights are non-default, save.
          2. Build Pattern 2: fresh backbone + fresh head with the
             same head name, fresh inference model.
          3. Pre-load: assert Pattern 2's predictions DIFFER from
             Pattern A's. This sanity-check breaks the symmetry —
             without it, a no-op `load_weights` could pass the
             post-load equality check on coincident initialization.
          4. Load weights with `skip_mismatch=False`.
          5. Post-load: assert Pattern 2's predictions match
             Pattern A's BITWISE (`np.array_equal`, not
             approximate). Pattern 2 should be exact.
        """
        weights_path = tmp_path / "test.weights.h5"

        # Step 1: Pattern A + save
        inf_model_a = self._build_pattern_a_and_save(
            fresh_backbone, fresh_head, weights_path,
        )

        # Step 2: Pattern 2 fresh build with matching configuration
        fresh_backbone_2 = _make_fake_backbone()
        fresh_head_2 = ClassificationHead(
            hidden_dim=HIDDEN_DIM, loss_fn=FLPULoss(prior=0.1), name="cca",
        )
        inf_model_b = build_inference_model(
            backbone=fresh_backbone_2,
            heads={"cca": fresh_head_2},
            seq_length=SEQ_LEN,
        )

        inf_inputs = _make_dummy_inputs(with_targets=False)
        preds_a = inf_model_a.predict(inf_inputs, verbose=0)
        preds_a = preds_a["cca"] if isinstance(preds_a, dict) else preds_a

        # Step 3: pre-load difference assertion
        preds_b_pre = inf_model_b.predict(inf_inputs, verbose=0)
        preds_b_pre = (
            preds_b_pre["cca"] if isinstance(preds_b_pre, dict) else preds_b_pre
        )
        assert not np.array_equal(preds_a, preds_b_pre), (
            "Pattern 2 predictions match Pattern A's BEFORE load — the test "
            "isn't actually exercising the load path. Likely the two fresh "
            "builds happened to have identical random initializations "
            "(astronomically unlikely with float32 Glorot init; investigate)."
        )

        # Step 4: load with skip_mismatch=False discipline
        inf_model_b.load_weights(str(weights_path), skip_mismatch=False)

        # Step 5: post-load bitwise equality
        preds_b_post = inf_model_b.predict(inf_inputs, verbose=0)
        preds_b_post = (
            preds_b_post["cca"] if isinstance(preds_b_post, dict) else preds_b_post
        )
        np.testing.assert_array_equal(
            preds_a, preds_b_post,
            err_msg=(
                "Pattern 2 predictions don't match Pattern A bitwise after "
                "load. Possible causes: dtype drift on save/load, missing "
                "weights, save-format precision loss, or non-deterministic "
                "forward pass (unlikely on CPU)."
            ),
        )

    def test_load_weights_actually_copies_head_weights(self, tmp_path):
        """
        Tier 3 closeout (addressing C1 from the adversarial review).
        The `test_round_trip_predictions_match_bitwise` test claims
        a "no-op load_weights could pass the post-load equality
        check on coincident initialization" but its setup uses
        `freeze_encoder=False` — so the trained backbone diverges
        from a fresh-init backbone, and *backbone* weight loading
        is what produces the post-load match. A `load_weights` that
        silently dropped *head* weights would still pass that test.

        This test pins the head-load invariant explicitly:
          - Backbones in both Pattern A and Pattern 2 are SEEDED
            IDENTICALLY before construction → backbone weights are
            bitwise-equal at construction.
          - `freeze_encoder=True` → backbone weights stay equal
            through training.
          - Head weights diverge: Pattern A's head is trained;
            Pattern 2's head is freshly initialized.
          - Pre-load: predictions differ because head weights
            differ.
          - Load weights into Pattern 2.
          - Post-load: if predictions match Pattern A bitwise,
            `load_weights` actually copied the trained head weights.
            A no-op load would leave Pattern 2's head random and
            predictions would still differ.

        The other round-trip test exercises the integrated path;
        this one isolates the no-op-protection invariant.
        """
        weights_path = tmp_path / "test.weights.h5"

        # --- Pattern A side ---
        # Seed BEFORE constructing fresh_backbone (in fixture). Since
        # fresh_backbone is already constructed by the fixture, we
        # build a new "seeded" backbone here for explicit control.
        keras.utils.set_random_seed(42)
        backbone_a = _make_fake_backbone()
        head_a = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=FLPULoss(prior=0.1),
            name="cca",
        )
        train_model = build_endpoint_model(
            backbone=backbone_a,
            heads={"cca": head_a},
            seq_length=SEQ_LEN,
            freeze_encoder=True,  # backbone stays put
        )
        inf_model_a = build_inference_model(
            backbone=backbone_a,
            heads={"cca": head_a},
            seq_length=SEQ_LEN,
        )
        train_model.compile(optimizer="adam")
        train_model.fit(
            _make_dummy_inputs(with_targets=True),
            epochs=1, batch_size=BATCH, verbose=0,
        )
        inf_model_a.save_weights(str(weights_path))

        # --- Pattern 2 side, with IDENTICALLY-SEEDED backbone ---
        keras.utils.set_random_seed(42)  # reset to same seed
        backbone_b = _make_fake_backbone()
        head_b = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=FLPULoss(prior=0.1),
            name="cca",
        )
        inf_model_b = build_inference_model(
            backbone=backbone_b,
            heads={"cca": head_b},
            seq_length=SEQ_LEN,
        )

        # Sanity check: backbones are bitwise-equal at construction.
        for w_a, w_b in zip(backbone_a.weights, backbone_b.weights):
            np.testing.assert_array_equal(
                w_a.numpy(), w_b.numpy(),
                err_msg=(
                    "Backbones not bitwise-equal at construction; "
                    "the same-seed assumption isn't holding. The "
                    "no-op-load-protection invariant this test is "
                    "supposed to pin can't be checked."
                ),
            )

        inf_inputs = _make_dummy_inputs(with_targets=False)
        preds_a = inf_model_a.predict(inf_inputs, verbose=0)
        preds_a = preds_a["cca"] if isinstance(preds_a, dict) else preds_a

        # Pre-load: Pattern A's trained head vs. Pattern 2's
        # fresh-init head produces different predictions even
        # though backbones are equal.
        preds_b_pre = inf_model_b.predict(inf_inputs, verbose=0)
        preds_b_pre = (
            preds_b_pre["cca"] if isinstance(preds_b_pre, dict) else preds_b_pre
        )
        assert not np.array_equal(preds_a, preds_b_pre), (
            "Pre-load predictions match — head weights happen to be "
            "identical despite Pattern A's having been trained. "
            "Either training had no effect on the head (unlikely) or "
            "the same-seed setup is producing the same head init AND "
            "fit was a no-op."
        )

        # Load weights — this is the operation under test.
        inf_model_b.load_weights(str(weights_path), skip_mismatch=False)

        # Post-load: if predictions now match bitwise, load_weights
        # actually copied the trained head's weights into Pattern 2's
        # head. A no-op load would leave them at fresh-init (and
        # predictions would still differ as in the pre-load assertion).
        preds_b_post = inf_model_b.predict(inf_inputs, verbose=0)
        preds_b_post = (
            preds_b_post["cca"] if isinstance(preds_b_post, dict) else preds_b_post
        )
        np.testing.assert_array_equal(
            preds_a, preds_b_post,
            err_msg=(
                "Pattern 2 predictions don't match Pattern A bitwise "
                "after load. With same-seed backbones and "
                "freeze_encoder=True, the only thing load_weights "
                "needs to copy is the head's trained weights. "
                "A failure here means head weights weren't actually "
                "loaded (no-op or partial load)."
            ),
        )

    def test_load_weights_raises_on_shape_mismatch(
        self, fresh_backbone, fresh_head, tmp_path,
    ):
        """
        Fail-loud invariant: loading weights into a model with a
        mismatched architectural shape (e.g., different `hidden_dim`)
        must raise rather than silently accepting a partial load.

        This test pins the `skip_mismatch=False` discipline. Without
        it, a future architectural drift between train and eval
        scripts (e.g., someone tweaks `hidden_dim` from 8 → 16 in
        the eval-side head config and forgets to retrain) would
        silently produce a model with zeroed-out or default-
        initialized weights of the wrong shape — predictions would
        still come out, but they would be nonsense.

        **Note on the test scope (per Piece 2 design doc empirical
        finding):** Keras's `.weights.h5` save format keys
        variables by *layer-class-name + positional index*, NOT by
        user-given name. So renaming a head from "cca" to "ccaa"
        does NOT break load (originally the design's planned
        mismatch test, abandoned after observation). What does
        break load — and what `skip_mismatch=False` reliably
        catches — is shape mismatch. That's what this test pins.

        Test pattern: build a Pattern A model with `hidden_dim=8`
        (HIDDEN_DIM, the test module's default), save. Build a
        fresh inference model with `hidden_dim=16` (deliberately
        doubled). Attempt load with `skip_mismatch=False`; expect
        `ValueError` (Keras raises this on shape mismatch, with a
        message about the target variable's shape not matching).
        """
        weights_path = tmp_path / "test.weights.h5"

        # Pattern A side with default HIDDEN_DIM=8
        self._build_pattern_a_and_save(
            fresh_backbone, fresh_head, weights_path,
        )

        # Mismatched fresh build: hidden_dim doubled. The head's
        # internal Dense layers will have weight shapes (8, 16) and
        # (16, 1) instead of (8, 8) and (8, 1) — incompatible with
        # the saved file's tensors.
        wrong_hidden_dim = HIDDEN_DIM * 2
        fresh_backbone_2 = _make_fake_backbone(hidden_dim=wrong_hidden_dim)
        head_wrong = ClassificationHead(
            hidden_dim=wrong_hidden_dim, loss_fn=FLPULoss(prior=0.1), name="cca",
        )
        inf_model_wrong = build_inference_model(
            backbone=fresh_backbone_2,
            heads={"cca": head_wrong},
            seq_length=SEQ_LEN,
        )

        with pytest.raises(ValueError):
            inf_model_wrong.load_weights(
                str(weights_path), skip_mismatch=False,
            )
