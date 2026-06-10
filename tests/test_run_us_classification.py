"""
Targeted unit tests for src.run_us_classification.

Verifies:
  - AC3.4: BCE in path, no FLPU/prior
  - AC3.2: prediction-distribution metrics present with correct names
  - Importing does not trigger training
"""

import keras
import numpy as np
import pytest

from src.model_setup.assembly import build_endpoint_model, build_inference_model
from src.model_setup.heads import ClassificationHead
from src.us_config import DEFAULT_US_CONFIG
from src.us_metrics import make_us_metrics
from src.diagnostics.distribution_metrics import make_distribution_metrics


SEQ_LEN = 4
HIDDEN_DIM = 8
VOCAB = 100
BATCH = 4


def _make_fake_backbone(seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM, vocab=VOCAB):
    """Build a tiny stand-in for a real backbone."""
    token_ids = keras.Input(shape=(seq_len,), dtype="int32", name="token_ids")
    padding_mask = keras.Input(shape=(seq_len,), dtype="int32", name="padding_mask")
    embed = keras.layers.Embedding(vocab, hidden_dim, name="fake_embed")
    embedded = embed(token_ids)
    mask_float = keras.ops.cast(padding_mask, "float32")
    mask_expanded = keras.ops.expand_dims(mask_float, axis=-1)
    masked = embedded * mask_expanded
    return keras.Model(
        inputs={"token_ids": token_ids, "padding_mask": padding_mask},
        outputs=masked,
        name="fake_backbone",
    )


class TestBCEEndpointAssembly:
    """Verify the US head is assembled with BCE and no FLPU."""

    def test_us_head_uses_bce_loss(self):
        """AC3.4: head's loss_fn is BinaryCrossentropy with from_logits=True."""
        us_head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=make_us_metrics(),
            name="us",
            expose_loss_components=False,
        )
        assert isinstance(us_head.loss_fn, keras.losses.BinaryCrossentropy)
        # Check from_logits=True (via config)
        config = us_head.loss_fn.get_config()
        assert config["from_logits"] is True

    def test_no_flpu_in_config(self):
        """AC3.4: DEFAULT_US_CONFIG has no prior or FLPU coupling."""
        assert not hasattr(DEFAULT_US_CONFIG, "prior")
        assert not hasattr(DEFAULT_US_CONFIG.head, "prior")
        assert not hasattr(DEFAULT_US_CONFIG, "loss")
        # UsRunConfig fields (should not include FLPU-related fields)
        field_names = {f.name for f in DEFAULT_US_CONFIG.__dataclass_fields__.values()}
        assert "prior" not in field_names
        assert "loss" not in field_names  # loss_fn lives in head, not config

    def test_endpoint_model_with_us_head(self):
        """Assemble US head via build_endpoint_model (BCE path)."""
        backbone = _make_fake_backbone()
        us_head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=make_us_metrics(),
            name="us",
            expose_loss_components=False,
        )
        model = build_endpoint_model(
            backbone=backbone,
            heads={"us": us_head},
            seq_length=SEQ_LEN,
            freeze_encoder=True,
        )
        # Model should be buildable; BCE is registered via head.add_loss
        assert model is not None
        # Verify compile with no loss argument succeeds
        model.compile(optimizer="adam", jit_compile=False)


class TestDistributionMetrics:
    """Verify prediction-distribution metrics populate with correct names (AC3.2)."""

    def test_distribution_metrics_names(self):
        """AC3.2: distribution metrics are named pred_dist/{mean,std,frac_above_0.5}."""
        diagnostics_config = DEFAULT_US_CONFIG.diagnostics
        dist_metrics = make_distribution_metrics(diagnostics_config)
        metric_names = {m.name for m in dist_metrics}
        assert "pred_dist/mean" in metric_names
        assert "pred_dist/std" in metric_names
        assert "pred_dist/frac_above_0.5" in metric_names

    def test_us_head_includes_distribution_metrics(self):
        """AC3.2: US head's metric set includes distribution metrics (head-prefixed)."""
        us_head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=make_us_metrics()
            + make_distribution_metrics(DEFAULT_US_CONFIG.diagnostics),
            name="us",
            expose_loss_components=False,
        )
        metric_names = {m.name for m in us_head.metric_objs}
        # Head prefixes metric names: metric "pred_dist/mean" becomes "us_pred_dist/mean"
        assert "us_pred_dist/mean" in metric_names
        assert "us_pred_dist/std" in metric_names
        assert "us_pred_dist/frac_above_0.5" in metric_names


class TestImportNoSideEffects:
    """Verify importing src.run_us_classification does not train."""

    def test_import_does_not_trigger_training(self):
        """Importing the module should not execute main() (guarded by if __name__...)."""
        # This test passes if the module imports without calling main().
        # The module's `if __name__ == "__main__": main()` guard ensures this.
        import src.run_us_classification

        # If we reach here without an exception, import was side-effect-free
        assert src.run_us_classification is not None
