"""Features-mode assembly + cached-embedding dataset builder.

Verifies the frozen-probe path reuses the instrumented stack:
- cca-doca.AC0.2: a features-mode endpoint model trains a head and, with
  diagnostics enabled, wires more metrics than the no-diagnostics build (parity
  with token mode, where diagnostics ride the head/loss level).
- cca-doca.AC0.3: a feature/head hidden_dim mismatch raises.
"""

from __future__ import annotations

import numpy as np
import pytest
import keras

from src.cca_config import DiagnosticsConfig
from src.loss_functions.loss import FLPULoss
from src.cca_metrics import make_cca_metrics
from src.diagnostics.distribution_metrics import make_distribution_metrics
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import (
    build_feature_endpoint_model,
    build_feature_inference_model,
)
from src.data_setup.data import dataset_from_embeddings

HID = 8


def _head(diag_on):
    diag = DiagnosticsConfig()
    return ClassificationHead(
        hidden_dim=HID,
        loss_fn=FLPULoss(prior=0.1),
        metrics=make_cca_metrics() + make_distribution_metrics(diag),
        name="cca",
        expose_loss_components=diag_on,
    )


def _batch(n=16):
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((n, HID)).astype("float32")
    targets = (rng.random(n) < 0.3).astype("float32")
    return {"features": feats, "cca_targets": targets}


def test_feature_endpoint_trains_and_diagnostics_add_metrics():
    # No-diagnostics baseline.
    m_plain = build_feature_endpoint_model({"cca": _head(False)}, hidden_dim=HID)
    m_plain.compile(optimizer="adam", jit_compile=False)
    logs_plain = m_plain.train_on_batch(_batch(), return_dict=True)
    assert "loss" in logs_plain  # training ran via head add_loss

    # Diagnostics-enabled build wires strictly more metrics (parity w/ token mode).
    diag = DiagnosticsConfig()
    m_diag = build_feature_endpoint_model(
        {"cca": _head(True)}, hidden_dim=HID, diagnostics=diag
    )
    m_diag.compile(optimizer="adam", jit_compile=False)
    logs_diag = m_diag.train_on_batch(_batch(), return_dict=True)
    assert "loss" in logs_diag
    assert len(m_diag.metrics) > len(m_plain.metrics)


def test_feature_inference_model_predicts():
    head = _head(False)
    # Pattern A: share the head instance with the endpoint model.
    build_feature_endpoint_model({"cca": head}, hidden_dim=HID)
    inf = build_feature_inference_model({"cca": head}, hidden_dim=HID)
    out = inf.predict({"features": _batch()["features"]}, verbose=0)
    assert out["cca"].shape == (16, 1)


def test_feature_endpoint_rejects_dim_mismatch():
    head = ClassificationHead(hidden_dim=HID, name="cca")
    with pytest.raises(ValueError, match="feature dim"):
        build_feature_endpoint_model({"cca": head}, hidden_dim=HID + 4)


def test_dataset_from_embeddings_ratio_batch_shapes_and_keys():
    pos = (np.ones((20, HID), dtype="float32"), np.ones(20, dtype="float32"))
    unl = (np.zeros((200, HID), dtype="float32"), np.zeros(200, dtype="float32"))
    ds = dataset_from_embeddings(
        shuffle_buffer=50, batch_size=32, data=[pos, unl], weights=[0.5, 0.5]
    )
    batch = next(iter(ds))
    assert set(batch.keys()) == {"features", "cca_targets"}
    assert tuple(batch["features"].shape) == (32, HID)
    assert tuple(batch["cca_targets"].shape) == (32,)
    # ~50/50 ratio batch: positives (label 1) should be well-represented.
    pos_frac = float(keras.ops.mean(batch["cca_targets"]))
    assert 0.2 < pos_frac < 0.8
