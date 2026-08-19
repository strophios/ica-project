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


# =============================================================================
# build_feature_inference_model: multi-source variant (branched-encoder support)
# =============================================================================


def _named_head(name):
    return ClassificationHead(hidden_dim=HID, loss_fn=FLPULoss(prior=0.1), name=name)


def test_feature_inference_model_legacy_default_single_shared_input():
    """feature_sources=None (the default) stays byte-identical: one shared
    'features' Input, no per-source Input names."""
    heads = {"cca": _named_head("cca"), "rel": _named_head("rel2")}
    model = build_feature_inference_model(heads, hidden_dim=HID)
    assert len(model.inputs) == 1
    assert model.inputs[0].name == "features"


def test_feature_inference_model_multi_source_creates_one_input_per_distinct_tag():
    heads = {"cca": _named_head("cca3"), "rel": _named_head("rel3")}
    model = build_feature_inference_model(
        heads, hidden_dim=HID, feature_sources={"cca": "base", "rel": "rel_branch"}
    )
    input_names = sorted(i.name for i in model.inputs)
    assert input_names == ["features_base", "features_rel_branch"]


def test_feature_inference_model_multi_source_shared_tag_collapses_to_one_input():
    """Two heads mapped to the same tag share a single Input (distinct-tag count,
    not per-head count)."""
    heads = {"us": _named_head("us4"), "cca": _named_head("cca4")}
    model = build_feature_inference_model(
        heads, hidden_dim=HID, feature_sources={"us": "base", "cca": "base"}
    )
    assert len(model.inputs) == 1
    assert model.inputs[0].name == "features_base"


def test_feature_inference_model_multi_source_each_head_reads_its_own_tag():
    """Each head is wired to ITS tag's input, not some other head's — verified
    by feeding distinct arrays per tag and checking outputs match direct calls.

    Uses the model's eager `__call__` (not `.predict()`): comparing a compiled
    `.predict()` graph against a bare eager `head(sample)` call trips the known
    tensorflow-metal dropout-path discrepancy on local MPS (see
    `apply_ica.assert_scoring_integrity`'s atol=0.05 rationale) — both eager
    paths here avoid that entirely, so exact equality is the right check.
    """
    head_cca = _named_head("cca5")
    head_rel = _named_head("rel5")
    heads = {"cca": head_cca, "rel": head_rel}
    model = build_feature_inference_model(
        heads, hidden_dim=HID, feature_sources={"cca": "base", "rel": "rel_branch"}
    )
    rng = np.random.default_rng(0)
    arr_base = rng.standard_normal((5, HID)).astype("float32")
    arr_variant = rng.standard_normal((5, HID)).astype("float32") * 10.0  # well-separated

    out = model({"features_base": arr_base, "features_rel_branch": arr_variant})
    direct_cca = np.asarray(head_cca(arr_base)).reshape(-1, 1)
    direct_rel = np.asarray(head_rel(arr_variant)).reshape(-1, 1)
    np.testing.assert_allclose(np.asarray(out["cca"]), direct_cca, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out["rel"]), direct_rel, rtol=1e-5, atol=1e-6)


def test_feature_inference_model_multi_source_missing_head_raises():
    heads = {"cca": _named_head("cca6"), "rel": _named_head("rel6")}
    with pytest.raises(ValueError, match="rel"):
        build_feature_inference_model(
            heads, hidden_dim=HID, feature_sources={"cca": "base"}
        )


def test_feature_inference_model_multi_source_unknown_head_key_raises():
    heads = {"cca": _named_head("cca7")}
    with pytest.raises(ValueError, match="extra"):
        build_feature_inference_model(
            heads, hidden_dim=HID, feature_sources={"cca": "base", "extra": "base"}
        )


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
