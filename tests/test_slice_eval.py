# pattern: Imperative Shell (applies model, reads/writes, computes metrics)
"""Tests for src.validation.slice_eval — transfer eval + proxy gap."""

import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import keras  # noqa: F401  (initializes backend)

from src.validation.slice_eval import (
    apply_us_model,
    evaluate_slice,
    proxy_gap,
)
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_inference_model
from src.us_config import UsRunConfig, UsHeadConfig
from src.cca_config import LRScheduleConfig, OptimizerConfig, DiagnosticsConfig
from src.calibration.calibrator import PlattCalibrator
import src.config as config


SEQ_LEN = 4
HIDDEN_DIM = 8
VOCAB = 100
BATCH = 4


def _make_fake_backbone(seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM, vocab=VOCAB,
                        name="fake_backbone"):
    """Build a tiny stand-in for a real backbone."""
    token_ids = keras.Input(shape=(seq_len,), dtype="int32", name="token_ids")
    padding_mask = keras.Input(shape=(seq_len,), dtype="int32", name="padding_mask")
    embed = keras.layers.Embedding(vocab, hidden_dim, name="fake_embed")
    embedded = embed(token_ids)
    mask_float = keras.ops.cast(padding_mask, "float32")
    mask_expanded = keras.ops.expand_dims(mask_float, axis=-1)
    masked = embedded * mask_expanded
    backbone = keras.Model(
        inputs={"token_ids": token_ids, "padding_mask": padding_mask},
        outputs=masked,
        name=name,
    )
    # Add hidden_dim attribute for config validation
    backbone.hidden_dim = hidden_dim
    return backbone


class TestApplyUsModel:
    """US model application with calibration."""

    def test_apply_us_model_returns_calibrated_scores(self, tmp_path):
        """apply_us_model returns calibrated probabilities [0, 1], shape correct."""
        # Build fake backbone + head + inference model
        backbone = _make_fake_backbone()
        us_head = ClassificationHead(hidden_dim=HIDDEN_DIM, name="us")

        inference_model = build_inference_model(
            backbone=backbone,
            heads={"us": us_head},
            seq_length=SEQ_LEN,
        )

        weights_path = tmp_path / "test.weights.h5"
        inference_model.save_weights(str(weights_path))

        # Write UsRunConfig sidecar
        config_sidecar = tmp_path / "test.config.json"
        config_path_val = UsRunConfig(
            seq_length=SEQ_LEN,
            text_key="headline_with_lead",
            target_dtype="float32",
            head=UsHeadConfig(name="us", source_column="us_label", hidden_dim=HIDDEN_DIM),
            epochs=1,
            backbone_weights_path=str(weights_path),  # Not actually used when backbone is injected
            lr_schedule=LRScheduleConfig(),
            optimizer=OptimizerConfig(),
            diagnostics=DiagnosticsConfig(enable_loss_components=False),
        )
        config_path_val.to_json(config_sidecar)

        # Write calibrator sidecar (identity: A=1, B=0)
        calibrator_json = tmp_path / "test.calibration.json"
        calibrator_json.write_text(json.dumps({
            "method": "platt",
            "A": 1.0,
            "B": 0.0,
            "fit_population": "train",
            "n": 100,
        }))

        # Create synthetic texts for apply_us_model
        texts = ["test article 1", "test article 2", "test article 3"]

        # Call apply_us_model with injected fake backbone
        scores = apply_us_model(texts, weights_path=weights_path, backbone=backbone)

        # Verify shape, finiteness, and range
        assert scores.shape == (len(texts),), f"Expected shape ({len(texts)},), got {scores.shape}"
        assert np.isfinite(scores).all(), "Non-finite scores produced"
        assert (scores >= 0).all() and (scores <= 1).all(), "Scores outside [0, 1]"

    def test_apply_us_model_handles_missing_calibration(self, tmp_path):
        """Missing calibration file raises clear error."""
        backbone = _make_fake_backbone()
        us_head = ClassificationHead(hidden_dim=HIDDEN_DIM, name="us")
        inference_model = build_inference_model(
            backbone=backbone,
            heads={"us": us_head},
            seq_length=SEQ_LEN,
        )

        weights_path = tmp_path / "test.weights.h5"
        inference_model.save_weights(str(weights_path))

        # Write UsRunConfig but NOT calibrator
        config_sidecar = tmp_path / "test.config.json"
        config_path_val = UsRunConfig(
            seq_length=SEQ_LEN,
            text_key="headline_with_lead",
            target_dtype="float32",
            head=UsHeadConfig(name="us", source_column="us_label", hidden_dim=HIDDEN_DIM),
            epochs=1,
            backbone_weights_path=str(weights_path),
            lr_schedule=LRScheduleConfig(),
            optimizer=OptimizerConfig(),
            diagnostics=DiagnosticsConfig(enable_loss_components=False),
        )
        config_path_val.to_json(config_sidecar)

        texts = ["test"]

        # Should raise when calibration is missing
        with pytest.raises(FileNotFoundError):
            apply_us_model(texts, weights_path=weights_path, backbone=backbone)


class TestEvaluateSlice:
    """Slice evaluation: precision, recall, F1."""

    def test_evaluate_slice_computes_metrics_from_known_probs(self):
        """Known probs/labels yield exact metrics."""
        # Create gold-set with known labels
        gold_df = pl.DataFrame({
            "id": ["1", "2", "3", "4"],
            "us_event": [True, True, False, False],
            "us_score": [0.9, 0.8, 0.3, 0.2],
        })

        # At threshold 0.5: pred = [True, True, False, False]
        # Exactly matches us_event, so P=1, R=1, F1=1
        metrics = evaluate_slice(gold_df, threshold=0.5)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["n_pos"] == 2
        assert metrics["n_neg"] == 2

    def test_evaluate_slice_threshold_affects_predictions(self):
        """Changing threshold changes TP/FP/FN counts."""
        gold_df = pl.DataFrame({
            "id": ["1", "2", "3", "4", "5"],
            "us_event": [True, True, False, False, False],
            "us_score": [0.9, 0.6, 0.5, 0.3, 0.1],
        })

        # Threshold 0.7: only id=1 predicted positive
        metrics_high = evaluate_slice(gold_df, threshold=0.7)
        # TP=1, FN=1, FP=0 => P=1, R=0.5, F1=2/3
        assert metrics_high["precision"] == 1.0
        assert abs(metrics_high["recall"] - 0.5) < 1e-6

        # Threshold 0.4: ids 1,2 predicted positive
        metrics_low = evaluate_slice(gold_df, threshold=0.4)
        # TP=2, FN=0, FP=1 => P=2/3, R=1, F1=4/5
        assert metrics_low["recall"] == 1.0

    def test_evaluate_slice_handles_all_negatives(self):
        """All negatives (no positives in gold set)."""
        gold_df = pl.DataFrame({
            "id": ["1", "2"],
            "us_event": [False, False],
            "us_score": [0.9, 0.8],
        })

        metrics = evaluate_slice(gold_df, threshold=0.5)

        # No positives, so precision/recall are 0 or nan
        assert metrics["n_pos"] == 0
        assert metrics["n_neg"] == 2

    def test_evaluate_slice_handles_all_positives(self):
        """All positives (no negatives in gold set)."""
        gold_df = pl.DataFrame({
            "id": ["1", "2"],
            "us_event": [True, True],
            "us_score": [0.9, 0.8],
        })

        metrics = evaluate_slice(gold_df, threshold=0.5)

        assert metrics["n_pos"] == 2
        assert metrics["n_neg"] == 0

    def test_evaluate_slice_returns_dict_with_required_keys(self):
        """Returned dict has all required keys."""
        gold_df = pl.DataFrame({
            "id": ["1"],
            "us_event": [True],
            "us_score": [0.8],
        })

        metrics = evaluate_slice(gold_df, threshold=0.5)

        required_keys = {"precision", "recall", "f1", "n_pos", "n_neg"}
        assert set(metrics.keys()) == required_keys


class TestProxyGap:
    """Dateline-vs-event-location proxy gap."""

    def test_proxy_gap_perfect_agreement(self):
        """When dateline and event_location coding agree perfectly, agreement=1."""
        gold_df = pl.DataFrame({
            "id": ["api_1", "api_2", "api_3"],
            "alt_corpus_id": ["ldc_1", "ldc_2", "ldc_3"],
            "us_event": [True, False, True],
            "event_location": ["US", "Foreign", "US"],
        })

        dateline_labels_df = pl.DataFrame({
            "ldc_id": ["ldc_1", "ldc_2", "ldc_3"],
            "us_label": [True, False, True],  # Perfectly aligned
        })

        gap = proxy_gap(gold_df, dateline_labels_df=dateline_labels_df)

        assert gap["dateline_event_agreement"] == 1.0
        assert gap["n"] == 3

    def test_proxy_gap_no_agreement(self):
        """When dateline and event_location coding disagree completely, agreement=0."""
        gold_df = pl.DataFrame({
            "id": ["api_1", "api_2"],
            "alt_corpus_id": ["ldc_1", "ldc_2"],
            "us_event": [True, False],
            "event_location": ["US", "Foreign"],
        })

        dateline_labels_df = pl.DataFrame({
            "ldc_id": ["ldc_1", "ldc_2"],
            "us_label": [False, True],  # Completely reversed
        })

        gap = proxy_gap(gold_df, dateline_labels_df=dateline_labels_df)

        assert gap["dateline_event_agreement"] == 0.0
        assert gap["n"] == 2

    def test_proxy_gap_partial_agreement(self):
        """Partial agreement yields expected fraction."""
        gold_df = pl.DataFrame({
            "id": ["api_1", "api_2", "api_3", "api_4"],
            "alt_corpus_id": ["ldc_1", "ldc_2", "ldc_3", "ldc_4"],
            "event_location": ["US", "US", "Foreign", "Foreign"],
        })

        dateline_labels_df = pl.DataFrame({
            "ldc_id": ["ldc_1", "ldc_2", "ldc_3", "ldc_4"],
            "us_label": [True, True, True, False],  # 3 agree, 1 disagree
        })

        gap = proxy_gap(gold_df, dateline_labels_df=dateline_labels_df)

        assert gap["dateline_event_agreement"] == 0.75
        assert gap["n"] == 4

    def test_proxy_gap_excludes_missing_alt_corpus_id(self):
        """Rows with null alt_corpus_id are excluded from join."""
        gold_df = pl.DataFrame({
            "id": ["api_1", "api_2", "api_3"],
            "alt_corpus_id": ["ldc_1", None, "ldc_3"],
            "event_location": ["US", "US", "Foreign"],
        })

        dateline_labels_df = pl.DataFrame({
            "ldc_id": ["ldc_1", "ldc_3"],
            "us_label": [True, False],
        })

        gap = proxy_gap(gold_df, dateline_labels_df=dateline_labels_df)

        # Only api_1 and api_3 join (api_2 has null alt_corpus_id)
        assert gap["n"] == 2
        assert gap["dateline_event_agreement"] == 1.0  # Both match

    def test_proxy_gap_excludes_missing_event_location(self):
        """Rows with null event_location are excluded from join."""
        gold_df = pl.DataFrame({
            "id": ["api_1", "api_2"],
            "alt_corpus_id": ["ldc_1", "ldc_2"],
            "event_location": ["US", None],
        })

        dateline_labels_df = pl.DataFrame({
            "ldc_id": ["ldc_1", "ldc_2"],
            "us_label": [True, False],
        })

        gap = proxy_gap(gold_df, dateline_labels_df=dateline_labels_df)

        # Only api_1 joins (api_2 has null event_location)
        assert gap["n"] == 1
        assert gap["dateline_event_agreement"] == 1.0

    def test_proxy_gap_empty_join(self):
        """When no rows join, returns n=0 and agreement=0.0."""
        gold_df = pl.DataFrame({
            "id": ["api_1"],
            "alt_corpus_id": [None],
            "event_location": ["US"],
        })

        dateline_labels_df = pl.DataFrame({
            "ldc_id": ["ldc_999"],
            "us_label": [True],
        })

        gap = proxy_gap(gold_df, dateline_labels_df=dateline_labels_df)

        assert gap["n"] == 0
        assert gap["dateline_event_agreement"] == 0.0

    def test_proxy_gap_returns_dict_with_required_keys(self):
        """proxy_gap returns dict with exactly the required keys."""
        gold_df = pl.DataFrame({
            "id": ["api_1"],
            "alt_corpus_id": ["ldc_1"],
            "event_location": ["US"],
        })

        dateline_labels_df = pl.DataFrame({
            "ldc_id": ["ldc_1"],
            "us_label": [True],
        })

        gap = proxy_gap(gold_df, dateline_labels_df=dateline_labels_df)

        required_keys = {"dateline_event_agreement", "n"}
        assert set(gap.keys()) == required_keys
