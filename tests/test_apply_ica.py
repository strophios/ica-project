"""
Tests for src.apply_ica (IcaModel application over corpora).

Verifies:
  - Output schema and dtypes (API and LDC paths)
  - Score ranges [0, 1]
  - LDC gold-first gating (gold label overrides ML)
  - Per-year output files
  - Ranked candidates parquet
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import polars as pl

import keras  # noqa: F401  (initializes backend)

from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_feature_inference_model
from src.calibration.calibrator import PlattCalibrator, platt_fit
from src.calibration.sidecar import save_calibration
from src.cca_config import (
    RunConfig,
    config_path_for_weights,
    LRScheduleConfig,
    OptimizerConfig,
    HeadConfig,
    FLPULossConfig,
    RatioBatchConfig,
)
from src.us_config import UsRunConfig, UsHeadConfig
from src.fusion.sidecar import save_fusion
from src.fusion.combiner import FusionConfig


HIDDEN_DIM = 768
BATCH = 10


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tiny_ica_model():
    """Create a minimal IcaModel-compatible artifact set for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        heads_data = {}

        # Create three tiny heads (us, cca, rel)
        for head_name in ["us", "cca", "rel"]:
            head = ClassificationHead(
                hidden_dim=HIDDEN_DIM,
                loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
                name=head_name,
            )

            model = build_feature_inference_model(
                {head_name: head}, hidden_dim=HIDDEN_DIM
            )

            # Dummy forward pass
            dummy_features = np.random.randn(BATCH, HIDDEN_DIM).astype(np.float32)
            _ = model.predict({"features": dummy_features}, verbose=0)

            # Save weights
            weights_path = tmpdir_path / f"{head_name}.weights.h5"
            model.save_weights(str(weights_path))

            # Save config
            if head_name == "us":
                us_config = UsRunConfig(
                    seq_length=512,
                    text_key="text",
                    target_dtype="float32",
                    backbone_weights_path="dummy.h5",
                    head=UsHeadConfig(
                        name="us",
                        source_column="us_label",
                        hidden_dim=HIDDEN_DIM,
                    ),
                    epochs=1,
                    lr_schedule=LRScheduleConfig(
                        initial_lr=2e-5,
                        warmup_target=5e-5,
                        decay_alpha=0.1,
                        warmup_steps_factor=0.25,
                        decay_steps_factor=3.0,
                    ),
                    optimizer=OptimizerConfig(weight_decay=0.01),
                )
                config_path = config_path_for_weights(weights_path)
                us_config.to_json(config_path)
            else:
                cca_config = RunConfig(
                    seq_length=512,
                    text_key="text",
                    target_dtype="float32",
                    backbone_weights_path="dummy.h5",
                    heads=(
                        HeadConfig(
                            name=head_name,
                            source_column=f"{head_name}_label",
                            hidden_dim=HIDDEN_DIM,
                            loss=FLPULossConfig(prior=0.02),
                        ),
                    ),
                    epochs=1,
                    ratio_batch=RatioBatchConfig(train_pos=0.1, val_pos=0.5, test_pos=0.5),
                    lr_schedule=LRScheduleConfig(
                        initial_lr=2e-5,
                        warmup_target=5e-5,
                        decay_alpha=0.1,
                        warmup_steps_factor=0.25,
                        decay_steps_factor=3.0,
                    ),
                    optimizer=OptimizerConfig(weight_decay=0.01),
                )
                config_path = config_path_for_weights(weights_path)
                cca_config.to_json(config_path)

            # Fit calibrator
            logits = np.random.randn(BATCH).astype(np.float32)
            labels = np.random.randint(0, 2, BATCH).astype(int)
            A, B = platt_fit(logits, labels)
            calibrator = PlattCalibrator(
                A=A, B=B, fit_population="test", n=BATCH, method="platt"
            )

            # Save calibration
            from src.calibration.sidecar import calibration_path_for_weights
            calibration_path = calibration_path_for_weights(weights_path)
            save_calibration(calibrator, str(calibration_path))

            heads_data[head_name] = str(weights_path)

        # Create and save fusion config
        fusion_config = FusionConfig(
            gate_threshold=0.5,
            combine="product",
            coefs=None,
            score_space="prob",
            includes_us=False,
            composed_platt=None,
            head_calibrators={"us": "test", "cca": "test", "rel": "test"},
        )
        fusion_path = tmpdir_path / "ica_fusion.fusion.json"
        save_fusion(fusion_config, str(fusion_path))

        heads_data["tmpdir"] = tmpdir_path
        heads_data["fusion_path"] = str(fusion_path)

        yield heads_data


@pytest.fixture
def synthetic_cache_api():
    """Create a synthetic API cache (meta + features) for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create synthetic metadata: id, year, us_logit, emb_row
        n_rows = 20
        ids = [f"api_{i:06d}" for i in range(n_rows)]
        years = [1960 + (i % 5) for i in range(n_rows)]
        us_logits = np.random.randn(n_rows).astype(np.float32).tolist()

        meta = pl.DataFrame({
            "id": ids,
            "year": years,
            "us_logit": us_logits,
        }).with_row_index("emb_row")

        # Create synthetic CLS features
        cls_features = np.random.randn(n_rows, HIDDEN_DIM).astype(np.float32)

        # Save as shards (simple: one shard for testing)
        cache_dir = tmpdir_path / "test_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        np.save(cache_dir / "shard_000_cls.npy", cls_features)
        meta.write_parquet(cache_dir / "shard_000_meta.parquet")

        # Save provenance
        import json
        provenance = {"timestamp": "20260626", "lead_column": "lead_paragraph"}
        with open(cache_dir / "provenance.json", "w") as f:
            json.dump(provenance, f)

        yield cache_dir, meta, cls_features


@pytest.fixture
def synthetic_cache_ldc():
    """Create a synthetic LDC cache (1996-2007) for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create synthetic metadata: id, year, us_logit, emb_row (1996-2007)
        n_rows = 30
        ids = [f"ldc_{i:06d}" for i in range(n_rows)]
        years = [1996 + (i % 12) for i in range(n_rows)]
        us_logits = np.random.randn(n_rows).astype(np.float32).tolist()

        meta = pl.DataFrame({
            "id": ids,
            "year": years,
            "us_logit": us_logits,
        }).with_row_index("emb_row")

        # Create synthetic CLS features
        cls_features = np.random.randn(n_rows, HIDDEN_DIM).astype(np.float32)

        # Save as shards
        cache_dir = tmpdir_path / "test_cache_ldc"
        cache_dir.mkdir(parents=True, exist_ok=True)

        np.save(cache_dir / "shard_000_cls.npy", cls_features)
        meta.write_parquet(cache_dir / "shard_000_meta.parquet")

        yield cache_dir, meta, cls_features


# =============================================================================
# Tests: Output schema and dtypes (API path)
# =============================================================================


def test_apply_ica_api_output_schema(tiny_ica_model, synthetic_cache_api):
    """Verify API apply produces correct schema and dtypes."""
    from src.apply_ica import apply_ica_api

    cache_dir, meta, cls_features = synthetic_cache_api

    with mock.patch("src.apply_ica.config") as mock_config, \
         mock.patch("src.apply_ica.load_cache") as mock_load_cache, \
         mock.patch("src.apply_ica.IcaModel") as mock_ica_model:

        # Mock config paths
        mock_config.CCA_EMBED_CACHE_DIR = cache_dir.parent
        output_dir_us = Path(mock_config.CCA_EMBED_CACHE_DIR) / "us_scores"
        output_dir_cca = Path(mock_config.CCA_EMBED_CACHE_DIR) / "cca_scores"
        output_dir_ica = Path(mock_config.CCA_EMBED_CACHE_DIR) / "ica_candidates"
        mock_config.US_FILTER_SCORES_DIR = output_dir_us
        mock_config.CCA_DOCA_SCORES_DIR = output_dir_cca
        mock_config.ICA_CANDIDATES_DIR = output_dir_ica

        # Mock load_cache
        mock_load_cache.return_value = (meta, cls_features)

        # Mock IcaModel
        model_instance = mock_ica_model.return_value
        us_scores = np.random.uniform(0, 1, meta.height).astype(np.float32)
        cca_scores = np.random.uniform(0, 1, meta.height).astype(np.float32)
        rel_scores = np.random.uniform(0, 1, meta.height).astype(np.float32)
        ica_scores = np.random.uniform(0, 1, meta.height).astype(np.float32)

        model_instance.predict_ica_from_features.return_value = {
            "us": us_scores,
            "cca": cca_scores,
            "rel": rel_scores,
            "ica_score": ica_scores,
        }
        model_instance.fusion_config.gate_threshold = 0.5

        # Call apply
        apply_ica_api(cache_suffix="test_cache")

        # Verify candidate file
        candidates_path = output_dir_ica / "api_1960_1995.parquet"
        assert candidates_path.exists()

        candidates = pl.read_parquet(candidates_path)
        expected_cols = ["id", "year", "us_score", "cca_score", "rel_score", "ica_score", "gated"]
        assert list(candidates.columns) == expected_cols

        # Verify dtypes
        assert candidates["id"].dtype == pl.String
        assert candidates["year"].dtype == pl.Int64  # polars default for integers
        assert candidates["us_score"].dtype == pl.Float32
        assert candidates["cca_score"].dtype == pl.Float32
        assert candidates["rel_score"].dtype == pl.Float32
        assert candidates["ica_score"].dtype == pl.Float32
        assert candidates["gated"].dtype == pl.Boolean

        # Verify score ranges [0, 1]
        assert (candidates["us_score"] >= 0).all()
        assert (candidates["us_score"] <= 1).all()
        assert (candidates["cca_score"] >= 0).all()
        assert (candidates["cca_score"] <= 1).all()
        assert (candidates["rel_score"] >= 0).all()
        assert (candidates["rel_score"] <= 1).all()
        assert (candidates["ica_score"] >= 0).all()
        assert (candidates["ica_score"] <= 1).all()


def test_apply_ica_ldc_output_schema(tiny_ica_model, synthetic_cache_ldc):
    """Verify LDC apply produces correct schema and dtypes, including gate_source."""
    from src.apply_ica import apply_ica_ldc

    cache_dir, meta, cls_features = synthetic_cache_ldc

    with mock.patch("src.apply_ica.config") as mock_config, \
         mock.patch("src.apply_ica.load_cache") as mock_load_cache, \
         mock.patch("src.apply_ica.IcaModel") as mock_ica_model:

        # Mock config
        mock_config.CCA_EMBED_CACHE_DIR = cache_dir.parent
        output_dir_ica = Path(mock_config.CCA_EMBED_CACHE_DIR) / "ica_candidates"
        mock_config.ICA_CANDIDATES_DIR = output_dir_ica

        # Create gold labels file
        gold_labels = pl.DataFrame({
            "id": meta["id"].to_list()[:10],  # Only label first 10
            "us_label": [True, False, True, None, True, False, True, None, True, False],
        })
        gold_labels_path = Path(mock_config.CCA_EMBED_CACHE_DIR) / "ldc_labeled.parquet"
        gold_labels.write_parquet(gold_labels_path)
        mock_config.US_FILTER_LABELED_PARQUET = gold_labels_path

        # Mock load_cache
        mock_load_cache.return_value = (meta, cls_features)

        # Mock IcaModel
        model_instance = mock_ica_model.return_value
        us_scores = np.random.uniform(0, 1, meta.height).astype(np.float32)
        cca_scores = np.random.uniform(0, 1, meta.height).astype(np.float32)
        rel_scores = np.random.uniform(0, 1, meta.height).astype(np.float32)
        ica_scores_ungatted = np.random.uniform(0, 1, meta.height).astype(np.float32)
        ica_scores_gated = np.where(us_scores >= 0.5, ica_scores_ungatted, 0.0)

        model_instance.predict_ica_from_features.side_effect = [
            {  # First call: ML gate
                "us": us_scores,
                "cca": cca_scores,
                "rel": rel_scores,
                "ica_score": ica_scores_ungatted,
            },
            {  # Second call: gate_override
                "us": us_scores,
                "cca": cca_scores,
                "rel": rel_scores,
                "ica_score": ica_scores_gated,
            },
        ]
        model_instance.fusion_config.gate_threshold = 0.5

        # Call apply
        apply_ica_ldc(cache_suffix="test_cache_ldc")

        # Verify candidate file
        candidates_path = output_dir_ica / "ldc_1996_2007.parquet"
        assert candidates_path.exists()

        candidates = pl.read_parquet(candidates_path)
        expected_cols = ["id", "year", "us_score", "cca_score", "rel_score", "ica_score", "gated", "gate_source"]
        assert list(candidates.columns) == expected_cols

        # Verify dtypes
        assert candidates["gate_source"].dtype == pl.String

        # Verify gate_source is either "gold" or "ml"
        assert candidates["gate_source"].is_in(["gold", "ml"]).all()


def test_apply_ica_ldc_gold_first_gating(synthetic_cache_ldc):
    """Verify LDC gold-first gating: gold label overrides ML score."""
    from src.apply_ica import apply_ica_ldc

    cache_dir, meta, cls_features = synthetic_cache_ldc

    with mock.patch("src.apply_ica.config") as mock_config, \
         mock.patch("src.apply_ica.load_cache") as mock_load_cache, \
         mock.patch("src.apply_ica.IcaModel") as mock_ica_model:

        # Mock config
        mock_config.CCA_EMBED_CACHE_DIR = cache_dir.parent
        output_dir_ica = Path(mock_config.CCA_EMBED_CACHE_DIR) / "ica_candidates"
        mock_config.ICA_CANDIDATES_DIR = output_dir_ica

        # Create gold labels: specific scenario
        # Row 0: gold=True (US), but ML says False (low us_score) → gate_override should be True
        # Row 1: gold=False (not-US), but ML says True (high us_score) → gate_override should be False
        # Row 2: gold=None, ML=True → use ML
        # Row 3: gold=None, ML=False → use ML
        gold_labels = pl.DataFrame({
            "id": meta["id"].to_list()[:4],
            "us_label": [True, False, None, None],
        })
        gold_labels_path = Path(mock_config.CCA_EMBED_CACHE_DIR) / "ldc_labeled.parquet"
        gold_labels.write_parquet(gold_labels_path)
        mock_config.US_FILTER_LABELED_PARQUET = gold_labels_path

        # Mock load_cache
        mock_load_cache.return_value = (meta, cls_features)

        # Mock IcaModel with specific scores
        model_instance = mock_ica_model.return_value
        n_rows = meta.height
        # Scores that test the gold-first logic
        us_scores = np.array(
            [0.1, 0.9] + [0.5] * (n_rows - 2), dtype=np.float32
        )  # Low, high, middle...
        cca_scores = np.random.uniform(0, 1, n_rows).astype(np.float32)
        rel_scores = np.random.uniform(0, 1, n_rows).astype(np.float32)
        ica_base = np.random.uniform(0, 1, n_rows).astype(np.float32)

        # Create a mock that tracks calls
        call_count = [0]

        def predict_side_effect(features, gate_override=None):
            call_count[0] += 1
            if gate_override is not None:
                # Gated version: apply the override
                ica_gated = np.where(gate_override, ica_base, 0.0)
            else:
                # ML gating
                ml_gate = us_scores >= 0.5
                ica_gated = np.where(ml_gate, ica_base, 0.0)
            return {
                "us": us_scores,
                "cca": cca_scores,
                "rel": rel_scores,
                "ica_score": ica_gated,
            }

        model_instance.predict_ica_from_features.side_effect = predict_side_effect
        model_instance.fusion_config.gate_threshold = 0.5

        # Call apply
        apply_ica_ldc(cache_suffix="test_cache_ldc")

        # Verify candidate file
        candidates_path = output_dir_ica / "ldc_1996_2007.parquet"
        assert candidates_path.exists()

        candidates = pl.read_parquet(candidates_path)

        # Row 0: gold=True (should be gated in despite low us_score)
        # Row 1: gold=False (should be gated out despite high us_score)
        # Rows 2-3: gold=None (should follow ML gate at 0.5 threshold)
        # Rows 4+: no gold (ML gate)

        row_0 = candidates.filter(pl.col("id") == meta["id"][0]).to_dicts()[0]
        row_1 = candidates.filter(pl.col("id") == meta["id"][1]).to_dicts()[0]

        # Row 0 should be gated in (gold=True overrides low us_score)
        assert row_0["gated"] is True, "Row 0 with gold=True should be gated in"
        assert row_0["gate_source"] == "gold", "Row 0 should use gold source"

        # Row 1 should be gated out (gold=False overrides high us_score)
        assert row_1["gated"] is False, "Row 1 with gold=False should be gated out"
        assert row_1["gate_source"] == "gold", "Row 1 should use gold source"
