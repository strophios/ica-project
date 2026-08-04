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


def test_apply_ica_api_output_schema(synthetic_cache_api):
    """Verify API apply produces correct schema and dtypes."""
    from src.apply_ica import apply_ica_api

    cache_dir, meta, cls_features = synthetic_cache_api

    with mock.patch("src.apply_ica.config") as mock_config, \
         mock.patch("src.apply_ica.load_cache") as mock_load_cache, \
         mock.patch("src.apply_ica.IcaModel") as mock_ica_model, \
         mock.patch("src.apply_ica.assert_scoring_integrity"):

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


def test_apply_ica_ldc_output_schema(synthetic_cache_ldc):
    """Verify LDC apply produces correct schema and dtypes, including gate_source."""
    from src.apply_ica import apply_ica_ldc

    cache_dir, meta, cls_features = synthetic_cache_ldc

    with mock.patch("src.apply_ica.config") as mock_config, \
         mock.patch("src.apply_ica.load_cache") as mock_load_cache, \
         mock.patch("src.apply_ica.IcaModel") as mock_ica_model, \
         mock.patch("src.apply_ica.assert_scoring_integrity"):

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
         mock.patch("src.apply_ica.IcaModel") as mock_ica_model, \
         mock.patch("src.apply_ica.assert_scoring_integrity"):

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


# =============================================================================
# Tests: out_name + years parameterization (the 1996-2025 forward apply)
# =============================================================================


def _mock_api_apply(mock_config, cache_dir, meta, cls_features, mock_load_cache, mock_ica_model):
    """Shared mock wiring for the API path (mirrors test_apply_ica_api_output_schema)."""
    mock_config.CCA_EMBED_CACHE_DIR = cache_dir.parent
    mock_config.US_FILTER_SCORES_DIR = Path(mock_config.CCA_EMBED_CACHE_DIR) / "us_scores"
    mock_config.CCA_DOCA_SCORES_DIR = Path(mock_config.CCA_EMBED_CACHE_DIR) / "cca_scores"
    mock_config.ICA_CANDIDATES_DIR = Path(mock_config.CCA_EMBED_CACHE_DIR) / "ica_candidates"
    mock_load_cache.return_value = (meta, cls_features)

    model_instance = mock_ica_model.return_value

    def predict(features, gate_override=None):
        n = features.shape[0]
        rng = np.random.default_rng(7)
        return {
            "us": rng.uniform(0, 1, n).astype(np.float32),
            "cca": rng.uniform(0, 1, n).astype(np.float32),
            "rel": rng.uniform(0, 1, n).astype(np.float32),
            "ica_score": rng.uniform(0, 1, n).astype(np.float32),
        }

    model_instance.predict_ica_from_features.side_effect = predict
    model_instance.fusion_config.gate_threshold = 0.5
    return model_instance


def test_apply_ica_api_custom_out_name(synthetic_cache_api):
    """out_name routes the candidates parquet (forward run: api_1996_2025.parquet)."""
    from src.apply_ica import apply_ica_api

    cache_dir, meta, cls_features = synthetic_cache_api
    with mock.patch("src.apply_ica.config") as mock_config, \
         mock.patch("src.apply_ica.load_cache") as mock_load_cache, \
         mock.patch("src.apply_ica.IcaModel") as mock_ica_model, \
         mock.patch("src.apply_ica.assert_scoring_integrity"):
        _mock_api_apply(mock_config, cache_dir, meta, cls_features,
                        mock_load_cache, mock_ica_model)
        apply_ica_api(cache_suffix="test_cache", out_name="api_1996_2025.parquet")
        out_dir = Path(mock_config.ICA_CANDIDATES_DIR)
        assert (out_dir / "api_1996_2025.parquet").exists()
        assert not (out_dir / "api_1960_1995.parquet").exists()


def test_apply_ica_api_years_filter(synthetic_cache_api):
    """years=(lo, hi) restricts scoring to that inclusive range, keeping
    meta rows and CLS features row-aligned."""
    from src.apply_ica import apply_ica_api

    cache_dir, meta, cls_features = synthetic_cache_api
    with mock.patch("src.apply_ica.config") as mock_config, \
         mock.patch("src.apply_ica.load_cache") as mock_load_cache, \
         mock.patch("src.apply_ica.IcaModel") as mock_ica_model, \
         mock.patch("src.apply_ica.assert_scoring_integrity"):
        model_instance = _mock_api_apply(mock_config, cache_dir, meta, cls_features,
                                         mock_load_cache, mock_ica_model)
        apply_ica_api(cache_suffix="test_cache", years=(1960, 1961))
        expected_rows = meta.filter(
            pl.col("year").cast(pl.Int64).is_between(1960, 1961)
        ).height
        candidates = pl.read_parquet(
            Path(mock_config.ICA_CANDIDATES_DIR) / "api_1960_1995.parquet"
        )
        assert candidates.height == expected_rows
        assert set(candidates["year"].to_list()) == {1960, 1961}
        # The model must only have been fed the filtered feature rows.
        (called_features,), _ = model_instance.predict_ica_from_features.call_args
        assert called_features.shape[0] == expected_rows


def test_apply_ica_ldc_years_parameterized(synthetic_cache_ldc):
    """apply_ica_ldc accepts years + out_name (default stays 1996-2007 /
    ldc_1996_2007.parquet — covered by the existing schema test)."""
    from src.apply_ica import apply_ica_ldc

    cache_dir, meta, cls_features = synthetic_cache_ldc
    with mock.patch("src.apply_ica.config") as mock_config, \
         mock.patch("src.apply_ica.load_cache") as mock_load_cache, \
         mock.patch("src.apply_ica.IcaModel") as mock_ica_model, \
         mock.patch("src.apply_ica.assert_scoring_integrity"):
        mock_config.CCA_EMBED_CACHE_DIR = cache_dir.parent
        mock_config.ICA_CANDIDATES_DIR = Path(mock_config.CCA_EMBED_CACHE_DIR) / "ica_candidates"
        gold_labels = pl.DataFrame({
            "id": meta["id"].to_list()[:4],
            "us_label": [True, False, None, None],
        })
        gold_labels_path = Path(mock_config.CCA_EMBED_CACHE_DIR) / "ldc_labeled.parquet"
        gold_labels.write_parquet(gold_labels_path)
        mock_config.US_FILTER_LABELED_PARQUET = gold_labels_path
        mock_load_cache.return_value = (meta, cls_features)

        model_instance = mock_ica_model.return_value

        def predict(features, gate_override=None):
            n = features.shape[0]
            rng = np.random.default_rng(7)
            return {
                "us": rng.uniform(0, 1, n).astype(np.float32),
                "cca": rng.uniform(0, 1, n).astype(np.float32),
                "rel": rng.uniform(0, 1, n).astype(np.float32),
                "ica_score": rng.uniform(0, 1, n).astype(np.float32),
            }

        model_instance.predict_ica_from_features.side_effect = predict
        model_instance.fusion_config.gate_threshold = 0.5

        apply_ica_ldc(cache_suffix="test_cache_ldc", years=(1996, 2000),
                      out_name="ldc_1996_2000.parquet")
        candidates = pl.read_parquet(
            Path(mock_config.ICA_CANDIDATES_DIR) / "ldc_1996_2000.parquet"
        )
        expected_rows = meta.filter(
            pl.col("year").cast(pl.Int64).is_between(1996, 2000)
        ).height
        assert candidates.height == expected_rows
        assert candidates["year"].cast(pl.Int64).is_between(1996, 2000).all()


def test_apply_ica_parser_defaults():
    """CLI parser: --out-name and --years default to None (function defaults
    preserve today's output names and ranges)."""
    from src.apply_ica import build_arg_parser

    args = build_arg_parser().parse_args(["--corpus", "api"])
    assert args.out_name is None
    assert args.years is None


def test_apply_ica_parser_accepts_out_name_and_years():
    from src.apply_ica import build_arg_parser

    args = build_arg_parser().parse_args([
        "--corpus", "api", "--cache-suffix", "api_9625",
        "--out-name", "api_1996_2025.parquet", "--years", "1996-2025",
    ])
    assert args.out_name == "api_1996_2025.parquet"
    assert args.years == "1996-2025"


# =============================================================================
# Tests: scoring-integrity guard (predict-vs-direct head check)
# =============================================================================


def test_scoring_integrity_check_passes_on_consistent_model(tiny_ica_model):
    """When model.predict agrees with direct head computation the guard passes.

    Hardware-independence note: predict is mocked to return exactly the direct
    computation, because on a local MPS machine the REAL predict path fails
    this guard — that is the live 2026-08-04 tensorflow-metal bug, and the
    guard firing there is the desired production behavior (CUDA/CPU stacks
    are exact and pass with real predict)."""
    from src.apply_ica import assert_scoring_integrity
    from src.assemble_ica import IcaModel

    with mock.patch("src.assemble_ica.config") as mock_config:
        tmpdir = tiny_ica_model["tmpdir"]
        mock_config.US_FILTER_FULL_WEIGHTS = tiny_ica_model["us"]
        mock_config.CCA_DOCA_WEIGHTS = tiny_ica_model["cca"]
        mock_config.RELEVANCE_DOCA_WEIGHTS = tiny_ica_model["rel"]
        mock_config.CCA_DOCA_DIR = tmpdir
        model = IcaModel(fusion_path=tiny_ica_model["fusion_path"])
    feats = np.random.default_rng(3).standard_normal((64, HIDDEN_DIM)).astype(np.float32) * 0.4

    def consistent(inputs, verbose=0):
        sample = inputs["features"]
        return {
            name: np.asarray(head(sample)).reshape(-1, 1)
            for name, head in [("us", model.us_head), ("cca", model.cca_head),
                               ("rel", model.rel_head)]
        }

    with mock.patch.object(model.model, "predict", side_effect=consistent):
        assert_scoring_integrity(model, feats)  # must not raise


def test_scoring_integrity_check_raises_on_distorted_predict(tiny_ica_model):
    """If the compiled predict path disagrees with direct head computation
    (the 2026-08-04 tensorflow-metal bug signature), the guard raises before
    any candidates are written."""
    from src.apply_ica import assert_scoring_integrity
    from src.assemble_ica import IcaModel

    with mock.patch("src.assemble_ica.config") as mock_config:
        tmpdir = tiny_ica_model["tmpdir"]
        mock_config.US_FILTER_FULL_WEIGHTS = tiny_ica_model["us"]
        mock_config.CCA_DOCA_WEIGHTS = tiny_ica_model["cca"]
        mock_config.RELEVANCE_DOCA_WEIGHTS = tiny_ica_model["rel"]
        mock_config.CCA_DOCA_DIR = tmpdir
        model = IcaModel(fusion_path=tiny_ica_model["fusion_path"])
    feats = np.random.default_rng(3).standard_normal((64, HIDDEN_DIM)).astype(np.float32) * 0.4

    real_predict = model.model.predict

    def distorted(*args, **kwargs):
        out = real_predict(*args, **kwargs)
        return {k: v + 0.9 for k, v in out.items()}

    with mock.patch.object(model.model, "predict", side_effect=distorted):
        with pytest.raises(RuntimeError, match="scoring integrity"):
            assert_scoring_integrity(model, feats)
