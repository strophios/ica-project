"""Tests for src.apply_us_filter — batch US filter application to API corpus.

Exercises apply_us_filter.main() with synthetic API parquet, fake backbone,
and stub calibrator. Verifies output columns, ranges, threshold semantics,
row count, and id preservation.
"""

from unittest import mock

import numpy as np
import polars as pl
import pytest

import keras

from src.apply_us_filter import main
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_inference_model
from src.us_config import UsRunConfig, UsHeadConfig, config_path_for_weights
from src.cca_config import LRScheduleConfig, OptimizerConfig
from src.calibration.calibrator import PlattCalibrator
from src.calibration.sidecar import save_calibration, calibration_path_for_weights


SEQ_LEN = 4
HIDDEN_DIM = 8
VOCAB = 100


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
    backbone.hidden_dim = hidden_dim
    return backbone


@pytest.fixture
def setup_model_artifact(tmp_path):
    """Build fake model, weights, config, calibrator; save to tmp_path.

    Returns: (weights_path, backbone) for use in main() tests.
    """
    # Build and save model
    backbone = _make_fake_backbone()
    us_head = ClassificationHead(hidden_dim=HIDDEN_DIM, name="us")

    inference_model = build_inference_model(
        backbone=backbone,
        heads={"us": us_head},
        seq_length=SEQ_LEN,
    )

    weights_path = tmp_path / "test.weights.h5"
    inference_model.save_weights(str(weights_path))

    # Create and save config sidecar
    config = UsRunConfig(
        seq_length=SEQ_LEN,
        text_key="headline_with_lead",
        target_dtype="float32",
        head=UsHeadConfig(name="us", source_column="us_label", hidden_dim=HIDDEN_DIM),
        epochs=1,
        backbone_weights_path="fake",
        lr_schedule=LRScheduleConfig(),
        optimizer=OptimizerConfig(),
    )
    config_path = config_path_for_weights(weights_path)
    config.to_json(config_path)

    # Create and save calibrator sidecar
    calibrator = PlattCalibrator(A=0.5, B=-0.1, fit_population="test", n=10, method="platt")
    cal_path = calibration_path_for_weights(weights_path)
    save_calibration(calibrator, cal_path)

    return weights_path, backbone


@pytest.fixture
def synthetic_api_corpus(tmp_path):
    """Create synthetic API parquet corpus with id, headline_with_lead, year.

    Returns: corpus_dir (parquets written by year).
    """
    corpus_dir = tmp_path / "api_corpus_synthetic"
    corpus_dir.mkdir()

    # Write two years of synthetic data
    for year in [1980, 1981]:
        df = pl.DataFrame({
            "id": [f"api_{year}_{i}" for i in range(3)],
            "headline_with_lead": [f"Headline {year} {i}</s>Lead paragraph {i}" for i in range(3)],
            "year": [year] * 3,
        })
        df.write_parquet(corpus_dir / f"{year}.parquet")

    return corpus_dir


class TestApplyUsFilterMain:
    """Tests driving apply_us_filter.main() with synthetic setup."""

    def test_main_writes_per_year_files(self, tmp_path, setup_model_artifact, synthetic_api_corpus):
        """main() writes one parquet per distinct year."""
        weights_path, backbone = setup_model_artifact
        corpus_dir = synthetic_api_corpus
        output_dir = tmp_path / "us_filter_scores"

        # Monkeypatch config and data_from_parquet
        with mock.patch("src.apply_us_filter.config") as mock_config, \
             mock.patch("src.apply_us_filter.data_from_parquet") as mock_data_from_parquet, \
             mock.patch("src.apply_us_filter.apply_us_model") as mock_apply:

            mock_config.PROJECT_ROOT = tmp_path
            mock_config.US_FILTER_SCORES_DIR = output_dir

            # Synthetic corpus: 6 rows (3 per year, 1980 and 1981)
            corpus_df = pl.concat([
                pl.read_parquet(corpus_dir / "1980.parquet"),
                pl.read_parquet(corpus_dir / "1981.parquet"),
            ])
            mock_data_from_parquet.return_value = corpus_df

            # Mock apply_us_model to return fixed scores
            mock_scores = np.array([0.2, 0.6, 0.8, 0.3, 0.7, 0.9])
            mock_apply.return_value = mock_scores

            # Call main with threshold=0.5
            main(threshold=0.5)

            # Verify two output files were written
            assert (output_dir / "1980.parquet").exists()
            assert (output_dir / "1981.parquet").exists()

            # Verify 1980 output
            year_1980 = pl.read_parquet(output_dir / "1980.parquet")
            assert list(year_1980.columns) == ["id", "us_score", "us"]
            assert year_1980.shape[0] == 3
            assert year_1980["id"].to_list() == ["api_1980_0", "api_1980_1", "api_1980_2"]
            assert np.allclose(year_1980["us_score"].to_numpy(), [0.2, 0.6, 0.8])
            assert year_1980["us"].to_list() == [False, True, True]

            # Verify 1981 output
            year_1981 = pl.read_parquet(output_dir / "1981.parquet")
            assert list(year_1981.columns) == ["id", "us_score", "us"]
            assert year_1981.shape[0] == 3
            assert year_1981["id"].to_list() == ["api_1981_0", "api_1981_1", "api_1981_2"]
            assert np.allclose(year_1981["us_score"].to_numpy(), [0.3, 0.7, 0.9])
            assert year_1981["us"].to_list() == [False, True, True]

    def test_main_id_dtype_preserved_as_string(self, tmp_path, setup_model_artifact):
        """main() preserves id column as Utf8 (string)."""
        weights_path, backbone = setup_model_artifact
        output_dir = tmp_path / "us_filter_scores"

        with mock.patch("src.apply_us_filter.config") as mock_config, \
             mock.patch("src.apply_us_filter.data_from_parquet") as mock_data_from_parquet, \
             mock.patch("src.apply_us_filter.apply_us_model") as mock_apply:

            mock_config.PROJECT_ROOT = tmp_path
            mock_config.US_FILTER_SCORES_DIR = output_dir

            # Create corpus with string ids
            corpus_df = pl.DataFrame({
                "id": ["api_abc", "api_def"],
                "headline_with_lead": ["text1", "text2"],
                "year": [1980, 1980],
            })
            mock_data_from_parquet.return_value = corpus_df
            mock_apply.return_value = np.array([0.3, 0.7])

            main(threshold=0.5)

            output = pl.read_parquet(output_dir / "1980.parquet")
            assert output["id"].dtype == pl.Utf8
            assert output["id"].to_list() == ["api_abc", "api_def"]

    def test_main_us_score_range_01(self, tmp_path, setup_model_artifact):
        """main() output us_score is in [0, 1]."""
        weights_path, backbone = setup_model_artifact
        output_dir = tmp_path / "us_filter_scores"

        with mock.patch("src.apply_us_filter.config") as mock_config, \
             mock.patch("src.apply_us_filter.data_from_parquet") as mock_data_from_parquet, \
             mock.patch("src.apply_us_filter.apply_us_model") as mock_apply:

            mock_config.PROJECT_ROOT = tmp_path
            mock_config.US_FILTER_SCORES_DIR = output_dir

            corpus_df = pl.DataFrame({
                "id": ["1", "2", "3"],
                "headline_with_lead": ["a", "b", "c"],
                "year": [1980, 1980, 1980],
            })
            mock_data_from_parquet.return_value = corpus_df
            mock_apply.return_value = np.array([0.0, 0.5, 1.0])

            main(threshold=0.5)

            output = pl.read_parquet(output_dir / "1980.parquet")
            scores = output["us_score"].to_numpy()
            assert (scores >= 0.0).all() and (scores <= 1.0).all()

    def test_main_threshold_semantics(self, tmp_path, setup_model_artifact):
        """main() applies threshold: us = (us_score >= threshold)."""
        weights_path, backbone = setup_model_artifact
        output_dir = tmp_path / "us_filter_scores"

        with mock.patch("src.apply_us_filter.config") as mock_config, \
             mock.patch("src.apply_us_filter.data_from_parquet") as mock_data_from_parquet, \
             mock.patch("src.apply_us_filter.apply_us_model") as mock_apply:

            mock_config.PROJECT_ROOT = tmp_path
            mock_config.US_FILTER_SCORES_DIR = output_dir

            corpus_df = pl.DataFrame({
                "id": ["1", "2", "3", "4"],
                "headline_with_lead": ["a", "b", "c", "d"],
                "year": [1980, 1980, 1980, 1980],
            })
            mock_data_from_parquet.return_value = corpus_df
            mock_apply.return_value = np.array([0.3, 0.5, 0.7, 0.9])

            main(threshold=0.5)

            output = pl.read_parquet(output_dir / "1980.parquet")
            # Verify us = (us_score >= 0.5)
            expected_us = output["us_score"] >= 0.5
            assert (output["us"] == expected_us).all()

    def test_main_row_count_preserved(self, tmp_path, setup_model_artifact):
        """main() output row count matches input."""
        weights_path, backbone = setup_model_artifact
        output_dir = tmp_path / "us_filter_scores"

        with mock.patch("src.apply_us_filter.config") as mock_config, \
             mock.patch("src.apply_us_filter.data_from_parquet") as mock_data_from_parquet, \
             mock.patch("src.apply_us_filter.apply_us_model") as mock_apply:

            mock_config.PROJECT_ROOT = tmp_path
            mock_config.US_FILTER_SCORES_DIR = output_dir

            # Create 10 rows across two years
            corpus_df = pl.DataFrame({
                "id": [f"id_{i}" for i in range(10)],
                "headline_with_lead": [f"text_{i}" for i in range(10)],
                "year": [1980] * 6 + [1981] * 4,
            })
            mock_data_from_parquet.return_value = corpus_df
            mock_apply.return_value = np.linspace(0.1, 0.9, 10)

            main(threshold=0.5)

            year_1980 = pl.read_parquet(output_dir / "1980.parquet")
            year_1981 = pl.read_parquet(output_dir / "1981.parquet")

            total_output = year_1980.shape[0] + year_1981.shape[0]
            assert total_output == 10

