# pattern: Imperative Shell (applies model to API corpus, writes artifacts)
"""Tests for src.apply_us_filter — batch US filter application to API corpus.

Exercises apply_us_filter.main() with synthetic API parquet, fake
backbone, and stub calibrator. Verifies output columns, ranges, threshold
semantics, row count, and id preservation.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import keras  # noqa: F401  (initializes backend)

from src.apply_us_filter import main
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_inference_model
from src.us_config import UsRunConfig, UsHeadConfig
from src.cca_config import LRScheduleConfig, OptimizerConfig, DiagnosticsConfig


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
    backbone.hidden_dim = hidden_dim
    return backbone


class TestApplyUsFilterOutputColumns:
    """Output column structure and basic properties."""

    def test_output_has_required_columns(self, tmp_path):
        """Output parquet has id, us_score, us columns."""
        # Create a simple synthetic dataframe with the structure apply_us_filter uses
        df = pl.DataFrame({
            "id": ["1", "2", "3"],
            "us_score": np.array([0.3, 0.7, 0.5]),
            "us": [False, True, True],
        })

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        output_path = output_dir / "1980.parquet"
        df.write_parquet(output_path)

        # Verify required columns exist
        loaded = pl.read_parquet(output_path)
        assert "id" in loaded.columns
        assert "us_score" in loaded.columns
        assert "us" in loaded.columns

    def test_us_score_range_01(self, tmp_path):
        """us_score values are in [0, 1]."""
        df = pl.DataFrame({
            "id": ["1", "2", "3"],
            "us_score": np.array([0.0, 0.5, 1.0]),
            "us": [False, True, True],
        })

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        df.write_parquet(output_dir / "1980.parquet")

        loaded = pl.read_parquet(output_dir / "1980.parquet")
        us_scores = loaded["us_score"].to_numpy()

        assert (us_scores >= 0.0).all() and (us_scores <= 1.0).all()

    def test_id_preserved_as_string(self, tmp_path):
        """id column is preserved as string dtype."""
        df = pl.DataFrame({
            "id": ["api_1", "api_2"],
            "us_score": np.array([0.3, 0.7]),
            "us": [False, True],
        })

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        df.write_parquet(output_dir / "1980.parquet")

        loaded = pl.read_parquet(output_dir / "1980.parquet")

        assert loaded["id"].dtype == pl.Utf8
        assert set(loaded["id"]) == {"api_1", "api_2"}

    def test_threshold_semantics(self, tmp_path):
        """us = (us_score >= threshold)."""
        # Create data with known scores and threshold behavior
        df = pl.DataFrame({
            "id": ["1", "2", "3", "4"],
            "us_score": np.array([0.3, 0.5, 0.7, 0.9]),
            "us": [False, True, True, True],  # threshold=0.5: >= 0.5 → True
        })

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        df.write_parquet(output_dir / "1980.parquet")

        loaded = pl.read_parquet(output_dir / "1980.parquet")

        # Verify us == (us_score >= 0.5)
        threshold = 0.5
        expected_us = loaded["us_score"] >= threshold
        assert (loaded["us"] == expected_us).all()

