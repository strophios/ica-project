"""
Tests for `src.preproc.preprocessor.ClassifierPreprocessor`.

Structured to act as an executable spec for the multi-head preprocessor
contract introduced in Tier 2 Piece 3:

  - TestConstruction: constructor accepts the expected parameters and
    stores them correctly.
  - TestSingleHead: the degenerate one-entry-dict case behaves the same
    way the old single-`label_key` preprocessor did, modulo dtype.
  - TestMultiHead: multi-entry dicts produce all expected keys, route
    source columns to output keys correctly.
  - TestModeShape: endpoint-mode returns a single dict; standard-mode
    returns a (features, targets) tuple. Both modes carry the same
    targets routing.
  - TestDtype: targets are cast to `target_dtype` (default float32),
    regardless of source-column dtype (bool, int, float). Token ids
    and padding mask retain their tokenizer-native dtypes.

Tests inject a single shared tokenizer via a module-scoped fixture to
avoid re-downloading the RoBERTa preset for each test.
"""

import numpy as np
import pytest

import keras  # noqa: F401  (initializes backend before keras_hub import)
import keras_hub
import tensorflow as tf

from src.preproc.preprocessor import ClassifierPreprocessor


SEQ_LENGTH = 16  # deliberately small for fast tests
BATCH_SIZE = 4


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    """Shared RoBERTa tokenizer; loaded once per test module."""
    return keras_hub.tokenizers.RobertaTokenizer.from_preset("roberta_base_en")


@pytest.fixture
def text_batch():
    """A small batch of strings for tokenization."""
    return tf.constant([
        "Demonstrators marched on the capitol.",
        "The senate debated the bill.",
        "Immigration officials announced new rules.",
        "A federal judge issued an injunction.",
    ])


@pytest.fixture
def labels_batch():
    """Two label columns with different source dtypes to exercise casting."""
    return {
        "cca_label": tf.constant([1, 0, 1, 0], dtype=tf.int64),
        "immig_label": tf.constant([False, False, True, False], dtype=tf.bool),
    }


@pytest.fixture
def inputs(text_batch, labels_batch):
    return {"text": text_batch, **labels_batch}


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------

class TestConstruction:
    def test_stores_config(self, tokenizer):
        pp = ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys={"cca_targets": "cca_label"},
            tokenizer=tokenizer,
            endpoint_model=True,
            target_dtype="float32",
        )
        assert pp.SEQ_LENGTH == SEQ_LENGTH
        assert pp.text_key == "text"
        assert pp.label_keys == {"cca_targets": "cca_label"}
        assert pp.endpoint_model is True
        assert pp.target_dtype == "float32"
        assert pp.tokenizer is tokenizer

    def test_default_target_dtype_is_float32(self, tokenizer):
        pp = ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys={"cca_targets": "cca_label"},
            tokenizer=tokenizer,
        )
        assert pp.target_dtype == "float32"

    def test_default_endpoint_model_is_false(self, tokenizer):
        pp = ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys={"cca_targets": "cca_label"},
            tokenizer=tokenizer,
        )
        assert pp.endpoint_model is False


# -----------------------------------------------------------------------------
# Single-head (degenerate one-entry dict)
# -----------------------------------------------------------------------------

class TestSingleHead:
    def test_endpoint_mode_returns_single_dict(self, tokenizer, inputs):
        pp = ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys={"cca_targets": "cca_label"},
            tokenizer=tokenizer,
            endpoint_model=True,
        )
        out = pp(inputs)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"token_ids", "padding_mask", "cca_targets"}
        assert out["token_ids"].shape == (BATCH_SIZE, SEQ_LENGTH)
        assert out["padding_mask"].shape == (BATCH_SIZE, SEQ_LENGTH)
        assert out["cca_targets"].shape == (BATCH_SIZE,)

    def test_standard_mode_returns_features_targets_tuple(self, tokenizer, inputs):
        pp = ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys={"cca_targets": "cca_label"},
            tokenizer=tokenizer,
            endpoint_model=False,
        )
        out = pp(inputs)
        assert isinstance(out, tuple)
        assert len(out) == 2
        features, targets = out
        assert set(features.keys()) == {"token_ids", "padding_mask"}
        assert set(targets.keys()) == {"cca_targets"}


# -----------------------------------------------------------------------------
# Multi-head
# -----------------------------------------------------------------------------

class TestMultiHead:
    @pytest.fixture
    def label_keys(self):
        return {
            "cca_targets": "cca_label",
            "immig_targets": "immig_label",
        }

    def test_endpoint_mode_includes_all_targets(self, tokenizer, inputs, label_keys):
        pp = ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys=label_keys,
            tokenizer=tokenizer,
            endpoint_model=True,
        )
        out = pp(inputs)
        assert set(out.keys()) == {
            "token_ids", "padding_mask", "cca_targets", "immig_targets",
        }
        assert out["cca_targets"].shape == (BATCH_SIZE,)
        assert out["immig_targets"].shape == (BATCH_SIZE,)

    def test_standard_mode_includes_all_targets(self, tokenizer, inputs, label_keys):
        pp = ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys=label_keys,
            tokenizer=tokenizer,
            endpoint_model=False,
        )
        features, targets = pp(inputs)
        assert set(features.keys()) == {"token_ids", "padding_mask"}
        assert set(targets.keys()) == {"cca_targets", "immig_targets"}

    def test_routing_source_to_output_keys(self, tokenizer, inputs, label_keys):
        """Verify that output key X carries the values from source column Y
        as configured in label_keys (not some other column by accident)."""
        pp = ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys=label_keys,
            tokenizer=tokenizer,
            endpoint_model=True,
        )
        out = pp(inputs)
        # cca_label is [1, 0, 1, 0]; immig_label is [F, F, T, F] -> [0,0,1,0]
        np.testing.assert_array_equal(
            np.asarray(out["cca_targets"]), [1.0, 0.0, 1.0, 0.0]
        )
        np.testing.assert_array_equal(
            np.asarray(out["immig_targets"]), [0.0, 0.0, 1.0, 0.0]
        )

    def test_swapped_routing_swaps_values(self, tokenizer, inputs):
        """Sanity check: swapping the source-column mapping swaps the
        output values, confirming routing is actually data-driven."""
        pp = ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            # output names same, but pointing at swapped source columns
            label_keys={
                "cca_targets": "immig_label",
                "immig_targets": "cca_label",
            },
            tokenizer=tokenizer,
            endpoint_model=True,
        )
        out = pp(inputs)
        np.testing.assert_array_equal(
            np.asarray(out["cca_targets"]), [0.0, 0.0, 1.0, 0.0]
        )
        np.testing.assert_array_equal(
            np.asarray(out["immig_targets"]), [1.0, 0.0, 1.0, 0.0]
        )


# -----------------------------------------------------------------------------
# Dtype contract
# -----------------------------------------------------------------------------

class TestDtype:
    def test_targets_cast_to_default_float32(self, tokenizer, inputs):
        pp = ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys={
                "cca_targets": "cca_label",     # source: int64
                "immig_targets": "immig_label", # source: bool
            },
            tokenizer=tokenizer,
            endpoint_model=True,
        )
        out = pp(inputs)
        assert out["cca_targets"].dtype == tf.float32
        assert out["immig_targets"].dtype == tf.float32

    def test_targets_cast_to_explicit_dtype(self, tokenizer, inputs):
        pp = ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys={"cca_targets": "cca_label"},
            tokenizer=tokenizer,
            endpoint_model=True,
            target_dtype="float16",
        )
        out = pp(inputs)
        assert out["cca_targets"].dtype == tf.float16

    def test_token_ids_retain_tokenizer_dtype(self, tokenizer, inputs):
        """target_dtype should not affect token_ids / padding_mask dtype."""
        pp = ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys={"cca_targets": "cca_label"},
            tokenizer=tokenizer,
            endpoint_model=True,
            target_dtype="float64",  # deliberately weird to make a change visible
        )
        out = pp(inputs)
        # token_ids should be integer-typed; we don't pin the exact int width
        # because keras_hub's packer dtype is implementation detail.
        assert out["token_ids"].dtype.is_integer
