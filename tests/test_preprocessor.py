"""
Tests for `src.preproc.preprocessor.ClassifierPreprocessor`.

Structured to act as an executable spec for the multi-head preprocessor
contract introduced in Tier 2 Piece 3 and the dual-boundary validation
introduced in Tier 3 Piece 1:

  - TestConstruction: constructor accepts the expected parameters and
    stores them correctly.
  - TestConstructionValidation (Tier 3 Piece 1): constructor *rejects*
    invalid configurations at __init__ time. Internal-config-validity
    bugs are caught here.
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
  - TestCallTimeInputValidation (Tier 3 Piece 1): __call__ rejects
    input batches missing the configured text_key or label_keys
    source columns. Config-vs-data-mismatch bugs are caught here.

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
# Construction-time validation (Tier 3 Piece 1)
# -----------------------------------------------------------------------------
# `__init__` validates the configuration for *internal-config-validity*
# bugs — bugs that are visible from the constructor arguments alone,
# without needing to see actual data. See `docs/notes/tier3-design.md`
# Piece 1 "Decision" / "Reasoning" / "Contracts" sections for the
# boundary-by-boundary validation framing.

class TestConstructionValidation:
    """Construction-time validation: configuration that is internally
    inconsistent or malformed should be rejected at __init__ time with
    `ValueError` and an informative message."""

    # --- text_key non-empty string -----------------------------------

    def test_text_key_none_rejected(self, tokenizer):
        with pytest.raises(ValueError, match="text_key"):
            ClassifierPreprocessor(
                SEQ_LENGTH=SEQ_LENGTH,
                text_key=None,
                label_keys={"cca_targets": "cca_label"},
                tokenizer=tokenizer,
            )

    def test_text_key_empty_string_rejected(self, tokenizer):
        with pytest.raises(ValueError, match="text_key"):
            ClassifierPreprocessor(
                SEQ_LENGTH=SEQ_LENGTH,
                text_key="",
                label_keys={"cca_targets": "cca_label"},
                tokenizer=tokenizer,
            )

    def test_text_key_non_string_rejected(self, tokenizer):
        with pytest.raises(ValueError, match="text_key"):
            ClassifierPreprocessor(
                SEQ_LENGTH=SEQ_LENGTH,
                text_key=42,
                label_keys={"cca_targets": "cca_label"},
                tokenizer=tokenizer,
            )

    # --- label_keys is a dict ----------------------------------------

    def test_label_keys_list_of_tuples_rejected(self, tokenizer):
        """The most plausible typo: hand-rolled list of (key, value)
        tuples instead of a dict."""
        with pytest.raises(ValueError, match="label_keys"):
            ClassifierPreprocessor(
                SEQ_LENGTH=SEQ_LENGTH,
                text_key="text",
                label_keys=[("cca_targets", "cca_label")],
                tokenizer=tokenizer,
            )

    def test_label_keys_none_rejected(self, tokenizer):
        with pytest.raises(ValueError, match="label_keys"):
            ClassifierPreprocessor(
                SEQ_LENGTH=SEQ_LENGTH,
                text_key="text",
                label_keys=None,
                tokenizer=tokenizer,
            )

    # --- target_dtype is a valid Keras dtype string (retires M2) -----

    def test_target_dtype_default_accepted(self, tokenizer):
        """Default target_dtype='float32' should not be rejected
        (sanity check that the validation isn't over-eager)."""
        ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys={"cca_targets": "cca_label"},
            tokenizer=tokenizer,
        )

    def test_target_dtype_invalid_string_rejected(self, tokenizer):
        with pytest.raises(ValueError, match="target_dtype"):
            ClassifierPreprocessor(
                SEQ_LENGTH=SEQ_LENGTH,
                text_key="text",
                label_keys={"cca_targets": "cca_label"},
                tokenizer=tokenizer,
                target_dtype="flot32",  # typo
            )

    # --- endpoint_model=False requires non-empty label_keys ---------
    # Standard mode emits (features, targets_dict); empty targets_dict
    # has nothing to route via compile(loss={...}) and is structurally
    # nonsensical. Empty label_keys is *only* valid in endpoint mode.

    def test_standard_mode_empty_label_keys_rejected(self, tokenizer):
        with pytest.raises(ValueError, match="label_keys"):
            ClassifierPreprocessor(
                SEQ_LENGTH=SEQ_LENGTH,
                text_key="text",
                label_keys={},
                tokenizer=tokenizer,
                endpoint_model=False,
            )

    # --- endpoint_model=True allows empty label_keys (predict-only) --
    # This pins the contract that the eval script's predict-only
    # configuration (label_keys={}, endpoint_model=True) remains
    # valid. **Important**: without this test, a future "tighten
    # validation" rewrite could regress the eval script silently.
    # When pinned question #3 lands (the preprocessor API refactor),
    # this test will flip to a rejection test; until then, it pins
    # the current contract.

    def test_endpoint_mode_empty_label_keys_accepted(self, tokenizer):
        """The eval script's predict-only configuration. Pin the
        contract — without this test, a tightening regress is silent."""
        ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys={},
            tokenizer=tokenizer,
            endpoint_model=True,
        )

    def test_endpoint_mode_with_label_keys_accepted(self, tokenizer):
        """The standard training configuration. Sanity-check the new
        validation doesn't accidentally reject the primary use case."""
        ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys={"cca_targets": "cca_label"},
            tokenizer=tokenizer,
            endpoint_model=True,
        )


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


# -----------------------------------------------------------------------------
# Call-time input validation (Tier 3 Piece 1)
# -----------------------------------------------------------------------------
# `__call__` validates the *input batch* against the preprocessor's
# configuration — config-vs-data-mismatch bugs that __init__ can't see
# (it has only Python strings, not the dataset's column set). See
# `docs/notes/tier3-design.md` Piece 1 "Reasoning" section for the
# why-both-boundaries argument.

class TestCallTimeInputValidation:
    """Call-time validation: input dicts missing the configured
    text_key or label_keys source columns should raise `KeyError` at
    the entry of __call__ (not deep inside the tokenizer call) with an
    informative message naming the missing columns, the configured
    expectation, and the keys that were present."""

    def test_missing_text_key_raises(self, tokenizer, labels_batch):
        pp = ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys={"cca_targets": "cca_label"},
            tokenizer=tokenizer,
            endpoint_model=True,
        )
        # Inputs dict is missing the text column entirely.
        bad_inputs = {**labels_batch}  # only label cols, no "text"
        with pytest.raises(KeyError) as exc:
            pp(bad_inputs)
        msg = str(exc.value)
        # Message names the missing column, the configured text_key, and
        # the available keys — the user needs all three to debug.
        assert "text" in msg
        assert "cca_label" in msg or "label_keys" in msg or "configured" in msg.lower()

    def test_missing_label_source_column_raises(self, tokenizer, text_batch):
        pp = ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys={"cca_targets": "cca_label"},
            tokenizer=tokenizer,
            endpoint_model=True,
        )
        # Inputs has the text column but not the label source column.
        bad_inputs = {"text": text_batch}
        with pytest.raises(KeyError) as exc:
            pp(bad_inputs)
        msg = str(exc.value)
        assert "cca_label" in msg

    def test_multiple_missing_columns_enumerated(self, tokenizer):
        """Pin the enumerate-all-missing contract: when both text_key
        AND a label source column are missing, the error names *both*
        rather than failing fast on the first."""
        pp = ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys={
                "cca_targets": "cca_label",
                "immig_targets": "immig_label",
            },
            tokenizer=tokenizer,
            endpoint_model=True,
        )
        # Empty inputs — every configured column is missing.
        bad_inputs = {"unrelated_column": tf.constant([1, 2, 3])}
        with pytest.raises(KeyError) as exc:
            pp(bad_inputs)
        msg = str(exc.value)
        # All three configured-but-missing columns should appear in
        # the message.
        assert "text" in msg
        assert "cca_label" in msg
        assert "immig_label" in msg

    def test_predict_mode_without_label_source_columns(self, tokenizer, text_batch):
        """Predict-only preprocessor (endpoint_model=True, label_keys={})
        should successfully process an input batch containing only the
        text column — no label source columns required. Pairs with
        `TestEndpointModeAllowsEmptyLabelKeys` from
        TestConstructionValidation to fully cover the predict-only flow."""
        pp = ClassifierPreprocessor(
            SEQ_LENGTH=SEQ_LENGTH,
            text_key="text",
            label_keys={},
            tokenizer=tokenizer,
            endpoint_model=True,
        )
        out = pp({"text": text_batch})
        # Should produce just the model inputs, no targets.
        assert isinstance(out, dict)
        assert set(out.keys()) == {"token_ids", "padding_mask"}
