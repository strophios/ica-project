"""Tests for src.run_relevance_text (text-mode rel trainer with encoder unfreeze).

Covers:
  - with_rel_label: pure PNU-label attachment (1.0/-1.0/0.0 per group)
  - _default_rel_text_config: head rename + canonical prior/eta + escalation defaults
  - 3-way weighted TEXT-stream composition via dataset_create (cardinalities,
    batch shapes, label routing) -- the never-before-exercised N-way weighted
    TEXT-stream path this module is the first caller of.
  - Importing does not trigger training.
"""

from __future__ import annotations

import dataclasses

import keras_hub
import polars as pl
import pytest
import tensorflow as tf

import keras

from src.data_setup.data import dataset_create
from src.preproc.preprocessor import ClassifierPreprocessor
from src.run_relevance_text import (
    DEFAULT_REL_TEXT_CONFIG,
    RELEVANCE_DIR,
    _default_rel_text_config,
    build_fit_callbacks,
    resolve_tensorboard_dir,
    with_rel_label,
)


# ---------------------------------------------------------------------------
# with_rel_label (pure)
# ---------------------------------------------------------------------------
class TestWithRelLabel:
    def test_pos_group_gets_label_1(self):
        df = pl.DataFrame({"id": ["a", "b"]})
        out = with_rel_label(df, "pos")
        assert out["rel_label"].to_list() == [1.0, 1.0]

    def test_neg_group_gets_label_minus_1(self):
        df = pl.DataFrame({"id": ["a"]})
        out = with_rel_label(df, "neg")
        assert out["rel_label"].to_list() == [-1.0]

    def test_unl_group_gets_label_0(self):
        df = pl.DataFrame({"id": ["a", "b", "c"]})
        out = with_rel_label(df, "unl")
        assert out["rel_label"].to_list() == [0.0, 0.0, 0.0]

    def test_dtype_is_float32(self):
        df = pl.DataFrame({"id": ["a"]})
        out = with_rel_label(df, "pos")
        assert out["rel_label"].dtype == pl.Float32

    def test_unknown_group_raises_keyerror(self):
        df = pl.DataFrame({"id": ["a"]})
        with pytest.raises(KeyError):
            with_rel_label(df, "bogus")

    def test_preserves_other_columns(self):
        df = pl.DataFrame({"id": ["a"], "headline_with_lead": ["h</s>l"]})
        out = with_rel_label(df, "pos")
        assert out["headline_with_lead"].to_list() == ["h</s>l"]

    def test_zero_row_dataframe(self):
        df = pl.DataFrame({"id": []}, schema={"id": pl.Utf8})
        out = with_rel_label(df, "unl")
        assert out.height == 0
        assert "rel_label" in out.columns


# ---------------------------------------------------------------------------
# _default_rel_text_config
# ---------------------------------------------------------------------------
class TestDefaultRelTextConfig:
    def test_head_renamed_to_rel(self):
        cfg = _default_rel_text_config()
        assert cfg.heads[0].name == "rel"

    def test_source_column_is_rel_label_not_cca_label(self):
        """The synthetic PNU-convention column, distinct from the text
        table's raw `cca_label` indicator column."""
        cfg = _default_rel_text_config()
        assert cfg.heads[0].source_column == "rel_label"

    def test_canonical_prior_and_eta(self):
        cfg = _default_rel_text_config()
        assert cfg.heads[0].loss.prior == 0.05
        assert cfg.heads[0].loss.nnpnu_eta == 0.0

    def test_label_keys_routes_rel_targets_to_rel_label(self):
        cfg = _default_rel_text_config()
        assert cfg.label_keys == {"rel_targets": "rel_label"}

    def test_escalation_defaults_are_frozen_probe(self):
        cfg = _default_rel_text_config()
        assert cfg.freeze_encoder is True
        assert cfg.unfreeze_top_n == 0
        assert cfg.layer_multipliers is None

    def test_epochs_override(self):
        cfg = _default_rel_text_config(epochs=1)
        assert cfg.epochs == 1

    def test_module_default_matches_default_epochs_7(self):
        assert DEFAULT_REL_TEXT_CONFIG.epochs == 7
        assert DEFAULT_REL_TEXT_CONFIG.heads[0].name == "rel"

    def test_unfreeze_variant_via_replace(self):
        """The documented escalation path: dataclasses.replace to turn on
        unfreezing (mirrors what the __main__ CLI and the smoke script do)."""
        cfg = dataclasses.replace(
            DEFAULT_REL_TEXT_CONFIG, unfreeze_top_n=1, freeze_encoder=False
        )
        assert cfg.unfreeze_top_n == 1
        assert cfg.freeze_encoder is False
        # Unrelated fields unaffected.
        assert cfg.heads[0].name == "rel"
        assert cfg.heads[0].loss.prior == 0.05


# ---------------------------------------------------------------------------
# resolve_tensorboard_dir (pure)
# ---------------------------------------------------------------------------
class TestResolveTensorboardDir:
    def test_off_by_default(self):
        assert resolve_tensorboard_dir(None, False, "20260101T000000Z") is None

    def test_explicit_dir_wins_when_tensorboard_flag_also_set(self):
        result = resolve_tensorboard_dir("/some/explicit/dir", True, "20260101T000000Z")
        assert result == "/some/explicit/dir"

    def test_explicit_dir_wins_when_tensorboard_flag_off(self):
        result = resolve_tensorboard_dir("/some/explicit/dir", False, "20260101T000000Z")
        assert result == "/some/explicit/dir"

    def test_tensorboard_flag_defaults_timestamped_path_under_relevance_dir(self):
        result = resolve_tensorboard_dir(None, True, "20260101T000000Z")
        assert result == str(RELEVANCE_DIR / "tb_logs" / "20260101T000000Z")

    def test_does_not_compute_its_own_timestamp(self):
        """Pure: passing a different timestamp changes the output deterministically
        (no hidden datetime.now() call inside the function)."""
        first = resolve_tensorboard_dir(None, True, "aaa")
        second = resolve_tensorboard_dir(None, True, "bbb")
        assert first != second
        assert first.endswith("aaa")
        assert second.endswith("bbb")


# ---------------------------------------------------------------------------
# build_fit_callbacks
# ---------------------------------------------------------------------------
class TestBuildFitCallbacks:
    def test_tensorboard_absent_when_dir_is_none(self, tmp_path):
        callbacks_list = build_fit_callbacks(tmp_path / "metrics.csv", tensorboard_log_dir=None)
        assert not any(isinstance(cb, keras.callbacks.TensorBoard) for cb in callbacks_list)

    def test_tensorboard_present_when_dir_given(self, tmp_path):
        callbacks_list = build_fit_callbacks(
            tmp_path / "metrics.csv", tensorboard_log_dir=str(tmp_path / "tb")
        )
        tb_callbacks = [cb for cb in callbacks_list if isinstance(cb, keras.callbacks.TensorBoard)]
        assert len(tb_callbacks) == 1
        assert tb_callbacks[0].log_dir == str(tmp_path / "tb")

    def test_csv_logger_and_early_stopping_always_present(self, tmp_path):
        callbacks_list = build_fit_callbacks(tmp_path / "metrics.csv", tensorboard_log_dir=None)
        assert any(isinstance(cb, keras.callbacks.CSVLogger) for cb in callbacks_list)
        assert any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in callbacks_list)

    def test_returns_list(self, tmp_path):
        callbacks_list = build_fit_callbacks(tmp_path / "metrics.csv", tensorboard_log_dir=None)
        assert isinstance(callbacks_list, list)


# ---------------------------------------------------------------------------
# 3-way weighted TEXT-stream composition
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tokenizer():
    """Shared RoBERTa tokenizer; loaded once per test module (mirrors
    tests/test_preprocessor.py's fixture)."""
    return keras_hub.tokenizers.RobertaTokenizer.from_preset("roberta_base_en")


def _group_dataset(texts, group_name):
    df = with_rel_label(pl.DataFrame({"headline_with_lead": texts}), group_name)
    return tf.data.Dataset.from_tensor_slices({
        "headline_with_lead": df["headline_with_lead"].to_list(),
        "rel_label": df["rel_label"].to_numpy(),
    })


class TestThreeStreamTextComposition:
    SEQ_LEN = 8
    BATCH = 8
    N_BATCHES_TO_SAMPLE = 40

    def _build_streams(self):
        pos_ds = _group_dataset([f"pos headline {i}</s>pos lede {i}" for i in range(20)], "pos")
        neg_ds = _group_dataset([f"neg headline {i}</s>neg lede {i}" for i in range(20)], "neg")
        unl_ds = _group_dataset([f"unl headline {i}</s>unl lede {i}" for i in range(20)], "unl")
        return pos_ds, neg_ds, unl_ds

    def test_composed_dataset_yields_correct_batch_shapes(self, tokenizer):
        preprocess = ClassifierPreprocessor(
            SEQ_LENGTH=self.SEQ_LEN,
            text_key="headline_with_lead",
            label_keys={"rel_targets": "rel_label"},
            tokenizer=tokenizer,
            endpoint_model=True,
        )
        pos_ds, neg_ds, unl_ds = self._build_streams()
        composed = dataset_create(
            shuffle_buffer=16, batch_size=self.BATCH, preprocessor=preprocess,
            data=[pos_ds, neg_ds, unl_ds], weights=[0.5, 0.3, 0.2],
        )
        batch = next(iter(composed.take(1)))
        assert batch["token_ids"].shape == (self.BATCH, self.SEQ_LEN)
        assert batch["padding_mask"].shape == (self.BATCH, self.SEQ_LEN)
        assert batch["rel_targets"].shape == (self.BATCH,)

    def test_label_routing_only_pnu_values_appear(self, tokenizer):
        """Every rel_targets value must be exactly one of {1.0, -1.0, 0.0} --
        the composed stream must not mix up which group contributed which
        label (the specific failure mode this test guards against)."""
        preprocess = ClassifierPreprocessor(
            SEQ_LENGTH=self.SEQ_LEN,
            text_key="headline_with_lead",
            label_keys={"rel_targets": "rel_label"},
            tokenizer=tokenizer,
            endpoint_model=True,
        )
        pos_ds, neg_ds, unl_ds = self._build_streams()
        composed = dataset_create(
            shuffle_buffer=16, batch_size=self.BATCH, preprocessor=preprocess,
            data=[pos_ds, neg_ds, unl_ds], weights=[0.5, 0.3, 0.2],
        )
        seen = set()
        for batch in composed.take(self.N_BATCHES_TO_SAMPLE):
            seen.update(batch["rel_targets"].numpy().tolist())
        assert seen <= {1.0, -1.0, 0.0}

    def test_all_three_groups_represented_over_many_batches(self, tokenizer):
        """With nonzero weight on every group, sampling enough batches should
        surface all three label values (statistically near-certain at 40
        batches x 8 = 320 draws against weights [0.5, 0.3, 0.2])."""
        preprocess = ClassifierPreprocessor(
            SEQ_LENGTH=self.SEQ_LEN,
            text_key="headline_with_lead",
            label_keys={"rel_targets": "rel_label"},
            tokenizer=tokenizer,
            endpoint_model=True,
        )
        pos_ds, neg_ds, unl_ds = self._build_streams()
        composed = dataset_create(
            shuffle_buffer=16, batch_size=self.BATCH, preprocessor=preprocess,
            data=[pos_ds, neg_ds, unl_ds], weights=[0.5, 0.3, 0.2],
        )
        seen = set()
        for batch in composed.take(self.N_BATCHES_TO_SAMPLE):
            seen.update(batch["rel_targets"].numpy().tolist())
        assert seen == {1.0, -1.0, 0.0}

    def test_composed_dataset_is_infinite(self, tokenizer):
        """dataset_create's repeat().batch(drop_remainder=True) path (the
        steps_per_epoch-driven training pipeline) yields an infinite dataset."""
        preprocess = ClassifierPreprocessor(
            SEQ_LENGTH=self.SEQ_LEN,
            text_key="headline_with_lead",
            label_keys={"rel_targets": "rel_label"},
            tokenizer=tokenizer,
            endpoint_model=True,
        )
        pos_ds, neg_ds, unl_ds = self._build_streams()
        composed = dataset_create(
            shuffle_buffer=16, batch_size=self.BATCH, preprocessor=preprocess,
            data=[pos_ds, neg_ds, unl_ds], weights=[0.5, 0.3, 0.2],
        )
        assert composed.cardinality().numpy() == tf.data.INFINITE_CARDINALITY


# ---------------------------------------------------------------------------
# Import side effects
# ---------------------------------------------------------------------------
class TestImportNoSideEffects:
    def test_import_does_not_trigger_training(self):
        import src.run_relevance_text

        assert src.run_relevance_text is not None
