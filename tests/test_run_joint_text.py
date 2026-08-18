"""Tests for src.run_joint_text (the joint CCA+rel text-mode trainer,
docs/design-plans/2026-08-18-stage4-joint-finetune.md Components item 3)
and its split-helper counterpart in src.data_setup.data.

Covers:
  - create_joint_text_data (src.data_setup.data): whole-table-first split,
    then per-split cca_pos/rel_pos/unl grouping, holdout drop, overlap
    behavior, us-restriction on the unlabeled pool.
  - derive_rel_target: pure rel-target derivation (1.0/-1.0/0.0), precedence.
  - validate_lam: (0, 1) range validation.
  - _default_joint_text_config: two-head RunConfig assembly (loss_weight
    mixing, shared prior/eta/hidden_dim, source columns).
  - _guard_weights_out: refuses known production weights paths.
  - resolve_tensorboard_dir / build_fit_callbacks: byte-identical ports from
    run_relevance_text.py (duplicated per this project's per-script
    self-containment precedent).
  - 4-way weighted TEXT-stream composition (two label keys per batch) via
    dataset_create -- the joint trainer's variant of
    test_run_relevance_text.py's TestThreeStreamTextComposition.
  - Importing does not trigger training.
"""

from __future__ import annotations

import dataclasses

import keras
import keras_hub
import polars as pl
import pytest
import tensorflow as tf

from src.data_setup.data import create_joint_text_data
from src.preproc.preprocessor import ClassifierPreprocessor
from src.run_joint_text import (
    DEFAULT_JOINT_TEXT_CONFIG,
    RELEVANCE_DIR,
    _default_joint_text_config,
    _guard_weights_out,
    build_fit_callbacks,
    derive_rel_target,
    resolve_tensorboard_dir,
    validate_lam,
)


# ---------------------------------------------------------------------------
# derive_rel_target (pure)
# ---------------------------------------------------------------------------
class TestDeriveRelTarget:
    def test_rel_positive_gets_1(self):
        df = pl.DataFrame({"rel_label": [1], "reliable_neg": [False]})
        out = derive_rel_target(df)
        assert out["rel_target"].to_list() == [1.0]

    def test_reliable_neg_gets_minus_1(self):
        df = pl.DataFrame({"rel_label": [0], "reliable_neg": [True]})
        out = derive_rel_target(df)
        assert out["rel_target"].to_list() == [-1.0]

    def test_neither_gets_0(self):
        df = pl.DataFrame({"rel_label": [0], "reliable_neg": [False]})
        out = derive_rel_target(df)
        assert out["rel_target"].to_list() == [0.0]

    def test_rel_positive_takes_precedence_over_reliable_neg(self):
        """Upstream invariant: reliable negatives exclude candidates, so this
        combination shouldn't occur in practice, but precedence must still
        favor the positive signal if it ever does."""
        df = pl.DataFrame({"rel_label": [1], "reliable_neg": [True]})
        out = derive_rel_target(df)
        assert out["rel_target"].to_list() == [1.0]

    def test_dtype_is_float32(self):
        df = pl.DataFrame({"rel_label": [1, 0, 0], "reliable_neg": [False, True, False]})
        out = derive_rel_target(df)
        assert out["rel_target"].dtype == pl.Float32

    def test_mixed_batch(self):
        df = pl.DataFrame({
            "rel_label": [1, 0, 0, 0],
            "reliable_neg": [False, True, False, False],
        })
        out = derive_rel_target(df)
        assert out["rel_target"].to_list() == [1.0, -1.0, 0.0, 0.0]

    def test_preserves_other_columns(self):
        df = pl.DataFrame({
            "id": ["a"], "rel_label": [1], "reliable_neg": [False],
            "cca_label": [0],
        })
        out = derive_rel_target(df)
        assert out["id"].to_list() == ["a"]
        assert out["cca_label"].to_list() == [0]

    def test_zero_row_dataframe(self):
        df = pl.DataFrame(
            {"rel_label": [], "reliable_neg": []},
            schema={"rel_label": pl.Int8, "reliable_neg": pl.Boolean},
        )
        out = derive_rel_target(df)
        assert out.height == 0
        assert "rel_target" in out.columns


# ---------------------------------------------------------------------------
# create_joint_text_data (src.data_setup.data, pure)
# ---------------------------------------------------------------------------
def _joint_table(n_cca_pos=6, n_rel_pos=6, n_overlap=2, n_unl=40, n_not_us=10):
    """Synthetic joint-table fixture with distinct id prefixes per role.

    n_overlap ids are counted within BOTH n_cca_pos and n_rel_pos (i.e. total
    distinct positive-tagged ids = n_cca_pos + n_rel_pos - n_overlap).
    """
    rows = []

    def add(prefix, n, cca, rel, us, reliable_neg=False):
        for i in range(n):
            rows.append({
                "id": f"{prefix}_{i}",
                "cca_label": cca,
                "rel_label": rel,
                "us": us,
                "reliable_neg": reliable_neg,
            })

    add("overlap", n_overlap, 1, 1, True)
    add("cca_only", n_cca_pos - n_overlap, 1, 0, True)
    add("rel_only", n_rel_pos - n_overlap, 0, 1, True)
    add("unl", n_unl, 0, 0, True)
    add("notus", n_not_us, 0, 0, False)
    return pl.DataFrame(rows)


class TestCreateJointTextData:
    def test_id_appears_in_exactly_one_split(self):
        table = _joint_table()
        splits = create_joint_text_data(table, seed=200)
        seen = {}
        for split_name in ("train", "val", "test"):
            for group_name in ("cca_pos", "rel_pos", "unl"):
                for i in splits[split_name][group_name]["id"].to_list():
                    seen.setdefault(i, set()).add(split_name)
        for i, split_names in seen.items():
            assert len(split_names) == 1, f"id {i} appears in splits {split_names}"

    def test_no_id_lost_or_duplicated_across_splits(self):
        """Every id in the input table that belongs to SOME group (cca_pos,
        rel_pos, or a US-passing unl row) lands in exactly one of
        train/val/test (accounting for cca_pos/rel_pos overlap WITHIN a
        split, not across). Non-US, non-positive-for-either-head rows
        (`notus_*`) are legitimately absent from every group -- they fit no
        Ratio-Batch stream (not a positive for either head, and `unl`
        requires `us`) -- so they're excluded from the expected set here."""
        table = _joint_table()
        splits = create_joint_text_data(table, seed=200)
        per_split_ids = {}
        for split_name in ("train", "val", "test"):
            ids = set()
            for group_name in ("cca_pos", "rel_pos", "unl"):
                ids |= set(splits[split_name][group_name]["id"].to_list())
            per_split_ids[split_name] = ids
        all_seen = per_split_ids["train"] | per_split_ids["val"] | per_split_ids["test"]
        expected = {i for i in table["id"].to_list() if not i.startswith("notus_")}
        assert all_seen == expected
        # pairwise disjoint
        assert per_split_ids["train"] & per_split_ids["val"] == set()
        assert per_split_ids["train"] & per_split_ids["test"] == set()
        assert per_split_ids["val"] & per_split_ids["test"] == set()

    def test_cca_pos_and_rel_pos_overlap_within_a_split(self):
        """An id positive for both heads appears in BOTH group frames of
        whichever single split it landed in -- the deliberate oversampling
        overlap (design doc: 'Streams overlap where a row is positive for
        both heads')."""
        table = _joint_table(n_cca_pos=20, n_rel_pos=20, n_overlap=20, n_unl=5)
        splits = create_joint_text_data(table, seed=200)
        for split_name in ("train", "val", "test"):
            cca_ids = set(splits[split_name]["cca_pos"]["id"].to_list())
            rel_ids = set(splits[split_name]["rel_pos"]["id"].to_list())
            overlap_ids = {i for i in cca_ids if i.startswith("overlap_")}
            # every overlap id present in this split's cca_pos is also in rel_pos
            assert overlap_ids <= rel_ids

    def test_unlabeled_group_excludes_non_us_rows(self):
        table = _joint_table(n_not_us=10, n_unl=10)
        splits = create_joint_text_data(table, seed=200)
        for split_name in ("train", "val", "test"):
            unl_ids = splits[split_name]["unl"]["id"].to_list()
            assert not any(i.startswith("notus_") for i in unl_ids)

    def test_unlabeled_group_excludes_positives(self):
        table = _joint_table()
        splits = create_joint_text_data(table, seed=200)
        for split_name in ("train", "val", "test"):
            unl = splits[split_name]["unl"]
            assert (unl["cca_label"] == 0).all()
            assert (unl["rel_label"] == 0).all()

    def test_holdout_ids_dropped_entirely(self):
        table = _joint_table()
        holdout = ["overlap_0", "unl_0"]
        splits = create_joint_text_data(table, seed=200, holdout_ids=holdout)
        for split_name in ("train", "val", "test"):
            for group_name in ("cca_pos", "rel_pos", "unl"):
                ids = splits[split_name][group_name]["id"].to_list()
                assert not (set(ids) & set(holdout))

    def test_empty_holdout_is_noop(self):
        table = _joint_table()
        with_none = create_joint_text_data(table, seed=200, holdout_ids=None)
        with_empty = create_joint_text_data(table, seed=200, holdout_ids=[])
        assert (
            with_none["train"]["unl"]["id"].to_list()
            == with_empty["train"]["unl"]["id"].to_list()
        )

    def test_deterministic_given_seed(self):
        table = _joint_table()
        a = create_joint_text_data(table, seed=200)
        b = create_joint_text_data(table, seed=200)
        assert a["train"]["unl"]["id"].to_list() == b["train"]["unl"]["id"].to_list()

    def test_roughly_90_5_5_split_sizes(self):
        table = _joint_table(n_unl=200)
        splits = create_joint_text_data(table, seed=200)
        n_train = splits["train"]["unl"].height
        n_val = splits["val"]["unl"].height
        n_test = splits["test"]["unl"].height
        total = n_train + n_val + n_test
        assert total == 200
        assert n_train / total == pytest.approx(0.9, abs=0.05)

    def test_duplicate_id_raises(self):
        table = pl.concat([_joint_table(), _joint_table()])
        with pytest.raises(AssertionError):
            create_joint_text_data(table, seed=200)


# ---------------------------------------------------------------------------
# validate_lam (pure)
# ---------------------------------------------------------------------------
class TestValidateLam:
    def test_valid_lam_returned_unchanged(self):
        assert validate_lam(0.5) == 0.5

    def test_boundary_zero_raises(self):
        with pytest.raises(ValueError):
            validate_lam(0.0)

    def test_boundary_one_raises(self):
        with pytest.raises(ValueError):
            validate_lam(1.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            validate_lam(-0.1)

    def test_above_one_raises(self):
        with pytest.raises(ValueError):
            validate_lam(1.5)

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            validate_lam("0.5")

    def test_bool_raises(self):
        with pytest.raises(ValueError):
            validate_lam(True)

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            validate_lam(float("nan"))


# ---------------------------------------------------------------------------
# _default_joint_text_config
# ---------------------------------------------------------------------------
class TestDefaultJointTextConfig:
    def test_two_heads_named_cca_and_rel(self):
        cfg = _default_joint_text_config(0.5)
        assert cfg.head_names == ("cca", "rel")

    def test_lam_splits_loss_weight(self):
        cfg = _default_joint_text_config(0.3)
        cca_head, rel_head = cfg.heads
        assert cca_head.loss_weight == pytest.approx(0.7)
        assert rel_head.loss_weight == pytest.approx(0.3)

    def test_source_columns(self):
        cfg = _default_joint_text_config(0.5)
        cca_head, rel_head = cfg.heads
        assert cca_head.source_column == "cca_label"
        assert rel_head.source_column == "rel_target"

    def test_shared_prior_and_eta(self):
        cfg = _default_joint_text_config(0.5)
        for head in cfg.heads:
            assert head.loss.prior == pytest.approx(0.02)
            assert head.loss.nnpnu_eta == 0.0

    def test_shared_hidden_dim_768(self):
        cfg = _default_joint_text_config(0.5)
        for head in cfg.heads:
            assert head.hidden_dim == 768

    def test_label_keys_routes_both_targets(self):
        cfg = _default_joint_text_config(0.4)
        assert cfg.label_keys == {"cca_targets": "cca_label", "rel_targets": "rel_target"}

    def test_invalid_lam_raises(self):
        with pytest.raises(ValueError):
            _default_joint_text_config(1.5)

    def test_epochs_override(self):
        cfg = _default_joint_text_config(0.5, epochs=1)
        assert cfg.epochs == 1

    def test_module_default_matches_lam_half_epochs_7(self):
        assert DEFAULT_JOINT_TEXT_CONFIG.epochs == 7
        assert DEFAULT_JOINT_TEXT_CONFIG.head_names == ("cca", "rel")

    def test_escalation_defaults_are_frozen_probe(self):
        cfg = _default_joint_text_config(0.5)
        assert cfg.freeze_encoder is True
        assert cfg.unfreeze_top_n == 0

    def test_unfreeze_variant_via_replace(self):
        cfg = dataclasses.replace(
            DEFAULT_JOINT_TEXT_CONFIG, unfreeze_top_n=1, freeze_encoder=False
        )
        assert cfg.unfreeze_top_n == 1
        assert cfg.head_names == ("cca", "rel")


# ---------------------------------------------------------------------------
# _guard_weights_out
# ---------------------------------------------------------------------------
class TestGuardWeightsOut:
    def test_arbitrary_path_ok(self, tmp_path):
        _guard_weights_out(tmp_path / "joint_N1_lam050_s200.weights.h5")  # no raise

    def test_production_relevance_weights_refused(self):
        import src.config as config

        with pytest.raises(ValueError):
            _guard_weights_out(config.RELEVANCE_DOCA_WEIGHTS)

    def test_production_relevance_text_weights_refused(self):
        import src.config as config

        with pytest.raises(ValueError):
            _guard_weights_out(config.RELEVANCE_TEXT_WEIGHTS)

    def test_production_cca_doca_weights_refused(self):
        import src.config as config

        with pytest.raises(ValueError):
            _guard_weights_out(config.CCA_DOCA_WEIGHTS)

    def test_production_us_filter_weights_refused(self):
        import src.config as config

        with pytest.raises(ValueError):
            _guard_weights_out(config.US_FILTER_FULL_WEIGHTS)

    def test_string_path_also_checked(self):
        import src.config as config

        with pytest.raises(ValueError):
            _guard_weights_out(str(config.CCA_DOCA_WEIGHTS))


# ---------------------------------------------------------------------------
# resolve_tensorboard_dir / build_fit_callbacks (ports from run_relevance_text)
# ---------------------------------------------------------------------------
class TestResolveTensorboardDir:
    def test_off_by_default(self):
        assert resolve_tensorboard_dir(None, False, "20260101T000000Z") is None

    def test_explicit_dir_wins(self):
        result = resolve_tensorboard_dir("/some/dir", True, "20260101T000000Z")
        assert result == "/some/dir"

    def test_tensorboard_flag_defaults_timestamped_path_under_relevance_dir(self):
        result = resolve_tensorboard_dir(None, True, "20260101T000000Z")
        assert result == str(RELEVANCE_DIR / "tb_logs" / "20260101T000000Z")


class TestBuildFitCallbacks:
    def test_tensorboard_absent_when_dir_is_none(self, tmp_path):
        callbacks_list = build_fit_callbacks(tmp_path / "metrics.csv", tensorboard_log_dir=None)
        assert not any(isinstance(cb, keras.callbacks.TensorBoard) for cb in callbacks_list)

    def test_tensorboard_present_when_dir_given(self, tmp_path):
        callbacks_list = build_fit_callbacks(
            tmp_path / "metrics.csv", tensorboard_log_dir=str(tmp_path / "tb")
        )
        assert any(isinstance(cb, keras.callbacks.TensorBoard) for cb in callbacks_list)

    def test_csv_logger_and_early_stopping_always_present(self, tmp_path):
        callbacks_list = build_fit_callbacks(tmp_path / "metrics.csv", tensorboard_log_dir=None)
        assert any(isinstance(cb, keras.callbacks.CSVLogger) for cb in callbacks_list)
        assert any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in callbacks_list)


# ---------------------------------------------------------------------------
# apply_cli_overrides -- N/A for this trainer? Kept out (priors are fixed
# per the design doc: "--prior not a knob here"). No test class: the trainer
# intentionally has no such function -- see the module docstring.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4-way weighted TEXT-stream composition (both heads' targets per batch)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tokenizer():
    return keras_hub.tokenizers.RobertaTokenizer.from_preset("roberta_base_en")


def _group_dataset(texts, cca_val, rel_val):
    return tf.data.Dataset.from_tensor_slices({
        "headline_with_lead": texts,
        "cca_label": tf.constant([cca_val] * len(texts), dtype=tf.float32),
        "rel_target": tf.constant([rel_val] * len(texts), dtype=tf.float32),
    })


class TestThreeStreamTextComposition:
    SEQ_LEN = 8
    BATCH = 8
    N_BATCHES_TO_SAMPLE = 40

    def _build_streams(self):
        cca_pos_ds = _group_dataset(
            [f"cca headline {i}</s>lede {i}" for i in range(20)], cca_val=1.0, rel_val=0.0
        )
        rel_pos_ds = _group_dataset(
            [f"rel headline {i}</s>lede {i}" for i in range(20)], cca_val=0.0, rel_val=1.0
        )
        unl_ds = _group_dataset(
            [f"unl headline {i}</s>lede {i}" for i in range(20)], cca_val=0.0, rel_val=0.0
        )
        return cca_pos_ds, rel_pos_ds, unl_ds

    def _preprocessor(self, tokenizer):
        return ClassifierPreprocessor(
            SEQ_LENGTH=self.SEQ_LEN,
            text_key="headline_with_lead",
            label_keys={"cca_targets": "cca_label", "rel_targets": "rel_target"},
            tokenizer=tokenizer,
            endpoint_model=True,
        )

    def test_composed_dataset_yields_both_target_keys(self, tokenizer):
        from src.data_setup.data import dataset_create

        preprocess = self._preprocessor(tokenizer)
        cca_pos_ds, rel_pos_ds, unl_ds = self._build_streams()
        composed = dataset_create(
            shuffle_buffer=16, batch_size=self.BATCH, preprocessor=preprocess,
            data=[cca_pos_ds, rel_pos_ds, unl_ds], weights=[0.1, 0.1, 0.8],
        )
        batch = next(iter(composed.take(1)))
        assert batch["token_ids"].shape == (self.BATCH, self.SEQ_LEN)
        assert batch["cca_targets"].shape == (self.BATCH,)
        assert batch["rel_targets"].shape == (self.BATCH,)

    def test_every_batch_carries_both_labels_for_all_rows(self, tokenizer):
        """Design doc: 'Every batch emits BOTH target keys for ALL rows' --
        both label arrays are always fully populated (no missing-value
        sentinel), regardless of which stream a row was drawn from."""
        from src.data_setup.data import dataset_create

        preprocess = self._preprocessor(tokenizer)
        cca_pos_ds, rel_pos_ds, unl_ds = self._build_streams()
        composed = dataset_create(
            shuffle_buffer=16, batch_size=self.BATCH, preprocessor=preprocess,
            data=[cca_pos_ds, rel_pos_ds, unl_ds], weights=[0.1, 0.1, 0.8],
        )
        for batch in composed.take(10):
            assert batch["cca_targets"].shape[0] == self.BATCH
            assert batch["rel_targets"].shape[0] == self.BATCH

    def test_label_routing_only_expected_values_appear(self, tokenizer):
        from src.data_setup.data import dataset_create

        preprocess = self._preprocessor(tokenizer)
        cca_pos_ds, rel_pos_ds, unl_ds = self._build_streams()
        composed = dataset_create(
            shuffle_buffer=16, batch_size=self.BATCH, preprocessor=preprocess,
            data=[cca_pos_ds, rel_pos_ds, unl_ds], weights=[0.1, 0.1, 0.8],
        )
        seen_cca, seen_rel = set(), set()
        for batch in composed.take(self.N_BATCHES_TO_SAMPLE):
            seen_cca.update(batch["cca_targets"].numpy().tolist())
            seen_rel.update(batch["rel_targets"].numpy().tolist())
        assert seen_cca <= {0.0, 1.0}
        assert seen_rel <= {0.0, 1.0}

    def test_composed_dataset_is_infinite(self, tokenizer):
        from src.data_setup.data import dataset_create

        preprocess = self._preprocessor(tokenizer)
        cca_pos_ds, rel_pos_ds, unl_ds = self._build_streams()
        composed = dataset_create(
            shuffle_buffer=16, batch_size=self.BATCH, preprocessor=preprocess,
            data=[cca_pos_ds, rel_pos_ds, unl_ds], weights=[0.1, 0.1, 0.8],
        )
        assert composed.cardinality().numpy() == tf.data.INFINITE_CARDINALITY


# ---------------------------------------------------------------------------
# Import side effects
# ---------------------------------------------------------------------------
class TestImportNoSideEffects:
    def test_import_does_not_trigger_training(self):
        import src.run_joint_text

        assert src.run_joint_text is not None
