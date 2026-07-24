from collections.abc import Collection

import polars as pl
import tensorflow as tf


def data_from_parquet(
    project_root, db_folder="ldc_corpus", addl_columns=None, lead_column="lead_paragraph",
    pattern=None,
):
    # `pattern` (relative to project_root) overrides the default recursive glob.
    # Needed when a folder holds multiple parquets with different schemas — e.g.
    # us_filter/ contains both ldc_labeled.parquet and audit/api_ldc_matched.parquet
    # (the latter lacking `id`), so the default `us_filter/**/*.parquet` glob fails
    # with ColumnNotFoundError. Pass pattern="us_filter/ldc_labeled.parquet" to read
    # exactly one file. Default (None) preserves the `{db_folder}/**/*.parquet` glob.
    glob = pattern if pattern is not None else f"{db_folder}/**/*.parquet"
    ldc_pq = pl.scan_parquet(
        f"{project_root}/{glob}", hive_partitioning=True
    )
    cols_to_select = ["id", "headline", lead_column]
    if addl_columns is not None:
        [cols_to_select.append(x) for x in addl_columns]

    ldc_data = ldc_pq.select(pl.col(cols_to_select))
    ldc_data = ldc_data.collect()
    # Replace missing headlines and leads with empty strings. Missing values
    # may appear either as the literal string "NA" (legacy upstream export
    # convention) or as a true polars null; both need to become "" before
    # the string concatenation below, which would otherwise raise on None.
    ldc_data = ldc_data.with_columns(
        pl.col("headline").fill_null(""),
        pl.col(lead_column).fill_null(""),
    )
    ldc_data = ldc_data.with_columns(
        pl.when(pl.col.headline == "NA")
        .then(pl.lit(""))
        .otherwise(pl.col.headline)
        .alias("headline"),
        pl.when(pl.col(lead_column) == "NA")
        .then(pl.lit(""))
        .otherwise(pl.col(lead_column))
        .alias(lead_column),
    )
    # joining the headlines and leads, including the RoBERTa separator token (which I'm fairly certain does not
    # get spaces on either side; at least not added ones)
    # note that, if we create "headline_with_lead" ourselves, we don't need to built in MaskedLM preprocessor,
    # since we'll have a single series of strings, each of which needs to be packed, padded, and masked separately.
    headline_lead = [
        x + "</s>" + y
        for x, y in zip(
            ldc_data.get_column("headline"), ldc_data.get_column(lead_column)
        )
    ]
    ldc_data = ldc_data.with_columns(
        pl.Series(name="headline_with_lead", values=headline_lead)
    )
    print(ldc_data.shape)  # LOG
    return ldc_data


def create_classifier_data(dataset, separate_labels=False):
    """
    Takes a dataset as a polars dataframe and returns a dictionary with keys a train/val/test split as polars dataframes.
    Note: currently hard coding a 90/5/5 train/val/test split, and also hard coding that we're looking at CCA specifically.

    :param: separate_labels sets whether we return a dict of three data sets (train/val/test) or six (train pos/train unlabeled/etc.)
    """
    # The train/val/test split below uses `sample(fraction=...)` followed by
    # `is_in(train_ids).not_()` to carve out the held-out splits. That only
    # gives correct splits when `id` is unique. Assert the invariant up front
    # so a silent labeling bug upstream surfaces here instead of producing
    # data leakage between splits.
    assert dataset["id"].n_unique() == dataset.shape[0], (
        f"`id` column is not unique: {dataset.shape[0]} rows but "
        f"{dataset['id'].n_unique()} distinct ids. The split logic in "
        f"`create_classifier_data` requires unique ids."
    )
    ldc_data = dataset.with_columns(
        cca_label=pl.when(pl.col("cca") | pl.col("cca_descriptor"))
        .then(1)
        .otherwise(0),
        immig_label=pl.when(pl.col("immig") | pl.col("immig_descriptor"))
        .then(1)
        .otherwise(0),
    )
    # Split into train/validate/test sets.
    # Based on @Ji2023, training = .9, validation = .05, test = .05

    # separate into labeled/unlabeled
    # now we actually need to decide which model we're training (cca or immigration)
    # working with cca for now
    ldc_unlabeled = ldc_data.filter(pl.col("cca_label") == 0)
    ldc_labeled = ldc_data.filter(pl.col("cca_label") == 1)

    print(ldc_unlabeled.shape)  # LOG
    print(ldc_labeled.shape)  # LOG

    # take .9/.05/.05 from each group
    ldc_unl_train = ldc_unlabeled.sample(
        fraction=0.9, seed=200
    )  # now sample .9 of the rows into the training set
    ldc_unl_val = ldc_unlabeled.filter(
        pl.col("id").is_in(ldc_unl_train["id"].implode()).not_()
    )
    ldc_unl_test = ldc_unl_val.sample(
        fraction=0.5, seed=200
    )  # and sample half of what's left (i.e., .05 of the total)
    ldc_unl_val = ldc_unl_val.filter(
        pl.col("id").is_in(ldc_unl_test["id"].implode()).not_()
    )

    ldc_lab_train = ldc_labeled.sample(fraction=0.9, seed=200)
    ldc_lab_val = ldc_labeled.filter(
        pl.col("id").is_in(ldc_lab_train["id"].implode()).not_()
    )
    ldc_lab_test = ldc_lab_val.sample(fraction=0.5, seed=200)
    ldc_lab_val = ldc_lab_val.filter(
        pl.col("id").is_in(ldc_lab_test["id"].implode()).not_()
    )
    # Note the use of .implode() above. This is because otherwise .is_in() with two columns is basically just
    # == (i.e., elementwise comparison) (this is not technically true, insofar as if the target column is a
    # column of lists, then you are indeed testing whether the elementwise corresponding value of the input is
    # in each one, but stil. especially cause I don't know that there's any more idiomatic way to do this in polars.)

    if separate_labels:
        print(
            f"Train: {ldc_lab_train.shape[0]} positives, {ldc_unl_train.shape[0]} unlabeled."
        )  # LOG
        print(
            f"Val: {ldc_lab_val.shape[0]} positives, {ldc_unl_val.shape[0]} unlabeled."
        )  # LOG
        print(
            f"Test: {ldc_lab_test.shape[0]} positives, {ldc_unl_test.shape[0]} unlabeled."
        )  # LOG

        return {
            "train": {"pos": ldc_lab_train, "unl": ldc_unl_train},
            "val": {"pos": ldc_lab_val, "unl": ldc_unl_val},
            "test": {"pos": ldc_lab_test, "unl": ldc_unl_test},
        }
    else:
        # stick the groups back together
        ldc_train = pl.concat([ldc_unl_train, ldc_lab_train])
        ldc_val = pl.concat([ldc_unl_val, ldc_lab_val])
        ldc_test = pl.concat([ldc_unl_test, ldc_lab_test])

        print(ldc_train.shape)  # LOG
        print(ldc_val.shape)  # LOG
        print(ldc_test.shape)  # LOG

        return {"train": ldc_train, "val": ldc_val, "test": ldc_test}


def create_us_filter_data(dataset):
    """Stratified 90/5/5 train/val/test split for the US filter (PN task).

    Drops rows with null `us_label` (unresolved/conflict), then splits the
    `us_label=True` and `us_label=False` groups separately (seed=200) and
    concatenates. Shuffles each split deterministically (seed=200) to ensure
    well-mixed class distribution throughout, preventing class-blocking when
    tf.data.from_tensor_slices preserves row order and downstream SHUFFLE_BUFFER
    is smaller than the full split.

    Returns {"train":..., "val":..., "test":...} polars DataFrames.
    """
    data = dataset.filter(pl.col("us_label").is_not_null())
    assert data["id"].n_unique() == data.shape[0], (
        f"`id` not unique: {data.shape[0]} rows, {data['id'].n_unique()} ids"
    )

    def _split(group):
        train = group.sample(fraction=0.9, seed=200)
        rest = group.filter(pl.col("id").is_in(train["id"].implode()).not_())
        test = rest.sample(fraction=0.5, seed=200)
        val = rest.filter(pl.col("id").is_in(test["id"].implode()).not_())
        return train, val, test

    pos = data.filter(pl.col("us_label"))
    neg = data.filter(pl.col("us_label").not_())
    p_tr, p_va, p_te = _split(pos)
    n_tr, n_va, n_te = _split(neg)
    return {
        "train": pl.concat([p_tr, n_tr]).sample(fraction=1.0, shuffle=True, seed=200),
        "val": pl.concat([p_va, n_va]).sample(fraction=1.0, shuffle=True, seed=200),
        "test": pl.concat([p_te, n_te]).sample(fraction=1.0, shuffle=True, seed=200),
    }


def assert_holdout_excluded(
    splits: dict, holdout_ids: Collection[str] | None
) -> None:
    """Verify no holdout id appears in any train/val pool.

    Raises ValueError enumerating offending ids if any holdout appears in
    train or val pools (test is fine — it's meant for evaluation only).
    Validates structure and raises if splits are malformed.
    No-op when holdout_ids is None or empty.

    Handles both CCA-doca splits (pos/unl) and relevance splits (pos/neg/unl).

    # pattern: Functional Core (pure, no side effects, raises on violation)
    """
    if not holdout_ids:
        return

    holdout_set = set(holdout_ids)
    leaked = set()

    for split_name in ("train", "val"):  # Only train/val matter; test is evaluation-only
        if split_name not in splits:
            raise ValueError(f"split '{split_name}' not found in splits dict")

        split_dict = splits[split_name]
        if not isinstance(split_dict, dict):
            raise ValueError(
                f"splits['{split_name}'] is not a dict: {type(split_dict)}"
            )

        # Guard both CCA (pos/unl) and relevance (pos/neg/unl) shapes.
        # pos and unl are always required; neg is optional (relevance only).
        for group_name in ("pos", "unl", "neg"):
            if group_name not in split_dict:
                # It's OK if "neg" doesn't exist (CCA splits have no neg).
                # But every split must have "pos" and "unl".
                if group_name in ("pos", "unl"):
                    raise ValueError(
                        f"group '{group_name}' not found in splits['{split_name}']"
                    )
                continue

            group_df = split_dict[group_name]
            if not isinstance(group_df, pl.DataFrame):
                raise ValueError(
                    f"splits['{split_name}']['{group_name}'] is not a DataFrame: "
                    f"{type(group_df)}"
                )

            group_ids = set(group_df["id"].to_list())
            leaked |= group_ids & holdout_set

    if leaked:
        raise ValueError(
            f"holdout ids leaked into train/val pools: {sorted(leaked)}"
        )


def create_cca_doca_data(table, seed=200, holdout_ids=None):
    """PU split for the CCA/DoCA retrain over cached embeddings.

    Operates on a polars table carrying at least `id`, `cca_label` (0/1), `us`
    (bool), and `emb_row` (row index into the cached CLS matrix). Returns the
    separate-labels shape `{"train":{"pos","unl"}, "val":..., "test":...}` of
    polars frames (each still carrying `emb_row`), so the caller gathers cached
    vectors per group and Ratio-Batch samples via `dataset_from_embeddings`.

    Positive/unlabeled definition (see docs/notes/cca-doca-retrain-design.md):
      - positives = `cca_label == 1` (DoCA-confirmed). Kept REGARDLESS of `us`:
        DoCA events are US by construction, so we never drop a confirmed positive
        because the US model scored it low.
      - unlabeled = `cca_label == 0 AND us` — the US-restricted background pool.

    `holdout_ids` (gold-set leakage guard): when given, those ids are dropped from
    the WHOLE table before splitting, so they never enter training in either role.
    The gold set is sampled from the unlabeled background and scored as (noisy)
    negatives; training on it then evaluating on it inflates apparent quality. The
    coding template carries no DoCA positives, but we drop from the whole table
    (not just the unlabeled pool) so the guard stays correct if that ever changes.
    `None`/empty is a strict no-op.

    Each group is split 90/5/5 separately (seed) and shuffled within split (seed),
    mirroring `create_us_filter_data` (prevents class-blocking under from_tensor_
    slices + a SHUFFLE_BUFFER smaller than the split).
    """
    assert table["id"].n_unique() == table.height, (
        f"`id` not unique: {table.height} rows, {table['id'].n_unique()} ids"
    )

    if holdout_ids:
        table = table.filter(pl.col("id").is_in(list(holdout_ids)).not_())

    def _split(group):
        train = group.sample(fraction=0.9, seed=seed)
        rest = group.filter(pl.col("id").is_in(train["id"].implode()).not_())
        test = rest.sample(fraction=0.5, seed=seed)
        val = rest.filter(pl.col("id").is_in(test["id"].implode()).not_())
        return train, val, test

    pos = table.filter(pl.col("cca_label") == 1)
    unl = table.filter((pl.col("cca_label") == 0) & pl.col("us"))
    p_tr, p_va, p_te = _split(pos)
    u_tr, u_va, u_te = _split(unl)

    def _shuf(d):
        return d.sample(fraction=1.0, shuffle=True, seed=seed)

    return {
        "train": {"pos": _shuf(p_tr), "unl": _shuf(u_tr)},
        "val": {"pos": _shuf(p_va), "unl": _shuf(u_va)},
        "test": {"pos": _shuf(p_te), "unl": _shuf(u_te)},
    }


def create_relevance_data(table, seed=200, holdout_ids=None):
    """PNU split for the relevance head: positives / reliable-negatives / unlabeled.

    The nnPNU counterpart to `create_cca_doca_data`. Operates on a table carrying
    `id`, `cca_label` (0/1), `reliable_neg` (bool), `us` (bool), and `emb_row`.
    Returns `{"train":{"pos","neg","unl"}, "val":..., "test":...}` so the caller
    gathers cached vectors per group and Ratio-Batch samples three streams.

    Group definition (differs from the CCA PU split):
      - positives  = `cca_label == 1`. Already US-restricted by the caller (unlike
        CCA, a relevance positive that scores non-US is foreign and out-of-domain).
      - reliable negatives = `reliable_neg` (confidently-foreign, no-US-footprint
        articles; selected US-passing). Fed to the loss as label -1.
      - unlabeled  = `cca_label == 0 AND us AND NOT reliable_neg` — US-restricted
        background minus the carved-out reliable negatives.

    `holdout_ids` drops gold-set ids from the whole table before splitting (same
    leakage guard as `create_cca_doca_data`). `None`/empty is a strict no-op.
    Each group is split 90/5/5 separately (seed) and shuffled within split (seed).
    """
    assert table["id"].n_unique() == table.height, (
        f"`id` not unique: {table.height} rows, {table['id'].n_unique()} ids"
    )
    if holdout_ids:
        table = table.filter(pl.col("id").is_in(list(holdout_ids)).not_())

    def _split(group):
        train = group.sample(fraction=0.9, seed=seed)
        rest = group.filter(pl.col("id").is_in(train["id"].implode()).not_())
        test = rest.sample(fraction=0.5, seed=seed)
        val = rest.filter(pl.col("id").is_in(test["id"].implode()).not_())
        return train, val, test

    pos = table.filter(pl.col("cca_label") == 1)
    neg = table.filter(pl.col("reliable_neg"))
    unl = table.filter(
        (pl.col("cca_label") == 0) & pl.col("us") & pl.col("reliable_neg").not_()
    )
    p_tr, p_va, p_te = _split(pos)
    g_tr, g_va, g_te = _split(neg)
    u_tr, u_va, u_te = _split(unl)

    def _shuf(d):
        return d.sample(fraction=1.0, shuffle=True, seed=seed)

    return {
        "train": {"pos": _shuf(p_tr), "neg": _shuf(g_tr), "unl": _shuf(u_tr)},
        "val": {"pos": _shuf(p_va), "neg": _shuf(g_va), "unl": _shuf(u_va)},
        "test": {"pos": _shuf(p_te), "neg": _shuf(g_te), "unl": _shuf(u_te)},
    }


def create_us_pnu_data(table, seed=200, holdout_ids=None):
    """90/5/5 P/N/U split for the US-head retrain table (src/build_us_pnu_table.py).

    # pattern: Functional Core (pure, no I/O)

    The nnPNU counterpart to `create_relevance_data`, keyed on this table's
    `pnu_label` Utf8 column ("pos"/"neg"/"unl") rather than cca_label/reliable_neg.
    Table must carry `id`, `pnu_label`, `cache`, `emb_row` (a per-cache feature-row
    index -- rows draw CLS vectors from FOUR different embed caches, so `emb_row`
    only makes sense together with `cache`; see `src.run_us_pnu.attach_emb_rows`,
    which produces this shape from the raw `us_pnu_table.parquet`).

    `holdout_ids` drops ids from the WHOLE table before splitting (belt-and-
    suspenders leakage guard, matching `create_cca_doca_data`/`create_relevance_data`
    -- the PNU table already excludes the ICA-eval holdout at build time; this is a
    second, independent check). `None`/empty is a strict no-op. Each group is split
    90/5/5 separately (seed) and shuffled within split (seed), mirroring the other
    `create_*_data` split functions (prevents class-blocking under
    `from_tensor_slices` + a `SHUFFLE_BUFFER` smaller than the split).
    """
    assert table["id"].n_unique() == table.height, (
        f"`id` not unique: {table.height} rows, {table['id'].n_unique()} ids"
    )
    if holdout_ids:
        table = table.filter(pl.col("id").is_in(list(holdout_ids)).not_())

    def _split(group):
        train = group.sample(fraction=0.9, seed=seed)
        rest = group.filter(pl.col("id").is_in(train["id"].implode()).not_())
        test = rest.sample(fraction=0.5, seed=seed)
        val = rest.filter(pl.col("id").is_in(test["id"].implode()).not_())
        return train, val, test

    pos = table.filter(pl.col("pnu_label") == "pos")
    neg = table.filter(pl.col("pnu_label") == "neg")
    unl = table.filter(pl.col("pnu_label") == "unl")
    p_tr, p_va, p_te = _split(pos)
    n_tr, n_va, n_te = _split(neg)
    u_tr, u_va, u_te = _split(unl)

    def _shuf(d):
        return d.sample(fraction=1.0, shuffle=True, seed=seed)

    return {
        "train": {"pos": _shuf(p_tr), "neg": _shuf(n_tr), "unl": _shuf(u_tr)},
        "val": {"pos": _shuf(p_va), "neg": _shuf(n_va), "unl": _shuf(u_va)},
        "test": {"pos": _shuf(p_te), "neg": _shuf(n_te), "unl": _shuf(u_te)},
    }


def dataset_create(
    shuffle_buffer,
    batch_size,
    preprocessor,
    path=None,
    data=None,
    weights=None,
    parallelism="default",
    seed=200,
):
    if path is not None:
        if weights is not None:
            dataset = [tf.data.Dataset.load(x) for x in path]
        else:
            dataset = tf.data.Dataset.load(path)
    elif data is not None:
        if weights is not None:
            if not isinstance(data[0], tf.data.Dataset):
                print(
                    "Note: creating a dataset from tensor slices can take several minutes."
                )
                dataset = [tf.data.Dataset.from_tensor_slices(x) for x in data]
            else:
                dataset = data
        else:
            if not isinstance(data, tf.data.Dataset):
                print(
                    "Note: creating a dataset from tensor slices can take several minutes."
                )
                dataset = tf.data.Dataset.from_tensor_slices(data)
            else:
                print(
                    "Passed a single dataset object. Check inputs, as this may be an error."
                )
                dataset = data
    else:
        raise ValueError("One of path or data must be provided.")
    if weights is not None:
        dataset = tf.data.Dataset.sample_from_datasets(
            dataset,
            weights=weights,
            seed=seed,
            stop_on_empty_dataset=True,
            rerandomize_each_iteration=True,
        )
    assert isinstance(dataset, tf.data.Dataset), "dataset not of type tf.data.Dataset"

    if shuffle_buffer == 0:
        dataset = dataset.repeat().batch(batch_size=batch_size, drop_remainder=True)
    else:
        dataset = (
            dataset.shuffle(buffer_size=shuffle_buffer)
            .repeat()
            .batch(batch_size=batch_size, drop_remainder=True)
        )

    if parallelism == "default":
        dataset = dataset.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(preprocessor, num_parallel_calls=parallelism)
        dataset = dataset.prefetch(parallelism)
    return dataset


def dataset_from_embeddings(
    shuffle_buffer,
    batch_size,
    data,
    weights=None,
    head_name="cca",
    repeat=True,
    seed=200,
):
    """Build a tf.data pipeline over CACHED CLS embeddings (features-mode).

    Counterpart to `dataset_create` for the frozen-backbone embedding cache:
    there is no tokenizer/preprocessor map because entries are already numeric.
    Each group in `data` is an `(features, labels)` pair of arrays — features
    `(N, hidden_dim)`, labels `(N,)`. With `weights`, groups are Ratio-Batch
    sampled exactly as `dataset_create` (e.g. `[0.1, 0.9]` pos:unl); without
    `weights`, `data` is a single `(features, labels)` group.

    Yields dicts shaped for `build_feature_endpoint_model`:
    `{"features": ..., f"{head_name}_targets": ...}`.

    `repeat=True` (training) gives an infinite pipeline for `steps_per_epoch`-
    driven `fit`; `repeat=False` (finite eval/predict) iterates once.
    """
    def _group_ds(group):
        feats, labels = group
        return tf.data.Dataset.from_tensor_slices(
            {
                "features": tf.convert_to_tensor(feats, dtype=tf.float32),
                f"{head_name}_targets": tf.convert_to_tensor(labels, dtype=tf.float32),
            }
        )

    if weights is not None:
        dataset = tf.data.Dataset.sample_from_datasets(
            [_group_ds(g) for g in data],
            weights=weights,
            seed=seed,
            stop_on_empty_dataset=True,
            rerandomize_each_iteration=True,
        )
    else:
        dataset = _group_ds(data)

    if shuffle_buffer != 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=repeat)
    return dataset.prefetch(tf.data.AUTOTUNE)


# the below version is the most recent working version, but it only works for the lu_classifier, since it doesn't implement path
# def dataset_create(
#     shuffle_buffer,
#     batch_size,
#     preprocessor,
#     data,
#     weights=None,
#     parallelism="default",
#     seed=200,
# ):
#     if not isinstance(data[0], tf.data.Dataset):
#         dataset = [tf.data.Dataset.from_tensor_slices(x) for x in data]
#     else:
#         dataset = data
#     dataset = tf.data.Dataset.sample_from_datasets(
#         dataset,
#         weights=weights,
#         seed=seed,
#         stop_on_empty_dataset=True,
#         rerandomize_each_iteration=True,
#     )
#     dataset = dataset.shuffle(buffer_size=shuffle_buffer).batch(
#         batch_size=batch_size, drop_remainder=True
#     )
#     if parallelism == "default":
#         dataset = dataset.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
#         dataset = dataset.prefetch(tf.data.AUTOTUNE)
#     else:
#         dataset = dataset.map(preprocessor, num_parallel_calls=parallelism)
#         dataset = dataset.prefetch(parallelism)
#     return dataset
