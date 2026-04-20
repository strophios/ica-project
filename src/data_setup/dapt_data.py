import polars as pl
import tensorflow as tf


def data_from_parquet(path_prefix, db_folder="ldc_corpus", addl_columns=None):
    ldc_pq = pl.scan_parquet(
        f"{path_prefix}/{db_folder}/**/*.parquet", hive_partitioning=True
    )
    cols_to_select = ["id", "headline", "lead_paragraph"]
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
        pl.col("lead_paragraph").fill_null(""),
    )
    ldc_data = ldc_data.with_columns(
        pl.when(pl.col.headline == "NA")
        .then(pl.lit(""))
        .otherwise(pl.col.headline)
        .alias("headline"),
        pl.when(pl.col.lead_paragraph == "NA")
        .then(pl.lit(""))
        .otherwise(pl.col.lead_paragraph)
        .alias("lead_paragraph"),
    )
    # joining the headlines and leads, including the RoBERTa separator token (which I'm fairly certain does not
    # get spaces on either side; at least not added ones)
    # note that, if we create "headline_with_lead" ourselves, we don't need to built in MaskedLM preprocessor,
    # since we'll have a single series of strings, each of which needs to be packed, padded, and masked separately.
    headline_lead = [
        x + "</s>" + y
        for x, y in zip(
            ldc_data.get_column("headline"), ldc_data.get_column("lead_paragraph")
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
