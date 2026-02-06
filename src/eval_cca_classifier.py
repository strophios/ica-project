import keras
import tensorflow as tf
import polars as pl
import numpy as np

import os
import math

import src.model_setup.dapt_setup
import src.data_setup.dapt_data  # should rename this, since it's not just dapt stuff anymore
import src.preproc.preprocessor
import src.model_setup.classification_setup
import src.loss_functions.loss

path_prefix = os.path.expanduser(
    "~/immigration_project/00_ML_data_expansion/00_explorer"
)
# path_prefix = os.path.abspath("/project/ahd")

BATCH_SIZE = 256
SEQ_LENGTH = 128
validation_steps = math.floor(1017 / (BATCH_SIZE / 2))

preprocess = src.preproc.preprocessor.ClassifierPreprocessor(
    SEQ_LENGTH=SEQ_LENGTH,
    text_key="headline_with_lead",
    label_key="cca_label",
    endpoint_model=False,
)

# load the data
test_pos = tf.data.Dataset.load(f"{path_prefix}/cca_set/test_pos.tf")
# set shuffle_buffer to 0, which now results in no shuffling
test_pos = src.data_setup.dapt_data.dataset_create(
    shuffle_buffer=0, batch_size=BATCH_SIZE, preprocessor=preprocess, data=test_pos
)
test_unl = tf.data.Dataset.load(f"{path_prefix}/cca_set/test_unl.tf")
test_unl = src.data_setup.dapt_data.dataset_create(
    shuffle_buffer=0, batch_size=BATCH_SIZE, preprocessor=preprocess, data=test_unl
)

# the following currently throws an error, not sure why?
# test_set = src.data_setup.dapt_data.dataset_create(
#     0,
#     BATCH_SIZE,
#     preprocess,
#     data=[test_pos, test_unl],
#     weights=[0.5, 0.5],
# )

# load the classifier
# cca_classifier = keras.models.load_model(f"{path_prefix}/cca_classifier.keras")

cca_classifier = src.model_setup.classification_setup.classifier_from_dapt_checkpoint(
    f"{path_prefix}/dapt_backbone.weights.h5",
    freeze_encoder=True,  # dropout = .2?,
)  # at the very least has identical shape to RobertaTextClassifier

cca_classifier.load_weights(f"{path_prefix}/cca_classifier.weights.h5")

# overall eval on test set. probably want to add more metrics?
# test_results = cca_classifier.evaluate(
#     test_set, steps=validation_steps, return_dict=True
# )

# positive and unlabeled scores separately, to be added to their respective polars subsets
pos_scores = cca_classifier.predict(
    test_pos, batch_size=BATCH_SIZE, steps=validation_steps
)

unl_scores = cca_classifier.predict(
    test_unl, batch_size=BATCH_SIZE, steps=validation_steps
)

# Now add the predictions to the data to facilitate easy qualitative evaluation
ldc_data = src.data_setup.dapt_data.data_from_parquet(
    path_prefix,
    "ldc_corpus",
    addl_columns=["cca", "cca_descriptor", "immig", "immig_descriptor"],
)  # the function includes "ldc_corpus" as a default arg

ldc_data = src.data_setup.dapt_data.create_classifier_data(
    ldc_data, separate_labels=True
)

# for convenience
pos_df = ldc_data["test"]["pos"]
unl_df = ldc_data["test"]["unl"]

# add the scores, taking just up to the number of df rows, since
# the overrun their end and have repetitions
pos_df = pos_df.with_columns(cca_logit=pos_scores[0 : pos_df.shape[0]].squeeze())
# we only scored as many unlabeled as we have positives, so we subset by pos_df shape here
unl_df = unl_df[0 : pos_df.shape[0]].with_columns(
    cca_logit=unl_scores[0 : pos_df.shape[0]].squeeze()
)
# sort(self, by, descending)
# bottom_k
# top_k
np.percentile(pos_scores, [1, 2, 3, 4, 5, 10, 20])
np.percentile(unl_scores, [50, 75, 80, 85, 90, 95, 98])

pos_df.select("id", "headline", "lead_paragraph", "cca_logit").bottom_k(
    by="cca_logit", k=200
).write_csv(f"{path_prefix}/cca_classifier/pos_top_200.csv")
