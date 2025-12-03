"""
Script to train a single head classification model for the CCA classification task.
"""

import keras
import keras_hub
import tensorflow as tf

import os
import warnings
import math

import src.model_setup.dapt_setup
import src.data_setup.dapt_data  # should rename this, since it's not just dapt stuff anymore
import src.preproc.preprocessor
import src.model_setup.classification_setup
import src.loss_functions.loss

keras.config.set_dtype_policy(
    "mixed_float16"
)  # want to make sure this works on Explorer

# Preprocessing params
# SEQ_LENGTH and BATCH_SIZE of 128 for local testing (see below for rough assessment of how
# much truncation that causes); maybe bump SEQ_LENGTH back to 256 for Explorer? Not sure.
BATCH_SIZE = 256
SEQ_LENGTH = 128

# Training params
EPOCHS = 3

path_prefix = os.path.expanduser(
    "~/immigration_project/00_ML_data_expansion/00_explorer"
)
path_prefix = os.path.abspath("/project/ahd")

backbone = keras_hub.models.Backbone.from_preset(
    "roberta_base_en", preprocessor=None, load_weights=False
)
backbone.load_weights(f"{path_prefix}/dapt_backbone.weights.h5")

cca_classifier = src.model_setup.classification_setup.classifier_from_dapt_checkpoint(
    f"{path_prefix}dapt_backbone.weights.h5"
)  # at the very least has identical shape to RobertaTextClassifier

# ---- Load and Process Data ----
ldc_data = src.data_setup.dapt_data.data_from_parquet(
    path_prefix,
    "ldc_corpus",
    addl_columns=["cca", "cca_descriptor", "immig", "immig_descriptor"],
)  # the function includes "ldc_corpus" as a default arg

ldc_data = src.data_setup.dapt_data.create_classifier_data(
    ldc_data, separate_labels=True
)

preprocess = src.preproc.preprocessor.ClassifierPreprocessor(
    SEQ_LENGTH=SEQ_LENGTH, text_key="headline_with_lead", label_key="cca_label"
)
# note: creating the dataset takes multiple minutes with the full dataset on Explorer
# so I check to see whether I've done it already, only do so if not (then save it)
if not os.path.isdir(f"{path_prefix}/cca_set"):
    for split in ldc_data.keys():
        for pu in ldc_data[split].keys():
            ldc_data[split][pu] = tf.data.Dataset.from_tensor_slices(
                ldc_data[split][pu]
                .select(["headline_with_lead", "cca_label"])
                .to_dict()
            )
            ldc_data[split][pu].save(f"{path_prefix}/cca_set/{split}_{pu}.tf")
else:
    split = ["train", "val", "test"]
    pu = ["pos", "unl"]
    ldc_data = {}
    for i in split:
        for t in pu:
            ldc_data[i][t] = tf.data.Dataset.load(f"{path_prefix}/cca_set/{i}_{t}.tf")


# **can now just load datasets directly with dataset_create()**

# Now do the preprocessing, shuffling, and batching
shuffle_buffer = 100000  # keep in mind that I ideally want to increase this, but may actually need to decrease it

# current batch ratio: 9 unl to 1 pos
training_set = src.data_setup.dapt_data.dataset_create(
    shuffle_buffer,
    BATCH_SIZE,
    preprocess,
    data=[ldc_data["train"]["pos"], ldc_data["train"]["unl"]],
    weights=[1 / 10, 9 / 10],
)
# NOTE: Not sure what weights I want for validation and test sets.
# Actually, pretty sure I should be using just positives and known
# negatives for validation and test, so this is maybe moot anyways.
validation_set = src.data_setup.dapt_data.dataset_create(
    shuffle_buffer,
    BATCH_SIZE,
    preprocess,
    data=[ldc_data["val"]["pos"], ldc_data["val"]["unl"]],
    weights=[0.5, 0.5],
)
test_set = src.data_setup.dapt_data.dataset_create(
    shuffle_buffer,
    BATCH_SIZE,
    preprocess,
    data=[ldc_data["test"]["pos"], ldc_data["test"]["unl"]],
    weights=[0.5, 0.5],
)

# Setting steps_per_epoch and validation_steps
# Train: 18300 positives, 1026418 unlabeled.
# Val: 1017 positives, 57024 unlabeled.
# NOTE: these are currently not working (in that they don't seem to stop us from exhausting the datasets and throwing warnings)
# (I think this is maybe fixed, since I'm repeating the datasets now)
steps_per_epoch = math.floor(18300 / (BATCH_SIZE / 10))
validation_steps = math.floor(1017 / (BATCH_SIZE / 2))

# ---- TRAINING ----
# (recall that we created the model above)

# Create the optimizer
# with loss scaling to deal with (potentially) problematically small gradients from AMP
optimizer = keras.optimizers.LossScaleOptimizer(  # not sure if we should be using a LossScaleOptimizer here, but meh
    keras.optimizers.AdamW(
        learning_rate=1e-3, weight_decay=1e-2
    )  # Need to better set these
)

# Create the losses
focal_loss = src.loss_functions.loss.FLPULoss()

# Compile the model
cca_classifier.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.Recall(thresholds=0, name="recall")],
    jit_compile="auto",  # probably set to true for Explorer?
)

# Set callbacks
callbacks_list = [
    # Early stopping
    keras.callbacks.EarlyStopping(monitor="recall", min_delta=0.005, verbose=1),
]

# ---- TRAIN THE MODEL ----
# train with class_weight
cca_classifier.fit(
    training_set,
    validation_data=validation_set,
    epochs=EPOCHS,
    class_weight={0: 1.0, 1: 15.0},
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks_list,
)

# There may be an issue with the profiler; leads to a warning during training, may or may not impact the usefulness of profiler

cca_classifier.save(f"{path_prefix}/cca_classifier.keras")
