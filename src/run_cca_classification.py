"""
Script to train a single head classification model for the CCA classification task.
"""

import keras
import keras_hub
import tensorflow as tf

import os
import warnings
import math
import datetime

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
EPOCHS = 7

path_prefix = os.path.expanduser(
    "~/immigration_project/00_ML_data_expansion/00_explorer"
)
path_prefix = os.path.abspath("/project/ahd")

# ---- Load and Process Data ----
ldc_data = src.data_setup.dapt_data.data_from_parquet(
    path_prefix,
    "ldc_corpus",
    addl_columns=["cca", "cca_descriptor", "immig", "immig_descriptor"],
)  # the function includes "ldc_corpus" as a default arg

ldc_data = src.data_setup.dapt_data.create_classifier_data(
    ldc_data, separate_labels=True
)

# note: creating the dataset takes multiple minutes with the full dataset on Explorer
# so I check to see whether I've done it already, only do so if not (then save it)
if not os.path.isdir(f"{path_prefix}/cca_set"):
    os.mkdir(f"{path_prefix}/cca_set")
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
    ldc_data = {"train": {}, "val": {}, "test": {}}
    for i in split:
        for t in pu:
            ldc_data[i][t] = tf.data.Dataset.load(f"{path_prefix}/cca_set/{i}_{t}.tf")


preprocess = src.preproc.preprocessor.ClassifierPreprocessor(
    SEQ_LENGTH=SEQ_LENGTH,
    text_key="headline_with_lead",
    label_key="cca_label",
    endpoint_model=False,
)

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
steps_per_epoch = math.floor(18300 / (BATCH_SIZE / 10))
validation_steps = math.floor(1017 / (BATCH_SIZE / 2))

# ---- CREATE AND TRAIN MODEL ----
cca_classifier = src.model_setup.classification_setup.classifier_from_dapt_checkpoint(
    f"{path_prefix}/dapt_backbone.weights.h5",
    freeze_encoder=True,  # dropout = .2?,
)  # at the very least has identical shape to RobertaTextClassifier
# NOTE: I'm currently training only the classification head (and freezing the encoder)
# because 1) I've already done a round of DAPT, so a lot of the easy wins for the encoder
# should be there already; and 2) ChatGPT suggests pretty different learning rates for the
# bottom of the encoder, top of the encoder, and classification head, as well as suggesting
# layer-wise learning rate decay. Given that, I'm planning on figuring out implementing
# multiple optimizers in order to facilitate those things (almost certainly via a custom
# training loop) eventually, but for the moment, we're just going to do the easy thing
# that allows us to have *something*.

# Create the optimizer
# First a learning rate scheduler; parameter recommendations from ChatGPT
# ChatGPT also suggested CosineDecay, but I was using it before that
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=steps_per_epoch * 3,
    alpha=1e-1,  # note that this is given as a fraction of the target rate
    warmup_target=1e-3,
    warmup_steps=steps_per_epoch / 4,
)
# with loss scaling to deal with (potentially) problematically small gradients from AMP
# optimizer = keras.optimizers.LossScaleOptimizer(
#     keras.optimizers.AdamW(
#         learning_rate=lr_schedule,
#         weight_decay=5e-3,  # not sure the optimum weight decay
#     )  # Need to better set these
# )
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=5e-3,  # not sure the optimum weight decay
)  # Need to better set these

# Create the losses
flpu_loss = src.loss_functions.loss.FLPULoss(prior=0.03, kiryo_clawback=False)
# note that the current implementation of flpu_loss assumes from_logits = True

metrics_list = [
    # keras.metrics.F1Score(threshold=0.0),  # threshold = 0 cause from_logits = True
    # keras.metrics.BinaryCrossentropy(from_logits=True),
    keras.metrics.BinaryAccuracy(),
]

# Compile the model
cca_classifier.compile(
    loss=flpu_loss,
    optimizer=optimizer,
    metrics=metrics_list,
    jit_compile="auto",  # probably set to true for Explorer?
)

if not os.path.isdir(f"{path_prefix}/cca_classifier"):
    os.mkdir(f"{path_prefix}/cca_classifier")

if not os.path.isdir(f"{path_prefix}/cca_logs"):
    os.mkdir(f"{path_prefix}/cca_logs")
# Set callbacks
callbacks_list = [
    # Saves the current weights after every epoch
    keras.callbacks.ModelCheckpoint(
        # Path to the destination model file
        filepath=f"{path_prefix}/cca_classifier/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_checkpoint.keras",
        # These two arguments mean you won't overwrite the model file
        # unless val_loss has improved, which allows you to keep the
        # best model seen during training.
        monitor="val_loss",
        save_best_only=True,
    ),
    # TensorBoard
    keras.callbacks.TensorBoard(
        log_dir=f"{path_prefix}/cca_logs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        histogram_freq=1,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=(500, 550),
    ),
    # Early Stopping
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        verbose=1,
        start_from_epoch=2,
    ),
]

# ---- TRAIN THE MODEL ----
# train with class_weight
cca_classifier.fit(
    training_set,
    validation_data=validation_set,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks_list,
)

# There may be an issue with the profiler; leads to a warning during training, may or may not impact the usefulness of profiler

cca_classifier.save(f"{path_prefix}/cca_classifier.keras")
cca_classifier.save_weights(f"{path_prefix}/cca_classifier.weights.h5")

test_results = cca_classifier.evaluate(
    test_set, steps=validation_steps, return_dict=True
)

test_pos = tf.data.Dataset.load(f"{path_prefix}/cca_set/test_pos.tf")
# test_pos = test_pos.map(preprocess, num_parallel_calls = tf.data.AUTOTUNE)
test_pos = src.data_setup.dapt_data.dataset_create(
    shuffle_buffer=0, batch_size=BATCH_SIZE, preprocessor=preprocess, data=test_pos
)
pos_scores = cca_classifier.predict(
    test_pos, batch_size=BATCH_SIZE, steps=validation_steps
)

test_unl = tf.data.Dataset.load(f"{path_prefix}/cca_set/test_unl.tf")
test_unl = src.data_setup.dapt_data.dataset_create(
    shuffle_buffer, BATCH_SIZE, preprocess, data=test_unl
)
unl_scores = cca_classifier.predict(
    test_unl, batch_size=BATCH_SIZE, steps=validation_steps
)
