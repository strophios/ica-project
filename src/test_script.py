"""
Script to create and train a toy model to facilitate (local) testing.
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

# Preprocessing params
# We make this a toy model by setting both BATCH_SIZE and SEQ_LENGTH to 128 (might consider
# dropping BATCH_SIZE even further)
BATCH_SIZE = 128
SEQ_LENGTH = 128

# Training params (though I will rarely be fully training a model here, I think)
EPOCHS = 5

path_prefix = os.path.expanduser(
    "~/immigration_project/00_ML_data_expansion/00_explorer"
)

backbone = keras_hub.models.Backbone.from_preset(
    "roberta_base_en", preprocessor=None, load_weights=False
)
backbone.load_weights(f"{path_prefix}/dapt_backbone.weights.h5")

# cca_classifier = src.model_setup.classification_setup.classifier_from_dapt_checkpoint(
#     f"{path_prefix}dapt_backbone.weights.h5"
# )  # at the very least has identical shape to RobertaTextClassifier

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
    SEQ_LENGTH=SEQ_LENGTH,
    text_key="headline_with_lead",
    label_key="cca_label",
    endpoint_model=True,
)

splits = ["val"]
pus = ["pos", "unl"]
if not os.path.isdir(f"{path_prefix}/trial_set"):
    os.mkdir(f"{path_prefix}/trial_set")
    for split in splits:
        for pu in pus:
            ldc_data[split][pu] = tf.data.Dataset.from_tensor_slices(
                ldc_data[split][pu]
                .select(["headline_with_lead", "cca_label"])
                .to_dict()
            )
            ldc_data[split][pu].save(f"{path_prefix}/trial_set/{split}_{pu}.tf")
else:
    split = "val"
    pu = ["pos", "unl"]
    trial_data = {}
    for i in split:
        for t in pu:
            trial_data[i][t] = tf.data.Dataset.load(
                f"{path_prefix}/trial_set/{i}_{t}.tf"
            )

# We make a toy model by just using the validation data
trial_data = {}
trial_data["pos"] = tf.data.Dataset.load(f"{path_prefix}/trial_set/val_pos.tf")
trial_data["unl"] = tf.data.Dataset.load(f"{path_prefix}/trial_set/val_unl.tf")

# Now do the preprocessing, shuffling, and batching
shuffle_buffer = 10000  # keep in mind that I ideally want to increase this, but may actually need to decrease it

# current batch ratio: 9 unl to 1 pos
trial_set = src.data_setup.dapt_data.dataset_create(
    shuffle_buffer,
    BATCH_SIZE,
    preprocess,
    data=[trial_data["pos"], trial_data["unl"]],
    weights=[1 / 10, 9 / 10],
)

# Setting steps_per_epoch and validation_steps
# Train: 18300 positives, 1026418 unlabeled.
# Val: 1017 positives, 57024 unlabeled.
# NOTE: these are currently not working (in that they don't seem to stop us from exhausting the datasets and throwing warnings)
# (I think this is maybe fixed, since I'm repeating the datasets now)
steps_per_epoch = math.floor(1017 / (BATCH_SIZE / 2))

# ---- TRAINING ----
# (recall that we created the model above)

# Create an optimizer
optimizer = keras.optimizers.AdamW(  # No loss optimizer, since AMP doesn't work locally
    learning_rate=1e-3, weight_decay=1e-2
)


class EndpointLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, logits, targets=None):
        if targets is not None:
            loss = self.loss_fn(targets, logits)
            self.add_loss(loss)
        return keras.ops.sigmoid(logits)


# Create the model
inputs = {
    **backbone.input,
    "targets": keras.Input(shape=(1,), dtype="int32", name="targets"),
}
x = backbone(inputs)[:, backbone.start_token_index, :]
logits = keras.layers.Dense(units=1, activation=None)(x)
outputs = EndpointLayer(name="preds")(logits, inputs["targets"])
trial_model = keras.Model(inputs, outputs)

trial_model.compile(
    optimizer=optimizer, metrics=["binary_crossentropy"], jit_compile=False
)

# ---- TRAIN THE MODEL ----
# train with class_weight
trial_model.fit(
    trial_set,
    # validation_data=validation_set, # no validation data at the moment
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    # validation_steps=validation_steps,
)

# There may be an issue with the profiler; leads to a warning during training, may or may not impact the usefulness of profiler
