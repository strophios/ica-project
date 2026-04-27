"""
Script to create and train a toy model to facilitate (local) testing.
"""

import keras
import keras_hub
import tensorflow as tf

import math

import src.config as config
import src.model_setup.dapt_setup
import src.data_setup.data
import src.preproc.preprocessor
# import src.model_setup.classification_setup  # retired in Tier 2 Piece 4c
import src.loss_functions.loss

# Preprocessing params
# We make this a toy model by setting both BATCH_SIZE and SEQ_LENGTH to 128 (might consider
# dropping BATCH_SIZE even further)
BATCH_SIZE = 128
SEQ_LENGTH = 128

# Training params (though I will rarely be fully training a model here, I think)
EPOCHS = 5

backbone = keras_hub.models.Backbone.from_preset(
    "roberta_base_en", preprocessor=None, load_weights=False
)
backbone.load_weights(str(config.DAPT_BACKBONE_WEIGHTS))

# cca_classifier = src.model_setup.classification_setup.classifier_from_dapt_checkpoint(
#     f"{path_prefix}dapt_backbone.weights.h5"
# )  # at the very least has identical shape to RobertaTextClassifier

# ---- Load and Process Data ----
# ldc_data = src.data_setup.dapt_data.data_from_parquet(
#     path_prefix,
#     "ldc_corpus",
#     addl_columns=["cca", "cca_descriptor", "immig", "immig_descriptor"],
# )  # the function includes "ldc_corpus" as a default arg
#
# ldc_data = src.data_setup.dapt_data.create_classifier_data(
#     ldc_data, separate_labels=True
# )


# splits = ["val"]
# pus = ["pos", "unl"]
# if not os.path.isdir(f"{path_prefix}/trial_set"):
#     os.mkdir(f"{path_prefix}/trial_set")
#     for split in splits:
#         for pu in pus:
#             ldc_data[split][pu] = tf.data.Dataset.from_tensor_slices(
#                 ldc_data[split][pu]
#                 .select(["headline_with_lead", "cca_label"])
#                 .to_dict()
#             )
#             ldc_data[split][pu].save(f"{path_prefix}/trial_set/{split}_{pu}.tf")
# else:
#     split = "val"
#     pu = ["pos", "unl"]
#     trial_data = {}
#     for i in split:
#         for t in pu:
#             trial_data[i][t] = tf.data.Dataset.load(
#                 f"{path_prefix}/trial_set/{i}_{t}.tf"
#             )

# We make a toy model by just using the validation data
trial_data = {}
trial_data["pos"] = tf.data.Dataset.load(
    str(config.PROJECT_ROOT / "trial_set" / "val_pos.tf")
)
trial_data["unl"] = tf.data.Dataset.load(
    str(config.PROJECT_ROOT / "trial_set" / "val_unl.tf")
)

preprocess = src.preproc.preprocessor.ClassifierPreprocessor(
    SEQ_LENGTH=SEQ_LENGTH,
    text_key="headline_with_lead",
    label_key="cca_label",
    endpoint_model=True,
)

# Now do the preprocessing, shuffling, and batching
shuffle_buffer = 10000  # keep in mind that I ideally want to increase this, but may actually need to decrease it

# current batch ratio: 9 unl to 1 pos
trial_set = src.data_setup.data.dataset_create(
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
    learning_rate=1e-3, weight_decay=1e-4
)

flpu_loss = src.loss_functions.loss.FLPULoss(prior=0.0175, kiryo_clawback=False)
# Not sure about whether I should be tweaking \alpha; the default (following the
# original focal loss paper @Lin2020) is 0.25. Note that this *down-weights* the
# less common positive class, which is different from how it's normally used; but
# they found it worked well given the impact of the \gamma scaling factor (though
# they also found that \alpha just didn't make much difference). But then also
# any kind of class balancing maybe interacts with the PU loss structure? For now
# I'm just sticking to the default (which is also what @Ji2023 used).


class EndpointLayer(keras.layers.Layer):
    def __init__(self, loss_fn, name=None):
        super().__init__(name=name)
        # self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_fn = loss_fn

    def call(self, logits, targets=None):
        # print(targets)
        # print(logits)
        if targets is not None:
            loss = self.loss_fn(targets, logits)
            self.add_loss(loss)
        return keras.ops.sigmoid(logits)


# Create the model
# inputs = {
#     **backbone.input,
#     "targets": keras.Input(shape=(1,), dtype="int32", name="targets"),
# }

feature_inputs = {
    "token_ids": keras.Input(shape=(SEQ_LENGTH,), dtype="int32", name="token_ids"),
    "padding_mask": keras.Input(
        shape=(SEQ_LENGTH,), dtype="int32", name="padding_mask"
    ),
}

inputs = {
    **feature_inputs,
    "targets": keras.Input(shape=(1,), dtype="int32", name="targets"),
}

# x = backbone(backbone.input)[:, backbone.start_token_index, :]
x = backbone(feature_inputs)[:, backbone.start_token_index, :]
logits = keras.layers.Dense(units=1, activation=None, name="create_logits")(x)
outputs = EndpointLayer(loss_fn=flpu_loss, name="endpoint")(logits, inputs["targets"])
trial_model = keras.Model(inputs, outputs)
# trial_model = keras.Model(feature_inputs, logits)

trial_model.compile(
    optimizer=optimizer,
    # loss = keras.losses.BinaryCrossentropy(from_logits = True),
    # metrics=["binary_crossentropy"],
    jit_compile=False,
)
# loss went from ~.55 to ~.61 (as high as ~.65)
# loss went from ~.8 to 1.5 in an epoch, but then down to .3-ish by the end of 5 epochs

# we've switched to using FLPULoss.
# First run: it works, but loss started around 2800 and ballooned by step 7 to 9600ish (it may have decreased at some point)
# I'm not sure how intrinsically worrying the size of the loss should be, but the whole "increasing" piece is worrying on its own
# Second (working) run: Yay, losses looking much better! Now we were at .177 at the end of epoch 1, down to .0162 by end of epoch 3

# ---- TRAIN THE MODEL ----
trial_model.fit(
    trial_set,
    # validation_data=validation_set, # no validation data at the moment
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    # validation_steps=validation_steps,
)

# There may be an issue with the profiler; leads to a warning during training, may or may not impact the usefulness of profiler

preprocess = src.preproc.preprocessor.ClassifierPreprocessor(
    SEQ_LENGTH=SEQ_LENGTH,
    text_key="headline_with_lead",
    label_key="cca_label",
    endpoint_model=False,
)

# cca_classifier = src.model_setup.classification_setup.classifier_from_dapt_checkpoint(
#     str(config.DAPT_BACKBONE_WEIGHTS)
# )  # retired in Tier 2 Piece 4c — use load_dapt_backbone + ClassificationHead +
#    # build_endpoint_model from src.model_setup instead. This scratch script is
#    # slated for Tier 4 hygiene cleanup.
raise RuntimeError(
    "test_script.py is partially broken pending Tier 4 cleanup; the "
    "classifier-construction path was retired in Piece 4c."
)
cca_classifier.compile(
    optimizer=optimizer,
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    jit_compile=False,
)
cca_classifier.fit(trial_set, epochs=EPOCHS, steps_per_epoch=steps_per_epoch)
