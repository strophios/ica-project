import polars as pl
import numpy as np

# Set keras backend
# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["KERAS_BACKEND"] = "torch"

import keras
import keras_hub

import tensorflow as tf
import datetime
import math

import src.config as config
import src.data_setup.data
from src.preproc.preprocessor import CustomPreprocessor
import src.model_setup.dapt_setup

# tf.config.list_physical_devices('GPU')

# Platform-conditional dtype policy. mixed_float16 on cluster CUDA;
# float32 locally (MPS mixed-precision support is patchy and the
# Tensor-Core motivation evaporates without CUDA).
keras.config.set_dtype_policy(config.DTYPE_POLICY)

# Seed Python, NumPy, and the Keras backend RNG. Matches the seed=200 used by
# polars `.sample()` calls elsewhere in the pipeline so the whole pipeline is
# reproducible. Needs to happen before any model construction or dataset
# creation (for sample_from_datasets / shuffle / MaskedLMMaskGenerator).
keras.utils.set_random_seed(200)

# Preprocessing params
# SEQ_LENGTH and BATCH_SIZE of 128 for local testing (see below for rough assessment of how
# much truncation that causes); maybe bump SEQ_LENGTH back to 256 for Explorer? Not sure.
BATCH_SIZE = 256  # will need to test; 256 may be too high
SEQ_LENGTH = 128
PREDICTIONS_PER_SEQ = (
    32  # in standard BERT / RoBERTa (with seq length of 512) this is 96
)
MASK_RATE = 0.15  # This is the RoBERTa default, I'm pretty sure.

# Training params
EPOCHS = 5

ldc_data = src.data_setup.data.data_from_parquet(
    config.PROJECT_ROOT, "ldc_corpus"
)  # the function includes "ldc_corpus" as a default arg

# Now we create our datasets, a 90/10 split for training/validation
ldc_train = ldc_data.sample(fraction=0.9, seed=200)
ldc_val = ldc_data.filter(pl.col("id").is_in(ldc_train["id"].implode()).not_())

# Setting steps_per_epoch and validation_steps
# steps_per_epoch = math.ceil(ldc_train.shape[0] / BATCH_SIZE)
# validation_steps = math.ceil(ldc_val.shape[0] / BATCH_SIZE)

steps_per_epoch = math.ceil((1160799 * 0.9) / BATCH_SIZE)  # 4081 with BATCH_SIZE = 256
validation_steps = math.ceil((1160799 * 0.1) / BATCH_SIZE)  # 454 with BATCH_SIZE = 256

# ---- PREPROCESSING ----
# The CustomPreprocessor now takes the arguments necessary for creating the tokenizer, packer, and masker
# components and creates them as part of initializing the class instance, rather than taking them as
# arguments. An alternative tokenizer may be provided (otherwise we get a from_preset roberta_base_en
# tokenizer). And you can still get preprocessing without masking by passing None for both MASK_RATE and
# PREDICTIONS_PER_SEQ.
preprocess = CustomPreprocessor(SEQ_LENGTH, MASK_RATE, PREDICTIONS_PER_SEQ)

# note: this takes multiple minutes with the full dataset on Explorer
# I'm now saving the dataset after this, so I can just load from there now
# training_set = tf.data.Dataset.from_tensor_slices(ldc_train["headline_with_lead"])
# validation_set = tf.data.Dataset.from_tensor_slices(ldc_val["headline_with_lead"])

# save dataset
# training_set.save(str(config.DAPT_TRAINING_SET))
# validation_set.save(str(config.DAPT_VALIDATION_SET))

training_set = tf.data.Dataset.load(str(config.DAPT_TRAINING_SET))
validation_set = tf.data.Dataset.load(str(config.DAPT_VALIDATION_SET))

# **can now just load datasets**

# Now do the preprocessing, shuffling, and batching
shuffle_buffer = 100000  # keep in mind that I ideally want to increase this, but may actually need to decrease it

training_set = src.data_setup.data.dataset_create(
    shuffle_buffer,
    BATCH_SIZE,
    preprocess,
    path=str(config.DAPT_TRAINING_SET),
)
validation_set = src.data_setup.data.dataset_create(
    shuffle_buffer,
    BATCH_SIZE,
    preprocess,
    path=str(config.DAPT_VALIDATION_SET),
)

# training_set = training_set.shuffle(buffer_size = shuffle_buffer).batch(batch_size = BATCH_SIZE)
# training_set = training_set.map(preprocess, num_parallel_calls = tf.data.AUTOTUNE)
# training_set = training_set.prefetch(tf.data.AUTOTUNE)
# validation_set = validation_set.shuffle(buffer_size = shuffle_buffer).batch(batch_size = BATCH_SIZE)
# validation_set = validation_set.map(preprocess, num_parallel_calls = tf.data.AUTOTUNE)
# validation_set = validation_set.prefetch(tf.data.AUTOTUNE)

# ---- CREATE THE MODEL ----
dapt_model = src.model_setup.dapt_setup.get_DAPT_model(
    PREDICTIONS_PER_SEQ, path=str(config.DAPT_LM_HEAD_WEIGHTS)
)

# ---- SETUP PRETRAINING ----

# Set the learning rate schedule
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=5e-6,
    decay_steps=12000,  # steps_per_epoch*3,
    alpha=1e-3,  # note that this is given as a fraction of the initial rate
    warmup_target=5e-5,  # higher LR w/ higher batch size. NOTE: when this was 5e-4, we were getting *reversed* training starting partway into warmup
    warmup_steps=1000,  # steps_per_epoch/4,
)
# lr_schedule = 1e-4


# seq length of 128 should take 1/16 the memory of seq length 512 (I think)
# Create the optimizer
# with loss scaling to deal with (potentially) problematically small gradients from AMP
optimizer = keras.optimizers.LossScaleOptimizer(
    keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.01)
)

# Compile the model
dapt_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    weighted_metrics=["sparse_categorical_accuracy"],
    jit_compile="auto",  # probably set to true for Explorer?
)

# Set callbacks
# from Deep Learning with Python
# Callbacks are passed to the model via the callbacks argument in
# fit(), which takes a list of callbacks. You can pass any number of
# callbacks.
_run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
callbacks_list = [
    # Saves the current weights after every epoch
    keras.callbacks.ModelCheckpoint(
        # Path to the destination model file
        filepath=str(config.PROJECT_ROOT / f"{_run_stamp}_dapt_checkpoint.keras"),
        # These two arguments mean you won't overwrite the model file
        # unless val_loss has improved, which allows you to keep the
        # best model seen during training.
        monitor="val_loss",
        save_best_only=True,
    ),
    # TensorBoard
    keras.callbacks.TensorBoard(
        log_dir=str(config.DAPT_LOGS_DIR / _run_stamp),
        histogram_freq=1,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=(500, 550),
        embeddings_freq=1,
    ),
]


# AMP off, still broken. loss drops for like 300 steps, then starts increasing

# ---- TRAIN THE MODEL ----
dapt_model.fit(
    training_set,
    validation_data=validation_set,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks_list,
)

# There may be an issue with the profiler; leads to a warning during training, may or may not impact the usefulness of profiler

dapt_model.save(str(config.DAPT_CURRENT_MODEL))

# This appears to be working locally, running at 50s/step, but not using the GPU at all with CPU maxed.
# Rerunning w/o AMP and with the torch backend drops this to 12s/step, with spiky GPU usage.
# Hypothesis: AMP not supported by Keras + MPS together (not sure on which side), leading to model
# fit running on the CPU. Dropping AMP puts us on the GPU, but still really inefficiently, so far.
# In both trials so far, we're seeing (training) loss starting around 0.54 and accuracy around .65

# Try:
# - set preprocessing parallelism / prefetching by hand instead of using tf.data.AUTOTUNE
# - increasing batch size
# - fully preprocessing the input ahead of time

# Tried:
# - dropping the dataset shuffling: no difference (slower, if anything)
# - putting the .batch() before the .map(preprocess) step in the datasets: may have provided slight speed up (11s/step),
#   didn't impact the spiky GPU usage.
# - setting prefetch() and map() number of parallel calls by hand has only made things worse. Of course, I haven't tried
#   all the options, but nothing I've tried so far has helped (I've only seen slowdowns and, if anything, worse GPU use).
# - batch size = 128: this has been the baseline this time around
# - batch size = 256: out of memory error
# - batch size = 64: dramatic improvement, 4s/step which is 3x faster with only a 2x smaller batch. GPU utilization still
#   spiky, but the average is much higher; overall way better. A full epoch this way would take under an hour.


# error: multiple warnings of the following type
# 2025-10-13 16:41:43.833582: I external/local_xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function 'gemm_fusion_dot_34176', 760 bytes spill stores, 760 bytes spill loads

# with batch size = 256
# 116ms/step (on H200); estimating ~7.5 minutes for the first epoch
# at end of Epoch 1 (started at .47/8 something)
# training loss: 0.4654 - sparse_categorical_accuracy: 0.68152025
# val_loss: 0.3971 - val_sparse_categorical_accuracy: 0.7207
# Epoch 2: val_loss: 0.3755 - val_sparse_categorical_accuracy: 0.7327
# Epoch 3: val_loss: 0.3755 - val_sparse_categorical_accuracy: 0.7324
# Epoch 4: val_loss: 0.3760 - val_sparse_categorical_accuracy: 0.7324
# Epoch 5:

# Epoch 1: Loss: 0.4525 - sparse_categorical_accuracy: 0.6880 - val_loss: 0.3909 - val_sparse_categorical_accuracy: 0.7239
# Epoch 2: Loss: 0.4231 - sparse_categorical_accuracy: 0.7047 - val_loss: 0.3743 - val_sparse_categorical_accuracy: 0.7331
# Epoch 3: Loss: 0.4024 - sparse_categorical_accuracy: 0.7157 - val_loss: 0.3660 - val_sparse_categorical_accuracy: 0.7380

# TF 2.19.1
