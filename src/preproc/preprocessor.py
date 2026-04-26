import keras
import keras_hub


class ClassifierPreprocessor:
    """
    Multi-head-aware classifier preprocessor.

    Maps a dict-valued batch (yielded by `tf.data.Dataset.map`) to the
    shape expected by either a standard-mode model (loss handled by
    `compile(loss=...)`) or an endpoint-layer model (loss handled by
    head-internal `add_loss`, our primary path for FLPU and eventual
    ALUM).

    Note on the endpoint-mode shape: in endpoint mode, *targets are
    model inputs*. The head's `call(features, targets=...)` consumes
    them inside the model graph via `add_loss`. The preprocessor's
    output dict therefore folds both features and targets into a
    single dict; Keras's `.fit()` routes the entries to named
    `keras.Input`s on the model side. This is the inherent shape of
    the endpoint-layer pattern — see `model_setup/heads.py` and the
    Piece 1 / Piece 3 sections of `docs/notes/tier2-design.md`.

    Dtype: targets are cast to `target_dtype` (default `"float32"`)
    here, so the cached preprocessed dataset has predictable dtype.
    Losses still cast `y_true` to `y_pred.dtype` at the loss boundary
    for mixed-precision robustness — these casts handle different
    invariants. See Piece 3 design doc for the layered framing.
    """

    def __init__(
        self,
        SEQ_LENGTH,
        text_key,
        label_keys,
        tokenizer=None,
        endpoint_model=False,
        target_dtype="float32",
    ):
        # --- Stash configuration ---------------------------------
        # `label_keys` is a dict[str, str]: output_dict_key -> source_column_name.
        # The output_dict_key must match either the corresponding `keras.Input`
        # name (endpoint mode) or the model output name (standard mode); the
        # preprocessor doesn't enforce that — the model-builder side has to
        # agree on the convention.
        self.SEQ_LENGTH = SEQ_LENGTH
        self.text_key = text_key
        self.label_keys = label_keys
        self.endpoint_model = endpoint_model
        self.target_dtype = target_dtype

        # --- Tokenizer + packer (unchanged from prior version) ---
        if tokenizer is None:
            self.tokenizer = keras_hub.tokenizers.RobertaTokenizer.from_preset(
                "roberta_base_en",
            )
        else:
            self.tokenizer = tokenizer
        self.packer = keras_hub.layers.StartEndPacker(
            sequence_length=self.SEQ_LENGTH,
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            return_padding_mask=True,
        )

    def __call__(self, inputs):
        """
        inputs: dict yielded by tf.data.Dataset (e.g.,
            {"text": <tensor>, "cca_label": <tensor>, "immig_label": <tensor>, ...}).

        Returns:
            - endpoint mode (`endpoint_model=True`): single dict
                {"token_ids": ..., "padding_mask": ..., <out_key>: <cast target>, ...}
                where <out_key>/<cast target> entries come from `label_keys`.
            - standard mode (`endpoint_model=False`): tuple
                ({"token_ids": ..., "padding_mask": ...},
                 {<out_key>: <cast target>, ...})
        """
        outputs = self.tokenizer(inputs[self.text_key])
        outputs = self.packer(outputs)
        outputs = {"token_ids": outputs[0], "padding_mask": outputs[1]}

        targets = dict()

        for out_key, source_col in self.label_keys.items():
            targets[out_key] = keras.ops.cast(
                inputs[source_col], dtype=self.target_dtype
            )

        if not self.endpoint_model:
            return (outputs, targets)
        else:
            return {**outputs, **targets}


class CustomPreprocessor:
    def __init__(self, SEQ_LENGTH, MASK_RATE, PREDICTIONS_PER_SEQ, tokenizer=None):
        self.SEQ_LENGTH = SEQ_LENGTH
        self.MASK_RATE = MASK_RATE
        self.PREDICTIONS_PER_SEQ = PREDICTIONS_PER_SEQ
        if tokenizer is None:
            self.tokenizer = keras_hub.tokenizers.RobertaTokenizer.from_preset(
                "roberta_base_en",  # also, they pass SEQ_LENGTH to their tokenizer, but I don't think it matters, since I'm packing next
            )
        else:
            self.tokenizer = tokenizer
        self.packer = keras_hub.layers.StartEndPacker(
            sequence_length=self.SEQ_LENGTH,
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            return_padding_mask=True,  # TESTING
            # return_padding_mask = False,
        )
        if MASK_RATE is None and PREDICTIONS_PER_SEQ is None:
            self.masker = None
        else:
            self.masker = keras_hub.layers.MaskedLMMaskGenerator(
                vocabulary_size=self.tokenizer.vocabulary_size(),
                mask_selection_rate=self.MASK_RATE,
                mask_selection_length=self.PREDICTIONS_PER_SEQ,
                mask_token_id=self.tokenizer.token_to_id("<mask>"),
                unselectable_token_ids=[
                    self.tokenizer.start_token_id,
                    self.tokenizer.end_token_id,
                    self.tokenizer.pad_token_id,
                ],
            )

    def __call__(self, inputs):
        inputs = self.tokenizer(inputs)
        inputs = self.packer(inputs)
        # allow for preprocessing for non-MLM models/tasks
        if self.masker is None:
            return {"token_ids": inputs[0], "padding_mask": inputs[1]}
        outputs = self.masker(
            inputs[0]
        )  # taking just the token ids, not the padding mask
        # Split the masking layer outputs into a (features, labels, and weights)
        # tuple that we can use with keras.Model.fit().
        features = {
            "token_ids": outputs["token_ids"],
            "padding_mask": inputs[1],
            "mask_positions": outputs["mask_positions"],
        }
        labels = outputs["mask_ids"]
        weights = outputs["mask_weights"]
        return features, labels, weights


# class CustomPreprocessor():
#     def __init__(self, tokenizer, packer, masker = None):
#         self.tokenizer = tokenizer
#         self.packer = packer
#         self.masker = masker
#     def __call__(self, inputs):
#         inputs = self.tokenizer(inputs)
#         inputs = self.packer(inputs)
#         # allow for preprocessing for non-MLM models/tasks
#         if self.masker is None:
#             return({"token_ids": inputs[0],
#                     "padding_mask": inputs[1]})
#         outputs = self.masker(inputs[0]) # taking just the token ids, not the padding mask
#         # Split the masking layer outputs into a (features, labels, and weights)
#         # tuple that we can use with keras.Model.fit().
#         features = {
#             "token_ids": outputs["token_ids"],
#             "padding_mask": inputs[1],
#             "mask_positions": outputs["mask_positions"],
#         }
#         labels = outputs["mask_ids"]
#         weights = outputs["mask_weights"]
#         return features, labels, weights
