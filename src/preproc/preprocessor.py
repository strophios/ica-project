import keras
import keras_hub


def _is_string_dtype(dtype) -> bool:
    """Robustly check whether a tensor dtype is string/bytes-typed.

    Handles both TensorFlow's `tf.dtypes.DType` (which has an `is_string`
    attribute) and Keras's dtype-string convention. Used by the Layer-2
    text-column check in `ClassifierPreprocessor.__call__`.
    """
    if hasattr(dtype, "is_string"):
        return bool(dtype.is_string)
    # Fallback: stringly-typed dtype (e.g., a numpy or string-name dtype).
    return str(dtype) in {"string", "<U", "bytes", "object"} or "string" in str(dtype).lower()


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
        # ============================================================
        # Boundary 1: construction-time validation (Tier 3 Piece 1)
        # ============================================================
        # Defense-in-depth: __init__ checks for *internal-config-validity*
        # bugs — bugs visible from the constructor arguments alone,
        # without needing to see actual data. __call__ has its own
        # check (Boundary 2) for config-vs-data mismatches that
        # __init__ can't see. Each boundary catches what the other
        # can't. See `docs/notes/tier3-design.md` Piece 1 for the full
        # framing. Validation block runs *first*, before any heavy
        # setup (tokenizer download, packer construction), so config
        # errors fail fast.

        # `text_key` must be a non-empty string. Catches: forgotten
        # default (text_key=None), empty-string typo, wrong type.
        if not isinstance(text_key, str) or not text_key:
            raise ValueError(
                f"text_key must be a non-empty string; "
                f"got {text_key!r} (type {type(text_key).__name__})."
            )

        # `label_keys` must be a dict. Catches the most plausible
        # typo: hand-rolled list of (key, value) tuples instead of
        # passing a dict literal. Without this check, the failure is
        # an AttributeError on `.items()` deep in __call__.
        if not isinstance(label_keys, dict):
            raise ValueError(
                f"label_keys must be a dict[str, str] mapping output_dict_key "
                f"-> source_column_name; got {type(label_keys).__name__}."
            )

        # `target_dtype` must be a Keras-recognized dtype string.
        # `keras.backend.standardize_dtype` raises on invalid input;
        # we catch and re-raise with a more contextual message naming
        # the bad value. Retires M2 from the Tier 2 review's deferred
        # Tier-4 list — natural home is alongside the other
        # construction-time checks.
        try:
            keras.backend.standardize_dtype(target_dtype)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"target_dtype must be a valid Keras dtype string; "
                f"got {target_dtype!r}. Underlying error: {e}"
            ) from e

        # Business rule: standard mode (endpoint_model=False) emits
        # `(features, targets_dict)`; an empty `targets_dict` has
        # nothing to route via `compile(loss={...})` and is
        # structurally nonsensical. Empty `label_keys` is *only*
        # valid in endpoint mode (predict-only configurations like
        # the eval script's `label_keys={}` pattern — see pinned
        # question #3 for the design smell this exposes and the
        # planned future refactor).
        if not endpoint_model and not label_keys:
            raise ValueError(
                "standard mode (endpoint_model=False) requires non-empty "
                "label_keys; got empty label_keys. Empty label_keys is only "
                "valid in endpoint mode (predict-only configuration)."
            )

        # ============================================================
        # Stash configuration
        # ============================================================
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

        Raises:
            KeyError: if `inputs` is missing `text_key` or any source
                column named in `label_keys`. See Boundary-2 check
                below.
        """
        # ============================================================
        # Boundary 2: call-time input validation (Tier 3 Piece 1)
        # ============================================================
        # __init__ already checked the configuration is internally
        # coherent (Boundary 1). This boundary catches a different
        # shape of bug: configuration-vs-actual-data mismatch — the
        # configured columns aren't in the batch. Without this check,
        # the failure surfaces deep inside the tokenizer / cast call
        # with a stack pointing at TF internals; with it, the failure
        # surfaces at the entry point with an informative message.
        #
        # Enumerate all missing columns in a single check rather than
        # failing fast on the first — diagnostic quality matters more
        # than microseconds in a once-per-dataset trace. The cost is
        # one set construction + one set difference per traced map
        # call (so ~once per dataset construction in normal tf.data
        # use), which is negligible.
        expected_cols = {self.text_key, *self.label_keys.values()}
        available_cols = set(inputs.keys())
        missing = expected_cols - available_cols
        if missing:
            # KeyError (not ValueError) for consistency with the
            # existing failure mode (`inputs[key]` raises KeyError);
            # callers catching KeyError to handle missing-column
            # errors keep working, just with a useful message and an
            # earlier stack location. The exception class signals
            # "the data you fed in is missing a column the
            # configuration expected — fix the dataset pipeline,"
            # distinct from __init__'s ValueError signal of "your
            # configuration is malformed."
            raise KeyError(
                f"ClassifierPreprocessor: input batch is missing required "
                f"column(s) {sorted(missing)}. "
                f"Configured text_key={self.text_key!r}; "
                f"label_keys source columns={sorted(self.label_keys.values())}. "
                f"Available columns in batch: {sorted(available_cols)}."
            )

        # Layer-2 check on the text column's dtype. Tier 3 closeout
        # (addressing I7 from the adversarial review): the column
        # is present, but if it's the wrong dtype (e.g., int64
        # because someone accidentally fed already-tokenized ids
        # instead of raw text), the failure surfaces deep inside
        # the tokenizer with a less helpful TF error. We only check
        # text_key here — source columns get cast via `keras.ops.cast`
        # below, which fails loudly on incompatible dtypes; tokenizer
        # input is the dtype-fragile boundary.
        text_tensor = inputs[self.text_key]
        text_dtype = getattr(text_tensor, "dtype", None)
        if text_dtype is not None and not _is_string_dtype(text_dtype):
            raise TypeError(
                f"ClassifierPreprocessor: column at text_key="
                f"{self.text_key!r} must be string-typed; got "
                f"dtype={text_dtype!r}. The most common cause is "
                f"feeding already-tokenized ids instead of raw "
                f"strings; the tokenizer expects text."
            )

        # ============================================================
        # Tokenize + pack
        # ============================================================
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
