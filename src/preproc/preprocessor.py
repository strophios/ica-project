import keras_hub


class CustomPreprocessor(): 
    def __init__(self, SEQ_LENGTH, MASK_RATE, PREDICTIONS_PER_SEQ, tokenizer = None):
        self.SEQ_LENGTH = SEQ_LENGTH
        self.MASK_RATE = MASK_RATE
        self.PREDICTIONS_PER_SEQ = PREDICTIONS_PER_SEQ
        if tokenizer is None: 
            self.tokenizer = keras_hub.tokenizers.RobertaTokenizer.from_preset(
                "roberta_base_en", # also, they pass SEQ_LENGTH to their tokenizer, but I don't think it matters, since I'm packing next
                )
        else: 
            self.tokenizer = tokenizer
        self.packer = keras_hub.layers.StartEndPacker(
            sequence_length = self.SEQ_LENGTH, 
            start_value = self.tokenizer.start_token_id, 
            end_value = self.tokenizer.end_token_id,
            pad_value = self.tokenizer.pad_token_id,
            return_padding_mask = True, # TESTING
            # return_padding_mask = False,
            )
        if MASK_RATE is None and PREDICTIONS_PER_SEQ is None: 
            self.masker = None
        else: 
            self.masker = keras_hub.layers.MaskedLMMaskGenerator(
                vocabulary_size = self.tokenizer.vocabulary_size(),
                mask_selection_rate = self.MASK_RATE,
                mask_selection_length = self.PREDICTIONS_PER_SEQ,
                mask_token_id = self.tokenizer.token_to_id("<mask>"), 
                unselectable_token_ids = [self.tokenizer.start_token_id, 
                                          self.tokenizer.end_token_id,
                                          self.tokenizer.pad_token_id],
                                          )
    def __call__(self, inputs):
        inputs = self.tokenizer(inputs)
        inputs = self.packer(inputs)
        # allow for preprocessing for non-MLM models/tasks
        if self.masker is None:
            return({"token_ids": inputs[0], 
                    "padding_mask": inputs[1]})
        outputs = self.masker(inputs[0]) # taking just the token ids, not the padding mask
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