import numpy as np
import keras
import keras_hub
import tensorflow as tf
import warnings


def get_DAPT_model(PREDICTIONS_PER_SEQ, weights=None, path=None):
    # Initialize the layers/models we need: the RoBERTa backbone, plus a MaskedLMHead
    backbone = keras_hub.models.Backbone.from_preset(
        "roberta_base_en", preprocessor=None, load_weights=True
    )
    lm_head = keras_hub.layers.MaskedLMHead(
        token_embedding=backbone.token_embedding,
        intermediate_activation="gelu",
        activation="softmax",
    )
    # note: there is no final activation by default, so by default this would output logits
    # Create the model
    inputs = {
        **backbone.input,
        "mask_positions": keras.Input(
            shape=(PREDICTIONS_PER_SEQ,), dtype="int32", name="mask_positions"
        ),
    }

    encoded_tokens = backbone(backbone.input)
    outputs = lm_head(encoded_tokens, mask_positions=inputs["mask_positions"])

    dapt_model = keras.Model(inputs, outputs)
    # Now we load in the pretrained MLM Head weights:
    update_lm_head_weights(dapt_model, weights=weights, path=path)

    return dapt_model


def update_lm_head_weights(model, weights=None, path=None):
    # so, to set the Keras weights, we want:
    # intermediate_dense.kernel = dense.weight # this one needs to be transposed
    # intermediate_dense.bias = dense.bias
    # intermediate_layer_norm.gamma = layer_norm.weight
    # intermediate_layer_norm.beta = layer_norm.bias
    # token_embedding.embedding = leave alone (though its equivalent is decoder.weight, I think)
    # output_bias = decoder.bias *or* bias (at least at present when I look at the model, these are identical)
    # for the overall output bias, I can't figure out a way to do it more "cleanly" (e.g., with .set_weights() or similar)
    # (at least not without custom building an MLM head, which I just don't think is otherwise worth it)
    if weights is None:
        assert path is not None, "Either weights or path must be provided."
        weights = np.load(path, allow_pickle=True).item()
    elif path is None:
        assert weights is not None, "Either weights or path must be provided."
    else:
        warnings.warn(
            "Only one of weights or path should be provided. Proceeding with weights."
        )

    model.layers[4]._bias.assign(
        weights["bias"]
    )  # or "decoder.bias" # (don't need to move to CPU now, technically)
    model.layers[4]._layers[1].set_weights(
        [weights["dense.weight"].transpose(), weights["dense.bias"]]
    )
    model.layers[4]._layers[2].set_weights(
        [weights["layer_norm.weight"], weights["layer_norm.bias"]]
    )
