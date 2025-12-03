"""
Convenience functions and classes for setting up a classification model.
"""

import keras
import keras_hub


def classifier_from_dapt_checkpoint(backbone_path, full_model_path=None):
    if backbone_path is None:
        dapt_model = keras.saving.load_model(full_model_path)
        backbone = dapt_model.layers[2]
    else:
        backbone = keras_hub.models.Backbone.from_preset(
            "roberta_base_en", preprocessor=None, load_weights=False
        )
        backbone.load_weights(backbone_path)
    # To create the binary classification head, I'm just implementing the classification layers
    # from the keras_hub.RobertaTextClassifier implementation
    # Parameters
    dropout = 0.1  # they default to 0, so I follow *Deep Learning with Python*, chapter 15, here
    num_classes = 1  # currently assuming we'll always want binary here
    # Initialize the layers
    pooled_dropout = keras.layers.Dropout(
        dropout,
        name="pooled_dropout",
    )
    pooled_dense = keras.layers.Dense(
        backbone.hidden_dim,
        activation="relu",  # keras_hub defaults to tanh here, I'm following *Deep Learning with Python* for the moment. Maybe consider gelu?
        name="pooled_dense",
    )
    output_dropout = keras.layers.Dropout(
        dropout,
        name="output_dropout",
    )
    output_dense = keras.layers.Dense(
        num_classes,
        activation=None,  # or sigmoid (binary classification) or softmax (multiple classes) if I want probabilities instead of logits.
        name="logits",
    )
    # Create the model
    inputs = backbone.input
    # use the hidden representation of the first token, as done in the keras_hub.RobertaTextClassifier implementation
    # and recommended in *Deep Learning with Python*, chapter 15
    x = backbone(inputs)[
        :, backbone.start_token_index, :
    ]  # backbone.start_token_index is just 0
    x = pooled_dropout(x)
    x = pooled_dense(x)
    x = output_dropout(x)
    outputs = output_dense(x)
    classifier = keras.Model(inputs, outputs)
    return classifier
