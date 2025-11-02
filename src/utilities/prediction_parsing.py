import re
import numpy as np
import keras
import tensorflow as tf
import torch


def get_predicted_tokens(preds, tokenizer, as_list=False):
    # Whether we've got logits or probabilities, we just get argmax
    pred_token_ids = keras.ops.argmax(preds, axis=-1)
    if keras.backend.backend() == "torch":
        pred_token_ids = pred_token_ids.detach().cpu().tolist()
    # by default, we just detokenize the predictions, outputting a single string per set of predictions
    if not as_list:
        pred_tokens = tokenizer.detokenize(pred_token_ids)
    # if as_list == True, then we instead turn each set of predictions into a set (list) or tokens
    # note that we do this by grabbing the vocabulary as a list and then using the predictions as
    # indices. this is *way* faster than repeated calls to .id_to_token() and is less complicated
    # (in terms of ensuring type compatibility) than using .id_to_token_map[]
    elif as_list:
        pred_tokens = get_tokens_as_list(pred_token_ids, tokenizer=tokenizer)
        # tmp_vocab = list(tokenizer.get_vocabulary())
        # pred_tokens_raw = tf.gather(tmp_vocab, pred_token_ids)
        # pred_tokens = []
        # for i in pred_tokens_raw.numpy():
        #     pred_tokens.append([s.decode("utf-8") for s in i])
    return pred_tokens


def get_tokens_as_list(token_ids, tokenizer):
    if keras.backend.backend() == "torch":
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
    tmp_vocab = list(tokenizer.get_vocabulary())
    tokens_raw = tf.gather(tmp_vocab, token_ids)
    tokens = []
    for i in tokens_raw.numpy():
        tokens.append([s.decode("utf-8") for s in i])
    return tokens


def preds_in_context(preds, input_batch, tokenizer):
    # Whether we've got logits or probabilities, we just get argmax
    pred_token_ids = keras.ops.argmax(preds, axis=-1)
    pred_tokens = get_tokens_as_list(pred_token_ids, tokenizer=tokenizer)
    ans_tokens = get_tokens_as_list(input_batch[1], tokenizer=tokenizer)
    predictions = []
    answers = []
    zero_token = tokenizer.id_to_token(
        0
    )  # assuming that 0 is the default "masked id" given when nothing is masked
    for i, j in zip(ans_tokens, pred_tokens):
        tmp = [[a, b] for a, b in zip(i, j) if a != zero_token]
        a, b = zip(*tmp)
        answers.append(list(a))
        predictions.append(list(b))
    input_seq = tokenizer.detokenize(input_batch[0]["token_ids"])
    for i in range(len(input_seq)):
        pad_start = re.search(
            "<pad>", input_seq[i]
        )  # for some reason python puts pattern first
        if pad_start is None:
            pass
        else:
            input_seq[i] = input_seq[i][0 : pad_start.start()]
    return {"input_seq": input_seq, "predictions": predictions, "answers": answers}

