import numpy as np
import keras
from keras import layers
import keras_hub
import tensorflow as tf
import warnings

from keras import ops

dapt_model = keras.saving.load_model(f"{path_prefix}/dapt_current_model.keras")
dapt_model.layers[2].save_weights(f"{path_prefix}/dapt_backbone.keras") # double check that I'm saving the correct layer

backbone = keras.saving.load_model(f"{path_prefix}/dapt_backbone.keras", compile = False)

def classifier_from_dapt_checkpoint(backbone_path, full_model_path = None):
    if backbone_path is None: 
         dapt_model = keras.saving.load_model(full_model_path)
         backbone = dapt_model.layers[2]
    else: 
        backbone = keras.saving.load_model(backbone_path)
    # To create the binary classification head, I'm just implementing the classification layers 
    # from the keras_hub.RobertaTextClassifier implementation
    
    # Parameters
    dropout = .1 # they default to 0, so I follow *Deep Learning with Python*, chapter 15, here
    num_classes = 1 # currently assuming we'll always want binary here
    
    # Initialize the layers
    pooled_dropout = keras.layers.Dropout(
        dropout,
        name="pooled_dropout",
    )
    hidden_dim = hidden_dim or backbone.hidden_dim
    pooled_dense = keras.layers.Dense(
        backbone.hidden_dim,
        activation="relu", # keras_hub defaults to tanh here, I'm following *Deep Learning with Python* for the moment. Maybe consider gelu?
        name="pooled_dense",
    )
    output_dropout = keras.layers.Dropout(
        dropout,
        name="output_dropout",
    )
    output_dense = keras.layers.Dense(
        num_classes,
        kernel_initializer=roberta_kernel_initializer(),
        activation=None, # or sigmoid (binary classification) or softmax (multiple classes) if I want probabilities instead of logits. 
        name="logits",
    )

    # Create the model
    inputs = backbone.input
    # use the hidden representation of the first token, as done in the keras_hub.RobertaTextClassifier implementation
    # and recommended in *Deep Learning with Python*, chapter 15
    x = backbone(inputs)[:, backbone.start_token_index, :] # backbone.start_token_index is just 0
    x = self.pooled_dropout(x)
    x = self.pooled_dense(x)
    x = self.output_dropout(x)
    outputs = self.output_dense(x)
    classifier = keras.Model(inputs, outputs)

    return classifier



class FLPULoss(keras.losses.Loss): 
    '''
    A non-negative PU learning implementation of the focal loss. 
    
    positive class prior * mean loss of positive samples + 
        max(0 | mean loss of unlabeled samples assuming they're negative - 
            positive class prior * mean loss of positive samples *assuming they're negative*)

    Note: it appears that @Kiryo2017 allow some flexibility in the actual implementation of nnPU, 
    parameterized by nn_beta between 0 and the max possible predicted value (I think) and nn_gamma, 
    between 0 and 1. nn_beta sets the strictness of the bound: instead of a max(0, Ru), we ask whether
    Ru is >= -nn_beta. Thus, if nn_beta = 0, we have a strict non-negative bound. 

    (nn_gamma is a scaling factor that discounts the contribution of Ru to the loss in cases where it 
    falls outside the bound (so you take a step of size nn_gamma * step_size along the gradient, rather 
    than just step_size).) - I actually think this is wrong, see below.

    If Ru is outside the bound, then the nnPU estimator tries to actively claw our way back from overfitting
    by taking a step in the *opposite direction* indicated by the Ru part of the loss and ignoring the positive
    piece entirely. nn_gamma scales the size of that step (as a fraction of a standard step). I'm not 100% sure
    that this is what's going on, since @Kiryo2017 is not super explicit about it, but it seems to be what they
    lay out and what they've implemented. 

    I'm also not totally sure whether we'd want the explicit overfitting walk-back if we're using this in 
    conjunction with ALUM. 
    '''
    def __init__(self, prior, focal_alpha = .25, focal_gamma = 2, nn_beta = 0, nn_gamma = 1)
        super().__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.nn_beta = nn_beta
        self.nn_gamma = nn_gamma
        if self.focal_alpha is not None: 
            self.apply_class_balancing = True
        self.focal_loss = keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing = self.apply_class_balancing, 
            alpha = self.focal_alpha, 
            gamma = self.focal_gamma, 
            from_logit = True, 
            reduction = None, # pretty sure about this
            )
        # Some useful constants
        self.positive = 1 # value of positive labels
        self.unlabeled = 0 # value of unlabeled labels
        self.min_count = 1

    def call(self, y_true, y_pred): 
        # For PU, we need to treat labeled and unlabeled cases differently
        # So we separate them by creating two boolean masks, then to integers
        positive, unlabeled = y_true == self.positive, y_true == self.unlabeled
        positive, unlabeled = ops.cast(positive, dtype = "int32"), ops.cast(unlabeled, dtype = "int32")
        # we cast to int so that we can elementwise set values of the loss to 0 (via multiplication)
        # we do this rather than boolean masking because I *think* this is maybe more efficient, in
        # terms of memory, assignment operations, etc. (I'm not actually sure about this, but I've seen
        # it done this way in other implementations (of similar things) that I've looked at).

        # Now count positive and negative samples; since we're using these for division, we set a minimum of 1
        n_positive, n_unlabeled = ops.minimum(ops.sum(positive), self.min_count), ops.minimum(ops.sum(unlabeled), self.min_count)

        # Now we calculate the losses for each subgroup (note: reduction = None for self.focal_loss has been set above)
        # We actually calculate for all inputs for each one, but then we zero out the out-of-group results
        y_positive = self.focal_loss(y_true, y_pred) * positive #  error for positive samples w/r/t positive ground truth
        y_positive_inv = self.focal_loss(ops.abs(y_true - 1), y_pred) * positive # error for positive samples assuming negative ground truth
        y_unlabeled = self.focal_loss(y_true, y_pred) * unlabeled # error for unlabeled samples assuming negative ground truth

        positive_risk = self.prior * ops.sum(y_positive) / n_positive
        negative_risk = ops.sum(y_unlabeled) / n_unlabeled - self.prior * ops.sum(y_positive_inv) / n_positive

        if negative_risk < -self.nn_beta:
            return -self.nn_gamma * negative_risk 
            # I'm pretty sure this is right, even if it seemed off at first glance
            # in particular, I think the idea is that putting the gamma here is equivalent to 
            # to having it directly scale step size. Also, I'm not 100% on the reason that 
            # the whole thing is negative, but that's correct to their paper and implementation. 
            # I *think* that we make the whole thing negative (and omit the positive risk part
            # of the loss) because the point here is to try and compensate for the overfitting
            # that's happening and actually claw our way back out of it, I think.

        return positive_risk + negative_risk


# backbone -> 


# Create the model
inputs = backbone.input
# use the hidden representation of the first token, as done in the keras_hub.RobertaTextClassifier implementation
# and recommended in *Deep Learning with Python*, chapter 15
encoded_representation = backbone(inputs)[:, backbone.start_token_index, :] # backbone.start_token_index is just 0
x = self.pooled_dropout(x)
x = self.pooled_dense(x)
x = self.output_dropout(x)
outputs = self.output_dense(x)
classifier = keras.Model(inputs, outputs)



class ALUMLayer(layers.Layer): 
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        self.add_loss()



class ALUMLoss(torch.nn.Module): 
    # reference implementations: https://github.com/lyakaap/VAT-pytorch (Virtual Adversarial Training)
    #                            https://github.com/9310gaurav/virtual-adversarial-training
    #                            https://github.com/namisan/mt-dnn/tree/v0.2/alum
    #                            https://discuss.pytorch.org/t/efficient-backprop-for-virtual-adversarial-training/3387
    #                            https://kevinmusgrave.github.io/pytorch-adapt/docs/layers/vat_loss/

    def __init__(self, inner_loss, noise_var = 8e-3, adv_step_size = 5e-5): 
        super().__init__()
        self.inner_loss = inner_loss
        self.noise_var = noise_var
        self.adv_step_size = adv_step_size

    def forward(self, model, sample_embedding, mask, inputs, targets):
        
        # We start by creating adversarial sample 1 (the "range finder") by adding a random disturbance to the sample_embedding
        # NOTE: we are copying as much as possible directly from https://github.com/namisan/mt-dnn/blob/v0.2/alum/adv_masked_lm.py
        noise = sample_embedding.new(sample_embedding.size()).normal_(0, 1) * self.noise_var
        noise.requires_grad_()
        range_finding_sample = sample_embedding.detach() + noise # adv sample 1
        range_finding_logits = model(inputs_embeds = range_finding_sample, attention_mask = mask, inputs_embeds_output = True) # adv logits 1
        range_finding_loss = self.binary_kl_div(range_finding_logits, inputs.detach(), 
                                                logit_input = True, logit_target = True, 
                                                reduction = "batchmean")

        range_finding_loss.backward()

        norm = noise.grad.norm()

        if (torch.isnan(norm) or torch.isinf(norm)):
            loss = self.inner_loss(inputs.squeeze(), targets)
        else: 
            noise = self.adv_project(noise.detach() + noise.grad * self.adv_step_size, norm_type = "l2", eps = 1e-6)
            # 1e-6 is their default value for "eps" in their method; in the call I'm replacing they set it to self.args.noise_gamma
            # whose value I can't find, so it's possible it should be different in this particular case, but we're rolling with it for now. 

            model.zero_grad()

            adv_sample = sample_embedding.detach() + noise # adv sample 2
            adv_sample = adv_sample.detach() # not totally sure what this does here (and why it's not used above)

            adv_logits = model(inputs_embeds = adv_sample, attention_mask = mask, inputs_embeds_output = True) # adv logits 2

            # @Liu2020 use "symmetric KL" here in their code (i.e., KL going in both directions, added together)
            # @Ji2023 make no mention of this, but I'm gonna follow along with @Liu2020
            # adv_loss_f = self.KL(adv_logits.squeeze(), inputs.squeeze().detach())
            # adv_loss_b = self.KL(inputs.squeeze().detach(), adv_logits.squeeze().detach())
            # adv_loss = (adv_loss_f + adv_loss_b)
            # actually, trying out just getting the divergence between adv_logits and logits
            adv_loss = self.binary_kl_div(adv_logits, inputs.detach(), 
                                          logit_input = True, logit_target = True, 
                                          reduction = "batchmean")
            # **Now that I've (hopefully) fixed the KL divergence, should I try going back to reduction = "sum" and/or the symmetric version?
            loss = self.inner_loss(inputs.squeeze(), targets)
            # print(f"Base loss: {loss}") # DEBUG
            # print(f"Range finding loss: {range_finding_loss}") # DEBUG
            # print(f"Adv loss: {adv_loss}") # DEBUG
            loss = loss + adv_loss
            # I'm not sure the detaches and stuff all wind up working correctly here (in particular the inputs.squeeze().detach() for adv_loss_b)

        return loss

    def binary_kl_div(self, inputs, targets, logit_input = True, logit_target = False, reduction = "batchmean"):
        # Note: KL Divergence wants its input to be *probability distributions*. Since we are doing binary 
        # classification, we only have one value per distribution (since, e.g.,  P(x = 1) = .6 implies that 
        # P(x = 0) = .4). So this function first turns the input tensors into full distributions, then uses
        # those to get the KL divergence

        if logit_input: 
            inputs = F.sigmoid(inputs)
        if logit_target: 
            targets = F.sigmoid(targets)

        inputs = torch.column_stack((inputs, 1 - inputs))
        targets = torch.column_stack((targets, 1 - targets))

        loss = F.kl_div(inputs.log(), targets, reduction=reduction, log_target = False)
        return loss

    def adv_project(self, grad, norm_type='inf', eps=1e-6):
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction





