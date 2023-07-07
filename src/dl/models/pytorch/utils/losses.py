# https://github.com/CuriousAI/mean-teacher/blob/546348ff863c998c26be4339021425df973b4a36/pytorch/mean_teacher/losses.py#L15
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes

def get_losses(scale, smooth, margin, args):
    """
    Getter for the losses.
    Args:
        scale: Scaler that was used, e.g. normalizer or binarize
        smooth: Parameter for label_smoothing
        margin: Parameter for the TripletMarginLoss

    Returns:
        sceloss: CrossEntropyLoss (with label smoothing)
        celoss: CrossEntropyLoss object (without label smoothing)
        mseloss: MSELoss object
        triplet_loss: TripletMarginLoss object
    """
    if args.classif_loss == 'ce':
        sceloss = nn.CrossEntropyLoss(label_smoothing=smooth)
    elif args.classif_loss == 'cosine':
        sceloss = nn.CosineSimilarity()
    celoss = nn.CrossEntropyLoss()
    if args.rec_loss == 'mse':
        mseloss = nn.MSELoss()
    elif args.rec_loss == 'l1':
        mseloss = nn.L1Loss()
    if scale == "binarize":
        mseloss = nn.BCELoss()

    if args.dloss == 'revTriplet':
        triplet_loss = nn.TripletMarginLoss(margin, p=2, swap=True)
    else:
        triplet_loss = nn.TripletMarginLoss(0, p=2, swap=False)

    return sceloss, celoss, mseloss, triplet_loss
