import torch.nn.functional as F
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)

def CrossEntropyLoss(output, target):
    loss_func = nn.CrossEntropyLoss()
    return loss_func(output, target)