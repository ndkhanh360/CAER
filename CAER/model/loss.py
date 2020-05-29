import torch.nn.functional as F


def cross_entropy(output, target):
    return F.cross_entropy(output, target)
