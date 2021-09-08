import torch
import torch.nn as nn


def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


class CharbonnierLoss(nn.Module):

    def __init__(self, loss_weight=1.0, eps=1e-12):
        super(CharbonnierLoss, self).__init__()

        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, pred, target, weight=1, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return torch.mean(self.loss_weight * charbonnier_loss(pred, target, eps=self.eps)) 
