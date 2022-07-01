import torch.nn as nn
import torch.nn.functional as F
import torch


class FocalLoss(nn.Module):
    """Implementation of Focal Loss"""

    def __init__(self, weight=None, label_smoothing=None, gamma=2, alpha=0.25, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.weighted_cs = nn.CrossEntropyLoss(weight=weight, reduction="none", label_smoothing=label_smoothing)
        self.cs = nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predicted, target):
        """
        predicted: [batch_size, n_classes]
        target: [batch_size]
        """
        pt = 1 / torch.exp(self.cs(predicted, target))
        entropy_loss = self.weighted_cs(predicted, target)
        focal_loss = ((1 - pt) ** self.gamma) * entropy_loss

        # if self.alpha >= 0:
        #     alpha_t = self.alpha * pt + (1 - self.alpha) * (1 - pt)
        #     focal_loss = alpha_t * focal_loss

        if self.reduction == "none":
            return focal_loss
        elif self.reduction == "mean":
            return focal_loss.mean()
        else:
            return focal_loss.sum()
