"""Mask-aware H3/T1 pairwise ranking and MSE loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F
PAIRWISE_ALPHA = 0.70

class CrossSectionalPairwiseRankingLoss(nn.Module):
    def __init__(self, alpha=PAIRWISE_ALPHA):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, y_pred, y_true, mask):
        y_pred, y_true, mask = y_pred.view(-1), y_true.view(-1), mask.view(-1)
        valid = (mask > 0).float()

        diff_pred = y_pred[:, None] - y_pred[None, :]
        diff_true = y_true[:, None] - y_true[None, :]
        pair_mask = valid[:, None] * valid[None, :] * (diff_true != 0).float()

        ranking_loss = (F.softplus(-torch.sign(diff_true) * diff_pred) * pair_mask).sum() / (pair_mask.sum() + 1e-8)
        mse_loss = (self.mse(y_pred, y_true) * mask).sum() / (mask.sum() + 1e-8)
        return self.alpha * ranking_loss + (1.0 - self.alpha) * mse_loss

