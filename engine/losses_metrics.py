import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    POS_WEIGHT,
    BCE_WEIGHT,
    DICE_WEIGHT,
    PRED_THRESHOLD
)
def dice_loss(pred, target, smooth=1.0):
    pred  = torch.sigmoid(pred)
    # Weight defect pixels more — they're rare and matter most
    weight = 1.0 + 7.0 * target
    inter  = (weight * pred * target).sum(dim=(1, 2, 3))
    union  = (weight * (pred + target)).sum(dim=(1, 2, 3))
    dice   = (2.0 * inter + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def bce_loss(pred, target):
    return F.binary_cross_entropy_with_logits(
        pred, target,
        pos_weight=torch.tensor(POS_WEIGHT, device=pred.device)
    )


class CombinedLoss(nn.Module):
    def forward(self, pred, target):
        return BCE_WEIGHT * bce_loss(pred, target) + DICE_WEIGHT * dice_loss(pred, target)




def dice_score(pred_logits, target, threshold=PRED_THRESHOLD):
    """Batch-level mean Dice. Used for validation reporting."""
    with torch.no_grad():
        pred  = (torch.sigmoid(pred_logits) > threshold).float()
        inter = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice  = (2.0 * inter + 1e-6) / (union + 1e-6)
    return dice.mean().item()


def recall_score_batch(pred_logits, target, threshold=PRED_THRESHOLD):
    """What fraction of real defect pixels did we catch?"""
    with torch.no_grad():
        pred = (torch.sigmoid(pred_logits) > threshold).float()
        tp   = (pred * target).sum()
        fn   = ((1 - pred) * target).sum()
    return (tp / (tp + fn + 1e-6)).item()

__all__ = [
    "dice_loss",
    "bce_loss",
    "CombinedLoss",
    "dice_score",
    "recall_score_batch"
]