# ── Smart Epoch Utilities ────────────────────────────────────────────────────
import os
import torch

from config import (
    EARLY_STOP_PATIENCE,
    PLATEAU_PATIENCE,
    PLATEAU_MIN_DELTA,
)

class EarlyStopping:
    """Saves best checkpoint; returns False when patience is exhausted."""
    def __init__(self, patience=EARLY_STOP_PATIENCE, min_delta=PLATEAU_MIN_DELTA,
                 checkpoint="best_model.pth"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.checkpoint = checkpoint
        self.best       = -float("inf")
        self.counter    = 0

    def step(self, score, model):
        if score > self.best + self.min_delta:
            self.best    = score
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint)
            return True          # improvement → keep going
        self.counter += 1
        if self.counter >= self.patience:
            return False         # patience exhausted → stop
        return True


class PlateauPruner:
    """Signals a plateau when val_dice barely moves for PLATEAU_PATIENCE epochs."""
    def __init__(self, patience=PLATEAU_PATIENCE, min_delta=PLATEAU_MIN_DELTA):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = -float("inf")
        self.counter   = 0

    def step(self, score):
        if score > self.best + self.min_delta:
            self.best    = score
            self.counter = 0
            return False   # not plateauing
        self.counter += 1
        return self.counter >= self.patience


class EpochPredictor:
    """Loosely estimates how many epochs to hit target_dice (informational only)."""
    def __init__(self, target_dice=0.80, window=5):
        self.target = target_dice
        self.window = window
        self.scores = []

    def update(self, epoch, score):
        self.scores.append(score)
        if len(self.scores) < 2:
            return
        recent = self.scores[-self.window:]
        if len(recent) >= 2:
            slope = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
            if slope > 1e-5 and recent[-1] < self.target:
                remaining = (self.target - recent[-1]) / slope
                print(f"    [EpochPredictor] ~{remaining:.0f} more epochs to "
                      f"reach Dice {self.target:.2f}  (slope={slope:.5f})")
                
__all__ = [
    "EarlyStopping",
    "PlateauPruner",
    "EpochPredictor"
]