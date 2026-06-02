# ── Smart Epoch Utilities ────────────────────────────────────────────────────
import os
import torch

import math

from config import (
    DEVICE,
    LR,
    MAX_EPOCHS,
    MIN_EPOCHS,
    EARLY_STOP_PATIENCE,
    PLATEAU_PATIENCE,
    PLATEAU_MIN_DELTA,
    IMG_H,
    IMG_W,
    BATCH_SIZE,
    BCE_WEIGHT,
    DICE_WEIGHT,
    POS_WEIGHT,
    DEFECT_RATIO_PER_BATCH,
)
from tqdm.auto import tqdm

from engine.losses_metrics import (
    CombinedLoss,
    dice_score,
    recall_score_batch,
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
                
def smart_train(model, train_loader, val_loader, save_path="best_model.pth", device=DEVICE):
    model     = model.to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Warmup for 2 epochs then cosine decay — avoids early instability
    def lr_lambda(epoch):
        if epoch < 2:
            return (epoch + 1) / 2   # linear warmup
        progress = (epoch - 2) / max(MAX_EPOCHS - 2, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler  = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE,
                            min_delta=PLATEAU_MIN_DELTA,
                            checkpoint=save_path)
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_recall": []}

    print(f"\n{'='*60}")
    print(f"  IMG: {IMG_H}×{IMG_W}  |  batch: {BATCH_SIZE}  |  LR: {LR}")
    print(f"  Loss: {BCE_WEIGHT}×BCE + {DICE_WEIGHT}×Dice  |  pos_weight: {POS_WEIGHT}")
    print(f"  Balanced sampler: {int(DEFECT_RATIO_PER_BATCH*100)}% defect per batch")
    print(f"{'='*60}\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"── Epoch {epoch}/{MAX_EPOCHS}   LR={cur_lr:.2e} ──")

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, leave=False, desc="  train"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type,
                                    enabled=(device.type == "cuda")):
                out  = model(imgs)
                loss = criterion(out, masks)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = val_dice = val_recall = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, leave=False, desc="  val"):
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                val_loss   += criterion(out, masks).item()
                val_dice   += dice_score(out, masks)
                val_recall += recall_score_batch(out, masks)

        n  = len(val_loader)
        tl = train_loss / len(train_loader)
        vl = val_loss   / n
        vd = val_dice   / n
        vr = val_recall / n

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_dice"].append(vd)
        history["val_recall"].append(vr)

        print(f"  loss={tl:.4f}  val_loss={vl:.4f}  "
              f"val_dice={vd:.4f}  val_recall={vr:.4f}")

        # Scheduler steps once per epoch
        scheduler.step()

        # Early stopping
        if not stopper.step(vd, model) and epoch >= MIN_EPOCHS:
            print(f"\n  Best Val Dice = {stopper.best:.4f}")
            break

    if os.path.exists(save_path):
        model.load_state_dict(
            torch.load(save_path, map_location=device, weights_only=True)
        )
        print(f"\n  Loaded best checkpoint  (Val Dice = {stopper.best:.4f})")

    print(f"\n  Epochs used   : {len(history['val_dice'])} / {MAX_EPOCHS}")
    print(f"  Best Val Dice : {max(history['val_dice']):.4f}")
    return model, history
__all__ = [
    "EarlyStopping",
    "PlateauPruner",
    "EpochPredictor",
    "smart_train"
]