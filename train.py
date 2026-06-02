import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models.models import ResNetUNet
from data.dataset import SteelSegDataset, BalancedSampler
from data.transforms import train_tfms, val_tfms
from engine.trainer import smart_train
from config import *


df_full = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
print("Columns:", df_full.columns.tolist())
print("CSV Shape:", df_full.shape)

# ── Scan actual directory to include clean images missing from CSV ────────────
all_images = os.listdir(os.path.join(DATA_PATH, "train_images"))
df_all     = pd.DataFrame({"ImageId": all_images})
print(f"Total images in folder: {len(all_images)}")

df_cls = df_full.copy()
df_cls["has_defect"] = df_cls["EncodedPixels"].notnull().astype(int)
df_cls = df_cls.groupby("ImageId")["has_defect"].max().reset_index()

# LEFT JOIN — clean images get has_defect = 0
df_cls = pd.merge(df_all, df_cls, on="ImageId", how="left")
df_cls["has_defect"] = df_cls["has_defect"].fillna(0).astype(int)

print(f"Defective images : {df_cls['has_defect'].sum()}")
print(f"Clean images     : {(df_cls['has_defect'] == 0).sum()}")

train_ids, val_ids = train_test_split(
    df_cls,
    test_size=0.2,
    stratify=df_cls["has_defect"],
    random_state=42
)

train_ids = train_ids["ImageId"].values
val_ids   = val_ids["ImageId"].values
print(f"Train: {len(train_ids)}  Val: {len(val_ids)}")

train_dataset = SteelSegDataset(
    train_ids,
    df_full,
    transforms=train_tfms,
    training=True
)

val_dataset = SteelSegDataset(
    val_ids,
    df_full,
    transforms=val_tfms,
    training=False
)

train_sampler = BalancedSampler(
    train_dataset,
    batch_size=BATCH_SIZE,
    defect_ratio=DEFECT_RATIO_PER_BATCH
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("\nTraining ResNetUNet...")

model = ResNetUNet()

model, history = smart_train(
    model,
    train_loader,
    val_loader,
    save_path="best_resnet_unet.pth"
)