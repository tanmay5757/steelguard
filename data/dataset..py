import numpy as np
import pandas as pd
import os
import random
import cv2

from torch.utils.data import Dataset
from config import DATA_PATH, IMG_H, IMG_W

def rle_decode(mask_rle, shape=(256, 1600)):
    """
    Decode run-length encoding to a binary mask.
    shape = (H, W)  <- consistent (height, width) convention.
    """
    if pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)

    s = list(map(int, mask_rle.split()))
    starts, lengths = s[::2], s[1::2]
    starts = np.array(starts) - 1
    ends   = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape[1], shape[0]).T


def build_mask(image_id, df):
    """Combine all class masks for one image into a single binary mask."""
    rows = df[df["ImageId"] == image_id]
    mask = np.zeros((IMG_H, 1600), dtype=np.uint8)
    for _, row in rows.iterrows():
        if pd.notnull(row["EncodedPixels"]):
            mask |= rle_decode(row["EncodedPixels"])

    return mask

class SteelSegDataset(Dataset):
    def __init__(self, image_ids, df_full, transforms=None, training=False):
        self.image_ids  = image_ids
        self.df         = df_full
        self.transforms = transforms
        self.training   = training

        # Pre-compute which indices have defects — used by BalancedSampler
        self.defect_indices = []
        self.clean_indices  = []
        for i, img_id in enumerate(self.image_ids):
            rows = df_full[df_full["ImageId"] == img_id]
            has_defect = rows["EncodedPixels"].notnull().any()
            if has_defect:
                self.defect_indices.append(i)
            else:
                self.clean_indices.append(i)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_path = os.path.join(DATA_PATH, "train_images", img_id)

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = build_mask(img_id, self.df).astype(np.float32)

        # Defect-centered crop — only during training, only on defective images
        # Crop to IMG_W wide window centered on a random defect pixel
        if self.training and mask.max() > 0 and random.random() < 0.8:
            ys, xs    = np.where(mask > 0)
            pick      = random.randint(0, len(xs) - 1)
            cx, cy    = int(xs[pick]), int(ys[pick])
            x1 = max(0, min(cx - IMG_W // 2, img.shape[1] - IMG_W))
            y1 = max(0, min(cy - IMG_H // 2, img.shape[0] - IMG_H))
            img  = img [y1:y1+IMG_H, x1:x1+IMG_W]
            mask = mask[y1:y1+IMG_H, x1:x1+IMG_W]

        if self.transforms:
            aug  = self.transforms(image=img, mask=mask)
            img  = aug["image"]
            mask = aug["mask"]

        return img, mask.unsqueeze(0).float()


