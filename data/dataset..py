import numpy as np
import pandas as pd
from config import IMG_H

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
