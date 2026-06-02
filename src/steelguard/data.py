from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def rle_decode(mask_rle: str, shape: tuple[int, int]) -> np.ndarray:
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def imagenet_norm(image: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (image - mean) / std


@dataclass
class Sample:
    image_path: Path
    label: int
    mask: np.ndarray | None = None


class SteelClassificationDataset(Dataset):
    def __init__(self, samples: list[Sample], size: int = 256, augment: bool = False):
        self.samples = samples
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = cv2.imread(str(sample.image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size))
        if self.augment and np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
        image = image.astype(np.float32) / 255.0
        image = imagenet_norm(image)
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        label = torch.tensor(sample.label, dtype=torch.float32)
        return image, label


class SteelSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        size: int = 256,
        mask_loader: Callable[[Sample], np.ndarray] | None = None,
    ):
        self.samples = samples
        self.size = size
        self.mask_loader = mask_loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = cv2.imread(str(sample.image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size))
        image = image.astype(np.float32) / 255.0
        image = imagenet_norm(image)
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)

        if sample.mask is not None:
            mask = sample.mask
        elif self.mask_loader is not None:
            mask = self.mask_loader(sample)
        else:
            mask = np.zeros((self.size, self.size), dtype=np.uint8)
        mask = cv2.resize(mask.astype(np.uint8), (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = torch.tensor(mask[None, :, :], dtype=torch.float32)
        return image, mask


def build_severstal_samples(dataset_root: Path, split: str = "train", test_size: float = 0.15) -> list[Sample]:
    csv_path = dataset_root / "train.csv"
    image_dir = dataset_root / "train_images"
    df = pd.read_csv(csv_path)
    if "ImageId_ClassId" in df.columns:
        df[["ImageId", "ClassId"]] = df["ImageId_ClassId"].str.split("_", expand=True)
    elif not {"ImageId", "ClassId"}.issubset(df.columns):
        raise ValueError("Unsupported Severstal CSV format. Expected 'ImageId_ClassId' or 'ImageId,ClassId'.")
    grouped = df.groupby("ImageId")["EncodedPixels"].apply(lambda x: [v for v in x if isinstance(v, str)]).reset_index()
    grouped["label"] = grouped["EncodedPixels"].apply(lambda x: 1 if len(x) > 0 else 0)

    train_df, val_df = train_test_split(
        grouped, test_size=test_size, random_state=42, stratify=grouped["label"]
    )
    source = train_df if split == "train" else val_df
    return [Sample(image_path=image_dir / row.ImageId, label=int(row.label)) for row in source.itertuples()]


def build_neu_samples(dataset_root: Path, split: str = "train") -> list[Sample]:
    img_root = dataset_root / split / "images"
    samples: list[Sample] = []
    for class_dir in img_root.iterdir():
        if not class_dir.is_dir():
            continue
        for image_path in class_dir.glob("*.jpg"):
            label = 1 if class_dir.name in {"crazing", "crack", "scratches"} else 0
            samples.append(Sample(image_path=image_path, label=label))
    return samples


def load_neu_mask_from_xml(dataset_root: Path, split: str, sample: Sample, size_hint: tuple[int, int]) -> np.ndarray:
    ann_root = dataset_root / split / "annotations"
    xml_path = ann_root / f"{sample.image_path.stem}.xml"
    mask = np.zeros(size_hint, dtype=np.uint8)
    if not xml_path.exists():
        return mask
    root = ET.parse(xml_path).getroot()
    for box in root.findall(".//bndbox"):
        xmin = int(float(box.findtext("xmin", "0")))
        ymin = int(float(box.findtext("ymin", "0")))
        xmax = int(float(box.findtext("xmax", "0")))
        ymax = int(float(box.findtext("ymax", "0")))
        mask[ymin:ymax, xmin:xmax] = 1
    return mask
