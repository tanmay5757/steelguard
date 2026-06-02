import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import IMG_H, IMG_W

train_tfms = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.HorizontalFlip(p=0.5),
    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),  # replaces ShiftScaleRotate
    A.OneOf([
        A.GridDistortion(p=1.0),
        A.ElasticTransform(alpha=1, sigma=10, p=1.0),
    ], p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, p=0.3),
    A.GaussNoise(std_range=(0.01, 0.04), p=0.3),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(8, 24),
                    hole_width_range=(8, 48), p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_tfms = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225)),
    ToTensorV2()
])