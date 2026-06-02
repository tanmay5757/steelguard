import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMG_H = 256
IMG_W = 512

BATCH_SIZE = 32

LR = 2e-4

POS_WEIGHT = 8.0
DICE_WEIGHT = 0.7
BCE_WEIGHT = 0.3

PRED_THRESHOLD = 0.91

MAX_EPOCHS = 40
MIN_EPOCHS = 8

EARLY_STOP_PATIENCE = 6
PLATEAU_PATIENCE = 4
PLATEAU_MIN_DELTA = 5e-4

DEFECT_RATIO_PER_BATCH = 0.5

DATA_PATH = "/kaggle/input/competitions/severstal-steel-defect-detection"