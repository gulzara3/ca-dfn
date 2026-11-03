"""
Global configuration for the CA-DFN framework.

All hyperparameters reported in the paper are defined here so that every
experiment is fully reproducible from a single source of truth.
"""
import os
import torch


class CFG:
    # ---- Data ----
    DATA_DIR = "./data"
    NUM_CLASSES = 101
    IMG_SIZE = 224

    # ---- Base-model training ----
    EPOCHS = 50
    BATCH_SIZE = 32
    ACCUM_STEPS = 2          # effective batch size = 64
    LR = 5e-4                # classification-head LR (backbone uses LR * 0.1)
    WD = 0.02
    LABEL_SMOOTH = 0.1
    WARMUP_EP = 5
    WORKERS = 2

    # ---- Augmentation ----
    CUTMIX_ALPHA = 1.0
    MIXUP_ALPHA = 0.8
    CUTMIX_PROB = 0.5

    # ---- CA-DFN fusion network ----
    FUSION_EP = 40
    FUSION_LR = 3e-4
    FUSION_HIDDEN = 512
    FUSION_TEMP = 3.0        # tau   (attention temperature)
    FUSION_ENT_W = 0.3       # lambda (entropy-regularisation weight)
    FUSION_IDROP = 0.25      # input-model dropout probability

    # ---- Curriculum knowledge distillation ----
    KD_EP = 40
    KD_TEMP = 4.0            # T
    KD_ALPHA = 0.7           # alpha

    # ---- Reproducibility ----
    SEEDS = [42, 123, 456]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Output paths ----
    CKPT = "./checkpoints"
    OUT = "./results"


# ---- Model registry ----
MODEL_NAMES = ["resnet50", "efficientnet_b4", "densenet121",
               "convnext_tiny", "swin_small"]

DISPLAY = {
    "resnet50": "ResNet-50",
    "efficientnet_b4": "EfficientNet-B4",
    "densenet121": "DenseNet-121",
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_small": "Swin-Small",
}

MODEL_TYPE = {
    "resnet50": "CNN",
    "efficientnet_b4": "CNN",
    "densenet121": "CNN",
    "convnext_tiny": "CNN",
    "swin_small": "Transformer",
}

# Architecture-type ids used by the CA-DFN embedding layer (CNN=0, ViT=1)
ARCH_TYPES = [0 if MODEL_TYPE[n] == "CNN" else 1 for n in MODEL_NAMES]


def seed_everything(seed: int) -> None:
    """Make a run fully deterministic for the given seed."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs() -> None:
    os.makedirs(CFG.CKPT, exist_ok=True)
    os.makedirs(CFG.OUT, exist_ok=True)
