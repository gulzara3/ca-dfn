"""
Food-101 data pipeline: transforms, CutMix/MixUp, and dataloaders.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import Food101

from .config import CFG

# ---- Transforms ----
train_tf = T.Compose([
    T.RandomResizedCrop(CFG.IMG_SIZE, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(0.4, 0.4, 0.4, 0.15),
    T.RandAugment(num_ops=2, magnitude=9),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.25),
])

test_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(CFG.IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---- CutMix / MixUp ----
def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, y1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    x2, y2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def apply_cutmix_or_mixup(images, labels, nc):
    """Apply CutMix (prob CUTMIX_PROB) or MixUp, returning soft labels."""
    if np.random.rand() < CFG.CUTMIX_PROB:
        lam = np.random.beta(CFG.CUTMIX_ALPHA, CFG.CUTMIX_ALPHA)
        ri = torch.randperm(images.size(0)).to(images.device)
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        images[:, :, x1:x2, y1:y2] = images[ri, :, x1:x2, y1:y2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
    else:
        lam = np.random.beta(CFG.MIXUP_ALPHA, CFG.MIXUP_ALPHA)
        ri = torch.randperm(images.size(0)).to(images.device)
        images = lam * images + (1 - lam) * images[ri]
    oh = F.one_hot(labels, nc).float()
    return images, lam * oh + (1 - lam) * F.one_hot(labels[ri], nc).float()


class _ValDS(Dataset):
    """Wrap a random_split subset so it uses the test-time transform."""
    def __init__(self, subset, tf):
        self.indices = subset.indices
        self.ds = copy.copy(subset.dataset)
        self.ds.transform = tf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.ds[self.indices[i]]


def build_dataloaders(download=True):
    """Return (train_ld, val_ld, test_ld, train_ds, test_ds, class_names)."""
    full_train = Food101(CFG.DATA_DIR, split="train", transform=train_tf, download=download)
    test_ds = Food101(CFG.DATA_DIR, split="test", transform=test_tf, download=download)

    n_val = int(0.1 * len(full_train))
    n_train = len(full_train) - n_val
    train_ds, val_raw = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    val_ds = _ValDS(val_raw, test_tf)

    train_ld = DataLoader(train_ds, CFG.BATCH_SIZE, shuffle=True,
                          num_workers=CFG.WORKERS, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, CFG.BATCH_SIZE, shuffle=False,
                        num_workers=CFG.WORKERS, pin_memory=True)
    test_ld = DataLoader(test_ds, CFG.BATCH_SIZE, shuffle=False,
                         num_workers=CFG.WORKERS, pin_memory=True)

    print(f"Food-101: Train:{n_train:,} Val:{n_val:,} Test:{len(test_ds):,} Classes:{CFG.NUM_CLASSES}")
    return train_ld, val_ld, test_ld, train_ds, test_ds, full_train.classes
