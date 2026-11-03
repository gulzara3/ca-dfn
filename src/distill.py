"""
Curriculum-based knowledge distillation from the CA-DFN ensemble teacher
into a single EfficientNet-B4 student.

Sample difficulty is the inter-model prediction variance; training starts
with the easiest 50% of samples and grows to 100% over the first 70% of
epochs. The KD loss combines a temperature-softened KL term with a
hard-label cross-entropy term.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import CFG, MODEL_NAMES
from .train import evaluate


class KDLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.7):
        super().__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, s_logits, t_probs, labels):
        s_soft = F.log_softmax(s_logits / self.T, dim=1)
        t_soft = (t_probs + 1e-8).pow(1.0 / self.T)
        t_soft = t_soft / t_soft.sum(1, keepdim=True)
        kd = F.kl_div(s_soft, t_soft, reduction="batchmean") * (self.T ** 2)
        ce = F.cross_entropy(s_logits, labels)
        return self.alpha * kd + (1 - self.alpha) * ce


class _KDDS(Dataset):
    def __init__(self, base, tp, indices=None):
        self.base = base
        self.tp = tp
        self.indices = indices

    def __len__(self):
        return len(self.indices) if self.indices is not None else min(len(self.base), len(self.tp))

    def __getitem__(self, i):
        idx = self.indices[i] if self.indices is not None else i
        img, lab = self.base[idx]
        return img, lab, self.tp[idx]


def run_distillation(cadfn, fus_tr, trn_P, trn_Y, train_ds, val_ld, test_ld, ca_t1):
    """Distil the CA-DFN ensemble into an EfficientNet-B4 student."""
    print("Generating teacher soft labels...")
    cadfn.eval()
    teacher_probs = []
    with torch.no_grad():
        for ps, _ in fus_tr:
            ps = [p.to(CFG.DEVICE) for p in ps]
            ep, _ = cadfn(ps, training=False)
            teacher_probs.append(ep.cpu())
    teacher_probs = torch.cat(teacher_probs)

    print("Computing sample difficulty...")
    disagreement = []
    for i in range(len(trn_Y)):
        preds = np.stack([trn_P[n][i].numpy() for n in MODEL_NAMES])
        disagreement.append(np.var(preds, axis=0).mean())
    disagreement = np.array(disagreement)
    difficulty_order = np.argsort(disagreement)

    student = models.efficientnet_b4(weights="IMAGENET1K_V1")
    student.classifier[1] = nn.Linear(student.classifier[1].in_features, CFG.NUM_CLASSES)
    student = student.to(CFG.DEVICE)
    st_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"Student: EfficientNet-B4 ({st_params:.1f}M)")

    kd_crit = KDLoss(CFG.KD_TEMP, CFG.KD_ALPHA)
    kd_opt = optim.AdamW([
        {"params": [p for n, p in student.named_parameters() if "classifier" not in n], "lr": 1e-4},
        {"params": list(student.classifier.parameters()), "lr": 5e-4},
    ], weight_decay=0.01)
    kd_sch = CosineAnnealingLR(kd_opt, T_max=CFG.KD_EP, eta_min=1e-6)

    best_st = 0.0
    print(f"Curriculum KD: {CFG.KD_EP} epochs...")
    for ep in range(CFG.KD_EP):
        frac = min(1.0, 0.5 + 0.5 * (ep / (CFG.KD_EP * 0.7)))
        n_samp = int(frac * len(difficulty_order))
        curr = difficulty_order[:n_samp].tolist()
        kd_ld = DataLoader(_KDDS(train_ds, teacher_probs, curr), CFG.BATCH_SIZE,
                           shuffle=True, num_workers=CFG.WORKERS,
                           pin_memory=True, drop_last=True)
        student.train()
        for imgs, labs, tp in kd_ld:
            imgs, labs, tp = imgs.to(CFG.DEVICE), labs.to(CFG.DEVICE), tp.to(CFG.DEVICE)
            kd_opt.zero_grad(set_to_none=True)
            lo = kd_crit(student(imgs), tp, labs)
            lo.backward()
            kd_opt.step()
        kd_sch.step()
        vr = evaluate(student, val_ld)
        flag = ""
        if vr["top1"] > best_st:
            best_st = vr["top1"]
            torch.save(student.state_dict(), f"{CFG.CKPT}/student.pth")
            flag = " *"
        if (ep + 1) % 5 == 0 or flag:
            print(f"  Ep {ep + 1:>2}/{CFG.KD_EP} Data {frac * 100:.0f}% Val {vr['top1']:.2f}%{flag}")

    student.load_state_dict(torch.load(f"{CFG.CKPT}/student.pth", weights_only=True))
    st_test = evaluate(student, test_ld)
    recovery = st_test["top1"] / ca_t1 * 100
    print(f"\nStudent: Top-1={st_test['top1']:.2f}%  Recovery={recovery:.1f}%  ({st_params:.1f}M params)")
    return student, st_test
