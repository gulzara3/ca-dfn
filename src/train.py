"""
Training and evaluation engine for the base models.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import f1_score

from .config import CFG, DISPLAY, seed_everything
from .models import build_model
from .data import apply_cutmix_or_mixup


class SoftCE(nn.Module):
    """Cross-entropy that accepts either hard labels (with label smoothing)
    or soft target distributions (for CutMix / MixUp)."""
    def __init__(self, ls=0.0):
        super().__init__()
        self.ls = ls

    def forward(self, logits, targets):
        if targets.dim() == 1:
            nc = logits.size(1)
            soft = F.one_hot(targets, nc).float() * (1 - self.ls) + self.ls / nc
        else:
            soft = targets
        return -(soft * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def train_epoch(model, loader, opt, crit, sched=None, use_mix=True):
    model.train()
    loss_sum = total = 0
    opt.zero_grad(set_to_none=True)
    for step, (imgs, labs) in enumerate(loader):
        imgs = imgs.to(CFG.DEVICE, non_blocking=True)
        labs = labs.to(CFG.DEVICE, non_blocking=True)
        if use_mix:
            imgs, soft_labs = apply_cutmix_or_mixup(imgs, labs, CFG.NUM_CLASSES)
        else:
            soft_labs = labs
        loss = crit(model(imgs), soft_labs) / CFG.ACCUM_STEPS
        loss.backward()
        if (step + 1) % CFG.ACCUM_STEPS == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        loss_sum += loss.item() * CFG.ACCUM_STEPS * imgs.size(0)
        total += labs.size(0)
    if sched:
        sched.step()
    return loss_sum / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_sum = correct = total = 0
    ce = nn.CrossEntropyLoss()
    all_probs, all_labs = [], []
    for imgs, labs in loader:
        imgs = imgs.to(CFG.DEVICE, non_blocking=True)
        labs = labs.to(CFG.DEVICE, non_blocking=True)
        out = model(imgs)
        loss_sum += ce(out, labs).item() * imgs.size(0)
        probs = F.softmax(out, dim=1)
        correct += out.argmax(1).eq(labs).sum().item()
        total += labs.size(0)
        all_probs.append(probs.cpu())
        all_labs.extend(labs.cpu().numpy())
    all_probs = torch.cat(all_probs)
    all_labs = np.array(all_labs)
    preds = all_probs.argmax(1).numpy()
    t5 = all_probs.topk(5, 1).indices
    t5a = 100 * np.mean([all_labs[i] in t5[i].numpy() for i in range(len(all_labs))])
    return dict(loss=loss_sum / total, top1=100 * correct / total, top5=t5a,
                preds=preds, labels=all_labs, probs=all_probs)


def train_full(name, seed, train_ld, val_ld, test_ld, epochs=CFG.EPOCHS):
    """Train one backbone end-to-end and return (model, history, test_result)."""
    print(f"\n{'=' * 60}\n  Training {DISPLAY[name]} (seed={seed}, {epochs} ep)\n{'=' * 60}")
    seed_everything(seed)
    model, pg = build_model(name)
    crit = SoftCE(CFG.LABEL_SMOOTH)
    opt = optim.AdamW(pg, weight_decay=CFG.WD)
    warm = LinearLR(opt, start_factor=0.01, total_iters=CFG.WARMUP_EP)
    cos = CosineAnnealingLR(opt, T_max=epochs - CFG.WARMUP_EP, eta_min=1e-7)
    sched = SequentialLR(opt, [warm, cos], milestones=[CFG.WARMUP_EP])

    best_acc = 0.0
    path = f"{CFG.CKPT}/{name}_s{seed}.pth"
    hist = {k: [] for k in ("tl", "ta", "vl", "va")}
    for ep in range(epochs):
        t0 = time.time()
        tl = train_epoch(model, train_ld, opt, crit, sched)
        vr = evaluate(model, val_ld)
        hist["tl"].append(tl)
        hist["ta"].append(vr["top1"])
        hist["vl"].append(vr["loss"])
        hist["va"].append(vr["top1"])
        flag = ""
        if vr["top1"] > best_acc:
            best_acc = vr["top1"]
            torch.save(model.state_dict(), path)
            flag = " *"
        if (ep + 1) % 5 == 0 or ep == 0 or flag:
            print(f"  Ep {ep + 1:>2}/{epochs} | Val {vr['top1']:.1f}% "
                  f"T5 {vr['top5']:.1f}% | {time.time() - t0:.0f}s{flag}")

    model.load_state_dict(torch.load(path, weights_only=True))
    tr = evaluate(model, test_ld)
    f1 = f1_score(tr["labels"], tr["preds"], average="macro")
    print(f"  Test: Top-1 {tr['top1']:.2f}%  Top-5 {tr['top5']:.2f}%  F1 {f1:.4f}")
    return model, hist, tr
