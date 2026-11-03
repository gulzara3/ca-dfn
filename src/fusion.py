"""
Prediction caching, static-fusion baselines, and CA-DFN training/evaluation.
"""
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score

from .config import CFG, MODEL_NAMES, DISPLAY, ARCH_TYPES, seed_everything
from .cadfn import CADFN, EntropyRegLoss


@torch.no_grad()
def cache_probs(models_dict, loader):
    """Pre-compute softmax outputs for every base model on a loader."""
    out = {n: [] for n in models_dict}
    labs = []
    for imgs, y in loader:
        imgs = imgs.to(CFG.DEVICE, non_blocking=True)
        labs.extend(y.numpy())
        for n, m in models_dict.items():
            m.eval()
            out[n].append(F.softmax(m(imgs), dim=1).cpu())
    for n in out:
        out[n] = torch.cat(out[n])
    return out, np.array(labs)


def ens_eval(probs, labs, weights=None):
    """Evaluate a (weighted) average-fusion ensemble from cached probs."""
    names = list(probs.keys())
    if weights is None:
        weights = {n: 1.0 / len(names) for n in names}
    ep = sum(weights[n] * probs[n] for n in names)
    preds = ep.argmax(1).numpy()
    t5 = ep.topk(5, 1).indices
    t5a = 100 * np.mean([labs[i] in t5[i].numpy() for i in range(len(labs))])
    return dict(top1=100 * accuracy_score(labs, preds), top5=t5a,
                f1=f1_score(labs, preds, average="macro"), preds=preds, probs=ep)


def static_baselines(test_P, test_Y, val_P, val_Y):
    """Compute uniform-average, performance-weighted, and majority-vote baselines."""
    uni = ens_eval(test_P, test_Y)

    va = {n: accuracy_score(val_Y, val_P[n].argmax(1).numpy()) for n in MODEL_NAMES}
    tot = sum(va.values())
    pw = {n: va[n] / tot for n in MODEL_NAMES}
    prf = ens_eval(test_P, test_Y, pw)

    vote_stack = np.stack([test_P[n].argmax(1).numpy() for n in MODEL_NAMES])
    vote_preds = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=CFG.NUM_CLASSES).argmax(), 0, vote_stack)
    vote_t1 = 100 * accuracy_score(test_Y, vote_preds)

    print(f"\nUniform Avg:   Top-1={uni['top1']:.2f}%  F1={uni['f1']:.4f}")
    print(f"Perf-Weighted: Top-1={prf['top1']:.2f}%  F1={prf['f1']:.4f}")
    print(f"Majority Vote: Top-1={vote_t1:.2f}%")
    return uni, prf, vote_t1


class _FusDS(Dataset):
    def __init__(self, P, Y):
        self.ps = [P[n] for n in MODEL_NAMES]
        self.ys = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        return [p[i] for p in self.ps], self.ys[i]


def _col(batch):
    pls, ys = zip(*batch)
    n = len(pls[0])
    return [torch.stack([pl[i] for pl in pls]) for i in range(n)], torch.stack(ys)


def train_cadfn(trn_P, trn_Y, val_P, val_Y, test_P, test_Y, results, seed=None):
    """Train CA-DFN on cached probabilities and evaluate on the test set."""
    if seed is None:
        seed = CFG.SEEDS[0]

    fus_tr = DataLoader(_FusDS(trn_P, trn_Y), 512, True, collate_fn=_col)
    fus_vl = DataLoader(_FusDS(val_P, val_Y), 512, False, collate_fn=_col)
    fus_te = DataLoader(_FusDS(test_P, test_Y), 512, False, collate_fn=_col)

    seed_everything(seed)
    cadfn = CADFN(len(MODEL_NAMES), CFG.NUM_CLASSES, ARCH_TYPES,
                  hidden=CFG.FUSION_HIDDEN, attn_temp=CFG.FUSION_TEMP,
                  input_drop=CFG.FUSION_IDROP).to(CFG.DEVICE)
    opt = optim.AdamW(cadfn.parameters(), lr=CFG.FUSION_LR, weight_decay=1e-3)
    sch = CosineAnnealingLR(opt, T_max=CFG.FUSION_EP, eta_min=1e-6)
    crit = EntropyRegLoss(CFG.FUSION_ENT_W)

    best_fa = 0.0
    print(f"Training CA-DFN ({CFG.FUSION_EP} epochs)...")
    for ep in range(CFG.FUSION_EP):
        cadfn.train()
        c = t = 0
        for ps, ys in fus_tr:
            ps = [p.to(CFG.DEVICE) for p in ps]
            ys = ys.to(CFG.DEVICE)
            opt.zero_grad()
            ep_probs, aw = cadfn(ps, training=True)
            loss, _, _ = crit(ep_probs, ys, aw)
            loss.backward()
            opt.step()
            c += ep_probs.argmax(1).eq(ys).sum().item()
            t += ys.size(0)
        sch.step()

        cadfn.eval()
        vc = vt = 0
        with torch.no_grad():
            for ps, ys in fus_vl:
                ps = [p.to(CFG.DEVICE) for p in ps]
                ys = ys.to(CFG.DEVICE)
                ep2, _ = cadfn(ps, training=False)
                vc += ep2.argmax(1).eq(ys).sum().item()
                vt += ys.size(0)
        va_acc = 100 * vc / vt
        flag = ""
        if va_acc > best_fa:
            best_fa = va_acc
            torch.save(cadfn.state_dict(), f"{CFG.CKPT}/cadfn.pth")
            flag = " *"
        if (ep + 1) % 5 == 0 or flag:
            print(f"  Ep {ep + 1:>2}/{CFG.FUSION_EP}  Tr {100 * c / t:.1f}%  Val {va_acc:.1f}%{flag}")

    cadfn.load_state_dict(torch.load(f"{CFG.CKPT}/cadfn.pth", weights_only=True))
    cadfn.eval()

    ca_preds_l, ca_probs_l, ca_w_l = [], [], []
    with torch.no_grad():
        for ps, _ in fus_te:
            ps = [p.to(CFG.DEVICE) for p in ps]
            ep, w = cadfn(ps, training=False)
            ca_preds_l.append(ep.argmax(1).cpu())
            ca_probs_l.append(ep.cpu())
            ca_w_l.append(w.cpu())
    ca_preds = torch.cat(ca_preds_l).numpy()
    ca_probs = torch.cat(ca_probs_l)
    ca_w = torch.cat(ca_w_l)

    ca_t1 = 100 * accuracy_score(test_Y, ca_preds)
    ca_f1 = f1_score(test_Y, ca_preds, average="macro")
    t5i = ca_probs.topk(5, 1).indices
    ca_t5 = 100 * np.mean([test_Y[i] in t5i[i].numpy() for i in range(len(test_Y))])

    best_single = max(results[n]["top1"] for n in MODEL_NAMES)
    best_nm = max(MODEL_NAMES, key=lambda n: results[n]["top1"])

    print(f"\nCA-DFN: Top-1={ca_t1:.2f}%  Top-5={ca_t5:.2f}%  F1={ca_f1:.4f}")
    print(f"  vs Best Single ({DISPLAY[best_nm]}): {ca_t1 - best_single:+.2f}%")
    mw = ca_w.mean(0).numpy()
    for i, n in enumerate(MODEL_NAMES):
        print(f"  {DISPLAY[n]:<20s} weight: {mw[i]:.4f}")

    return dict(model=cadfn, preds=ca_preds, probs=ca_probs, weights=ca_w,
                top1=ca_t1, top5=ca_t5, f1=ca_f1, fus_tr=fus_tr)
