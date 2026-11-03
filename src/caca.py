"""
Cross-Architecture Complementarity Analysis (CACA).

Generates Grad-CAM saliency maps for each base model, thresholds them to
binary attention masks, and computes a pairwise Grad-CAM IoU matrix. The
complementarity score C(i,j) = 1 - IoU(M_i, M_j) is correlated against the
pairwise fusion gain to reveal the diversity-vs-benefit relationship.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as T
from torchvision.datasets import Food101
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .config import CFG, MODEL_NAMES, DISPLAY
from .fusion import ens_eval


def _target_layers(model, name):
    if name == "resnet50":
        return [model.layer4[-1]]
    if name == "efficientnet_b4":
        return [model.features[-1]]
    if name == "densenet121":
        return [model.features.denseblock4]
    if name == "convnext_tiny":
        return [model.features[-1]]
    if name == "swin_small":
        return [model.layers[-1].blocks[-1].norm1]
    raise ValueError(name)


def _swin_reshape(t, h=7, w=7):
    return t.reshape(t.size(0), h, w, t.size(-1)).permute(0, 3, 1, 2)


def run_caca(trained, test_ds, test_P, test_Y, results, class_names,
             n_cam=150, n_vis=5, thr=0.5):
    """Compute the CACA IoU/complementarity matrices and figures.

    Returns (iou_mx, comp_mx, mean_comp, correlation).
    """
    raw_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    raw_ds = Food101(CFG.DATA_DIR, split="test", transform=raw_tf)
    idxs = np.random.RandomState(0).choice(len(test_ds), n_cam, replace=False)

    print(f"Grad-CAM: {n_cam} images x {len(MODEL_NAMES)} models...")
    heatmaps = {n: [] for n in MODEL_NAMES}
    for nm in MODEL_NAMES:
        m = trained[nm]
        m.eval()
        rf = _swin_reshape if "swin" in nm else None
        cam = GradCAM(model=m, target_layers=_target_layers(m, nm), reshape_transform=rf)
        for idx in idxs:
            img, lab = test_ds[idx]
            hm = cam(input_tensor=img.unsqueeze(0).to(CFG.DEVICE),
                     targets=[ClassifierOutputTarget(lab)])[0]
            heatmaps[nm].append(hm)
        del cam
        torch.cuda.empty_cache()

    # Pairwise Grad-CAM IoU and complementarity
    iou_mx = np.zeros((len(MODEL_NAMES), len(MODEL_NAMES)))
    for i, ni in enumerate(MODEL_NAMES):
        for j, nj in enumerate(MODEL_NAMES):
            ious = []
            for k in range(n_cam):
                a = (heatmaps[ni][k] > thr).astype(float)
                b = (heatmaps[nj][k] > thr).astype(float)
                ious.append((a * b).sum() / (np.clip(a + b, 0, 1).sum() + 1e-8))
            iou_mx[i, j] = np.mean(ious)
    comp_mx = 1 - iou_mx
    mask_tri = np.triu(np.ones_like(comp_mx, dtype=bool), k=1)
    mean_comp = comp_mx[mask_tri].mean()

    # Complementarity vs pairwise fusion gain
    pair_comp, pair_gain = [], []
    for i, ni in enumerate(MODEL_NAMES):
        for j, nj in enumerate(MODEL_NAMES):
            if i >= j:
                continue
            cs = comp_mx[i, j]
            pr = ens_eval({ni: test_P[ni], nj: test_P[nj]}, test_Y)
            bi = max(results[ni]["top1"], results[nj]["top1"])
            g = pr["top1"] - bi
            pair_comp.append(cs)
            pair_gain.append(g)
            print(f"  {DISPLAY[ni]:>18s}+{DISPLAY[nj]:<18s} Comp={cs:.3f} Gain={g:+.2f}%")
    correlation = np.corrcoef(pair_comp, pair_gain)[0, 1]
    print(f"\nCorrelation (comp vs gain): r={correlation:.3f}")

    # Figure: IoU / complementarity / scatter
    disp = [DISPLAY[n] for n in MODEL_NAMES]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.heatmap(iou_mx, annot=True, fmt=".3f", cmap="YlOrRd", xticklabels=disp,
                yticklabels=disp, ax=axes[0], vmin=0, vmax=1, square=True)
    axes[0].set_title("Grad-CAM IoU", fontweight="bold")
    sns.heatmap(comp_mx, annot=True, fmt=".3f", cmap="YlGn", xticklabels=disp,
                yticklabels=disp, ax=axes[1], vmin=0, vmax=1, square=True)
    axes[1].set_title("Complementarity (1-IoU)", fontweight="bold")
    axes[2].scatter(pair_comp, pair_gain, s=120, c="#2e86de", edgecolors="k", zorder=3)
    z = np.polyfit(pair_comp, pair_gain, 1)
    xl = np.linspace(min(pair_comp), max(pair_comp), 50)
    axes[2].plot(xl, np.polyval(z, xl), "r--", lw=2, label=f"r={correlation:.3f}")
    axes[2].set_xlabel("Complementarity")
    axes[2].set_ylabel("Fusion Gain (%)")
    axes[2].set_title("Complementarity Predicts Fusion Benefit", fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(f"{CFG.OUT}/caca_iou.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure: qualitative heatmaps
    fig, axes = plt.subplots(n_vis, len(MODEL_NAMES) + 1,
                             figsize=(3 * (len(MODEL_NAMES) + 1), 3 * n_vis))
    for row, si in enumerate(idxs[:n_vis]):
        ri = raw_ds[si][0].permute(1, 2, 0).numpy()
        lab = raw_ds[si][1]
        axes[row, 0].imshow(ri)
        axes[row, 0].set_title(class_names[lab], fontsize=9, fontweight="bold")
        axes[row, 0].axis("off")
        for col, nm in enumerate(MODEL_NAMES):
            axes[row, col + 1].imshow(show_cam_on_image(ri, heatmaps[nm][row], use_rgb=True))
            axes[row, col + 1].set_title(DISPLAY[nm], fontsize=8)
            axes[row, col + 1].axis("off")
    plt.suptitle("CACA: Cross-Architecture Complementarity", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{CFG.OUT}/caca_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Mean Complementarity: {mean_comp:.4f}")
    return iou_mx, comp_mx, mean_comp, correlation
