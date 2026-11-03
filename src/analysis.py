"""
Ablation studies and additional analysis plots:
  - fusion-method comparison, ensemble-size curve, leave-one-out
  - mean attention weights
  - per-category adaptive weighting
  - training curves and confusion matrix
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

from .config import CFG, MODEL_NAMES, DISPLAY
from .fusion import ens_eval


def run_ablations(test_P, test_Y, results, ca_t1, ca_w, vote_t1, uni, prf):
    """Fusion-method, ensemble-size, and leave-one-out ablations + figure."""
    print("Ablation 1: Fusion Methods")
    print("=" * 55)
    abl = [("Majority Vote", vote_t1), ("Uniform Avg", uni["top1"]),
           ("Perf-Weight", prf["top1"]), ("CA-DFN (Ours)", ca_t1)]
    for m, a in abl:
        print(f"  {m:<25s} Top-1: {a:.2f}%")

    print("\nAblation 2: Ensemble Size")
    print("=" * 55)
    ranked = sorted(MODEL_NAMES, key=lambda n: results[n]["top1"], reverse=True)
    sz_accs = []
    for k in range(1, len(ranked) + 1):
        r = ens_eval({n: test_P[n] for n in ranked[:k]}, test_Y)
        sz_accs.append(r["top1"])
        print(f"  {k} model(s) Top-1={r['top1']:.2f}%")

    print("\nAblation 3: Leave-One-Out")
    print("=" * 55)
    for drop in MODEL_NAMES:
        r = ens_eval({n: test_P[n] for n in MODEL_NAMES if n != drop}, test_Y)
        print(f"  w/o {DISPLAY[drop]:<18s} Top-1={r['top1']:.2f}%")

    mw = ca_w.mean(0).numpy()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5))
    ms, acs = zip(*abl)
    a1.barh(ms, acs, color=["#bdc3c7", "#95a5a6", "#3498db", "#2ecc71"], edgecolor="k")
    for i, v in enumerate(acs):
        a1.text(v + .1, i, f"{v:.2f}%", va="center", fontweight="bold")
    a1.set_xlabel("Top-1 (%)")
    a1.set_title("Fusion Comparison", fontweight="bold")
    a2.plot(range(1, len(ranked) + 1), sz_accs, "o-", color="#2e86de", lw=2, ms=8)
    a2.set_xlabel("Models")
    a2.set_ylabel("Top-1 (%)")
    a2.set_title("Ensemble Size", fontweight="bold")
    a3.bar([DISPLAY[n] for n in MODEL_NAMES], mw,
           color=["#e74c3c", "#3498db", "#f39c12", "#2ecc71", "#9b59b6"], edgecolor="k")
    a3.set_ylabel("Weight")
    a3.set_title("CA-DFN Weights", fontweight="bold")
    a3.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(f"{CFG.OUT}/ablations.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_and_confusion(hists, ca_preds, test_Y, class_names):
    """Training/validation curves and the top-25 confusion matrix."""
    cl = {"resnet50": "#e74c3c", "efficientnet_b4": "#3498db", "densenet121": "#f39c12",
          "convnext_tiny": "#2ecc71", "swin_small": "#9b59b6"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for nm in MODEL_NAMES:
        a1.plot(hists[nm]["ta"], label=DISPLAY[nm], color=cl[nm], lw=1.5)
        a2.plot(hists[nm]["va"], label=DISPLAY[nm], color=cl[nm], lw=1.5)
    a1.set_title("Training", fontweight="bold")
    a1.legend(fontsize=8)
    a2.set_title("Validation", fontweight="bold")
    a2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{CFG.OUT}/training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(test_Y, ca_preds)
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    confused = set()
    for idx in np.argsort(cm_off.ravel())[-30:]:
        confused.update(divmod(idx, CFG.NUM_CLASSES))
    sel = sorted(confused)[:25]
    cm_sub = cm[np.ix_(sel, sel)]
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_sub, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=[class_names[i] for i in sel],
                yticklabels=[class_names[i] for i in sel])
    ax.set_title("Confusion Matrix (Top-25 Confused Classes)", fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{CFG.OUT}/confusion.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_per_category_weights(ca_w, test_Y, class_names):
    """Per-category CA-DFN attention weights for the 15 most diverse classes."""
    pcw = np.zeros((CFG.NUM_CLASSES, len(MODEL_NAMES)))
    for c in range(CFG.NUM_CLASSES):
        m = (test_Y == c)
        if m.sum():
            pcw[c] = ca_w[m].mean(0).numpy()
    div = pcw.std(1)
    top_div = np.argsort(div)[-15:]
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(top_div))
    w = 0.15
    cw = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71", "#9b59b6"]
    for i, nm in enumerate(MODEL_NAMES):
        ax.bar(x + i * w, pcw[top_div, i], w, label=DISPLAY[nm], color=cw[i], edgecolor="k", lw=.5)
    ax.set_xticks(x + w * 2)
    ax.set_xticklabels([class_names[i] for i in top_div], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Avg Weight")
    ax.set_title("CA-DFN Weights per Category (Top-15 Most Diverse)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, .55)
    plt.tight_layout()
    plt.savefig(f"{CFG.OUT}/cadfn_weights.png", dpi=300, bbox_inches="tight")
    plt.close()
