"""
Calibration and uncertainty analysis: Expected Calibration Error (ECE),
reliability diagram, entropy distribution, and selective-prediction curve.
"""
import numpy as np
import matplotlib.pyplot as plt

from .config import CFG, MODEL_NAMES, DISPLAY


def ece_calc(probs, labs, n_bins=15):
    """Expected Calibration Error with equal-width confidence bins."""
    confs = probs.max(1)
    preds = probs.argmax(1)
    accs = (preds == labs).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_acc, bin_conf = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs > lo) & (confs <= hi)
        if m.sum():
            ba, bc = accs[m].mean(), confs[m].mean()
            ece += m.sum() / len(labs) * abs(ba - bc)
            bin_acc.append(ba)
            bin_conf.append(bc)
    return ece, bin_acc, bin_conf


def run_calibration(ca_probs, ca_preds, test_Y, ca_t1, results):
    """Compute ECE, Acc@80% coverage, and produce the uncertainty figure."""
    ens_np = ca_probs.numpy()
    ent = -(ens_np * np.log(ens_np + 1e-10)).sum(1)
    mx_c = ens_np.max(1)
    correct = (ca_preds == test_Y)

    ece_val, bas, bcs = ece_calc(ens_np, test_Y)

    # Selective prediction (accuracy at coverage)
    si = np.argsort(-mx_c)
    cum_c = np.cumsum(correct[si])
    aac = cum_c / np.arange(1, len(si) + 1)
    i80 = int(0.8 * len(si))

    print("Calibration Comparison:")
    for nm in MODEL_NAMES:
        e, _, _ = ece_calc(results[nm]["probs"].numpy(), results[nm]["labels"])
        print(f"  {DISPLAY[nm]:<22s} ECE={e:.4f}  Top-1={results[nm]['top1']:.2f}%")
    print(f"  {'CA-DFN':<22s} ECE={ece_val:.4f}  Top-1={ca_t1:.2f}%")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5))
    a1.hist(ent[correct], bins=50, alpha=.7, label="Correct", color="#2ecc71", density=True)
    a1.hist(ent[~correct], bins=50, alpha=.7, label="Incorrect", color="#e74c3c", density=True)
    a1.set_xlabel("Entropy")
    a1.set_title("Entropy Distribution", fontweight="bold")
    a1.legend()

    a2.bar(bcs, bas, width=.05, alpha=.8, color="#3498db", edgecolor="k")
    a2.plot([0, 1], [0, 1], "r--", lw=2)
    a2.set_title(f"Reliability (ECE={ece_val:.4f})", fontweight="bold")
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 1)

    covs = np.arange(1, len(si) + 1) / len(si)
    a3.plot(covs * 100, aac * 100, color="#2e86de", lw=2)
    a3.axhline(ca_t1, color="gray", ls="--", alpha=.5, label=f"Full: {ca_t1:.1f}%")
    a3.annotate(f"80%: {aac[i80] * 100:.1f}%", xy=(80, aac[i80] * 100),
                xytext=(55, aac[i80] * 100 - 5),
                arrowprops=dict(arrowstyle="->", color="red"))
    a3.set_xlabel("Coverage (%)")
    a3.set_title("Selective Prediction", fontweight="bold")
    a3.legend()

    plt.tight_layout()
    plt.savefig(f"{CFG.OUT}/uncertainty.png", dpi=300, bbox_inches="tight")
    plt.close()

    acc80 = aac[i80] * 100
    print(f"ECE={ece_val:.4f}  Acc@80%={acc80:.2f}%")
    return ece_val, acc80
