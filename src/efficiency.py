"""
Computational-efficiency benchmarking: parameters, GFLOPs, latency,
throughput, and the accuracy-vs-compute Pareto plot.
"""
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info

from .config import CFG, MODEL_NAMES, DISPLAY


def benchmark(trained, student, student_top1, results, ca_t1):
    """Measure efficiency metrics for all models and the full ensemble."""
    eff = []
    dummy = torch.randn(1, 3, 224, 224).to(CFG.DEVICE)

    def _bench(model, label, top1):
        flops, params = get_model_complexity_info(
            model, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
        model.eval()
        with torch.no_grad():
            for _ in range(50):
                model(dummy)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(200):
                model(dummy)
            torch.cuda.synchronize()
            lat = (time.time() - t0) / 200 * 1000
        eff.append(dict(Model=label, Params_M=params / 1e6, GFLOPs=flops / 1e9,
                        Latency_ms=lat, Throughput=1000 / lat, Top1=top1))

    for nm in MODEL_NAMES:
        _bench(trained[nm], DISPLAY[nm], results[nm]["top1"])
    _bench(student, "Student (EffNet-B4 KD)", student_top1)

    ens_idx = list(range(len(MODEL_NAMES)))
    eff.append(dict(
        Model="CA-DFN Ensemble",
        Params_M=sum(eff[i]["Params_M"] for i in ens_idx),
        GFLOPs=sum(eff[i]["GFLOPs"] for i in ens_idx),
        Latency_ms=sum(eff[i]["Latency_ms"] for i in ens_idx),
        Throughput=1000 / sum(eff[i]["Latency_ms"] for i in ens_idx),
        Top1=ca_t1,
    ))

    df = pd.DataFrame(eff)
    print(df.to_string(index=False, float_format="%.2f"))
    df.to_csv(f"{CFG.OUT}/efficiency.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71", "#9b59b6", "#1abc9c", "#e67e22"]
    for i, row in df.iterrows():
        ax.scatter(row.GFLOPs, row.Top1, s=row.Params_M * 3 + 50,
                   c=colors[i % len(colors)], alpha=.8, edgecolors="k", lw=1, zorder=3)
        ax.annotate(row.Model, (row.GFLOPs, row.Top1), xytext=(8, 4),
                    textcoords="offset points", fontsize=7)
    ax.set_xlabel("GFLOPs")
    ax.set_ylabel("Top-1 (%)")
    ax.set_title("Accuracy vs Compute", fontweight="bold")
    ax.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(f"{CFG.OUT}/pareto.png", dpi=300, bbox_inches="tight")
    plt.close()
    return df
