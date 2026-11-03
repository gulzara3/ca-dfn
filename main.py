"""
End-to-end pipeline for the CA-DFN framework on Food-101.

Stages:
  1. Train five diverse base models (ResNet-50, EfficientNet-B4, DenseNet-121,
     ConvNeXt-Tiny, Swin-Small).
  2. Cache predictions and evaluate static-fusion baselines.
  3. Train the Confidence-Aware Dynamic Fusion Network (CA-DFN).
  4. Cross-Architecture Complementarity Analysis (CACA).
  5. Calibration / uncertainty analysis.
  6. Curriculum knowledge distillation into an EfficientNet-B4 student.
  7. Efficiency benchmarking, ablations, and figures.

Usage:
    python main.py                 # full pipeline, default seed (42)
    python main.py --seed 123      # reproduce with a different seed
    python main.py --no-download   # skip the Food-101 download

Reproduces the results reported in the paper (Food-101, seed 42):
    CA-DFN Top-1 = 92.63%, Top-5 = 98.87%, F1 = 0.9261,
    ECE = 0.0423, Acc@80% coverage = 98.35%.
"""
import argparse
import gc
import os

import torch
from sklearn.metrics import f1_score

from src.config import CFG, MODEL_NAMES, DISPLAY, seed_everything, ensure_dirs
from src.models import summarise_models
from src.data import build_dataloaders
from src.train import train_full
from src.fusion import cache_probs, static_baselines, train_cadfn
from src.caca import run_caca
from src.calibrate import run_calibration
from src.distill import run_distillation
from src.efficiency import benchmark
from src.analysis import (run_ablations, plot_training_and_confusion,
                          plot_per_category_weights)


def main(seed, download):
    ensure_dirs()
    print(f"Device: {CFG.DEVICE} | base-model seed={seed}")
    summarise_models()

    # 1. Data
    train_ld, val_ld, test_ld, train_ds, test_ds, class_names = build_dataloaders(download)

    # 2. Train base models
    trained, hists, results = {}, {}, {}
    for nm in MODEL_NAMES:
        m, h, r = train_full(nm, seed, train_ld, val_ld, test_ld)
        trained[nm], hists[nm], results[nm] = m, h, r
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 65}\n{'Model':<22s} {'Top-1':>8s} {'Top-5':>8s} {'F1':>8s}\n{'-' * 50}")
    for nm in MODEL_NAMES:
        r = results[nm]
        f1 = f1_score(r["labels"], r["preds"], average="macro")
        print(f"{DISPLAY[nm]:<22s} {r['top1']:>7.2f}% {r['top5']:>7.2f}% {f1:>8.4f}")

    # 3. Cache predictions + static-fusion baselines
    print("\nCaching predictions...")
    test_P, test_Y = cache_probs(trained, test_ld)
    val_P, val_Y = cache_probs(trained, val_ld)
    trn_P, trn_Y = cache_probs(trained, train_ld)
    uni, prf, vote_t1 = static_baselines(test_P, test_Y, val_P, val_Y)

    # 4. CA-DFN
    ca = train_cadfn(trn_P, trn_Y, val_P, val_Y, test_P, test_Y, results, seed)

    # 5. CACA
    iou_mx, comp_mx, mean_comp, correlation = run_caca(
        trained, test_ds, test_P, test_Y, results, class_names)

    # 6. Calibration
    ece_val, acc80 = run_calibration(ca["probs"], ca["preds"], test_Y, ca["top1"], results)

    # 7. Curriculum KD
    student, st_test = run_distillation(
        ca["model"], ca["fus_tr"], trn_P, trn_Y, train_ds, val_ld, test_ld, ca["top1"])

    # 8. Efficiency + ablations + figures
    benchmark(trained, student, st_test["top1"], results, ca["top1"])
    run_ablations(test_P, test_Y, results, ca["top1"], ca["weights"], vote_t1, uni, prf)
    plot_training_and_confusion(hists, ca["preds"], test_Y, class_names)
    plot_per_category_weights(ca["weights"], test_Y, class_names)

    # Summary
    print("\n" + "=" * 70)
    print("  CA-DFN - FINAL RESULTS")
    print("=" * 70)
    print("\nENSEMBLE METHODS:")
    print(f"  Majority Vote      Top-1: {vote_t1:.2f}%")
    print(f"  Uniform Avg        Top-1: {uni['top1']:.2f}%")
    print(f"  Perf-Weight        Top-1: {prf['top1']:.2f}%")
    print(f"  CA-DFN (Ours)      Top-1: {ca['top1']:.2f}%  "
          f"Top-5: {ca['top5']:.2f}%  F1: {ca['f1']:.4f}")
    print(f"\nKNOWLEDGE DISTILLATION:")
    print(f"  Student (EffNet-B4) Top-1: {st_test['top1']:.2f}%  "
          f"Recovery: {st_test['top1'] / ca['top1'] * 100:.1f}%")
    print(f"\nANALYSIS:")
    print(f"  Mean Complementarity: {mean_comp:.4f}  |  Comp-Gain r={correlation:.3f}")
    print(f"  ECE: {ece_val:.4f}  |  Acc@80%: {acc80:.2f}%")
    print(f"\nFigures and tables saved to {CFG.OUT}/")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CA-DFN end-to-end pipeline")
    ap.add_argument("--seed", type=int, default=CFG.SEEDS[0],
                    help="random seed for base-model training (default: 42)")
    ap.add_argument("--no-download", action="store_true",
                    help="assume Food-101 is already present in ./data")
    args = ap.parse_args()
    seed_everything(args.seed)
    main(args.seed, download=not args.no_download)
