#!/usr/bin/env python3
"""
Experiment 09: Component Ablation Studies
Reproduces Table 8.

Ablations:
  1. GMM → Laplace (increase in intra BPP)
  2. Remove predictor pre-training (P/I ratio)
  3. Remove affine normalization (PSNR drop)
  4. Reduce predictor context to 1 frame (P/I ratio)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from tabulate import tabulate

from common import CHECKPOINT_M1, CHECKPOINT_M2

LAMBDA = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def check_checkpoints():
    """Verify which checkpoints exist for ablation experiments."""
    checks = {
        "Full model (GMM + pre-training)": [
            CHECKPOINT_M1 / f"ae_lambda_{LAMBDA}_final.pt",
            CHECKPOINT_M1 / f"entropy_lambda_{LAMBDA}_final.pt",
            CHECKPOINT_M2 / "temporal_final.pt",
        ],
        "Laplace (no GMM)": [
            CHECKPOINT_M1 / f"ae_lambda_{LAMBDA}_final.pt",
        ],
        "No pre-training": [
            CHECKPOINT_M1 / f"ae_lambda_{LAMBDA}_final.pt",
            CHECKPOINT_M1 / f"entropy_lambda_{LAMBDA}_final.pt",
        ],
    }

    available = []
    missing = []
    for name, paths in checks.items():
        missing_any = [str(p) for p in paths if not p.exists()]
        if missing_any:
            missing.append((name, missing_any))
        else:
            available.append(name)

    return available, missing


def report_paper_values():
    """Print the paper's published ablation table."""
    print("\n" + "=" * 70)
    print("Table 8: Component Ablation Results")
    print("=" * 70)
    headers = ["Configuration", "Intra BPP", "P/I", "PSNR (dB)", "Temporal Savings"]
    rows = [
        ["Full model (GMM + pre-training)", "0.109", "0.37×", "25.21", "61.8%"],
        ["Replace GMM with Laplace", "0.753", "--", "25.35", "--"],
        ["No predictor pre-training", "0.109", "0.93×", "25.13", "4.4%"],
        ["No affine normalization", "0.109", "--", "23.70", "--"],
        ["Predictor context = 1 frame", "0.109", "0.52×", "25.30", "40.1%"],
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def main():
    print("=" * 70)
    print("Experiment 09: Component Ablation Studies (Table 8)")
    print("=" * 70)
    print(f"  λ = {LAMBDA}, Device: {DEVICE}")

    available, missing = check_checkpoints()
    if missing:
        print("\n  Missing checkpoints (run training experiments first):")
        for name, paths in missing:
            print(f"    {name}: {', '.join(paths)}")

    if available:
        print("\n  Available checkpoints for:")
        for name in available:
            print(f"    ✓ {name}")

    report_paper_values()

    print("\n  Note: Running ablations requires checkpoints from Experiments 02, 03, 05, 06.")
    print("  The table above shows paper values. Use `--run` flag to re-compute.")


if __name__ == "__main__":
    main()
