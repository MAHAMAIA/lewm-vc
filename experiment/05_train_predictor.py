#!/usr/bin/env python3
"""
Experiment 05: JEPA Predictor Pre-training (Phase 1)
Trains the 8-layer transformer predictor with frozen encoder/decoder/entropy.

Output: checkpoints_milestone2/temporal_epoch20.pt
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import subprocess

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 70)
    print("Experiment 05: JEPA Predictor Pre-training (Phase 1)")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print("  Expected wall time: ~2 hours on MI300X")

    # Delegate to the existing pipeline script
    cmd = [
        sys.executable,
        os.path.join(ROOT, "pipeline", "jepa_train.py"),
        "--phase",
        "1",  # pre-train only
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)

    print("\n=== Experiment 05 complete ===")
    print("  Checkpoint: checkpoints_milestone2/temporal_epoch20.pt")


if __name__ == "__main__":
    main()
