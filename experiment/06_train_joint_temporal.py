#!/usr/bin/env python3
"""
Experiment 06: Joint Temporal Fine-tuning (Phase 2)
Fine-tunes all components with rate-distortion loss.

Output: checkpoints_milestone2/temporal_final.pt
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
    print("Experiment 06: Joint Temporal Fine-tuning (Phase 2)")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print("  Expected wall time: ~6 hours on MI300X")

    cmd = [
        sys.executable,
        os.path.join(ROOT, "pipeline", "jepa_train.py"),
        "--phase",
        "2",  # joint fine-tune
        "--checkpoint",
        os.path.join(ROOT, "checkpoints_milestone2", "temporal_epoch20.pt"),
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)

    print("\n=== Experiment 06 complete ===")
    print("  Checkpoint: checkpoints_milestone2/temporal_final.pt")


if __name__ == "__main__":
    main()
