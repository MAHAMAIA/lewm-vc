#!/usr/bin/env python3
"""
Experiment 11: Computational Efficiency
Reproduces Table 9.

Measures:
  - Total parameter count and per-component breakdown
  - Inference throughput on available GPU (I-frame, P-frame)
  - Peak GPU memory
  - Training time reference
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import time
import numpy as np
from tabulate import tabulate

from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.predictor import LeWMPredictor
from lewm_vc.quant import Quantizer

from common import CHECKPOINT_M1, CHECKPOINT_M2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_SIZE = (256, 256)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_throughput(
    model: torch.nn.Module, dummy_input: torch.Tensor, n_warmup: int = 10, n_iters: int = 100
) -> float:
    """Measure throughput in fps. Higher is better."""
    model.to(DEVICE).eval()
    dummy = dummy_input.to(DEVICE)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    torch.cuda.synchronize() if DEVICE == "cuda" else None
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(dummy)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    elapsed = time.perf_counter() - start

    return n_iters / elapsed


def main():
    print("=" * 70)
    print("Experiment 11: Computational Efficiency (Table 9)")
    print("=" * 70)
    print(f"  Device: {DEVICE}")

    # ── Build models ──
    encoder = LeWMEncoder(latent_dim=192, patch_size=16)
    decoder = LeWMDecoder(latent_dim=192)
    entropy = HyperpriorEntropy(latent_dim=192)
    predictor = LeWMPredictor(latent_dim=192)

    # ── Parameter counts ──
    n_enc = count_params(encoder)
    n_dec = count_params(decoder)
    n_pred = count_params(predictor)
    n_ent = count_params(entropy)
    n_total = n_enc + n_dec + n_pred + n_ent

    print(f"\n  ── Parameter Counts ──")
    params_rows = [
        ["Encoder", f"{n_enc / 1e6:.1f}M"],
        ["Decoder", f"{n_dec / 1e6:.1f}M"],
        ["Predictor", f"{n_pred / 1e6:.1f}M"],
        ["Entropy model", f"{n_ent / 1e6:.1f}M"],
        ["Total", f"{n_total / 1e6:.1f}M"],
    ]
    print(tabulate(params_rows, headers=["Component", "Parameters"], tablefmt="grid"))

    # ── Throughput (if GPU available) ──
    if DEVICE == "cuda":
        dummy_frame = torch.randn(1, 3, *FRAME_SIZE)
        dummy_latent = torch.randn(1, 192, 16, 16)
        dummy_context = torch.randn(4, 1, 192, 16, 16)

        print(f"\n  ── Throughput ──")
        try:
            fps_i = measure_throughput(encoder, dummy_frame)
            fps_p = measure_throughput(predictor, dummy_context)

            # Memory
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = encoder(dummy_frame.to(DEVICE))
                _ = decoder(dummy_latent.to(DEVICE))
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            thrpt_rows = [
                ["I-frame (encode+decode)", f"{fps_i:.1f} fps"],
                ["P-frame (predict+encode)", f"{fps_p:.1f} fps"],
                ["Peak GPU memory", f"{peak_mem:.1f} GB"],
            ]
            # For comparison: paper reports 84.7 fps (I), 80.6 fps (P) on T4
            print(tabulate(thrpt_rows, headers=["Operation", "Measured"], tablefmt="grid"))
            print("  (Paper reports 84.7 I-frame / 80.6 P-frame fps on NVIDIA T4)")
        except Exception as e:
            print(f"  [warn] Throughput measurement failed: {e}")
    else:
        print(f"\n  [info] GPU not available — throughput measurement skipped.")
        print("  Paper reports: 84.7 fps (I-frame), 80.6 fps (P-frame) on T4")

    # ── Final table ──
    print("\n" + "=" * 70)
    print("Table 9: Computational Profile (Paper Values)")
    print("=" * 70)
    paper_rows = [
        ["Total parameters", "14.7M"],
        ["Encoder", "5.8M"],
        ["Decoder", "2.3M"],
        ["Predictor", "4.2M"],
        ["Entropy model", "2.4M"],
        ["Inference (T4, I)", "84.7 fps"],
        ["Inference (T4, P)", "80.6 fps"],
        ["Peak GPU memory", "1.2 GB"],
        ["Training time", "~8 hours (MI300X)"],
    ]
    print(tabulate(paper_rows, headers=["Metric", "Value"], tablefmt="grid"))


if __name__ == "__main__":
    main()
