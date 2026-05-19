#!/usr/bin/env python3
"""
Experiment 12: Cross-Dataset Evaluation on UVG
Reproduces Table 10.

Evaluates LeWM-VC (trained on PEViD-HD) on UVG natural scenes
to demonstrate domain specificity / failure mode.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
from tabulate import tabulate

from common import (
    load_frames,
    encode_frames,
    decode_frames,
    compute_psnr,
    compute_bpp_from_entropy,
    CHECKPOINT_M1,
    DATASET_DIR,
)

LAMBDAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_SIZE = (256, 256)


@torch.no_grad()
def evaluate_video(video_path: str, lambda_val: float, max_frames: int = 50):
    """Evaluate LeWM-VC on a single video at given λ."""
    from lewm_vc.encoder import LeWMEncoder
    from lewm_vc.working_decoder import LeWMDecoder
    from lewm_vc.entropy import HyperpriorEntropy

    encoder = LeWMEncoder(latent_dim=192, patch_size=16).to(DEVICE)
    decoder = LeWMDecoder(latent_dim=192).to(DEVICE)
    entropy = HyperpriorEntropy(latent_dim=192).to(DEVICE)

    ae_path = CHECKPOINT_M1 / f"ae_lambda_{lambda_val}_final.pt"
    entropy_path = CHECKPOINT_M1 / f"entropy_lambda_{lambda_val}_final.pt"

    if ae_path.exists():
        state = torch.load(ae_path, map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(
            {k.replace("encoder.", ""): v for k, v in state.items() if "encoder" in k},
            strict=False,
        )
        decoder.load_state_dict(
            {k.replace("decoder.", ""): v for k, v in state.items() if "decoder" in k},
            strict=False,
        )
    else:
        print(f"    [warn] No checkpoint at {ae_path}")
        return None, None

    if entropy_path.exists():
        entropy.load_state_dict(torch.load(entropy_path, map_location=DEVICE, weights_only=True))

    encoder.eval()
    decoder.eval()
    entropy.eval()

    frames = load_frames(str(video_path), max_frames=min(30, max_frames))
    latents = encode_frames(frames, encoder, DEVICE)
    recon = decode_frames(latents, decoder, DEVICE, FRAME_SIZE)

    psnrs = [compute_psnr(recon[i : i + 1], frames[i : i + 1]) for i in range(frames.shape[0])]
    bpp = compute_bpp_from_entropy(latents, entropy)
    bpp = compute_bpp_from_entropy(latents[:1], entropy)

    return bpp, np.mean(psnrs)


def main():
    print("=" * 70)
    print("Experiment 12: Cross-Dataset Evaluation on UVG (Table 10)")
    print("=" * 70)
    print(f"  Device: {DEVICE}")

    uvg_files = sorted((DATASET_DIR / "uvg").glob("*.mp4"))
    if not uvg_files:
        # Try other extensions
        uvg_files = sorted((DATASET_DIR / "uvg").glob("*.*"))
        uvg_files = [
            f for f in uvg_files if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".yuv")
        ]

    if not uvg_files:
        print("  No UVG files found. Run 01_download_data.sh first.")
        print("  Publishing paper values below.\n")

    # Compute per-λ results across UVG sequences
    results_by_lambda = {lam: {"bpp": [], "psnr": []} for lam in LAMBDAS}

    if uvg_files:
        # Use first UVG video as representative test
        test_video = str(uvg_files[0])
        print(f"\n  Test video: {os.path.basename(test_video)}")

        for lam in LAMBDAS:
            print(f"  λ = {lam}...", end=" ", flush=True)
            try:
                bpp, psnr = evaluate_video(test_video, lam, max_frames=30)
                if bpp is not None:
                    results_by_lambda[lam]["bpp"].append(bpp)
                    results_by_lambda[lam]["psnr"].append(psnr)
                    print(f"BPP={bpp:.3f}, PSNR={psnr:.2f} dB")
                else:
                    print("skip (no checkpoint)")
            except Exception as e:
                print(f"error: {e}")

    # Print results
    print("\n" + "=" * 70)
    print("Table 10: UVG Cross-Dataset Results")
    print("=" * 70)
    headers = ["λ", "Mean BPP", "Mean PSNR (dB)"]

    rows = []
    for lam in LAMBDAS:
        bpps = results_by_lambda[lam]["bpp"]
        psnrs = results_by_lambda[lam]["psnr"]
        if bpps:
            rows.append(
                [
                    f"{lam:.3f}",
                    f"{np.mean(bpps):.3f}",
                    f"{np.mean(psnrs):.2f} ± {np.std(psnrs):.2f}",
                ]
            )
        else:
            rows.append([f"{lam:.3f}", "N/A", "N/A"])

    print(tabulate(rows, headers=headers, tablefmt="grid"))

    print("\n  ── Expected (paper values) ──")
    paper_rows = [
        ["0.001", "1.950", "14.86 ± 1.78"],
        ["0.005", "1.963", "14.45 ± 1.75"],
        ["0.01", "1.962", "13.33 ± 1.95"],
        ["0.05", "1.951", "13.75 ± 1.86"],
        ["0.1", "1.951", "13.75 ± 1.83"],
        ["0.5", "1.952", "11.20 ± 1.54"],
    ]
    print(tabulate(paper_rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
