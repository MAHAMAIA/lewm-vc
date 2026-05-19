#!/usr/bin/env python3
"""
Experiment 04: Intra-frame Rate-Distortion Evaluation
Reproduces Table 4.

Loads each λ checkpoint, encodes 100 test frames,
computes BPP (via entropy model) and PSNR.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
from tabulate import tabulate

from common import (
    DATASET_DIR,
    CHECKPOINT_M1,
    CHECKPOINT_M2,
    load_frames,
    encode_frames,
    decode_frames,
    compute_psnr,
    compute_bpp_from_entropy,
)

LAMBDAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
FRAME_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(lambda_val: float, device: str = DEVICE):
    """Load encoder + decoder + entropy model for a given λ."""
    from lewm_vc.encoder import LeWMEncoder
    from lewm_vc.working_decoder import LeWMDecoder
    from lewm_vc.entropy import HyperpriorEntropy

    encoder = LeWMEncoder(latent_dim=192, patch_size=16).to(device)
    decoder = LeWMDecoder(latent_dim=192).to(device)
    entropy = HyperpriorEntropy(latent_dim=192).to(device)

    ae_path = CHECKPOINT_M1 / f"ae_lambda_{lambda_val}_final.pt"
    entropy_path = CHECKPOINT_M1 / f"entropy_lambda_{lambda_val}_final.pt"

    if ae_path.exists():
        print(f"  Loading autoencoder: {ae_path}")
        ae_state = torch.load(ae_path, map_location=device, weights_only=True)
        encoder.load_state_dict(
            {k.replace("encoder.", ""): v for k, v in ae_state.items() if "encoder" in k},
            strict=False,
        )
        decoder.load_state_dict(
            {k.replace("decoder.", ""): v for k, v in ae_state.items() if "decoder" in k},
            strict=False,
        )
    else:
        print(f"  [warn] No checkpoint at {ae_path}, using random init")

    if entropy_path.exists():
        print(f"  Loading entropy: {entropy_path}")
        entropy.load_state_dict(torch.load(entropy_path, map_location=device, weights_only=True))

    return encoder, decoder, entropy


@torch.no_grad()
def evaluate_lambda(lambda_val: float, test_frames: torch.Tensor, device: str = DEVICE):
    """Evaluate a single λ model and return (BPP, PSNR)."""
    encoder, decoder, entropy = load_models(lambda_val, device)
    encoder.eval()
    decoder.eval()
    entropy.eval()

    latents = encode_frames(test_frames, encoder, device)
    recon = decode_frames(latents, decoder, device, FRAME_SIZE)

    psnr = compute_psnr(recon[:1], test_frames[:1])  # single frame
    psnrs = []
    for i in range(min(test_frames.shape[0], 100)):
        psnrs.append(compute_psnr(recon[i : i + 1], test_frames[i : i + 1]))
    psnr_avg = np.mean(psnrs)
    psnr_std = np.std(psnrs)

    bpp = compute_bpp_from_entropy(latents[:1], entropy)

    return bpp, psnr_avg, psnr_std


def main():
    print("=" * 70)
    print("Experiment 04: Intra-frame RD Evaluation (Table 4)")
    print("=" * 70)

    # Load test frames
    pevid_test = sorted((DATASET_DIR / "pevid-hd").glob("*.mpg"))
    if not pevid_test:
        print("No PEViD-HD files found. Run 01_download_data.sh first.")
        sys.exit(1)

    print(f"\nLoading test frames from: {pevid_test[0]}")
    test_frames = load_frames(str(pevid_test[0]), max_frames=10)
    print(f"  Loaded {test_frames.shape[0]} frames")

    results = []
    for lam in LAMBDAS:
        print(f"\n--- λ = {lam} ---")
        try:
            bpp, psnr, psnr_std = evaluate_lambda(lam, test_frames, DEVICE)
            results.append((lam, bpp, psnr, psnr_std))
            print(f"  BPP = {bpp:.4f}, PSNR = {psnr:.2f} dB ± {psnr_std:.2f}")
        except Exception as e:
            print(f"  [error] {e}")
            results.append((lam, None, None, None))

    print("\n" + "=" * 70)
    print("Table 4: Intra-frame Rate-Distortion Results")
    print("=" * 70)
    headers = ["λ", "LeWM BPP", "LeWM PSNR (dB)", "x265 CRF", "x265 BPP", "x265 PSNR (dB)"]
    # x265 reference values from paper
    x265_refs = {
        0.001: (18, 0.083, 26.10),
        0.005: (23, 0.044, 25.79),
        0.01: (28, 0.021, 25.22),
        0.05: (33, 0.011, 24.36),
        0.1: (38, 0.006, 23.27),
    }
    rows = []
    for lam, bpp, psnr, psnr_std in results:
        x265 = x265_refs.get(lam, ("--", "--", "--"))
        rows.append(
            [
                f"{lam:.3f}",
                f"{bpp:.3f}" if bpp else "ERR",
                f"{psnr:.2f}" if psnr else "ERR",
                str(x265[0]),
                f"{x265[1]:.3f}",
                f"{x265[2]:.2f}",
            ]
        )
    print(tabulate(rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
