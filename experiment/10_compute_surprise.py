#!/usr/bin/env python3
"""
Experiment 10: Surprise Metric Calibration
Reproduces Section 5.5 and Figure 2.

Computes per-frame surprise scores s_t on PEViD-HD test videos,
reports aggregate statistics (mean, std, percentiles).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
from tabulate import tabulate

from common import load_frames, encode_frames, CHECKPOINT_M1, CHECKPOINT_M2, DATASET_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_SIZE = (256, 256)


def compute_surprise_for_video(video_path: str, max_frames: int = 100) -> dict:
    """Compute surprise scores for all frames in a video."""
    from lewm_vc.encoder import LeWMEncoder
    from lewm_vc.predictor import LeWMPredictor

    encoder = LeWMEncoder(latent_dim=192, patch_size=16).to(DEVICE)
    predictor = LeWMPredictor(latent_dim=192).to(DEVICE)

    # Load from checkpoint
    ae_path = CHECKPOINT_M1 / "ae_lambda_0.05_final.pt"
    if ae_path.exists():
        state = torch.load(ae_path, map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(
            {k.replace("encoder.", ""): v for k, v in state.items() if "encoder" in k},
            strict=False,
        )
    temp_path = CHECKPOINT_M2 / "temporal_final.pt"
    if temp_path.exists():
        predictor.load_state_dict(torch.load(temp_path, map_location=DEVICE, weights_only=True))

    encoder.eval()
    predictor.eval()

    frames = load_frames(str(video_path), max_frames=max_frames)
    latents = encode_frames(frames, encoder, DEVICE)

    surprise_scores = []
    context = []
    EPS = 1e-8

    with torch.no_grad():
        for t in range(latents.shape[0]):
            z_t = latents[t : t + 1]
            if len(context) > 0:
                context_tensor = torch.stack([c for c in context[-4:]], dim=0).unsqueeze(1)
                mu, _ = predictor(context_tensor.to(DEVICE))
                z_hat = mu[-1:]
                mse = torch.mean((z_t - z_hat.to(DEVICE)) ** 2).item()
                norm = torch.mean(torch.abs(z_hat)).item() + EPS
                s_t = mse / norm
                surprise_scores.append(s_t)
            context.append(z_t.cpu())

    return {
        "video": os.path.basename(video_path),
        "frames": len(latents),
        "surprise_scores": surprise_scores,
        "mean": float(np.mean(surprise_scores)) if surprise_scores else 0,
        "std": float(np.std(surprise_scores)) if surprise_scores else 0,
        "p95": float(np.percentile(surprise_scores, 95)) if surprise_scores else 0,
        "p99": float(np.percentile(surprise_scores, 99)) if surprise_scores else 0,
        "max": float(np.max(surprise_scores)) if surprise_scores else 0,
        "min": float(np.min(surprise_scores)) if surprise_scores else 0,
    }


def report_paper_values():
    """Paper's published surprise statistics."""
    print("\n" + "=" * 70)
    print("Section 5.5: Surprise Metric Statistics (Paper Values)")
    print("=" * 70)
    stats = [
        ("Mean", "0.596"),
        ("Std Dev", "0.008"),
        ("95th %ile", "0.608"),
        ("99th %ile", "0.612"),
        ("Maximum", "0.623"),
        ("τ_low", "0.4  (never breached)"),
        ("τ_high", "0.8  (never breached)"),
    ]
    print(tabulate(stats, headers=["Statistic", "Value"], tablefmt="grid"))


def main():
    print("=" * 70)
    print("Experiment 10: Surprise Metric Calibration (Section 5.5)")
    print("=" * 70)
    print(f"  Device: {DEVICE}\n")

    pevid_files = sorted((DATASET_DIR / "pevid-hd").glob("*.mpg"))
    if not pevid_files:
        print("  No PEViD-HD files found. Run 01_download_data.sh first.")
        sys.exit(1)

    all_surprise = []
    for vid in pevid_files[:3]:
        print(f"  Processing {vid.name}...")
        try:
            result = compute_surprise_for_video(str(vid), max_frames=50)
            all_surprise.extend(result["surprise_scores"])
            print(
                f"    mean={result['mean']:.4f}, max={result['max']:.4f}, n={len(result['surprise_scores'])}"
            )
        except Exception as e:
            print(f"    [warn] {e}")

    if all_surprise:
        print("\n" + "-" * 70)
        print("Computed Surprise Statistics:")
        print("-" * 70)
        computed = [
            ("Mean", f"{np.mean(all_surprise):.4f}"),
            ("Std Dev", f"{np.std(all_surprise):.4f}"),
            ("p95", f"{np.percentile(all_surprise, 95):.4f}"),
            ("p99", f"{np.percentile(all_surprise, 99):.4f}"),
            ("Max", f"{np.max(all_surprise):.4f}"),
        ]
        print(tabulate(computed, headers=["Statistic", "Value"], tablefmt="grid"))

    report_paper_values()


if __name__ == "__main__":
    main()
