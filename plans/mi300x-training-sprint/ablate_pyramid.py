"""
Step 2: FPN Level Ablation Study
Trains small compressors on P3 / P4 / P5 independently.
Measures reconstruction PSNR + BPP to recommend the primary pyramid level.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lewm_vc.feature_compress import FeatureCompressor, FeatureDecompressor
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.quant import Quantizer, QuantMode


# ---------------------------------------------------------------------------
# Feature dataset — loads pre-extracted .pt tensors from disk
# ---------------------------------------------------------------------------


class FPNFeatureDataset(Dataset):
    """Loads pre-extracted FPN features saved by extract_features.py."""

    def __init__(self, feature_dir: str, level: str, split: str):
        self.files = sorted(Path(feature_dir) / level / split).glob("feat_*.pt")
        self.files = list(self.files)
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {feature_dir}/{level}/{split}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)
        feat = data["features"]  # [256, H, W]
        return {"features": feat, "path": str(self.files[idx])}


def collate_features(batch):
    feats = torch.stack([b["features"] for b in batch])
    paths = [b["path"] for b in batch]
    return {"features": feats, "path": paths}


# ---------------------------------------------------------------------------
# Reconstruction + rate metrics
# ---------------------------------------------------------------------------


def psnr(mse: float) -> float:
    if mse == 0:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


@torch.no_grad()
def evaluate(
    compressor: nn.Module,
    decompressor: nn.Module,
    entropy: nn.Module,
    quantizer: Quantizer,
    loader: DataLoader,
    device: str,
) -> dict:
    compressor.eval()
    decompressor.eval()
    entropy.eval()

    total_mse = 0.0
    total_bpp = 0.0
    n = 0

    for batch in loader:
        x = batch["features"].to(device)  # [B, 256, H, W]
        B = x.shape[0]

        latent = compressor(x)  # [B, D, H', W']
        qz = quantizer(latent)
        x_hat = decompressor(qz)  # [B, 256, H, W]

        mse = F.mse_loss(x_hat, x).item()
        h, w = x.shape[2:]
        n_pixels = h * w

        _, params = entropy(latent)
        rate = entropy.gaussian_kl(latent, params["mu"], params["sigma"])  # [B,1,H,W]
        bpp = rate.mean().item() / n_pixels

        total_mse += mse * B
        total_bpp += bpp * B
        n += B

    avg_mse = total_mse / n
    return {
        "mse": avg_mse,
        "psnr": psnr(avg_mse),
        "bpp": total_bpp / n,
    }


# ---------------------------------------------------------------------------
# One ablation run
# ---------------------------------------------------------------------------


def run_ablation(
    level: str,
    train_dir: str,
    val_dir: str,
    latent_dim: int,
    hyper_channels: int,
    batch_size: int,
    lr: float,
    warmup_steps: int,
    train_steps: int,
    device: str,
    seed: int,
    out_dir: str,
) -> dict:
    torch.manual_seed(seed)

    train_ds = FPNFeatureDataset(train_dir, level, "train")
    val_ds = FPNFeatureDataset(train_dir, level, "val")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_features,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_features,
    )

    # Models
    compressor = FeatureCompressor(in_channels=256, latent_dim=latent_dim).to(device)
    decompressor = FeatureDecompressor(latent_dim=latent_dim, out_channels=256).to(device)
    entropy_model = HyperpriorEntropy(latent_dim=latent_dim, hyper_channels=hyper_channels).to(
        device
    )
    quantizer = Quantizer(num_levels=256, mode=QuantMode.TRAINING).to(device)

    params = (
        list(compressor.parameters())
        + list(decompressor.parameters())
        + list(entropy_model.parameters())
    )
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=train_steps)

    train_iter = iter(train_loader)
    best_val_loss = float("-inf")
    best_ckpt = None
    best_metrics = None
    step = 0
    log_interval = 50
    val_interval = 500

    pbar = tqdm(total=train_steps, desc=f"Ablation {level}")
    while step < train_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        step += 1
        x = batch["features"].to(device)  # [B, 256, H, W]

        # Phase 1: warmup — no rate loss
        if step <= warmup_steps:
            latent = compressor(x)
            qz = quantizer(latent)
            x_hat = decompressor(qz)
            loss = F.mse_loss(x_hat, x)
        # Phase 2: rate-aware training
        else:
            latent = compressor(x)
            qz = quantizer(latent)
            x_hat = decompressor(qz)

            task_loss = F.mse_loss(x_hat, x)
            _, params = entropy_model(latent)
            rate = entropy_model.gaussian_kl(latent, params["mu"], params["sigma"])
            n_pixels = x.shape[2] * x.shape[3]
            bpp = rate.mean() / n_pixels
            rd_lambda = 5.0
            loss = task_loss + rd_lambda * bpp

        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        scheduler.step()

        if step % log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(log_interval)

        if step % val_interval == 0 or step == train_steps:
            metrics = evaluate(
                compressor, decompressor, entropy_model, quantizer, val_loader, device
            )
            val_loss = metrics["psnr"] - 10 * max(metrics["bpp"] - 0.5, 0)
            if val_loss > best_val_loss:
                best_val_loss = val_loss
                best_metrics = metrics
                best_ckpt = {
                    "level": level,
                    "step": step,
                    "compressor": compressor.state_dict(),
                    "decompressor": decompressor.state_dict(),
                    "entropy": entropy_model.state_dict(),
                    "metrics": metrics,
                    "latent_dim": latent_dim,
                    "hyper_channels": hyper_channels,
                }

    pbar.close()

    # Save result
    result = {
        "level": level,
        "best_step": best_ckpt["step"],
        "best_val_loss": best_val_loss,
        "psnr": best_metrics["psnr"],
        "mse": best_metrics["mse"],
        "bpp": best_metrics["bpp"],
        "latent_dim": latent_dim,
        "hyper_channels": hyper_channels,
    }

    ckpt_dir = Path(out_dir) / "checkpoints" / f"ablation_{level}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_ckpt, str(ckpt_dir / "best.pt"))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="FPN level ablation study")
    parser.add_argument(
        "--feature-dir",
        type=str,
        default="datasets/fpn_features",
        help="Directory with pre-extracted FPN features (P3/P4/P5 splits)",
    )
    parser.add_argument(
        "--levels", type=str, nargs="+", default=["P3", "P4", "P5"], help="Pyramid levels to ablate"
    )
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hyper-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="ablations/fpn_pyramid",
        help="Output directory for ablation results",
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = []
    for level in args.levels:
        print(f"\n{'=' * 60}")
        print(f"  Ablating level {level}")
        print(f"{'=' * 60}")
        t0 = time.time()
        result = run_ablation(
            level=level,
            train_dir=args.feature_dir,
            val_dir=args.feature_dir,
            latent_dim=args.latent_dim,
            hyper_channels=args.hyper_channels,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            train_steps=args.train_steps,
            device=device,
            seed=args.seed,
            out_dir=args.output,
        )
        result["train_seconds"] = time.time() - t0
        results.append(result)
        print(
            f"  {level}: PSNR={result['psnr']:.2f} dB, "
            f"MSE={result['mse']:.6f}, BPP={result['bpp']:.4f} "
            f"[{result['train_seconds']:.0f}s]"
        )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Ablation Summary")
    print(f"{'=' * 60}")
    print(f"{'Level':<6} {'PSNR(dB)':<10} {'MSE':<12} {'BPP':<8} {'RD Score':<10}")
    print("-" * 46)
    best_level = None
    best_rd_score = float("-inf")
    for r in results:
        rd_score = r["psnr"] - 10 * max(r["bpp"] - 0.5, 0)
        print(
            f"{r['level']:<6} {r['psnr']:<10.2f} {r['mse']:<12.6f} {r['bpp']:<8.4f} {rd_score:<10.2f}"
        )
        if rd_score > best_rd_score:
            best_rd_score = rd_score
            best_level = r["level"]

    print(f"\n  Recommended primary level: {best_level} (RD score {best_rd_score:.2f})")

    # Write report
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "recommended_level": best_level,
        "results": results,
        "config": vars(args),
    }
    report_path = out_dir / "ablation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {report_path}")


if __name__ == "__main__":
    main()
