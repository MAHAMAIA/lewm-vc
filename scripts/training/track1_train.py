#!/usr/bin/env python3
"""
Track 1: Expanded Dataset Training for LeWM-VC

Two-phase training on VIRAT Ground + full PEViD-HD:

Phase 1 — Predictor pre-training (20 epochs):
    Freeze encoder/decoder/entropy. Train only JEPA predictor on latent MSE.

Phase 2 — Joint fine-tuning (80 epochs):
    Unfreeze all. Train with rate-distortion loss L = λR + D.

Usage:
    # Phase 1
    python3 track1_train.py --data-dir datasets/virat_preprocessed/ \
        --phase pretrain --epochs 20 --batch-size 8 --lr 1e-3 \
        --checkpoint checkpoints/v0.1_encoder.pt

    # Phase 2
    python3 track1_train.py --data-dir datasets/virat_preprocessed/ \
        --phase joint --epochs 80 --batch-size 8 --lr 5e-5 \
        --lambda-rd 0.05 --checkpoint checkpoints/track1_pretrained.pt \
        --output checkpoints/track1_lambda_0.05.pt
"""

import os
import sys
import glob
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure lewm_vc is importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.predictor import LeWMPredictor
from lewm_vc.quant import Quantizer


# ── Dataset ────────────────────────────────────────────────────
class SurveillanceDataset(Dataset):
    """Load preprocessed PNG frames from a directory tree.
    Expects subdirectories per video, each containing sequential
    frame_0001.png, frame_0002.png, etc.
    """

    def __init__(
        self,
        data_dir: str,
        target_size: tuple = (256, 256),
        max_frames_per_video: int = 1000,
        gop_size: int = 16,
    ):
        self.target_size = target_size
        self.gop_size = gop_size
        self.samples = []  # list of (video_dir, start_idx)

        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Collect all PNG frame sequences
        print(f"Scanning {data_path} for frame sequences...")
        for video_dir in sorted(data_path.iterdir()):
            if not video_dir.is_dir():
                continue
            frames = sorted(video_dir.glob("frame_*.png"))
            if not frames:
                frames = sorted(video_dir.glob("*.png"))
            if not frames:
                continue

            n_frames = min(len(frames), max_frames_per_video)
            for i in range(0, n_frames - gop_size, gop_size):
                self.samples.append((str(video_dir), i))

        print(
            f"  Found {len(self.samples)} GOP segments across "
            f"{len(set(s[0] for s in self.samples))} videos"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, start = self.samples[idx]
        frames = []
        for i in range(start, start + self.gop_size):
            # Try several naming conventions
            for pattern in [
                f"frame_{i + 1:04d}.png",
                f"{i + 1:04d}.png",
                f"frame_{i:04d}.png",
                f"{i:04d}.png",
            ]:
                path = os.path.join(video_dir, pattern)
                if os.path.exists(path):
                    img = Image.open(path).convert("RGB").resize(self.target_size)
                    frames.append(np.array(img).astype(np.float32) / 255.0)
                    break

        # Pad if fewer frames than GOP
        while len(frames) < self.gop_size:
            frames.append(frames[-1].copy())

        tensor = torch.from_numpy(np.stack(frames)).float().permute(0, 3, 1, 2)
        return tensor  # [GOP, 3, H, W]


# ── Training ───────────────────────────────────────────────────
class Trainer:
    def __init__(
        self,
        encoder,
        decoder,
        predictor,
        entropy,
        quantizer,
        lambda_rd: float = 0.05,
        device: str = "cuda",
    ):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.predictor = predictor.to(device)
        self.entropy = entropy.to(device)
        self.quantizer = quantizer.to(device)
        self.lambda_rd = lambda_rd
        self.device = device
        self.mse_loss = nn.MSELoss()

    def phase1_pretrain(
        self, dataloader, epochs: int, lr: float, checkpoint_path: str, output_path: str
    ):
        """
        Phase 1: Freeze encoder/decoder/entropy, train only predictor.
        """
        # Freeze everything except predictor
        for model in [self.encoder, self.decoder, self.entropy]:
            for param in model.parameters():
                param.requires_grad_(False)
        for param in self.predictor.parameters():
            param.requires_grad_(True)

        optimizer = optim.Adam(self.predictor.parameters(), lr=lr)
        self.predictor.train()

        print(f"\nPhase 1: Predictor pre-training ({epochs} epochs)")
        print(f"  Learning rate: {lr}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Output: {output_path}")

        # Load pre-trained encoder/decoder weights
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.encoder.load_state_dict(
                {k.replace("encoder.", ""): v for k, v in state.items() if "encoder" in k},
                strict=False,
            )
            self.decoder.load_state_dict(
                {k.replace("decoder.", ""): v for k, v in state.items() if "decoder" in k},
                strict=False,
            )
            print(f"  Loaded encoder/decoder from {checkpoint_path}")
        else:
            print(f"  [warn] No checkpoint at {checkpoint_path}, using random init")

        best_loss = float("inf")
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
                batch = batch.to(self.device)  # [B, GOP, 3, H, W]
                B, G, C, H, W = batch.shape

                # Encode all frames in the GOP
                latents = []
                flat = batch.view(B * G, C, H, W)
                all_z = self.encoder(flat)
                latents = all_z.view(B, G, *all_z.shape[1:])

                # Predict each frame from up to 4 previous latents
                pred_loss = 0.0
                for t in range(1, G):
                    context = latents[:, max(0, t - 4) : t]
                    # Pad context to length 4 if needed
                    if context.shape[1] < 4:
                        pad = latents[:, :1].repeat(1, 4 - context.shape[1], 1, 1, 1)
                        context = torch.cat([pad, context], dim=1)
                    mu, _ = self.predictor(context)
                    pred_loss += self.mse_loss(mu[:, -1], latents[:, t])

                pred_loss = pred_loss / (G - 1)

                optimizer.zero_grad()
                pred_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
                optimizer.step()

                total_loss += pred_loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            print(f"  Loss: {avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.predictor.state_dict(), output_path.replace(".pt", "_best.pt"))

        # Save final
        torch.save(
            {
                "predictor": self.predictor.state_dict(),
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "entropy": self.entropy.state_dict(),
                "epoch": epochs,
                "best_loss": best_loss,
            },
            output_path,
        )
        print(f"  Saved to {output_path}")

    def phase2_joint(
        self, dataloader, epochs: int, lr: float, checkpoint_path: str, output_path: str
    ):
        """
        Phase 2: Joint fine-tuning of all components with RD loss.
        Loss: L = λR + D  where D = MSE(x, x_hat), R = entropy rate
        """
        # Unfreeze everything
        for model in [self.encoder, self.decoder, self.entropy, self.predictor]:
            for param in model.parameters():
                param.requires_grad_(True)

        params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.entropy.parameters())
            + list(self.predictor.parameters())
        )
        optimizer = optim.Adam(params, lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.encoder.train()
        self.decoder.train()
        self.entropy.train()
        self.predictor.train()

        # Load Phase 1 checkpoint
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            if "predictor" in state:
                self.predictor.load_state_dict(state["predictor"], strict=False)
                if "encoder" in state:
                    self.encoder.load_state_dict(
                        {k: v for k, v in state["encoder"].items() if "encoder" in k},
                        strict=False,
                    )
                print(f"  Loaded Phase 1 weights from {checkpoint_path}")
            else:
                self.predictor.load_state_dict(state, strict=False)
                print(f"  Loaded predictor from {checkpoint_path}")

        best_loss = float("inf")
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
                batch = batch.to(self.device)
                B, G, C, H, W = batch.shape

                # Batch-encode all frames: [B*G, 3, H, W] -> [B*G, D, H', W']
                flat = batch.view(B * G, C, H, W)
                all_z = self.encoder(flat)
                all_qz = self.quantizer(all_z)
                all_rec = self.decoder(all_qz)

                # Reshape back to [B, G, ...]
                all_z = all_z.view(B, G, *all_z.shape[1:])
                all_qz = all_qz.view(B, G, *all_qz.shape[1:])
                all_rec = all_rec.view(B, G, C, H, W)

                # Rate + distortion in one pass
                all_rate = self.entropy(all_qz.view(B * G, *all_qz.shape[2:]))[0]
                all_rate = all_rate.view(B, G, -1).sum(dim=(2,))
                all_dist = (all_rec - batch).pow(2).mean(dim=(2, 3, 4))

                rd_loss = (self.lambda_rd * all_rate + all_dist).mean()

                optimizer.zero_grad()
                rd_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                total_loss += rd_loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  Loss: {avg_loss:.4f}  (λ={self.lambda_rd})")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    {
                        "encoder": self.encoder.state_dict(),
                        "decoder": self.decoder.state_dict(),
                        "predictor": self.predictor.state_dict(),
                        "entropy": self.entropy.state_dict(),
                        "lambda": self.lambda_rd,
                        "epoch": epoch + 1,
                        "loss": avg_loss,
                    },
                    output_path.replace(".pt", "_best.pt"),
                )

        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "predictor": self.predictor.state_dict(),
                "entropy": self.entropy.state_dict(),
                "lambda": self.lambda_rd,
                "epoch": epochs,
                "loss": avg_loss,
            },
            output_path,
        )
        print(f"  Saved to {output_path}")


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Track 1: Expanded dataset training")
    parser.add_argument("--data-dir", required=True, help="Path to preprocessed frames")
    parser.add_argument("--phase", choices=["pretrain", "joint"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-rd", type=float, default=0.05)
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint to load")
    parser.add_argument("--output", default="checkpoints/track1_model.pt")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.05)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Track 1 Training — device: {device}")
    print(f"Phase: {args.phase}, epochs: {args.epochs}, batch: {args.batch_size}")
    print(f"λ_RD: {args.lambda_rd}, lr: {args.lr}")
    print(f"Data: {args.data_dir}")

    # Build models
    encoder = LeWMEncoder(latent_dim=192, patch_size=16)
    decoder = LeWMDecoder(latent_dim=192)
    predictor = LeWMPredictor(latent_dim=192, context_len=6)
    entropy = HyperpriorEntropy(latent_dim=192)
    quantizer = Quantizer(num_levels=256)

    n_params = (
        sum(p.numel() for p in encoder.parameters())
        + sum(p.numel() for p in decoder.parameters())
        + sum(p.numel() for p in predictor.parameters())
        + sum(p.numel() for p in entropy.parameters())
    )
    print(f"Total parameters: {n_params / 1e6:.1f}M")

    # Dataset
    dataset = SurveillanceDataset(args.data_dir)
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    print(f"Train: {train_size} samples, Val: {val_size} samples")

    # Trainer
    trainer = Trainer(
        encoder, decoder, predictor, entropy, quantizer, lambda_rd=args.lambda_rd, device=device
    )

    if args.phase == "pretrain":
        trainer.phase1_pretrain(
            train_loader, args.epochs, args.lr, args.checkpoint or "", args.output
        )
    else:
        trainer.phase2_joint(train_loader, args.epochs, args.lr, args.checkpoint or "", args.output)

    print("Done.")


if __name__ == "__main__":
    main()
