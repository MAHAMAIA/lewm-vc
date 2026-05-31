#!/usr/bin/env python3
"""
LeWM-VC Training Script for AMD/NVIDIA GPUs
Supports: AMD ROCm, NVIDIA CUDA, CPU fallback

Usage:
    python train_local.py --video-dir ./datasets/pevid-hd --epochs 100 --resolution 512
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from lewm_vc.encoder import LeWMEncoder


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.nn.functional.gelu(self.norm1(x))
        x = self.conv1(x)
        x = torch.nn.functional.gelu(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class LeWMDecoder(nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=512, output_channels=3):
        super().__init__()
        self.proj = nn.Conv2d(latent_dim, hidden_dim, kernel_size=1)

        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1)
        self.res1 = ResidualBlock(hidden_dim // 2)

        self.up2 = nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1)
        self.res2 = ResidualBlock(hidden_dim // 4)

        self.up3 = nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, 4, 2, 1)
        self.res3 = ResidualBlock(hidden_dim // 8)

        self.up4 = nn.ConvTranspose2d(hidden_dim // 8, hidden_dim // 16, 4, 2, 1)
        self.res4 = ResidualBlock(hidden_dim // 16)

        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim // 16, hidden_dim // 32, 3, padding=1),
            nn.InstanceNorm2d(hidden_dim // 32),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 32, output_channels, 3, padding=1),
        )

    def forward(self, latent, target_size=None):
        x = self.proj(latent)
        x = self.up1(x); x = self.res1(x)
        x = self.up2(x); x = self.res2(x)
        x = self.up3(x); x = self.res3(x)
        x = self.up4(x); x = self.res4(x)
        x = self.final(x)
        x = torch.sigmoid(x)
        if target_size:
            x = torch.nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class VideoDataset(Dataset):
    def __init__(self, video_paths, frame_size=(512, 512), frames_per_clip=4):
        self.videos = video_paths
        self.frame_size = frame_size
        self.frames_per_clip = frames_per_clip

    def __len__(self):
        return len(self.videos) * 200

    def __getitem__(self, idx):
        video_idx = idx % len(self.videos)
        cap = cv2.VideoCapture(self.videos[video_idx])

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = np.random.randint(0, max(1, total - self.frames_per_clip))

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for _ in range(self.frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)

        cap.release()
        return torch.from_numpy(np.stack(frames)).float()


class VideoAutoencoder(nn.Module):
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        x_flat = x.view(batch_size * num_frames, c, h, w)
        latent = self.encoder(x_flat, return_surprise=False)
        reconstructed = self.decoder(latent, target_size=(h, w))
        reconstructed = reconstructed.view(batch_size, num_frames, c, h, w)
        return reconstructed, latent


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda'), torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, 'hip') and torch.backends.hip.is_available():
        return torch.device('hip'), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "AMD GPU"
    return torch.device('cpu'), 'CPU'


def save_demo_video(model, dataloader, device, output_path, num_frames=50):
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader)).to(device)
        reconstructed, _ = model(batch[:1])
        
        frames = []
        for i in range(min(num_frames, reconstructed.shape[1])):
            recon_frame = reconstructed[0, i].permute(1, 2, 0).cpu().numpy()
            recon_frame = (recon_frame * 255).astype('uint8')
            
            orig_frame = batch[0, i].permute(1, 2, 0).cpu().numpy()
            orig_frame = (orig_frame * 255).astype('uint8')
            
            combined = np.hstack([orig_frame, recon_frame])
            frames.append(combined)
        
        if frames:
            h, w = frames[0].shape[:2]
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
            for frame in frames:
                out.write(frame)
            out.release()
            print(f"Demo saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='LeWM-VC Training')
    parser.add_argument('--video-dir', type=str, default='./datasets/pevid-hd', help='Directory with video files')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--resolution', type=int, default=512, help='Frame resolution (square)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_improved', help='Checkpoint directory')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    args = parser.parse_args()

    device, device_name = get_device()
    print(f"Training on: {device} ({device_name})")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")

    video_paths = (
        glob.glob(f'{args.video_dir}/*.mpg') +
        glob.glob(f'{args.video_dir}/*.mp4') +
        glob.glob(f'{args.video_dir}/*.avi')
    )

    if not video_paths:
        print(f"No videos found in {args.video_dir}")
        print("Download PEViD-HD dataset:")
        print("  wget https://cv.khu.ac.kr/PEViD-HD/PEViD-HD.zip")
        return

    print(f"Found {len(video_paths)} videos")

    dataset = VideoDataset(
        video_paths,
        frame_size=(args.resolution, args.resolution),
        frames_per_clip=4
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    model = VideoAutoencoder(latent_dim=192).to(device)

    perceptual_loss_fn = None
    try:
        import lpips
        perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
        print("LPIPS perceptual loss enabled")
    except ImportError:
        print("LPIPS not available, using only L1 loss")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    l1_loss_fn = nn.L1Loss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_l1 = 0
        total_perc = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            reconstructed, latent = model(batch)
            loss_l1 = l1_loss_fn(reconstructed, batch)

            if perceptual_loss_fn is not None and epoch > 5:
                b, f, c, h, w = reconstructed.shape
                recon_4d = reconstructed.view(b * f, c, h, w)
                batch_4d = batch.view(b * f, c, h, w)
                loss_perc = perceptual_loss_fn(recon_4d * 2 - 1, batch_4d * 2 - 1).mean()
                loss = loss_l1 + 0.1 * loss_perc
                total_perc += loss_perc.item()
            else:
                loss_perc = torch.tensor(0.0, device=device)
                loss = loss_l1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_l1 += loss_l1.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_loss = total_loss / num_batches
        avg_l1 = total_l1 / num_batches
        avg_perc = total_perc / max(1, num_batches)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, L1={avg_l1:.4f}, Perc={avg_perc:.4f}")

        if (epoch + 1) % 10 == 0:
            ckpt_path = f'{args.checkpoint_dir}/autoencoder_e{epoch+1}.pt'
            torch.save({
                'encoder': model.encoder.state_dict(),
                'decoder': model.decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }, ckpt_path)
            print(f"Saved {ckpt_path}")

            model.eval()
            with torch.no_grad():
                val_batch = next(iter(dataloader)).to(device)
                val_recon, _ = model(val_batch)
                val_loss = l1_loss_fn(val_recon, val_batch)
                print(f"Validation L1: {val_loss.item():.4f}")

    final_path = f'{args.checkpoint_dir}/autoencoder_final.pt'
    torch.save({
        'encoder': model.encoder.state_dict(),
        'decoder': model.decoder.state_dict(),
    }, final_path)
    print(f"Saved {final_path}")

    print("Generating demo video...")
    save_demo_video(model, dataloader, device, 'demo_trained_amd.mp4')
    print("Training complete!")


if __name__ == "__main__":
    main()
