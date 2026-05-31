#!/usr/bin/env python3
"""
Phase 1 Training: Joint Rate-Distortion Optimization
- Trains autoencoder + hyperprior entropy model end-to-end
- Uses λ to trade off rate vs. distortion
- Saves RD curve checkpoints
"""

import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from tqdm import tqdm

# Add repo to path
sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.quant import Quantizer

# ------------------------------
# Device
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ------------------------------
# Residual block and decoder (same as before)
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        residual = x
        x = torch.nn.functional.gelu(self.norm1(x))
        x = self.conv1(x)
        x = torch.nn.functional.gelu(self.norm2(x))
        x = self.conv2(x)
        return x + residual

class LeWMDecoder(nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(latent_dim, hidden_dim, 1)
        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4,2,1)
        self.res1 = ResidualBlock(hidden_dim//2)
        self.up2 = nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 4,2,1)
        self.res2 = ResidualBlock(hidden_dim//4)
        self.up3 = nn.ConvTranspose2d(hidden_dim//4, hidden_dim//8, 4,2,1)
        self.res3 = ResidualBlock(hidden_dim//8)
        self.up4 = nn.ConvTranspose2d(hidden_dim//8, hidden_dim//16, 4,2,1)
        self.res4 = ResidualBlock(hidden_dim//16)
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim//16, hidden_dim//32, 3,1,1),
            nn.InstanceNorm2d(hidden_dim//32),
            nn.GELU(),
            nn.Conv2d(hidden_dim//32, 3, 3,1,1),
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

class VideoAutoencoder(nn.Module):
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
    def forward(self, x):
        b, t, c, h, w = x.shape
        x_flat = x.view(b*t, c, h, w)
        latent = self.encoder(x_flat, return_surprise=False)
        recon = self.decoder(latent, target_size=(h,w))
        recon = recon.view(b, t, c, h, w)
        return recon, latent

# ------------------------------
# Dataset (same as before)
# ------------------------------
class VideoDataset(Dataset):
    def __init__(self, video_paths, frame_size=(256,256), frames_per_clip=4):
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
            frame = np.transpose(frame, (2,0,1))
            frames.append(frame)
        cap.release()
        return torch.from_numpy(np.stack(frames)).float()

# ------------------------------
# Full model with entropy model and quantizer
# ------------------------------
class FullCodec(nn.Module):
    def __init__(self, autoencoder, entropy_model, quantizer):
        super().__init__()
        self.autoencoder = autoencoder
        self.entropy_model = entropy_model
        self.quantizer = quantizer

    def forward(self, x, lambda_val):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x_flat = x.view(B*T, C, H, W)
        # Encode
        latent = self.autoencoder.encoder(x_flat, return_surprise=False)
        # Quantize
        quantized = self.quantizer(latent)
        # Rate estimation (in bits)
        rate_nats, _ = self.entropy_model(quantized)
        rate_bits = rate_nats.sum() * torch.log2(torch.tensor(np.e)).to(x.device)
        # Decode
        recon = self.autoencoder.decoder(quantized, target_size=(H,W))
        recon = recon.view(B, T, C, H, W)
        # Distortion (MSE)
        mse = torch.nn.functional.mse_loss(recon, x)
        # Perceptual loss (simplified placeholder – you can add LPIPS later)
        perceptual = mse  # replace with LPIPS if available
        # Surprise (placeholder)
        surprise = torch.tensor(0.0, device=x.device)
        # Total loss
        loss = lambda_val * rate_bits + (0.7 * mse + 0.3 * perceptual) + 0.01 * surprise
        return loss, recon, rate_bits, mse

# ------------------------------
# Training function
# ------------------------------
def train_phase1(lambda_val, epochs=30, batch_size=4, lr=1e-4):
    # Load pre-trained autoencoder (from Phase 0)
    autoencoder = VideoAutoencoder().to(device)
    checkpoint_path = '/root/le-maia/checkpoints/autoencoder_final.pt'
    if os.path.exists(checkpoint_path):
        autoencoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded pre-trained autoencoder")
    else:
        print("No pre-trained autoencoder found; training from scratch")

    entropy_model = HyperpriorEntropy(latent_dim=192, hyper_channels=320).to(device)
    quantizer = Quantizer(num_levels=256, mode='training').to(device)

    model = FullCodec(autoencoder, entropy_model, quantizer).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Dataset
    video_paths = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')
    if not video_paths:
        raise FileNotFoundError("No videos found")
    dataset = VideoDataset(video_paths, frame_size=(256,256), frames_per_clip=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    os.makedirs(f'/root/le-maia/checkpoints/phase1_lambda_{lambda_val}', exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        total_rate = 0
        total_mse = 0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"λ={lambda_val} Epoch {epoch}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, recon, rate_bits, mse = model(batch, lambda_val)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_rate += rate_bits.item()
            total_mse += mse.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item(), rate=rate_bits.item())
        scheduler.step()
        avg_loss = total_loss / num_batches
        avg_rate = total_rate / num_batches
        avg_mse = total_mse / num_batches
        print(f"λ={lambda_val} Epoch {epoch}: Loss={avg_loss:.4f}, Rate={avg_rate:.2f} bits, MSE={avg_mse:.6f}")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'/root/le-maia/checkpoints/phase1_lambda_{lambda_val}/epoch_{epoch}.pt')

    # Save final model for this λ
    torch.save(model.state_dict(), f'/root/le-maia/checkpoints/phase1_lambda_{lambda_val}/final.pt')
    print(f"Training for λ={lambda_val} complete.")

# ------------------------------
# Main: train multiple λ values to build RD curve
# ------------------------------
if __name__ == "__main__":
    # List of λ values to sweep (increase for lower bitrate, higher distortion)
    lambda_list = [0.001, 0.01, 0.1, 1.0, 10.0]
    for lam in lambda_list:
        train_phase1(lambda_val=lam, epochs=30)
