#!/usr/bin/env python3
"""
Stable entropy model training with discrete Gaussian likelihood.
Prevents sigma collapse, uses correct quantizer step size.
"""

import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.distributions.normal import Normal
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---------- Model definitions (same as before) ----------
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

class AffineNormalization(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    def forward(self, x):
        return x * self.scale + self.shift

class VideoAutoencoderWithAffine(nn.Module):
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
        self.affine = AffineNormalization(latent_dim)
    def encode(self, x):
        latent = self.encoder(x, return_surprise=False)
        return self.affine(latent)

# ---------- Load frozen autoencoder ----------
autoencoder = VideoAutoencoderWithAffine(latent_dim=192).to(device)
ae_checkpoint = '/root/le-maia/checkpoints_affine/autoencoder_affine_final.pt'
autoencoder.load_state_dict(torch.load(ae_checkpoint, map_location=device), strict=False)
autoencoder.eval()
for param in autoencoder.parameters():
    param.requires_grad = False
print("Autoencoder loaded and frozen.")

# ---------- Dataset ----------
class LatentDataset(Dataset):
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

video_paths = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')
dataset = LatentDataset(video_paths, frame_size=(256,256), frames_per_clip=4)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

# ---------- Quantizer diagnostic ----------
print("\n🔍 Quantizer diagnostic (first batch):")
with torch.no_grad():
    sample_batch = next(iter(dataloader)).to(device)
    B, T, C, H, W = sample_batch.shape
    flat = sample_batch.view(B*T, C, H, W)
    latent_norm = autoencoder.encode(flat)
    quantizer = Quantizer(num_levels=256, mode='inference').to(device)
    quantized = quantizer(latent_norm)
    print(f"  latent_norm: min={latent_norm.min():.4f}, max={latent_norm.max():.4f}, std={latent_norm.std():.4f}")
    print(f"  quantized:   min={quantized.min():.4f}, max={quantized.max():.4f}, std={quantized.std():.4f}")
    uniq = torch.unique(quantized.flatten()[:5000])
    print(f"  unique values (first 5000): {len(uniq)}")
    if len(uniq) > 1:
        step = (uniq.max() - uniq.min()) / (len(uniq) - 1)
        print(f"  approx step size: {step.item():.6f}")
print("===================================\n")

# Determine quantizer step (assuming uniform 256 levels over [-1,1])
QUANT_STEP = 2.0 / 255  # because 256 levels cover [-1,1] → step = (1 - (-1)) / (256-1) ≈ 0.00784
print(f"Using quantizer step = {QUANT_STEP:.6f} (256 levels over [-1,1])\n")

# ---------- Entropy model and quantizer (training mode) ----------
entropy_model = HyperpriorEntropy(latent_dim=192, hyper_channels=320).to(device)
quantizer_train = Quantizer(num_levels=256, mode='training').to(device)

# Resume if checkpoint exists
checkpoint_dir = '/root/le-maia/checkpoints_entropy_stable'
os.makedirs(checkpoint_dir, exist_ok=True)
resume_ckpt = f"{checkpoint_dir}/entropy_final.pt"
if os.path.exists(resume_ckpt):
    entropy_model.load_state_dict(torch.load(resume_ckpt, map_location=device))
    print(f"Resumed from {resume_ckpt}")

# ---------- Corrected discrete likelihood with step size ----------
def gaussian_likelihood_discrete(y, mu, log_sigma, step, epsilon=1e-9):
    """Discrete Gaussian likelihood using CDF difference, with given step size."""
    sigma = torch.nn.functional.softplus(log_sigma) + 0.1   # floor to prevent collapse
    sigma = torch.clamp(sigma, min=0.1, max=10.0)
    # CDF bounds for bin [y - step/2, y + step/2]
    lower = (y - 0.5 * step - mu) / sigma
    upper = (y + 0.5 * step - mu) / sigma
    normal = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
    cdf_upper = normal.cdf(upper)
    cdf_lower = normal.cdf(lower)
    pmf = cdf_upper - cdf_lower
    pmf = torch.clamp(pmf, min=epsilon, max=1.0)
    nll = -torch.log(pmf)
    return nll.mean()

# Optional: sigma regularizer to prevent collapse
def sigma_regularizer(sigma, quantized, mu):
    """Encourage sigma not to be too small compared to absolute residual."""
    residual_std = (quantized - mu).abs().mean()
    # Penalize sigma < 0.5 * residual_std
    penalty = torch.mean(torch.relu(0.5 * residual_std - sigma))
    return penalty

optimizer = optim.AdamW(entropy_model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

EPOCHS = 100

for epoch in range(1, EPOCHS+1):
    entropy_model.train()
    total_nll = 0.0
    total_bpp = 0.0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    
    for batch in pbar:
        batch = batch.to(device)
        B, T, C, H, W = batch.shape
        batch_flat = batch.view(B*T, C, H, W)
        
        with torch.no_grad():
            latent_norm = autoencoder.encode(batch_flat)
        
        # Quantize (training mode)
        quantized = quantizer_train(latent_norm)
        
        # Predict mu, log_sigma
        params = entropy_model.hyperprior_cnn(quantized)
        mu = params[:, :192, :, :]
        log_sigma = params[:, 192:, :, :]
        
        # Stabilized sigma
        sigma = torch.nn.functional.softplus(log_sigma) + 0.1
        sigma = torch.clamp(sigma, min=0.1, max=10.0)
        
        # Loss: NLL + small sigma regularizer
        nll = gaussian_likelihood_discrete(quantized, mu, log_sigma, step=QUANT_STEP)
        reg = sigma_regularizer(sigma, quantized, mu)
        loss = nll + 0.01 * reg
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(entropy_model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        total_nll += nll.item()
        bpp = nll.item() / np.log(2)
        total_bpp += bpp
        num_batches += 1
        
        pbar.set_postfix(
            bpp=f"{bpp:.4f}",
            sigma=f"{sigma.mean().item():.4f}",
            loss=f"{loss.item():.4f}"
        )
    
    scheduler.step()
    avg_nll = total_nll / num_batches
    avg_bpp = total_bpp / num_batches
    print(f"Epoch {epoch}: Avg NLL = {avg_nll:.6f} → {avg_bpp:.4f} bpp")
    
    if epoch % 10 == 0 or epoch == EPOCHS:
        torch.save(entropy_model.state_dict(), f'{checkpoint_dir}/entropy_epoch{epoch}.pt')

torch.save(entropy_model.state_dict(), f'{checkpoint_dir}/entropy_final.pt')
print("✅ Stable entropy model training complete.")
