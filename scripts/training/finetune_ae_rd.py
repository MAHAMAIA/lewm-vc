#!/usr/bin/env python3
"""
finetune_ae_rd.py – Fine‑tune affine autoencoder with frozen entropy model.
Sweeps λ values, saves checkpoints, logs RD points.
"""

import os
import sys
import glob
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.quant import Quantizer

# ---------- Device ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---------- Dataset (same as before) ----------
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

# ---------- Affine autoencoder architecture (same as training) ----------
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
    def forward(self, x):
        b, t, c, h, w = x.shape
        x_flat = x.view(b*t, c, h, w)
        latent = self.encoder(x_flat, return_surprise=False)
        latent_norm = self.affine(latent)
        recon = self.decoder(latent_norm, target_size=(h,w))
        recon = recon.view(b, t, c, h, w)
        return recon, latent_norm
    def encode(self, x):
        latent = self.encoder(x, return_surprise=False)
        return self.affine(latent)

# ---------- Load pre-trained models ----------
# Affine autoencoder (reconstruction trained)
ae_checkpoint = '/root/le-maia/checkpoints_affine/autoencoder_affine_final.pt'
autoencoder = VideoAutoencoderWithAffine(latent_dim=192).to(device)
autoencoder.load_state_dict(torch.load(ae_checkpoint, map_location=device), strict=False)
autoencoder.train()
print("Autoencoder loaded.")

# Entropy model (NLL trained, should give non-zero bits)
entropy_model = HyperpriorEntropy(latent_dim=192, hyper_channels=320).to(device)
entropy_ckpt = '/root/le-maia/checkpoints_entropy_affine_v3/entropy_final.pt'  # update if you have a better one
entropy_model.load_state_dict(torch.load(entropy_ckpt, map_location=device))
entropy_model.eval()
for param in entropy_model.parameters():
    param.requires_grad = False
print("Entropy model loaded and frozen.")

# Quantizer (with uniform noise for training)
quantizer = Quantizer(num_levels=256, mode='training').to(device)

# ---------- Dataset ----------
video_paths = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')
if not video_paths:
    raise FileNotFoundError("No videos found")
dataset = VideoDataset(video_paths, frame_size=(256,256), frames_per_clip=4)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

# ---------- Loss functions ----------
criterion_mse = nn.MSELoss()
try:
    import lpips
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    use_perceptual = True
except:
    use_perceptual = False
    perceptual_loss_fn = None

def rate_loss(quantized, entropy_model):
    """Compute rate (bits) using entropy model's KL divergence."""
    with torch.no_grad():
        rate_nats, _ = entropy_model(quantized)
        bits = rate_nats.sum() * np.log2(np.e)
    return bits

# ---------- Fine-tune for each λ ----------
lambda_list = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
output_dir = '/root/le-maia/checkpoints_finetuned_rd'
os.makedirs(output_dir, exist_ok=True)

for lam in lambda_list:
    print(f"\n========== Fine-tuning with λ = {lam} ==========")
    # Reset autoencoder to original weights for each λ
    autoencoder.load_state_dict(torch.load(ae_checkpoint, map_location=device), strict=False)
    autoencoder.train()
    optimizer = optim.AdamW(autoencoder.parameters(), lr=1e-5, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    best_psnr = 0.0
    for epoch in range(1, 101):
        total_loss = 0.0
        total_rate = 0.0
        total_mse = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"λ={lam} Epoch {epoch}/100")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward
            recon, latent_norm = autoencoder(batch)
            quantized = quantizer(latent_norm)
            
            # Rate (KL divergence) – no gradient to entropy model
            rate_bits = rate_loss(quantized, entropy_model)
            
            # Distortion
            mse = criterion_mse(recon, batch)
            if use_perceptual and epoch > 10:
                b, t, c, h, w = recon.shape
                recon_4d = recon.view(b*t, c, h, w)
                batch_4d = batch.view(b*t, c, h, w)
                perceptual = perceptual_loss_fn(recon_4d*2-1, batch_4d*2-1).mean()
                distortion = mse + 0.1 * perceptual
            else:
                distortion = mse
            
            loss = lam * rate_bits + distortion
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_rate += rate_bits.item()
            total_mse += mse.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item(), rate=rate_bits.item(), mse=mse.item())
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        avg_rate = total_rate / num_batches
        avg_mse = total_mse / num_batches
        psnr = 10 * np.log10(1.0 / avg_mse) if avg_mse > 0 else 0
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Rate={avg_rate:.2f} bits, MSE={avg_mse:.6f}, PSNR={psnr:.2f} dB")
        
        if epoch % 10 == 0:
            ckpt_path = os.path.join(output_dir, f"ae_lambda_{lam}_epoch_{epoch}.pt")
            torch.save(autoencoder.state_dict(), ckpt_path)
            # Save RD point (bits per pixel)
            bpp = avg_rate / (256 * 256)  # 256x256 frames
            with open(os.path.join(output_dir, f"rd_lambda_{lam}.csv"), 'a') as f:
                f.write(f"{epoch},{bpp},{psnr}\n")
    
    # Final checkpoint for this λ
    final_ckpt = os.path.join(output_dir, f"ae_lambda_{lam}_final.pt")
    torch.save(autoencoder.state_dict(), final_ckpt)
    print(f"Finished λ={lam}. Final PSNR: {psnr:.2f} dB, Rate: {avg_rate:.2f} bits, bpp: {avg_rate/(256*256):.4f}")

print("\nAll λ runs completed.")
