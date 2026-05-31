#!/usr/bin/env python3
"""
Phase 1 training for affine autoencoder: joint rate-distortion optimization.
Trains autoencoder + entropy model together to minimize: loss = lambda * rate + distortion
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

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---------- Affine autoencoder architecture ----------
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
    def inverse(self, y):
        return (y - self.shift) / (self.scale + 1e-8)

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
    def decode(self, latent_norm, target_size):
        return self.decoder(latent_norm, target_size=target_size)

# ---------- Dataset ----------
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

video_paths = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')
if not video_paths:
    raise FileNotFoundError("No videos found")
print(f"Found {len(video_paths)} videos")

dataset = VideoDataset(video_paths, frame_size=(256,256), frames_per_clip=4)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

# ---------- Model ----------
autoencoder = VideoAutoencoderWithAffine(latent_dim=192).to(device)
entropy_model = HyperpriorEntropy(latent_dim=192, hyper_channels=320).to(device)
quantizer = Quantizer(num_levels=256, mode='training').to(device)

# Optionally load pre-trained affine weights (from earlier reconstruction training)
checkpoint_affine = '/root/le-maia/checkpoints_affine/autoencoder_affine_final.pt'
if os.path.exists(checkpoint_affine):
    autoencoder.load_state_dict(torch.load(checkpoint_affine, map_location=device), strict=False)
    print("Loaded pre-trained affine autoencoder weights (will fine-tune).")

# Optimizer for all components
optimizer = optim.AdamW(
    list(autoencoder.parameters()) + list(entropy_model.parameters()),
    lr=1e-4, weight_decay=0.01
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Loss functions
criterion_mse = nn.MSELoss()

# Perceptual loss (optional)
try:
    import lpips
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    use_perceptual = True
    print("LPIPS enabled")
except:
    use_perceptual = False
    print("LPIPS not available, using MSE only")

# Rate-distortion tradeoff parameter (lambda)
LAMBDA = 0.01  # adjust this: higher = more bitrate penalty

EPOCHS = 100
os.makedirs('/root/le-maia/checkpoints_affine_phase1', exist_ok=True)

for epoch in range(1, EPOCHS+1):
    autoencoder.train()
    entropy_model.train()
    total_loss = 0
    total_rate = 0
    total_dist = 0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward through autoencoder
        recon, latent_norm = autoencoder(batch)  # latent_norm: [B*T, 192, 16, 16]
        
        # Quantize and compute rate
        quantized = quantizer(latent_norm)
        rate_nats, _ = entropy_model(quantized)
        rate_bits = rate_nats.sum() * np.log2(np.e)
        
        # Distortion loss (MSE + perceptual)
        mse_loss = criterion_mse(recon, batch)
        if use_perceptual and epoch > 5:
            b, t, c, h, w = recon.shape
            recon_4d = recon.view(b*t, c, h, w)
            batch_4d = batch.view(b*t, c, h, w)
            perceptual_loss = perceptual_loss_fn(recon_4d*2-1, batch_4d*2-1).mean()
            distortion = mse_loss + 0.1 * perceptual_loss
        else:
            distortion = mse_loss
        
        # Total loss: rate + distortion
        loss = LAMBDA * rate_bits + distortion
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(entropy_model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_rate += rate_bits.item()
        total_dist += distortion.item()
        num_batches += 1
        pbar.set_postfix(loss=loss.item(), rate=rate_bits.item(), dist=distortion.item())
    
    scheduler.step()
    avg_loss = total_loss / num_batches
    avg_rate = total_rate / num_batches
    avg_dist = total_dist / num_batches
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Rate={avg_rate:.2f} bits, Dist={avg_dist:.6f}")
    
    if epoch % 20 == 0:
        torch.save({
            'autoencoder': autoencoder.state_dict(),
            'entropy_model': entropy_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }, f'/root/le-maia/checkpoints_affine_phase1/checkpoint_epoch{epoch}.pt')

# Save final models
torch.save(autoencoder.state_dict(), '/root/le-maia/checkpoints_affine_phase1/autoencoder_final.pt')
torch.save(entropy_model.state_dict(), '/root/le-maia/checkpoints_affine_phase1/entropy_final.pt')
print("Training complete. Models saved to checkpoints_affine_phase1/")
