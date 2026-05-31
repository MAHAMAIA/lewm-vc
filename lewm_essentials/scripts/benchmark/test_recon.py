#!/usr/bin/env python3
import sys
import os
import glob
import cv2
import torch
import numpy as np

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---------- Define the affine autoencoder (same as eval script) ----------
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = torch.nn.InstanceNorm2d(channels)
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = torch.nn.InstanceNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        residual = x
        x = torch.nn.functional.gelu(self.norm1(x))
        x = self.conv1(x)
        x = torch.nn.functional.gelu(self.norm2(x))
        x = self.conv2(x)
        return x + residual

class LeWMDecoder(torch.nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=512):
        super().__init__()
        self.proj = torch.nn.Conv2d(latent_dim, hidden_dim, 1)
        self.up1 = torch.nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4,2,1)
        self.res1 = ResidualBlock(hidden_dim//2)
        self.up2 = torch.nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 4,2,1)
        self.res2 = ResidualBlock(hidden_dim//4)
        self.up3 = torch.nn.ConvTranspose2d(hidden_dim//4, hidden_dim//8, 4,2,1)
        self.res3 = ResidualBlock(hidden_dim//8)
        self.up4 = torch.nn.ConvTranspose2d(hidden_dim//8, hidden_dim//16, 4,2,1)
        self.res4 = ResidualBlock(hidden_dim//16)
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim//16, hidden_dim//32, 3,1,1),
            torch.nn.InstanceNorm2d(hidden_dim//32),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden_dim//32, 3, 3,1,1),
        )
        self.post_filter = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3,1,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 3, 3,1,1),
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
        x = self.post_filter(x)
        x = self.post_filter(x)
        return x

class AffineNormalization(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.shift = torch.nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    def forward(self, x):
        return x * self.scale + self.shift

class VideoAutoencoderWithAffine(torch.nn.Module):
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
        self.affine = AffineNormalization(latent_dim)
    def encode(self, x):
        latent = self.encoder(x, return_surprise=False)
        return self.affine(latent)
    def decode(self, latent_norm, target_size):
        return self.decoder(latent_norm, target_size=target_size)

# Load checkpoint
lam = 0.05
ae_ckpt = f'/root/le-maia/checkpoints_rd_scratch/ae_lambda_{lam}_final.pt'
autoencoder = VideoAutoencoderWithAffine().to(device)
autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device, weights_only=False), strict=False)
autoencoder.eval()

# Load a single frame from test video
test_video = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')[0]
cap = cv2.VideoCapture(test_video)
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Could not read frame")
frame = cv2.resize(frame, (64,64))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
frame_t = frame_t.to(device)

# Encode and decode (no quantization, no entropy)
with torch.no_grad():
    latent = autoencoder.encode(frame_t)
    recon = autoencoder.decode(latent, target_size=(64,64))

# Compute PSNR
mse = torch.nn.functional.mse_loss(recon, frame_t).item()
psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100
print(f"PSNR (autoencoder only): {psnr:.2f} dB")

# Save images
orig_np = (frame_t.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
recon_np = (recon.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
cv2.imwrite('/tmp/orig.png', cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR))
cv2.imwrite('/tmp/recon.png', cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR))
print("Saved original to /tmp/orig.png and reconstructed to /tmp/recon.png")
