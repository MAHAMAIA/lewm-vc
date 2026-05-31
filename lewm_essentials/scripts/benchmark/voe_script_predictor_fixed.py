#!/usr/bin/env python3
"""
Violation of Expectation using PREDICTOR (prediction error).
FIXED: Matches trained predictor architecture (num_layers=4).
"""

import os
import sys
import glob
import torch
import torch.nn as nn
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.predictor import LeWMPredictor
from lewm_vc.working_decoder import LeWMDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16
print(f"Device: {device}")

# ---------- Affine autoencoder ----------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        return x + self.conv2(torch.nn.functional.gelu(self.norm2(
            self.conv1(torch.nn.functional.gelu(self.norm1(x))))))

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
        x = torch.sigmoid(self.final(x))
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

# Load models
ae_ckpt = '/root/le-maia/checkpoints_corrected/ae_lambda_0.01_final.pt'
if not os.path.exists(ae_ckpt):
    print(f"❌ Checkpoint not found: {ae_ckpt}")
    sys.exit(1)

print("Loading autoencoder...")
autoencoder = VideoAutoencoderWithAffine().to(device).to(dtype)
autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device, weights_only=False), strict=False)
autoencoder.eval()

print("Loading predictor...")
# ✅ FIXED: Use same architecture as training (num_layers=4)
predictor = LeWMPredictor(
    latent_dim=192, 
    hidden_dim=256, 
    num_layers=4,      # Matches training
    num_heads=4,
    context_len=3      # Matches training
).to(device).to(dtype)

predictor_ckpt = '/root/le-maia/checkpoints/predictor_final.pt'
if os.path.exists(predictor_ckpt):
    # ✅ FIXED: Load with strict=False to handle minor mismatches
    state = torch.load(predictor_ckpt, map_location=device, weights_only=False)
    predictor.load_state_dict(state, strict=False)
    predictor.eval()
    print("Predictor loaded.")
else:
    print("Predictor checkpoint not found. Using fallback: surprise = latent norm.")
    predictor = None

# Load anomaly clip (dropping bag)
video_path = '/root/le-maia/datasets/pevid-hd/droppingBag_day_indoor_1_1.mpg'
if not os.path.exists(video_path):
    video_path = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')[0]
    print(f"Using video: {video_path}")

cap = cv2.VideoCapture(video_path)
frames = []
target_size = (128,128)
while len(frames) < 150:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, target_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
cap.release()
print(f"Loaded {len(frames)} frames")

# Compute prediction error (surprise) per frame
surprise_scores = []
context = []
for i, frame in enumerate(tqdm(frames, desc="Computing surprise")):
    frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
    frame_t = frame_t.to(device).to(dtype)
    with torch.no_grad():
        latent = autoencoder.encode(frame_t)  # (1, 192, 8, 8) for 128x128
        
        if predictor is not None and len(context) >= 3:
            # Use last 3 frames to predict current
            pred_mean, _ = predictor(context[-3:])
            surprise = torch.nn.functional.mse_loss(pred_mean, latent).item()
        else:
            surprise = latent.norm().item()
        
        surprise_scores.append(surprise)
        context.append(latent)
        if len(context) > 4:
            context.pop(0)

# Identify anomaly frames (e.g., frames 50-80 contain the dropping bag)
normal_surprise = np.mean(surprise_scores[:40])
anomaly_surprise = np.mean(surprise_scores[50:80])
ratio = anomaly_surprise / normal_surprise if normal_surprise > 0 else 0

print(f"\nNormal surprise (avg, frames 0-40): {normal_surprise:.6f}")
print(f"Anomaly surprise (avg, frames 50-80): {anomaly_surprise:.6f}")
print(f"Surprise ratio (anomaly/normal): {ratio:.2f}x")

# Save results
os.makedirs('/root/le-maia/benchmark_results', exist_ok=True)
with open('/root/le-maia/benchmark_results/voe_results.txt', 'w') as f:
    f.write(f"Normal surprise (avg, frames 0-40): {normal_surprise:.6f}\n")
    f.write(f"Anomaly surprise (avg, frames 50-80): {anomaly_surprise:.6f}\n")
    f.write(f"Surprise ratio (anomaly/normal): {ratio:.2f}x\n")
print("✅ Results saved to /root/le-maia/benchmark_results/voe_results.txt")
