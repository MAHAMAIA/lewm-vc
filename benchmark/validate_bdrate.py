#!/usr/bin/env python3
"""
Validate Phase 1 models using entropy model's rate estimates.
Computes BD-rate without bitstream writer stub.
"""

import os
import sys
import subprocess
import csv
import glob
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.predictor import LeWMPredictor
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ------------------------------
# Model architecture (same as training)
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

class FullCodec(nn.Module):
    def __init__(self, autoencoder, entropy_model, quantizer):
        super().__init__()
        self.autoencoder = autoencoder
        self.entropy_model = entropy_model
        self.quantizer = quantizer

# Load pre-trained autoencoder (Phase 0)
autoencoder = VideoAutoencoder().to(device)
checkpoint_auto = '/root/le-maia/checkpoints/autoencoder_final.pt'
autoencoder.load_state_dict(torch.load(checkpoint_auto, map_location=device))
autoencoder.eval()
print("Autoencoder loaded.")

# ------------------------------
# Helper functions
# ------------------------------
def compute_psnr(orig, recon):
    mse = np.mean((orig.astype(float) - recon.astype(float))**2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def extract_frames(video_path, max_frames, target_size=(256,256)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def encode_video_with_lambda(model, frames):
    """Encode frames using entropy model rate (no actual bitstream). Returns total bits and decoded frames."""
    T = len(frames)
    h, w = frames[0].shape[:2]
    total_bits = 0
    decoded_frames = []
    for t in range(T):
        frame_t = torch.from_numpy(frames[t]).float().permute(2,0,1).unsqueeze(0) / 255.0
        frame_t = frame_t.to(device)
        with torch.no_grad():
            latent = model.autoencoder.encoder(frame_t, return_surprise=False)
            quantized = model.quantizer(latent)
            # Compute rate in bits using the trained entropy model
            rate_nats, _ = model.entropy_model(quantized)
            bits = (rate_nats * np.log2(np.e)).sum().item()
            total_bits += bits
            # Decode for PSNR
            recon = model.autoencoder.decoder(quantized, target_size=(h,w))
            recon_np = (recon.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            decoded_frames.append(recon_np)
    return decoded_frames, total_bits

def encode_with_x265(video_path, crf, output_path):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-c:v", "libx265", "-crf", str(crf), "-preset", "medium", output_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    decoded_dir = Path(output_path).stem + "_decoded"
    os.makedirs(decoded_dir, exist_ok=True)
    subprocess.run(["ffmpeg", "-i", output_path, os.path.join(decoded_dir, "frame_%06d.png")],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cap = cv2.VideoCapture(video_path)
    orig_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig_frames.append(frame)
    cap.release()
    decoded_frames = []
    for p in sorted(glob.glob(os.path.join(decoded_dir, "*.png"))):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        decoded_frames.append(img)
    min_len = min(len(orig_frames), len(decoded_frames))
    psnr_vals = [compute_psnr(orig_frames[i], decoded_frames[i]) for i in range(min_len)]
    avg_psnr = np.mean(psnr_vals)
    file_size = os.path.getsize(output_path) * 8
    total_pixels = min_len * orig_frames[0].shape[0] * orig_frames[0].shape[1]
    bpp = file_size / total_pixels if total_pixels > 0 else 0
    shutil.rmtree(decoded_dir)
    return bpp, avg_psnr

def bd_rate(rate1, psnr1, rate2, psnr2):
    import numpy as np
    idx1 = np.argsort(psnr1)
    idx2 = np.argsort(psnr2)
    psnr1_s = np.array(psnr1)[idx1]
    psnr2_s = np.array(psnr2)[idx2]
    rate1_s = np.array(rate1)[idx1]
    rate2_s = np.array(rate2)[idx2]
    psnr_min = max(psnr1_s.min(), psnr2_s.min())
    psnr_max = min(psnr1_s.max(), psnr2_s.max())
    if psnr_min >= psnr_max:
        return float('nan')
    psnr_interp = np.linspace(psnr_min, psnr_max, 100)
    rate1_interp = np.interp(psnr_interp, psnr1_s, np.log(rate1_s))
    rate2_interp = np.interp(psnr_interp, psnr2_s, np.log(rate2_s))
    avg_diff = np.mean(rate2_interp - rate1_interp)
    return (np.exp(avg_diff) - 1) * 100

# ------------------------------
# Main: benchmark all λ checkpoints
# ------------------------------
test_video = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')[0]
frames = extract_frames(test_video, max_frames=150, target_size=(256,256))
print(f"Extracted {len(frames)} frames from {test_video}")

lambda_list = [0.001, 0.01, 0.1, 1.0, 10.0]
lewm_results = []

for lam in lambda_list:
    ckpt_dir = f'/root/le-maia/checkpoints/phase1_lambda_{lam}'
    ckpt_path = os.path.join(ckpt_dir, 'final.pt')
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint for λ={lam} not found, skipping")
        continue
    print(f"\nLoading λ={lam} model...")
    entropy_model = HyperpriorEntropy(latent_dim=192, hyper_channels=320).to(device)
    quantizer = Quantizer(num_levels=256, mode='inference').to(device)
    model = FullCodec(autoencoder, entropy_model, quantizer).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    # The checkpoint saved by train_phase1.py contains the full model state dict.
    # We need to load it into our model.
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"Encoding with λ={lam}...")
    decoded, total_bits = encode_video_with_lambda(model, frames)
    psnr_vals = [compute_psnr(frames[i], decoded[i]) for i in range(len(decoded))]
    avg_psnr = np.mean(psnr_vals)
    total_pixels = len(frames) * frames[0].shape[0] * frames[0].shape[1]
    bpp = total_bits / total_pixels
    lewm_results.append((bpp, avg_psnr, lam))
    print(f"  λ={lam}: bpp={bpp:.6f}, PSNR={avg_psnr:.2f} dB")

# x265 benchmark
crf_list = [23, 28, 32, 36]
x265_results = []
for crf in crf_list:
    out_path = f"/tmp/x265_crf{crf}.mp4"
    print(f"Encoding x265 CRF={crf}...")
    bpp, psnr = encode_with_x265(test_video, crf, out_path)
    x265_results.append((bpp, psnr, crf))
    print(f"  CRF={crf}: bpp={bpp:.6f}, PSNR={psnr:.2f} dB")

# Compute BD-rate
if lewm_results:
    lewm_bpp = [r[0] for r in lewm_results]
    lewm_psnr = [r[1] for r in lewm_results]
    x265_bpp = [r[0] for r in x265_results]
    x265_psnr = [r[1] for r in x265_results]
    bd = bd_rate(lewm_bpp, lewm_psnr, x265_bpp, x265_psnr)
    print(f"\nBD-rate (LeWM-VC vs. x265): {bd:.2f}%")
    if bd < 0:
        print("✅ LeWM-VC saves bits compared to x265")
    else:
        print("⚠️ LeWM-VC uses more bits – more training needed")
else:
    print("No LeWM-VC results – check checkpoint paths.")

# Save CSV
os.makedirs('/root/le-maia/benchmark_results', exist_ok=True)
csv_path = '/root/le-maia/benchmark_results/phase1_bdrate_entropy.csv'
with open(csv_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Codec", "λ/CRF", "bpp", "PSNR"])
    for bpp, psnr, lam in lewm_results:
        writer.writerow(["LeWM-VC", lam, bpp, psnr])
    for bpp, psnr, crf in x265_results:
        writer.writerow(["x265", crf, bpp, psnr])
print(f"Results saved to {csv_path}")
