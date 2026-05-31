#!/usr/bin/env python3
"""
Evaluate all GMM λ checkpoints + compare with x265.
Produces RD curve CSV for BD-rate computation.
"""

import os
import sys
import glob
import csv
import subprocess
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.distributions.normal import Normal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---------- Model definitions ----------
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

class VideoAutoencoder(nn.Module):
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

class GMMEntropyModel(nn.Module):
    def __init__(self, latent_dim=192, hyper_channels=256, num_components=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.hyperprior = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hyper_channels, hyper_channels, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(hyper_channels, hyper_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hyper_channels, latent_dim * num_components * 3, 3, padding=1),
        )
        self.softplus = nn.Softplus()
    def forward(self, x):
        params = self.hyperprior(x)
        B, C, H, W = params.shape
        channels_per_comp = C // self.num_components
        params = params.view(B, self.num_components, channels_per_comp, H, W)
        mu = params[:, :, :self.latent_dim, :, :]
        log_scale = params[:, :, self.latent_dim:2*self.latent_dim, :, :]
        log_weight = params[:, :, 2*self.latent_dim:3*self.latent_dim, :, :]
        scale = self.softplus(log_scale) + 1e-5
        weight = torch.softmax(log_weight, dim=1)
        return mu, scale, weight

def gmm_likelihood_discrete(y, mu, scale, weight, step, epsilon=1e-9):
    B, C, H, W = y.shape
    num_comp = mu.shape[1]
    y_expanded = y.unsqueeze(1).expand(-1, num_comp, -1, -1, -1)
    normal = Normal(mu, scale)
    cdf_upper = normal.cdf(y_expanded + 0.5 * step)
    cdf_lower = normal.cdf(y_expanded - 0.5 * step)
    pmf = cdf_upper - cdf_lower
    pmf = torch.clamp(pmf, min=epsilon, max=1.0)
    mixture_pmf = (weight * pmf).sum(dim=1)
    nll = -torch.log(mixture_pmf)
    return nll.mean()

# ---------- Config ----------
lambda_list = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
BASE_DIR = os.environ.get('LEWM_BASE', '/root/le-maia')
CHECKPOINT_DIR = os.environ.get('LEWM_CHECKPOINT_DIR', os.path.join(BASE_DIR, 'checkpoints_gmm'))
DATASET_DIR = os.environ.get('LEWM_DATASET', os.path.join(BASE_DIR, 'datasets/pevid-hd'))
BENCHMARK_DIR = os.environ.get('LEWM_BENCHMARK_DIR', os.path.join(BASE_DIR, 'benchmark_results'))
QUANT_STEP = 2.0 / 255

# Test video
test_video = glob.glob(os.path.join(DATASET_DIR, '*.mpg'))[0]
cap = cv2.VideoCapture(test_video)
frames = []
target_size = (256, 256)
while len(frames) < 150:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, target_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
cap.release()
print(f"Loaded {len(frames)} frames at {target_size[0]}x{target_size[1]}")

quantizer = Quantizer(num_levels=256, mode='inference').to(device)

# ---------- Evaluate GMM ----------
def evaluate_gmm(ae_ckpt, ent_ckpt):
    autoencoder = VideoAutoencoder().to(device)
    autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device, weights_only=False), strict=False)
    autoencoder.eval()
    entropy_model = GMMEntropyModel().to(device)
    entropy_model.load_state_dict(torch.load(ent_ckpt, map_location=device, weights_only=False))
    entropy_model.eval()
    total_bits = 0
    total_mse = 0
    for frame in tqdm(frames, desc=f"GMM {os.path.basename(ae_ckpt)}"):
        frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
        frame_t = frame_t.to(device)
        with torch.no_grad():
            latent_norm = autoencoder.encode(frame_t)
            quantized = quantizer(latent_norm)
            mu, scale, weight = entropy_model(quantized)
            nll = gmm_likelihood_discrete(quantized, mu, scale, weight, step=QUANT_STEP)
            bits = nll * quantized.numel() / np.log(2)
            total_bits += bits.item()
            recon = autoencoder.decode(quantized, target_size=target_size)
            mse = torch.nn.functional.mse_loss(recon, frame_t).item()
            total_mse += mse
    bpp = total_bits / (len(frames) * target_size[0] * target_size[1])
    psnr = 20 * np.log10(1.0 / np.sqrt(total_mse / len(frames)))
    return bpp, psnr

results = []
for lam in lambda_list:
    ae_ckpt = os.path.join(CHECKPOINT_DIR, f'ae_lambda_{lam}_final.pt')
    ent_ckpt = os.path.join(CHECKPOINT_DIR, f'entropy_lambda_{lam}_final.pt')
    if not os.path.exists(ae_ckpt) or not os.path.exists(ent_ckpt):
        print(f"Missing checkpoints for λ={lam}, skipping")
        continue
    print(f"\nEvaluating GMM λ={lam}...")
    bpp, psnr = evaluate_gmm(ae_ckpt, ent_ckpt)
    results.append((lam, bpp, psnr))
    print(f"  GMM λ={lam}: bpp={bpp:.4f}, PSNR={psnr:.2f} dB")

# ---------- x265 benchmark ----------
def encode_x265(video_path, crf, target_size):
    out_path = f'/tmp/x265_crf{crf}.mp4'
    cmd = ['ffmpeg', '-y', '-i', video_path, '-vf', f'scale={target_size[0]}:{target_size[1]}', '-c:v', 'libx265', '-crf', str(crf), '-preset', 'medium', out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    decoded_dir = f'/tmp/x265_decoded_{crf}'
    os.makedirs(decoded_dir, exist_ok=True)
    subprocess.run(['ffmpeg', '-i', out_path, os.path.join(decoded_dir, 'frame_%06d.png')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cap = cv2.VideoCapture(video_path)
    orig = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig.append(frame)
    cap.release()
    dec = []
    for p in sorted(glob.glob(os.path.join(decoded_dir, '*.png'))):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dec.append(img)
    min_len = min(len(orig), len(dec))
    psnr_sum = 0
    for i in range(min_len):
        mse = np.mean((orig[i].astype(float) - dec[i].astype(float))**2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100
        psnr_sum += psnr
    avg_psnr = psnr_sum / min_len
    file_size = os.path.getsize(out_path) * 8
    total_pixels = min_len * target_size[0] * target_size[1]
    bpp = file_size / total_pixels
    shutil.rmtree(decoded_dir)
    os.remove(out_path)
    return bpp, avg_psnr

print("\nEncoding x265...")
crf_list = [23, 28, 32, 36]
x265_results = []
for crf in crf_list:
    bpp, psnr = encode_x265(test_video, crf, target_size)
    x265_results.append((crf, bpp, psnr))
    print(f"x265 CRF={crf}: bpp={bpp:.4f}, PSNR={psnr:.2f} dB")

# Save CSV
os.makedirs(BENCHMARK_DIR, exist_ok=True)
csv_path = os.path.join(BENCHMARK_DIR, 'rd_curve_gmm.csv')
with open(csv_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['λ', 'bpp', 'PSNR'])
    for lam, bpp, psnr in results:
        writer.writerow([lam, bpp, psnr])
    writer.writerow([])
    writer.writerow(['x265 CRF', 'bpp', 'PSNR'])
    for crf, bpp, psnr in x265_results:
        writer.writerow([crf, bpp, psnr])
print(f"Results saved to {csv_path}")
