#!/usr/bin/env python3
"""
Simplified evaluation with KL estimate (no torchac).
Uses λ=0.05, 256x256 resolution, I-frames only.
"""

import os
import sys
import glob
import subprocess
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.distributions.laplace import Laplace

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---------- Model definitions (simplified, matching training) ----------
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
        self.post_filter = nn.Sequential(
            nn.Conv2d(3, 16, 3,1,1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3,1,1),
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
    def decode(self, latent_norm, target_size):
        return self.decoder(latent_norm, target_size=target_size)

class CheckerboardContext(nn.Module):
    def __init__(self, channels, hidden_dim=128):
        super().__init__()
        self.mask_conv = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.refine = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, 3, padding=1),
        )
        self.register_buffer('mask', None)
    def forward(self, x, full_context=False):
        if self.mask is None or self.mask.shape[2:] != x.shape[2:]:
            h, w = x.shape[2], x.shape[3]
            mask = torch.zeros(1, 1, h, w, device=x.device)
            mask[..., 0::2, 0::2] = 1
            mask[..., 1::2, 1::2] = 1
            self.mask = mask
        out = self.mask_conv(x)
        out = self.refine(out)
        if not full_context:
            out = out * self.mask
        return out

class ContextualEntropyModel(nn.Module):
    def __init__(self, latent_dim=192, hyper_channels=512, context_hidden=128):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_channels, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2, stride=2),
            nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2),
            nn.GELU(),
        )
        self.skip_proj = nn.Conv2d(latent_dim, hyper_channels, 1)
        self.head = nn.Conv2d(hyper_channels, latent_dim * 2, 1)
        self.context = CheckerboardContext(latent_dim, context_hidden)
        self.refine_mu = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        self.refine_scale = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
    def forward(self, x):
        x_down = self.down(x)
        x_up = self.up(x_down)
        x_skip = nn.functional.interpolate(x, size=x_up.shape[2:], mode='bilinear', align_corners=False)
        x_skip = self.skip_proj(x_skip)
        features = x_up + x_skip
        base_params = self.head(features)
        mu_base = base_params[:, :192, :, :]
        log_scale_base = base_params[:, 192:, :, :]
        ctx = self.context(x)
        mu_offset = self.refine_mu(ctx)
        scale_offset = self.refine_scale(ctx)
        mu = mu_base + mu_offset
        log_scale = log_scale_base + scale_offset
        return mu, log_scale

# ---------- Choose λ = 0.05 ----------
lam = 0.05
checkpoint_dir = '/root/le-maia/checkpoints_rd_scratch'
ae_ckpt = os.path.join(checkpoint_dir, f'ae_lambda_{lam}_final.pt')
ent_ckpt = os.path.join(checkpoint_dir, f'entropy_lambda_{lam}_final.pt')

print(f"Using λ = {lam}")
print(f"Autoencoder: {ae_ckpt}")
print(f"Entropy model: {ent_ckpt}")

if not os.path.exists(ae_ckpt) or not os.path.exists(ent_ckpt):
    raise FileNotFoundError(f"Checkpoints for λ={lam} not found")

print("Loading models...")
autoencoder = VideoAutoencoderWithAffine().to(device)
autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device, weights_only=False), strict=False)
autoencoder.eval()

entropy_model = ContextualEntropyModel().to(device)
state = torch.load(ent_ckpt, map_location=device, weights_only=False)
for key in list(state.keys()):
    if 'mask' in key:
        del state[key]
entropy_model.load_state_dict(state, strict=False)
entropy_model.eval()

quantizer = Quantizer(num_levels=256, mode='inference').to(device)

# Test video
test_video = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')[0]
cap = cv2.VideoCapture(test_video)
frames = []
target_size = (64, 64)
while len(frames) < 150:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, target_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
cap.release()
print(f"Loaded {len(frames)} frames at {target_size[0]}x{target_size[1]}")

QUANT_STEP = 2.0 / 255

def laplace_likelihood_discrete(y, mu, log_scale, step, sigma_floor=0.003, epsilon=1e-9):
    scale = torch.nn.functional.softplus(log_scale) + sigma_floor
    laplace = Laplace(mu, scale)
    cdf_upper = laplace.cdf(y + 0.5 * step)
    cdf_lower = laplace.cdf(y - 0.5 * step)
    pmf = cdf_upper - cdf_lower
    pmf = torch.clamp(pmf, min=epsilon, max=1.0)
    nll = -torch.log(pmf)
    return nll.mean()  # average NLL per element

# Encode frames (I-frames only, no predictor)
total_bits = 0
decoded_frames = []
for frame in tqdm(frames, desc="Encoding"):
    frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
    frame_t = frame_t.to(device)
    with torch.no_grad():
        latent_norm = autoencoder.encode(frame_t)
        quantized = quantizer(latent_norm)
        mu, log_scale = entropy_model(quantized)
        nll = laplace_likelihood_discrete(quantized, mu, log_scale, step=QUANT_STEP)
        bits = nll * quantized.numel() / np.log(2)
        total_bits += bits.item()
        recon = autoencoder.decode(quantized, target_size=target_size)
        recon_np = (recon.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        decoded_frames.append(recon_np)

# Compute PSNR
psnr_sum = 0
for orig, recon in zip(frames, decoded_frames):
    mse = np.mean((orig.astype(float) - recon.astype(float))**2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100
    psnr_sum += psnr
avg_psnr = psnr_sum / len(frames)
total_pixels = len(frames) * target_size[0] * target_size[1]
bpp = total_bits / total_pixels

print(f"\nLeWM-VC (λ={lam}): bpp = {bpp:.4f}, PSNR = {avg_psnr:.2f} dB")

# x265 benchmark
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
    bpp_x, psnr_x = encode_x265(test_video, crf, target_size)
    x265_results.append((crf, bpp_x, psnr_x))
    print(f"x265 CRF={crf}: bpp={bpp_x:.4f}, PSNR={psnr_x:.2f} dB")

os.makedirs('/root/le-maia/benchmark_results', exist_ok=True)
with open('/root/le-maia/benchmark_results/improved_eval.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Codec', 'Setting', 'bpp', 'PSNR'])
    writer.writerow(['LeWM-VC', f'λ={lam}', bpp, avg_psnr])
    for crf, bpp_x, psnr_x in x265_results:
        writer.writerow(['x265', crf, bpp_x, psnr_x])
print("Results saved to /root/le-maia/benchmark_results/improved_eval.csv")
