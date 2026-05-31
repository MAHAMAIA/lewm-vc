#!/usr/bin/env python3
"""
CORRECTED Evaluation Script for LeWM-VC
- Matches training BPP calculation exactly
- Fixed λ range to match corrected training
- Proper Y-PSNR calculation
- BD-rate with overlap detection
- Updated checkpoint paths
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
dtype = torch.bfloat16
print(f"Device: {device}, Dtype: {dtype}")

# ---------- Model definitions (matching corrected training) ----------
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
    def __init__(self, latent_dim=192, hidden_dim=512, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.proj = nn.Conv2d(latent_dim, hidden_dim, 1)
        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4, 2, 1)
        self.res1 = ResidualBlock(hidden_dim//2)
        self.up2 = nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 4, 2, 1)
        self.res2 = ResidualBlock(hidden_dim//4)
        self.up3 = nn.ConvTranspose2d(hidden_dim//4, hidden_dim//8, 4, 2, 1)
        self.res3 = ResidualBlock(hidden_dim//8)
        self.up4 = nn.ConvTranspose2d(hidden_dim//8, hidden_dim//16, 4, 2, 1)
        self.res4 = ResidualBlock(hidden_dim//16)
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim//16, hidden_dim//32, 3, 1, 1),
            nn.InstanceNorm2d(hidden_dim//32),
            nn.GELU(),
            nn.Conv2d(hidden_dim//32, 3, 3, 1, 1),
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
    def forward(self, x): return x * self.scale + self.shift

class VideoAutoencoderWithAffine(nn.Module):
    def __init__(self, latent_dim=192, decoder_layers=4):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim, num_layers=decoder_layers)
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
        self.refine = nn.Sequential(nn.GELU(), nn.Conv2d(hidden_dim, channels, 3, padding=1))
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
        return out * self.mask if not full_context else out

class ContextualEntropyModel(nn.Module):
    def __init__(self, latent_dim=192, hyper_channels=512, context_hidden=128):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_channels, 5, padding=2), nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2, stride=2), nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2), nn.GELU()
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2), nn.GELU()
        )
        self.skip_proj = nn.Conv2d(latent_dim, hyper_channels, 1)
        self.head = nn.Conv2d(hyper_channels, latent_dim * 2, 1)
        self.context = CheckerboardContext(latent_dim, context_hidden)
        self.refine_mu = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        self.refine_scale = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
    def forward(self, x):
        x_down = self.down(x)
        x_up = self.up(x_down)
        x_skip = self.skip_proj(nn.functional.interpolate(x, size=x_up.shape[2:], mode='bilinear', align_corners=False))
        base = self.head(x_up + x_skip)
        mu_b, sc_b = base[:, :192], base[:, 192:]
        ctx = self.context(x).to(x.dtype)
        return mu_b + self.refine_mu(ctx), sc_b + self.refine_scale(ctx)

# Load test video
test_video = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')[0]
print(f"Test video: {os.path.basename(test_video)}")

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
frames = frames[:150]
print(f"Loaded {len(frames)} frames at {target_size[0]}x{target_size[1]}")

quantizer = Quantizer(num_levels=256, mode='inference').to(device)
QUANT_STEP = 2.0 / 255

def rgb_to_yuv_torch(rgb):
    r, g, b = rgb[:,0], rgb[:,1], rgb[:,2]
    y = 0.299*r + 0.587*g + 0.114*b
    return y

def laplace_likelihood_discrete(y, mu, log_scale, step, sigma_floor=0.01, epsilon=1e-9):
    """Matches corrected training: sigma_floor=0.01, max=10.0"""
    scale = torch.nn.functional.softplus(log_scale) + sigma_floor
    scale = torch.clamp(scale, min=sigma_floor, max=10.0)
    laplace = Laplace(mu, scale)
    cdf_upper = laplace.cdf(y + 0.5 * step)
    cdf_lower = laplace.cdf(y - 0.5 * step)
    pmf = torch.clamp(cdf_upper - cdf_lower, min=epsilon, max=1.0)
    return -torch.log(pmf).mean()

def compute_bd_rate(rate1, psnr1, rate2, psnr2):
    """Compute Bjontegaard Delta Rate with proper sorting."""
    from scipy.interpolate import CubicSpline
    
    # Sort by PSNR
    sorted1 = sorted(zip(psnr1, rate1))
    sorted2 = sorted(zip(psnr2, rate2))
    
    psnr1 = np.array([x[0] for x in sorted1])
    rate1 = np.array([x[1] for x in sorted1])
    psnr2 = np.array([x[0] for x in sorted2])
    rate2 = np.array([x[1] for x in sorted2])
    
    psnr_min = max(psnr1.min(), psnr2.min())
    psnr_max = min(psnr1.max(), psnr2.max())
    
    if psnr_min >= psnr_max:
        return float('nan')
    
    cs1 = CubicSpline(psnr1, np.log2(rate1), extrapolate=False)
    cs2 = CubicSpline(psnr2, np.log2(rate2), extrapolate=False)
    
    psnr_range = np.linspace(psnr_min, psnr_max, 1000)
    log_rate1 = cs1(psnr_range)
    log_rate2 = cs2(psnr_range)
    
    valid = ~(np.isnan(log_rate1) | np.isnan(log_rate2))
    if valid.sum() < 10:
        return float('nan')
    
    avg_rate1 = np.mean(2**log_rate1[valid])
    avg_rate2 = np.mean(2**log_rate2[valid])
    
    return 100 * (avg_rate2 / avg_rate1 - 1)  # Note: anchor is rate2 (x265)

# ✅ FIXED: λ values matching corrected training
lambda_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
checkpoint_dir = '/root/le-maia/checkpoints_corrected'
results = []

for lam in lambda_list:
    ae_ckpt = f'{checkpoint_dir}/ae_lambda_{lam}_final.pt'
    ent_ckpt = f'{checkpoint_dir}/entropy_lambda_{lam}_final.pt'
    
    if not os.path.exists(ae_ckpt) or not os.path.exists(ent_ckpt):
        ae_ckpt = f'{checkpoint_dir}/ae_lambda_{lam}_best.pt'
        ent_ckpt = f'{checkpoint_dir}/entropy_lambda_{lam}_best.pt'
        if not os.path.exists(ae_ckpt) or not os.path.exists(ent_ckpt):
            print(f"⚠️ Checkpoints for λ={lam} not found, skipping")
            continue
    
    print(f"\n📦 Loading λ={lam} models...")
    
    # Determine decoder layers (matching training)
    decoder_layers = 6 if lam >= 0.01 else 4
    
    autoencoder = VideoAutoencoderWithAffine(latent_dim=192, decoder_layers=decoder_layers).to(device).to(dtype)
    state_ae = torch.load(ae_ckpt, map_location=device, weights_only=False)
    autoencoder.load_state_dict(state_ae, strict=False)
    autoencoder.eval()
    
    entropy_model = ContextualEntropyModel(latent_dim=192).to(device).to(dtype)
    state_ent = torch.load(ent_ckpt, map_location=device, weights_only=False)
    for key in list(state_ent.keys()):
        if 'mask' in key:
            del state_ent[key]
    entropy_model.load_state_dict(state_ent, strict=False)
    entropy_model.eval()
    
    total_bpp = 0
    total_mse_y = 0
    
    with torch.no_grad():
        for frame in tqdm(frames, desc=f"  Encoding λ={lam}"):
            frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
            frame_t = frame_t.to(device).to(dtype)
            
            latent_norm = autoencoder.encode(frame_t)
            quantized = quantizer(latent_norm)
            mu, log_scale = entropy_model(quantized)
            
            nll = laplace_likelihood_discrete(quantized, mu, log_scale, step=QUANT_STEP)
            
            # ✅ FIXED: BPP calculation exactly matches training
            # rate = (nll * num_latents / ln(2)) / num_pixels
            num_pixels = frame_t.numel()  # 3 * 256 * 256 = 196,608
            bpp = (nll.item() * quantized.numel() / np.log(2)) / num_pixels
            total_bpp += bpp
            
            recon = autoencoder.decode(quantized, target_size=target_size)
            y_recon = rgb_to_yuv_torch(recon.float())
            y_orig = rgb_to_yuv_torch(frame_t.float())
            mse_y = torch.mean((y_recon - y_orig)**2).item()
            total_mse_y += mse_y
    
    avg_bpp = total_bpp / len(frames)
    avg_psnr_y = 10 * np.log10(1.0 / (total_mse_y / len(frames))) if total_mse_y > 0 else 100
    results.append((lam, avg_bpp, avg_psnr_y))
    print(f"  ✅ λ={lam}: BPP={avg_bpp:.6f}, Y-PSNR={avg_psnr_y:.2f} dB")

# x265 baseline
def encode_x265(video_path, crf, target_size, num_frames=150):
    out_path = f'/tmp/x265_crf{crf}.mp4'
    cmd = ['ffmpeg', '-y', '-i', video_path, '-vf', f'scale={target_size[0]}:{target_size[1]}', 
           '-c:v', 'libx265', '-crf', str(crf), '-preset', 'veryslow', out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    decoded_dir = f'/tmp/x265_decoded_{crf}'
    os.makedirs(decoded_dir, exist_ok=True)
    subprocess.run(['ffmpeg', '-i', out_path, os.path.join(decoded_dir, 'frame_%06d.png')], 
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    cap = cv2.VideoCapture(video_path)
    orig = []
    while len(orig) < num_frames:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig.append(frame)
    cap.release()
    orig = orig[:num_frames]
    
    dec = []
    for p in sorted(glob.glob(os.path.join(decoded_dir, '*.png')))[:num_frames]:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dec.append(img)
    dec = dec[:num_frames]
    
    min_len = min(len(orig), len(dec))
    psnr_sum = 0
    for i in range(min_len):
        orig_np = orig[i].astype(np.float32) / 255.0
        dec_np = dec[i].astype(np.float32) / 255.0
        y_orig = 0.299*orig_np[:,:,0] + 0.587*orig_np[:,:,1] + 0.114*orig_np[:,:,2]
        y_dec = 0.299*dec_np[:,:,0] + 0.587*dec_np[:,:,1] + 0.114*dec_np[:,:,2]
        mse = np.mean((y_orig - y_dec)**2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100
        psnr_sum += psnr
    
    avg_psnr = psnr_sum / min_len
    file_size = os.path.getsize(out_path) * 8
    total_pixels = min_len * target_size[0] * target_size[1]
    bpp = file_size / total_pixels
    
    shutil.rmtree(decoded_dir)
    os.remove(out_path)
    return bpp, avg_psnr

print("\n🎬 Encoding x265 baseline...")
crf_list = [18, 23, 28, 33, 38, 43]  # Wider range for better overlap
x265_results = []
for crf in crf_list:
    bpp, psnr = encode_x265(test_video, crf, target_size, num_frames=150)
    x265_results.append((crf, bpp, psnr))
    print(f"  x265 CRF={crf:2}: BPP={bpp:.6f}, Y-PSNR={psnr:.2f} dB")

# Compute BD-rate
print("\n📊 Computing BD-rate...")
valid_results = [(lam, bpp, psnr) for lam, bpp, psnr in results if psnr > 0 and bpp > 0]
valid_x265 = [(crf, bpp, psnr) for crf, bpp, psnr in x265_results if psnr > 0 and bpp > 0]

if len(valid_results) >= 4 and len(valid_x265) >= 4:
    valid_results_sorted = sorted(valid_results, key=lambda x: x[2])
    valid_x265_sorted = sorted(valid_x265, key=lambda x: x[2])
    
    my_rates = [r[1] for r in valid_results_sorted]
    my_psnrs = [r[2] for r in valid_results_sorted]
    x265_rates = [r[1] for r in valid_x265_sorted]
    x265_psnrs = [r[2] for r in valid_x265_sorted]
    
    psnr_min = max(min(my_psnrs), min(x265_psnrs))
    psnr_max = min(max(my_psnrs), max(x265_psnrs))
    
    if psnr_min < psnr_max:
        bd_rate = compute_bd_rate(my_rates, my_psnrs, x265_rates, x265_psnrs)
        if not np.isnan(bd_rate):
            print(f"  ✅ BD-rate (LeWM-VC vs x265): {bd_rate:+.2f}%")
        else:
            print("  ⚠️ BD-rate could not be computed (interpolation failed)")
            bd_rate = None
    else:
        print(f"  ⚠️ No PSNR overlap: LeWM-VC [{min(my_psnrs):.2f}, {max(my_psnrs):.2f}] vs x265 [{min(x265_psnrs):.2f}, {max(x265_psnrs):.2f}]")
        bd_rate = None
else:
    print(f"  ⚠️ Insufficient points: LeWM-VC={len(valid_results)}, x265={len(valid_x265)}")
    bd_rate = None

# Save CSV
os.makedirs('/root/le-maia/benchmark_results', exist_ok=True)
csv_path = '/root/le-maia/benchmark_results/rd_curve_corrected.csv'
with open(csv_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['λ', 'BPP', 'Y-PSNR'])
    for lam, bpp, psnr in results:
        writer.writerow([lam, bpp, psnr])
    writer.writerow([])
    writer.writerow(['x265 CRF', 'BPP', 'Y-PSNR'])
    for crf, bpp, psnr in x265_results:
        writer.writerow([crf, bpp, psnr])
    if bd_rate is not None and not np.isnan(bd_rate):
        writer.writerow([])
        writer.writerow(['BD-rate vs x265', f'{bd_rate:+.2f}%'])

print(f"\n📁 Results saved to {csv_path}")

# Summary
print("\n" + "="*60)
print("LEWM-VC RD POINTS (Corrected)")
print("="*60)
for lam, bpp, psnr in sorted(results, key=lambda x: x[2]):
    print(f"λ={lam:6} | BPP={bpp:.6f} | Y-PSNR={psnr:.2f} dB")

print("\n" + "="*60)
print("X265 BASELINE")
print("="*60)
for crf, bpp, psnr in sorted(x265_results, key=lambda x: x[2]):
    print(f"CRF={crf:3} | BPP={bpp:.6f} | Y-PSNR={psnr:.2f} dB")
