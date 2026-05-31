#!/usr/bin/env python3
"""
Evaluate all λ checkpoints: compute bpp and PSNR on test video.
Compare to x265 and compute BD-rate.
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

# ---------- Model definitions (same as training) ----------
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

# Load test video
test_video = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')[0]
cap = cv2.VideoCapture(test_video)
frames = []
target_size = (256,256)
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

def laplace_likelihood_discrete(y, mu, log_scale, step, sigma_floor=0.05, epsilon=1e-9):
    scale = torch.nn.functional.softplus(log_scale) + sigma_floor
    scale = torch.clamp(scale, min=sigma_floor, max=5.0)
    laplace = Laplace(mu, scale)
    cdf_upper = laplace.cdf(y + 0.5 * step)
    cdf_lower = laplace.cdf(y - 0.5 * step)
    pmf = cdf_upper - cdf_lower
    pmf = torch.clamp(pmf, min=epsilon, max=1.0)
    nll = -torch.log(pmf)
    return nll.mean()

def compute_bd_rate(rate1, psnr1, rate2, psnr2):
    import numpy as np
    from scipy.interpolate import CubicSpline
    psnr1 = np.array(psnr1)
    psnr2 = np.array(psnr2)
    rate1 = np.array(rate1)
    rate2 = np.array(rate2)
    cs1 = CubicSpline(psnr1, np.log2(rate1))
    cs2 = CubicSpline(psnr2, np.log2(rate2))
    psnr_min = max(min(psnr1), min(psnr2))
    psnr_max = min(max(psnr1), max(psnr2))
    psnr_range = np.linspace(psnr_min, psnr_max, 1000)
    avg_rate1 = np.mean(2**cs1(psnr_range))
    avg_rate2 = np.mean(2**cs2(psnr_range))
    return 100 * (avg_rate1 / avg_rate2 - 1)

# λ values to evaluate
lambda_list = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
results = []
for lam in lambda_list:
    ae_ckpt = f'/root/le-maia/checkpoints_rd_scratch/ae_lambda_{lam}_best.pt'
    ent_ckpt = f'/root/le-maia/checkpoints_rd_scratch/entropy_lambda_{lam}_best.pt'
    if not os.path.exists(ae_ckpt) or not os.path.exists(ent_ckpt):
        ae_ckpt = f'/root/le-maia/checkpoints_rd_scratch/ae_lambda_{lam}_final.pt'
        ent_ckpt = f'/root/le-maia/checkpoints_rd_scratch/entropy_lambda_{lam}_final.pt'
        if not os.path.exists(ae_ckpt) or not os.path.exists(ent_ckpt):
            print(f"Checkpoints for λ={lam} not found, skipping")
            continue
    
    print(f"\nLoading λ={lam} models...")
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
    
    total_bits = 0
    total_mse_y = 0
    with torch.no_grad():
        for frame in tqdm(frames, desc=f"Encoding λ={lam}"):
            frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
            frame_t = frame_t.to(device)
            latent_norm = autoencoder.encode(frame_t)
            quantized = quantizer(latent_norm)
            mu, log_scale = entropy_model(quantized)
            nll = laplace_likelihood_discrete(quantized, mu, log_scale, step=QUANT_STEP, sigma_floor=0.05)
            bits = nll.item() * quantized.numel() / np.log(2)
            total_bits += bits
            recon = autoencoder.decode(quantized, target_size=target_size)
            y_recon = rgb_to_yuv_torch(recon)
            y_orig = rgb_to_yuv_torch(frame_t)
            mse_y = torch.mean((y_recon - y_orig)**2).item()
            total_mse_y += mse_y
    bpp = total_bits / (len(frames) * target_size[0] * target_size[1])
    psnr_y = 10 * np.log10(1.0 / (total_mse_y / len(frames)))
    results.append((lam, bpp, psnr_y))
    print(f"λ={lam}: bpp={bpp:.6f}, Y-PSNR={psnr_y:.2f} dB")

# x265 benchmark
def encode_x265(video_path, crf, target_size, num_frames=150):
    out_path = f'/tmp/x265_crf{crf}.mp4'
    cmd = ['ffmpeg', '-y', '-i', video_path, '-vf', f'scale={target_size[0]}:{target_size[1]}', '-c:v', 'libx265', '-crf', str(crf), '-preset', 'medium', out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    decoded_dir = f'/tmp/x265_decoded_{crf}'
    os.makedirs(decoded_dir, exist_ok=True)
    subprocess.run(['ffmpeg', '-i', out_path, os.path.join(decoded_dir, 'frame_%06d.png')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cap = cv2.VideoCapture(video_path)
    orig = []
    while len(orig) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
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

print("\nEncoding x265...")
crf_list = [23, 28, 32, 36]
x265_results = []
for crf in crf_list:
    bpp, psnr = encode_x265(test_video, crf, target_size, num_frames=150)
    x265_results.append((crf, bpp, psnr))
    print(f"x265 CRF={crf}: bpp={bpp:.6f}, Y-PSNR={psnr:.2f} dB")

# Compute BD-rate
if len(results) >= 2 and len(x265_results) >= 2:
    my_rates = [r[1] for r in results]
    my_psnrs = [r[2] for r in results]
    x265_rates = [r[1] for r in x265_results]
    x265_psnrs = [r[2] for r in x265_results]
    bd_rate = compute_bd_rate(my_rates, my_psnrs, x265_rates, x265_psnrs)
    print(f"\nBD-rate vs x265: {bd_rate:+.2f}%")

# Save CSV
os.makedirs('/root/le-maia/benchmark_results', exist_ok=True)
with open('/root/le-maia/benchmark_results/rd_curve.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['λ', 'bpp', 'Y-PSNR'])
    for lam, bpp, psnr in results:
        writer.writerow([lam, bpp, psnr])
    writer.writerow([])
    writer.writerow(['x265 CRF', 'bpp', 'Y-PSNR'])
    for crf, bpp, psnr in x265_results:
        writer.writerow([crf, bpp, psnr])
    if len(results) >= 2 and len(x265_results) >= 2:
        writer.writerow([])
        writer.writerow(['BD-rate vs x265', f'{bd_rate:+.2f}%'])
print("Results saved to /root/le-maia/benchmark_results/rd_curve.csv")
