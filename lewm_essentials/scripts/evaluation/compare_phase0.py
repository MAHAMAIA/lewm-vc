#!/usr/bin/env python3
"""
Compare LeWM-VC (Phase 0) vs x265 on a single video.
LeWM-VC uses stub bitstream writer (constant bitrate).
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

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.quant import Quantizer
from lewm_vc.bitstream.writer import BitstreamWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---------- Phase 0 autoencoder (with affine) ----------
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

# Load Phase 0 checkpoint
ckpt_path = '/root/le-maia/checkpoints_rd_scratch/autoencoder_final.pt'
model = VideoAutoencoder().to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
model.eval()
print("Phase 0 autoencoder loaded.")

quantizer = Quantizer(num_levels=256, mode='inference').to(device)
writer = BitstreamWriter(version=1)

def encode_lewm(video_path, target_size=(256,256)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 150:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    total_bytes = 0
    psnr_sum = 0
    for frame in tqdm(frames, desc="LeWM encoding"):
        frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
        frame_t = frame_t.to(device)
        with torch.no_grad():
            latent_norm = model.encode(frame_t)
            quantized = quantizer(latent_norm)
            frame_data = {"latent": quantized.cpu()}
            nal_bytes = writer.write_frame(frame_data, is_iframe=True)
            total_bytes += len(nal_bytes)
            recon = model.decode(quantized, target_size=target_size)
            recon_np = (recon.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            mse = np.mean((frame.astype(float) - recon_np.astype(float))**2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100
            psnr_sum += psnr
    total_bits = total_bytes * 8
    total_pixels = len(frames) * target_size[0] * target_size[1]
    bpp = total_bits / total_pixels
    avg_psnr = psnr_sum / len(frames)
    return bpp, avg_psnr

def encode_x265(video_path, crf, target_size=(256,256)):
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

def main():
    test_video = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')[0]
    print(f"Test video: {test_video}")

    # LeWM-VC
    lewm_bpp, lewm_psnr = encode_lewm(test_video, target_size=(256,256))
    print(f"\nLeWM-VC (Phase 0): bpp = {lewm_bpp:.4f}, PSNR = {lewm_psnr:.2f} dB")

    # x265 at multiple CRF
    crf_list = [23, 28, 32, 36]
    x265_results = []
    for crf in crf_list:
        bpp, psnr = encode_x265(test_video, crf, target_size=(256,256))
        x265_results.append((crf, bpp, psnr))
        print(f"x265 CRF={crf}: bpp = {bpp:.4f}, PSNR = {psnr:.2f} dB")

    # Find x265 point with closest PSNR to LeWM-VC
    closest = min(x265_results, key=lambda x: abs(x[2] - lewm_psnr))
    crf, x265_bpp, x265_psnr = closest
    savings = (1 - lewm_bpp / x265_bpp) * 100 if x265_bpp > 0 else 0
    print(f"\nAt PSNR ≈ {lewm_psnr:.2f} dB:")
    print(f"  LeWM-VC bitrate: {lewm_bpp:.4f} bpp")
    print(f"  x265 (CRF {crf}) bitrate: {x265_bpp:.4f} bpp")
    print(f"  Bitrate savings: {savings:.2f}% (negative = worse)")

    # Save CSV
    os.makedirs('/root/le-maia/benchmark_results', exist_ok=True)
    with open('/root/le-maia/benchmark_results/phase0_comparison.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Codec', 'Setting', 'bpp', 'PSNR'])
        writer.writerow(['LeWM-VC', 'Phase 0', lewm_bpp, lewm_psnr])
        for crf, bpp, psnr in x265_results:
            writer.writerow(['x265', crf, bpp, psnr])
    print("Results saved to /root/le-maia/benchmark_results/phase0_comparison.csv")

if __name__ == '__main__':
    main()
