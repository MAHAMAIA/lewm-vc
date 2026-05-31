#!/usr/bin/env python3
"""
Create demo videos using Phase 0 autoencoder.
Left: x265 (CRF 36), Right: LeWM-VC with heatmap (patch variance).
"""

import os
import sys
import glob
import subprocess
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

def encode_lewm_frames(frames, target_size):
    decoded = []
    bitmaps = []  # per‑patch variance (proxy for bits)
    for frame in tqdm(frames, desc="LeWM encoding"):
        frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
        frame_t = frame_t.to(device)
        with torch.no_grad():
            latent_norm = model.encode(frame_t)  # [1,192,H/16,W/16]
            quantized = quantizer(latent_norm)
            # Write to bitstream (stub) – just to simulate bits
            frame_data = {"latent": quantized.cpu()}
            nal_bytes = writer.write_frame(frame_data, is_iframe=True)
            # For heatmap: use standard deviation per patch (higher variance = more bits)
            patch_var = quantized.var(dim=1, keepdim=True).cpu().numpy()[0,0]  # [H/16,W/16]
            bitmaps.append(patch_var)
            # Decode
            recon = model.decode(quantized, target_size=target_size)
            recon_np = (recon.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            decoded.append(recon_np)
    return decoded, bitmaps

def encode_x265_video(video_path, crf=36, target_size=(256,256)):
    out_path = '/tmp/x265_temp.mp4'
    cmd = ['ffmpeg', '-y', '-i', video_path, '-vf', f'scale={target_size[0]}:{target_size[1]}', '-c:v', 'libx265', '-crf', str(crf), '-preset', 'medium', out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    decoded_dir = '/tmp/x265_decoded'
    os.makedirs(decoded_dir, exist_ok=True)
    subprocess.run(['ffmpeg', '-i', out_path, os.path.join(decoded_dir, 'frame_%06d.png')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = []
    for p in sorted(glob.glob(os.path.join(decoded_dir, '*.png'))):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    shutil.rmtree(decoded_dir)
    os.remove(out_path)
    return frames

def overlay_heatmap(img, heatmap, alpha=0.6, color_map=cv2.COLORMAP_JET):
    h, w = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
    heatmap_norm = (heatmap_resized / (heatmap_resized.max() + 1e-8) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_norm, color_map)
    blended = cv2.addWeighted(img, 1-alpha, colored, alpha, 0)
    return blended

def create_demo(video_path, output_path, target_size=(256,256)):
    print(f"Processing: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    # Encode with x265
    frames_x265 = encode_x265_video(video_path, crf=36, target_size=target_size)
    # Encode with LeWM-VC
    frames_lewm, bitmaps = encode_lewm_frames(frames, target_size)
    min_len = min(len(frames), len(frames_x265), len(frames_lewm))
    frames = frames[:min_len]
    frames_x265 = frames_x265[:min_len]
    frames_lewm = frames_lewm[:min_len]
    bitmaps = bitmaps[:min_len]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (target_size[1]*2, target_size[0]))
    for i, (orig, x265, lewm, bm) in enumerate(zip(frames, frames_x265, frames_lewm, bitmaps)):
        left = cv2.cvtColor(x265, cv2.COLOR_RGB2BGR)
        cv2.putText(left, 'x265 (CRF 36)', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        right = cv2.cvtColor(lewm, cv2.COLOR_RGB2BGR)
        right = overlay_heatmap(right, bm, alpha=0.6)
        cv2.putText(right, 'LeWM-VC (Phase 0)', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(right, 'Heatmap: patch variance (red=high)', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        combined = np.hstack((left, right))
        out.write(combined)
    out.release()
    print(f"Saved: {output_path}")

def main():
    dataset_dir = '/root/le-maia/datasets/pevid-hd'
    video_paths = glob.glob(os.path.join(dataset_dir, '*.mpg'))
    output_dir = '/root/le-maia/demo_videos_phase0'
    os.makedirs(output_dir, exist_ok=True)
    for vp in video_paths:
        basename = os.path.splitext(os.path.basename(vp))[0]
        out_path = os.path.join(output_dir, f'{basename}_demo.mp4')
        create_demo(vp, out_path)
    print("All demo videos created.")

if __name__ == '__main__':
    main()
