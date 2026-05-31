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
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# Load affine autoencoder
autoencoder = VideoAutoencoderWithAffine(latent_dim=192).to(device)
checkpoint_affine = '/root/le-maia/checkpoints_affine/autoencoder_affine_final.pt'
autoencoder.load_state_dict(torch.load(checkpoint_affine, map_location=device), strict=False)
autoencoder.eval()
print("Affine autoencoder loaded.")

# Load trained entropy model (NLL version)
entropy_model = HyperpriorEntropy(latent_dim=192, hyper_channels=320).to(device)
entropy_ckpt = '/root/le-maia/checkpoints_entropy_affine_v3/entropy_final.pt'
entropy_model.load_state_dict(torch.load(entropy_ckpt, map_location=device))
entropy_model.eval()
print("Entropy model loaded.")

quantizer = Quantizer(num_levels=256, mode='inference').to(device)

# Helper functions
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

def encode_video(model, entropy_model, quantizer, frames):
    T = len(frames)
    h, w = frames[0].shape[:2]
    total_bits = 0.0
    decoded_frames = []
    for t in range(T):
        frame_t = torch.from_numpy(frames[t]).float().permute(2,0,1).unsqueeze(0) / 255.0
        frame_t = frame_t.to(device)
        with torch.no_grad():
            latent_norm = model.encode(frame_t)
            quantized = quantizer(latent_norm)
            # Compute rate using KL divergence (for comparison)
            rate_nats, _ = entropy_model(quantized)
            bits = rate_nats.sum().item() * np.log2(np.e)
            total_bits += bits
            recon = model.decode(quantized, target_size=(h,w))
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

# Run benchmark
test_video = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')[0]
frames = extract_frames(test_video, max_frames=150, target_size=(256,256))
print(f"Extracted {len(frames)} frames")

print("Encoding with LeWM-VC (affine + NLL entropy model)...")
decoded, total_bits = encode_video(autoencoder, entropy_model, quantizer, frames)
psnr_vals = [compute_psnr(frames[i], decoded[i]) for i in range(len(decoded))]
avg_psnr = np.mean(psnr_vals)
total_pixels = len(frames) * frames[0].shape[0] * frames[0].shape[1]
bpp = total_bits / total_pixels
print(f"LeWM-VC: bpp={bpp:.6f}, PSNR={avg_psnr:.2f} dB")

crf_list = [23, 28, 32, 36]
x265_results = []
for crf in crf_list:
    out_path = f"/tmp/x265_crf{crf}.mp4"
    print(f"Encoding x265 CRF={crf}...")
    bpp_x, psnr_x = encode_with_x265(test_video, crf, out_path)
    x265_results.append((bpp_x, psnr_x, crf))
    print(f"  CRF={crf}: bpp={bpp_x:.6f}, PSNR={psnr_x:.2f} dB")

best_idx = np.argmin([abs(psnr_x - avg_psnr) for _, psnr_x, _ in x265_results])
bpp_x265, psnr_x265, crf = x265_results[best_idx]
savings = (1 - bpp / bpp_x265) * 100 if bpp_x265 > 0 else 0
print(f"\nAt PSNR ≈ {avg_psnr:.2f} dB:")
print(f"  LeWM-VC bitrate: {bpp:.6f} bpp")
print(f"  x265 (CRF {crf}) bitrate: {bpp_x265:.6f} bpp")
print(f"  Bitrate savings: {savings:.2f}%")
if savings > 0:
    print("✅ LeWM-VC saves bits compared to x265")
else:
    print("⚠️ LeWM-VC uses more bits – more training needed")
