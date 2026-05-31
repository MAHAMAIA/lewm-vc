#!/usr/bin/env python3
"""
LeWM-VC Full Codec Benchmark
- Trains predictor (using frozen autoencoder)
- Encodes video with I/P-frames and entropy model
- Computes BD-rate vs x265
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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add repo to path
sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.predictor import LeWMPredictor
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.quant import Quantizer

# ------------------------------
# Device
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ------------------------------
# Autoencoder (must match training)
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
    def forward(self, x):
        b, t, c, h, w = x.shape
        x_flat = x.view(b*t, c, h, w)
        latent = self.encoder(x_flat, return_surprise=False)
        recon = self.decoder(latent, target_size=(h,w))
        recon = recon.view(b, t, c, h, w)
        return recon, latent

# ------------------------------
# Load trained autoencoder
# ------------------------------
autoencoder = VideoAutoencoder().to(device)
checkpoint_path = '/root/le-maia/checkpoints/autoencoder_final.pt'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
autoencoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
autoencoder.eval()
for param in autoencoder.parameters():
    param.requires_grad = False
print("Autoencoder loaded and frozen.")

# ------------------------------
# Dataset for predictor training
# ------------------------------
class PredictorDataset(Dataset):
    def __init__(self, video_paths, frame_size=(256,256), seq_len=5):
        self.videos = video_paths
        self.frame_size = frame_size
        self.seq_len = seq_len
    def __len__(self):
        return len(self.videos) * 50
    def __getitem__(self, idx):
        video_idx = idx % len(self.videos)
        cap = cv2.VideoCapture(self.videos[video_idx])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = np.random.randint(0, max(1, total - self.seq_len))
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2,0,1))
            frames.append(frame)
        cap.release()
        return torch.from_numpy(np.stack(frames)).float()

# ------------------------------
# Train predictor
# ------------------------------
video_paths = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')
if not video_paths:
    raise FileNotFoundError("No .mpg files found in dataset")
print(f"Found {len(video_paths)} videos for predictor training")

dataset = PredictorDataset(video_paths, frame_size=(256,256), seq_len=5)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

predictor = LeWMPredictor(latent_dim=192, hidden_dim=256, num_layers=8, num_heads=4).to(device)
optimizer = optim.AdamW(predictor.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.MSELoss()

print("Training predictor...")
for epoch in range(1, 31):
    predictor.train()
    total_loss = 0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Predictor Epoch {epoch}/30")
    for batch in pbar:
        batch = batch.to(device)
        B, T = batch.shape[:2]
        # Extract latents
        latents = []
        for t in range(T):
            frame = batch[:, t]
            with torch.no_grad():
                latent = autoencoder.encoder(frame, return_surprise=False)
            latents.append(latent)
        # Predict next from first 4
        context = latents[:4]
        target = latents[4]
        pred_mean, _ = predictor(context)
        loss = criterion(pred_mean, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=loss.item())
    scheduler.step()
    print(f"Epoch {epoch}: Loss = {total_loss/num_batches:.4f}")

# Save predictor
torch.save(predictor.state_dict(), '/root/le-maia/checkpoints/predictor_final.pt')
print("Predictor saved.")

# ------------------------------
# Entropy model and quantizer
# ------------------------------
entropy_model = HyperpriorEntropy(latent_dim=192, hyper_channels=320).to(device)
entropy_model.eval()
quantizer = Quantizer(num_levels=256, mode='inference').to(device)

# ------------------------------
# Encoding functions with I/P-frames
# ------------------------------
def encode_frame_i(frame_tensor):
    with torch.no_grad():
        latent = autoencoder.encoder(frame_tensor, return_surprise=False)
        quantized = quantizer(latent)
        rate_nats, _ = entropy_model(quantized)
        bits = int(rate_nats.sum().item() * np.log2(np.e))
        return quantized, bits

def encode_frame_p(frame_tensor, prev_quantized):
    with torch.no_grad():
        pred_mean, _ = predictor([prev_quantized])
        latent = autoencoder.encoder(frame_tensor, return_surprise=False)
        residual = latent - pred_mean
        quantized_residual = quantizer(residual)
        rate_nats, _ = entropy_model(quantized_residual)
        bits = int(rate_nats.sum().item() * np.log2(np.e))
        reconstructed_latent = pred_mean + quantized_residual
        return reconstructed_latent, bits

def decode_frame(quantized_latent, target_size):
    with torch.no_grad():
        recon = autoencoder.decoder(quantized_latent, target_size=target_size)
    return recon

def encode_video_frames(frames, gop_size=16):
    T = len(frames)
    h, w = frames[0].shape[:2]
    # Convert frames to tensor [1, T, C, H, W]
    frame_tensor = torch.stack([torch.from_numpy(f).float().permute(2,0,1)/255.0 for f in frames]).unsqueeze(0).to(device)
    decoded_frames = []
    total_bits = 0
    prev_quantized = None
    for t in range(T):
        current = frame_tensor[:, t]
        if t % gop_size == 0 or prev_quantized is None:
            quantized, bits = encode_frame_i(current)
            total_bits += bits
            prev_quantized = quantized
        else:
            quantized, bits = encode_frame_p(current, prev_quantized)
            total_bits += bits
            prev_quantized = quantized
        recon = decode_frame(quantized, target_size=(h,w))
        recon_np = (recon.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        decoded_frames.append(recon_np)
    return decoded_frames, total_bits

# ------------------------------
# x265 encoding and PSNR
# ------------------------------
def compute_psnr(orig, recon):
    mse = np.mean((orig.astype(float) - recon.astype(float))**2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def encode_with_x265(video_path, crf, output_path):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-c:v", "libx265", "-crf", str(crf), "-preset", "medium", output_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Decode to frames
    decoded_dir = Path(output_path).stem + "_decoded"
    os.makedirs(decoded_dir, exist_ok=True)
    subprocess.run(["ffmpeg", "-i", output_path, os.path.join(decoded_dir, "frame_%06d.png")],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Load original frames
    cap = cv2.VideoCapture(video_path)
    orig_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig_frames.append(frame)
    cap.release()
    # Load decoded frames
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

# ------------------------------
# Extract frames from test video
# ------------------------------
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

# ------------------------------
# BD-rate function
# ------------------------------
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
# Main benchmark
# ------------------------------
def main():
    # Use first video as test
    test_video = video_paths[0]
    print(f"Test video: {test_video}")

    frames = extract_frames(test_video, max_frames=150, target_size=(256,256))
    print(f"Extracted {len(frames)} frames")

    # Vary quantization scale to get RD curve
    scales = [0.5, 1.0, 2.0, 4.0]
    lewm_results = []
    for scale in scales:
        quantizer.step_size = torch.tensor(2.0/256 * scale).to(device)
        print(f"Encoding with LeWM-VC, scale={scale}")
        decoded, total_bits = encode_video_frames(frames, gop_size=16)
        psnr_vals = [compute_psnr(frames[i], decoded[i]) for i in range(len(decoded))]
        avg_psnr = np.mean(psnr_vals)
        total_pixels = len(frames) * frames[0].shape[0] * frames[0].shape[1]
        bpp = total_bits / total_pixels
        lewm_results.append((bpp, avg_psnr, scale))
        print(f"  bpp={bpp:.4f}, PSNR={avg_psnr:.2f} dB")

    # x265 encoding
    crf_list = [23, 28, 32, 36]
    x265_results = []
    for crf in crf_list:
        out_path = f"/tmp/x265_crf{crf}.mp4"
        print(f"Encoding with x265 CRF={crf}")
        bpp, psnr = encode_with_x265(test_video, crf, out_path)
        x265_results.append((bpp, psnr, crf))
        print(f"  bpp={bpp:.4f}, PSNR={psnr:.2f} dB")

    # Compute BD-rate
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

    # Save CSV
    os.makedirs("/root/le-maia/benchmark_results", exist_ok=True)
    csv_path = "/root/le-maia/benchmark_results/full_codec_benchmark.csv"
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Codec", "Scale/CRF", "bpp", "PSNR"])
        for bpp, psnr, s in lewm_results:
            writer.writerow(["LeWM-VC", s, bpp, psnr])
        for bpp, psnr, c in x265_results:
            writer.writerow(["x265", c, bpp, psnr])
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()
