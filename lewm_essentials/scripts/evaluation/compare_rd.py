#!/usr/bin/env python3
"""
Compare RD curves of LeWM-VC (best checkpoints) vs x265 on validation videos.
Computes BD-rate (positive = LeWM-VC uses more bits).
"""

import os
import sys
import glob
import subprocess
import csv
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

sys.path.insert(0, '/root/le-maia/src')
from train_joint_phase0 import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---------- Prepare validation videos as a single video file for x265 ----------
video_paths = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')
if not video_paths:
    raise FileNotFoundError("No videos found")

# We'll create a temporary concatenated video of the first few seconds of each validation clip
# For simplicity, we use the same validation loader but extract frames and encode with x265.
# Since x265 works on video files, we need to write frames to a video. Let's use the first video only.
test_video = video_paths[0]
print(f"Using test video: {test_video}")

# ---------- Evaluate LeWM-VC on this video (using best checkpoint) ----------
lam = 0.05
ae_ckpt = f'/root/le-maia/checkpoints_joint_phase0/ae_lambda_{lam}_best.pt'
ent_ckpt = f'/root/le-maia/checkpoints_joint_phase0/entropy_lambda_{lam}_best.pt'

autoencoder = VideoAutoencoder().to(device)
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
QUANT_STEP = 2.0 / 255

def laplace_likelihood_discrete(y, mu, log_scale, step, sigma_floor=0.003, epsilon=1e-9):
    scale = torch.nn.functional.softplus(log_scale) + sigma_floor
    laplace = Laplace(mu, scale)
    cdf_upper = laplace.cdf(y + 0.5 * step)
    cdf_lower = laplace.cdf(y - 0.5 * step)
    pmf = cdf_upper - cdf_lower
    pmf = torch.clamp(pmf, min=epsilon, max=1.0)
    nll = -torch.log(pmf)
    return nll.mean()

# Extract frames from test video
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
print(f"Loaded {len(frames)} frames from {test_video}")

total_bits = 0
total_mse = 0
for frame in tqdm(frames, desc="LeWM-VC encoding"):
    frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
    frame_t = frame_t.to(device)
    with torch.no_grad():
        latent_norm = autoencoder.encode(frame_t)
        quantized = quantizer(latent_norm)
        mu, log_scale = entropy_model(quantized)
        nll = laplace_likelihood_discrete(quantized, mu, log_scale, step=QUANT_STEP, sigma_floor=0.003)
        bits = nll * quantized.numel() / np.log(2)
        total_bits += bits.item()
        recon = autoencoder.decode(quantized, target_size=target_size)
        mse = torch.nn.functional.mse_loss(recon, frame_t).item()
        total_mse += mse
bpp_lewm = total_bits / (len(frames) * target_size[0] * target_size[1])
psnr_lewm = 20 * np.log10(1.0 / np.sqrt(total_mse / len(frames)))
print(f"LeWM-VC (λ={lam}): bpp={bpp_lewm:.4f}, PSNR={psnr_lewm:.2f} dB")

# ---------- x265 encoding at multiple CRF ----------
def encode_x265(video_path, crf, target_size):
    out_path = f'/tmp/x265_crf{crf}.mp4'
    cmd = ['ffmpeg', '-y', '-i', video_path, '-vf', f'scale={target_size[0]}:{target_size[1]}', '-c:v', 'libx265', '-crf', str(crf), '-preset', 'medium', out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Decode to frames and compute PSNR
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

crf_list = [18, 22, 26, 30, 34]  # covers range around 35 dB
x265_results = []
for crf in crf_list:
    bpp, psnr = encode_x265(test_video, crf, target_size)
    x265_results.append((crf, bpp, psnr))
    print(f"x265 CRF={crf}: bpp={bpp:.4f}, PSNR={psnr:.2f} dB")

# ---------- Compute BD-rate (LeWM-VC vs x265) ----------
def bd_rate(rate1, psnr1, rate2, psnr2):
    import numpy as np
    from scipy.interpolate import interp1d
    # Sort by PSNR
    idx1 = np.argsort(psnr1)
    idx2 = np.argsort(psnr2)
    psnr1_s = np.array(psnr1)[idx1]
    psnr2_s = np.array(psnr2)[idx2]
    rate1_s = np.array(rate1)[idx1]
    rate2_s = np.array(rate2)[idx2]
    # Common PSNR range
    psnr_min = max(psnr1_s.min(), psnr2_s.min())
    psnr_max = min(psnr1_s.max(), psnr2_s.max())
    if psnr_min >= psnr_max:
        return float('nan')
    psnr_interp = np.linspace(psnr_min, psnr_max, 100)
    # Interpolate log rates
    rate1_interp = np.interp(psnr_interp, psnr1_s, np.log(rate1_s))
    rate2_interp = np.interp(psnr_interp, psnr2_s, np.log(rate2_s))
    avg_diff = np.mean(rate2_interp - rate1_interp)
    return (np.exp(avg_diff) - 1) * 100

# We only have one point for LeWM-VC, so BD-rate cannot be computed.
# Instead, compare at the PSNR of LeWM-VC.
# Find x265 point with closest PSNR.
closest = min(x265_results, key=lambda x: abs(x[2] - psnr_lewm))
crf_closest, bpp_x265, psnr_x265 = closest
savings = (1 - bpp_lewm / bpp_x265) * 100 if bpp_x265 > 0 else 0
print(f"\nAt PSNR ≈ {psnr_lewm:.2f} dB:")
print(f"  LeWM-VC bitrate: {bpp_lewm:.4f} bpp")
print(f"  x265 (CRF {crf_closest}) bitrate: {bpp_x265:.4f} bpp")
print(f"  Bitrate savings: {savings:.2f}% (negative = LeWM-VC worse)")

# Save results
os.makedirs('/root/le-maia/benchmark_results', exist_ok=True)
with open('/root/le-maia/benchmark_results/rd_comparison.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Codec', 'Setting', 'bpp', 'PSNR'])
    writer.writerow(['LeWM-VC', f'λ={lam}', bpp_lewm, psnr_lewm])
    for crf, bpp, psnr in x265_results:
        writer.writerow(['x265', crf, bpp, psnr])
print("Results saved to /root/le-maia/benchmark_results/rd_comparison.csv")
