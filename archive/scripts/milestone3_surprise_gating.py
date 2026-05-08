#!/usr/bin/env python3
"""
milestone3_surprise_gating.py
Adds VOE surprise detection to the encoding loop.
High-surprise frames get finer quantization (more bits preserved).
Evaluates BPP ratio between normal and anomaly videos.
Uses milestone2 temporal checkpoint.
Usage: python3 milestone3_surprise_gating.py
"""

import os, sys, glob, csv, datetime
import cv2, numpy as np
import torch, torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.predictor import LeWMPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESOLUTION = 256
QUANT_STEP = 2.0 / 255
TAU_HIGH = 0.8
TAU_LOW = 0.4
DATASET_DIR = 'datasets/pevid-hd'
BENCHMARK_DIR = 'benchmark_milestone3'
os.makedirs(BENCHMARK_DIR, exist_ok=True)

LOG_FILE = os.path.join(BENCHMARK_DIR, f'surprise_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()
sys.stdout = Tee(sys.stdout, open(LOG_FILE, 'w'))
print(f"Logging to {LOG_FILE}")
print(f"Device: {DEVICE}")
print(f"Thresholds: TAU_HIGH={TAU_HIGH}, TAU_LOW={TAU_LOW}")

# --- Architecture ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels); self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels); self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        r = x; x = torch.nn.functional.gelu(self.norm1(x)); x = self.conv1(x)
        x = torch.nn.functional.gelu(self.norm2(x)); x = self.conv2(x)
        return x + r

class LeWMDecoder(nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(latent_dim, hidden_dim, 1)
        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4,2,1); self.res1 = ResidualBlock(hidden_dim//2)
        self.up2 = nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 4,2,1); self.res2 = ResidualBlock(hidden_dim//4)
        self.up3 = nn.ConvTranspose2d(hidden_dim//4, hidden_dim//8, 4,2,1); self.res3 = ResidualBlock(hidden_dim//8)
        self.up4 = nn.ConvTranspose2d(hidden_dim//8, hidden_dim//16, 4,2,1); self.res4 = ResidualBlock(hidden_dim//16)
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim//16, hidden_dim//32, 3,1,1), nn.InstanceNorm2d(hidden_dim//32),
            nn.GELU(), nn.Conv2d(hidden_dim//32, 3,3,1,1),
        )
    def forward(self, latent, target_size=None):
        x = self.proj(latent); x = self.up1(x); x = self.res1(x); x = self.up2(x); x = self.res2(x)
        x = self.up3(x); x = self.res3(x); x = self.up4(x); x = self.res4(x)
        x = torch.sigmoid(self.final(x))
        if target_size: x = torch.nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class AffineNormalization(nn.Module):
    def __init__(self, n): super().__init__(); self.s = nn.Parameter(torch.ones(1,n,1,1)); self.b = nn.Parameter(torch.zeros(1,n,1,1))
    def forward(self, x): return x * self.s + self.b

class GMMEntropyModel(nn.Module):
    def __init__(self, latent_dim=192, hyper_channels=256, nc=2):
        super().__init__(); self.latent_dim = latent_dim; self.nc = nc
        self.hp = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hyper_channels, hyper_channels, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(hyper_channels, hyper_channels, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hyper_channels, latent_dim * nc * 3, 3, padding=1),
        )
        self.sp = nn.Softplus()
    def forward(self, x):
        p = self.hp(x); B, C, H, W = p.shape; cp = C // self.nc
        p = p.view(B, self.nc, cp, H, W)
        mu = p[:,:,:self.latent_dim]; ls = p[:,:,self.latent_dim:2*self.latent_dim]
        lw = p[:,:,2*self.latent_dim:3*self.latent_dim]
        return mu, self.sp(ls)+1e-5, torch.softmax(lw, dim=1)

from torch.distributions.normal import Normal
def gmm_bits(y, mu, scale, weight, step, eps=1e-12):
    B,C,H,W = y.shape; nc = mu.shape[1]; ye = y.unsqueeze(1).expand(-1,nc,-1,-1,-1)
    n = Normal(mu, scale)
    pmf = torch.clamp(n.cdf(ye+0.5*step)-n.cdf(ye-0.5*step), min=eps, max=1.0)
    nll = -torch.log((weight*pmf).sum(dim=1)).mean()
    return (nll / np.log(2)) * y.numel()

class SurpriseGatedCodec(nn.Module):
    def __init__(self, ckpt_path=None):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=192, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=192)
        self.affine = AffineNormalization(192)
        self.predictor = LeWMPredictor(latent_dim=192)
        self.entropy = GMMEntropyModel()
        self.predictor_trained = False
        if ckpt_path and os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            if 'codec' in sd:
                self.load_state_dict(sd['codec'], strict=False)
                self.predictor_trained = True
            if 'entropy' in sd:
                self.entropy.load_state_dict(sd['entropy'], strict=False)
        self.to(DEVICE)
        self.eval()

    def encode_frame(self, x, prev_latents):
        latent = self.affine(self.encoder(x, return_surprise=False))
        if len(prev_latents) == 0 or not self.predictor_trained:
            xq = torch.round(latent / QUANT_STEP) * QUANT_STEP
            q = xq + (latent - xq.detach()) * 0.5
            return latent, q, 0.5, True
        pred_mean, _ = self.predictor(prev_latents)
        residual = latent - pred_mean
        mse = torch.nn.functional.mse_loss(latent, pred_mean, reduction='none').mean().item()
        pred_mag = pred_mean.abs().mean().item() + 1e-8
        surprise = mse / pred_mag
        if surprise >= TAU_HIGH:
            step = QUANT_STEP * 0.5
        elif surprise <= TAU_LOW:
            step = QUANT_STEP * 2.0
        else:
            step = QUANT_STEP
        xq = torch.round(residual / step) * step
        q = xq + (residual - xq.detach()) * 0.5
        return latent, (q, step, pred_mean), surprise, False

    def decode_frame(self, coded, prev_latents):
        if isinstance(coded, tuple):
            q, step, pred_mean = coded
            latent = pred_mean + q
        else:
            latent = coded
        return self.decoder(latent, target_size=(RESOLUTION, RESOLUTION))

def evaluate_video(codec, video_path, label):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 100:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (RESOLUTION, RESOLUTION))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
    cap.release()

    prev_latents = []; total_bits = 0; high_bits = 0; low_bits = 0; med_bits = 0
    high_count = 0; low_count = 0; med_count = 0
    total_mse = 0.0; surprises = []

    with torch.no_grad():
        for frame in tqdm(frames, desc=f"Encoding {label}"):
            x = torch.from_numpy(np.transpose(frame, (2,0,1))).unsqueeze(0).float().to(DEVICE)
            latent, coded, surprise, is_I = codec.encode_frame(x, prev_latents)
            surprises.append(surprise)
            if is_I:
                q_for_rate, step_for_rate = coded, QUANT_STEP
            else:
                q_for_rate, step_for_rate, _ = coded
            mu, scale, weight = codec.entropy(q_for_rate)
            bits = gmm_bits(q_for_rate, mu, scale, weight, step=step_for_rate).item()
            total_bits += bits
            if surprise >= TAU_HIGH:
                high_bits += bits; high_count += 1
            elif surprise <= TAU_LOW:
                low_bits += bits; low_count += 1
            else:
                med_bits += bits; med_count += 1
            decoded = codec.decode_frame(coded if is_I else coded, prev_latents)
            total_mse += torch.nn.functional.mse_loss(decoded, x).item()
            if is_I:
                prev_latents = [latent.detach()]
            else:
                pred_mean = coded[2]
                prev_latents.append((pred_mean + q_for_rate).detach())
            if len(prev_latents) > 4: prev_latents.pop(0)

    n = len(frames)
    bpp = total_bits / (n * 3 * RESOLUTION * RESOLUTION)
    psnr = 10 * np.log10(1.0 / (total_mse / n))
    avg_s = np.mean(surprises)
    print(f"  {label}: BPP={bpp:.4f}, PSNR={psnr:.2f} dB, Avg surprise={avg_s:.4f}")
    print(f"          Hi ({high_count}): {high_bits/max(1,high_count):.0f} bits, Med ({med_count}): {med_bits/max(1,med_count):.0f} bits, Lo ({low_count}): {low_bits/max(1,low_count):.0f} bits")
    return bpp, psnr, high_bits, low_bits, med_bits, high_count, low_count, med_count, avg_s

if __name__ == '__main__':
    ckpt = 'checkpoints_milestone2/temporal_final.pt'
    if not os.path.exists(ckpt):
        print(f"ERROR: {ckpt} not found. Run milestone2 first."); exit(1)
    codec = SurpriseGatedCodec(ckpt_path=ckpt)

    normal_video = sorted(glob.glob(os.path.join(DATASET_DIR, 'walking*.mpg')))
    anomaly_video = sorted(glob.glob(os.path.join(DATASET_DIR, 'droppingBag*.mpg')))
    if not normal_video: normal_video = sorted(glob.glob(os.path.join(DATASET_DIR, '*.mpg')))[:1]
    if not anomaly_video: anomaly_video = sorted(glob.glob(os.path.join(DATASET_DIR, '*.mpg')))[1:2]
    if not normal_video or not anomaly_video: print("Need 2 videos."); exit(1)

    print(f"\n{'='*60}\nNormal:  {normal_video[0]}\nAnomaly: {anomaly_video[0]}\n{'='*60}\n")
    nb, np_, nhb, nlb, nmb, nhc, nlc, nmc, ns = evaluate_video(codec, normal_video[0], "Normal")
    ab, ap_, ahb, alb, amb, ahc, alc, amc, as_ = evaluate_video(codec, anomaly_video[0], "Anomaly")

    print(f"\n{'='*60}\nSURPRISE GATING RESULTS\n{'='*60}")
    print(f"Thresholds: Hi>{TAU_HIGH}, Lo<{TAU_LOW}")
    print(f"Normal:   BPP={nb:.4f}, PSNR={np_:.2f} dB, Avg surprise={ns:.4f}")
    print(f"          Hi={nhc} ({nhb/max(1,nhc):.0f} b), Med={nmc} ({nmb/max(1,nmc):.0f} b), Lo={nlc} ({nlb/max(1,nlc):.0f} b)")
    print(f"Anomaly:  BPP={ab:.4f}, PSNR={ap_:.2f} dB, Avg surprise={as_:.4f}")
    print(f"          Hi={ahc} ({ahb/max(1,ahc):.0f} b), Med={amc} ({amb/max(1,amc):.0f} b), Lo={alc} ({alb/max(1,alc):.0f} b)")
    print(f"BPP ratio (anomaly/normal): {ab/max(nb,1e-10):.2f}x")
    print(f"Surprise ratio (anomaly/normal): {as_/max(ns,1e-10):.2f}x")

    with open(f'{BENCHMARK_DIR}/surprise_gating.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video','bpp','psnr','hi_bits','med_bits','lo_bits','hi_count','med_count','lo_count','avg_surprise'])
        w.writerow(['normal', nb, np_, nhb, nmb, nlb, nhc, nmc, nlc, ns])
        w.writerow(['anomaly', ab, ap_, ahb, amb, alb, ahc, amc, alc, as_])
    print(f"\nResults saved to {BENCHMARK_DIR}/surprise_gating.csv")
