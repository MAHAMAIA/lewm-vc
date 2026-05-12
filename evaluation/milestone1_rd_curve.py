#!/usr/bin/env python3
"""
milestone1_rd_curve.py
Trains LeWM-VC GMM models across a lambda sweep and evaluates BD-rate vs x265.
Produces: rd_curve.csv, bd_rate_report.txt, training log.
Usage: python3 milestone1_rd_curve.py
"""

import os, sys, glob, subprocess, shutil, csv, datetime
import cv2, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.distributions.normal import Normal
from tqdm import tqdm

# --- Paths ---
sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

RESOLUTION = 256
QUANT_STEP = 2.0 / 255
BATCH_SIZE = 8
EPOCHS = 100
LAMBDAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
OUTPUT_DIR = 'checkpoints_milestone1'
DATASET_DIR = 'datasets/pevid-hd'
BENCHMARK_DIR = 'benchmark_milestone1'
NUM_EVAL_FRAMES = 100
X265_CRFS = [18, 23, 28, 33, 38]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BENCHMARK_DIR, exist_ok=True)

# --- Logging ---
LOG_FILE = os.path.join(BENCHMARK_DIR, f'training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, open(LOG_FILE, 'w'))
print(f"Logging to {LOG_FILE}")

# ============================================================
# ARCHITECTURE (exact match to working GMM evaluation)
# ============================================================
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
    def forward(self, x):
        return x * self.scale + self.shift

class GMMEntropyModel(nn.Module):
    def __init__(self, latent_dim=192, hyper_channels=256, num_components=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.hyperprior = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hyper_channels, hyper_channels, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(hyper_channels, hyper_channels, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 4, stride=2, padding=1), nn.ReLU(),
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

class VideoAutoencoder(nn.Module):
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
        self.affine = AffineNormalization(latent_dim)
    def encode(self, x):
        return self.affine(self.encoder(x, return_surprise=False))
    def decode(self, latent, target_size):
        return self.decoder(latent, target_size=target_size)

# ============================================================
# LIKELIHOOD
# ============================================================
def gmm_likelihood_discrete(y, mu, scale, weight, step, epsilon=1e-12):
    B, C, H, W = y.shape
    num_comp = mu.shape[1]
    y_expanded = y.unsqueeze(1).expand(-1, num_comp, -1, -1, -1)
    normal = Normal(mu, scale)
    cdf_upper = normal.cdf(y_expanded + 0.5 * step)
    cdf_lower = normal.cdf(y_expanded - 0.5 * step)
    pmf = torch.clamp(cdf_upper - cdf_lower, min=epsilon, max=1.0)
    mixture_pmf = (weight * pmf).sum(dim=1)
    return -torch.log(mixture_pmf).mean()

# ============================================================
# DATASET
# ============================================================
class VideoDataset(Dataset):
    def __init__(self, video_paths, frame_size=(RESOLUTION, RESOLUTION), frames_per_clip=1):
        self.videos = video_paths
        self.frame_size = frame_size
        self.frames_per_clip = frames_per_clip
    def __len__(self):
        return len(self.videos) * 200
    def __getitem__(self, idx):
        video_idx = idx % len(self.videos)
        cap = cv2.VideoCapture(self.videos[video_idx])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = np.random.randint(0, max(1, total - self.frames_per_clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(self.frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(np.transpose(frame, (2, 0, 1)))
        cap.release()
        return torch.from_numpy(frames[0]).float()

# ============================================================
# X265 BENCHMARK
# ============================================================
def encode_x265(video_path, crf, target_size):
    out_path = f'/tmp/x265_crf{crf}.mp4'
    cmd = ['ffmpeg', '-y', '-i', video_path, '-vf', f'scale={target_size[0]}:{target_size[1]}',
           '-c:v', 'libx265', '-crf', str(crf), '-preset', 'medium', out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    decoded_dir = f'/tmp/x265_decoded_{crf}'
    os.makedirs(decoded_dir, exist_ok=True)
    subprocess.run(['ffmpeg', '-i', out_path, os.path.join(decoded_dir, 'frame_%06d.png')],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cap_orig = cv2.VideoCapture(video_path)
    orig, dec = [], []
    while True:
        ret, frame = cap_orig.read()
        if not ret: break
        frame = cv2.resize(frame, target_size)
        orig.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap_orig.release()
    for p in sorted(glob.glob(os.path.join(decoded_dir, '*.png'))):
        dec.append(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
    min_len = min(len(orig), len(dec))
    psnr_sum = sum(20 * np.log10(255.0 / np.sqrt(np.mean((orig[i].astype(float) - dec[i].astype(float))**2 + 1e-10)))
                   for i in range(min_len))
    file_bits = os.path.getsize(out_path) * 8
    bpp = file_bits / (min_len * target_size[0] * target_size[1] * 3)
    shutil.rmtree(decoded_dir, ignore_errors=True)
    os.remove(out_path)
    return bpp, psnr_sum / max(1, min_len)

# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(ae, ent, frames):
    autoencoder = VideoAutoencoder().to(DEVICE)
    autoencoder.load_state_dict(torch.load(ae, map_location=DEVICE, weights_only=False), strict=False)
    autoencoder.eval()
    entropy_model = GMMEntropyModel().to(DEVICE)
    state = torch.load(ent, map_location=DEVICE, weights_only=False)
    for k in list(state.keys()):
        if 'mask' in k: del state[k]
    entropy_model.load_state_dict(state, strict=False)
    entropy_model.eval()
    total_bits, total_mse, total_pix = 0.0, 0.0, 0
    with torch.no_grad():
        for frame in frames:
            x = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
            latent = autoencoder.encode(x)
            x_quant = torch.round(latent / QUANT_STEP) * QUANT_STEP
            q = x_quant + (latent - x_quant.detach()) * 0.5
            mu, scale, weight = entropy_model(q)
            nll = gmm_likelihood_discrete(q, mu, scale, weight, step=QUANT_STEP)
            bits = (nll.item() / np.log(2)) * q.numel()
            total_bits += bits
            recon = autoencoder.decode(q, target_size=(RESOLUTION, RESOLUTION))
            total_mse += torch.nn.functional.mse_loss(recon, x).item()
            total_pix += x.numel()
    return total_bits / total_pix, 10 * np.log10(1.0 / (total_mse / len(frames)))

# ============================================================
# TRAINING
# ============================================================
def train_one_lambda(lam):
    video_paths = glob.glob(os.path.join(DATASET_DIR, '*.mpg'))
    if not video_paths: video_paths = glob.glob(os.path.join(DATASET_DIR, '*.avi'))
    if not video_paths: video_paths = glob.glob(os.path.join(DATASET_DIR, '*.mov'))
    if not video_paths: raise FileNotFoundError(f"No videos in {DATASET_DIR}")
    dataset = VideoDataset(video_paths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    autoencoder = VideoAutoencoder().to(DEVICE)
    entropy_model = GMMEntropyModel().to(DEVICE)
    opt = optim.AdamW([
        {'params': autoencoder.parameters(), 'lr': 1e-4},
        {'params': entropy_model.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-6)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for epoch in range(1, EPOCHS+1):
        autoencoder.train(); entropy_model.train()
        total_loss_epoch, total_rate_epoch, total_dist_epoch, n_batches = 0.0, 0.0, 0.0, 0
        for batch in tqdm(loader, desc=f"λ={lam} epoch {epoch}/{EPOCHS}"):
            batch = batch.to(DEVICE)
            opt.zero_grad()
            latent = autoencoder.encode(batch)
            x_quant = torch.round(latent / QUANT_STEP) * QUANT_STEP
            q = x_quant + (latent - x_quant.detach()) * 0.5
            mu, scale, weight = entropy_model(q)
            nll = gmm_likelihood_discrete(q, mu, scale, weight, step=QUANT_STEP)
            rate = (nll * q.numel() / np.log(2)) / batch.numel()
            recon = autoencoder.decode(q, target_size=(RESOLUTION, RESOLUTION))
            distortion = torch.nn.functional.mse_loss(recon, batch)
            loss = lam * rate + distortion
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(autoencoder.parameters()) + list(entropy_model.parameters()), 1.0)
            opt.step()
            total_loss_epoch += loss.item()
            total_rate_epoch += rate.item()
            total_dist_epoch += distortion.item()
            n_batches += 1
        sched.step()
        avg_loss = total_loss_epoch / n_batches
        avg_rate = total_rate_epoch / n_batches
        avg_dist = total_dist_epoch / n_batches
        avg_psnr = 10 * np.log10(1.0 / avg_dist) if avg_dist > 0 else 100.0
        print(f"λ={lam} epoch {epoch}/{EPOCHS}: Loss={avg_loss:.4f}, Rate={avg_rate:.4f} bpp, Dist={avg_dist:.6f}, PSNR={avg_psnr:.2f} dB")

    torch.save(autoencoder.state_dict(), f'{OUTPUT_DIR}/ae_lambda_{lam}_final.pt')
    torch.save(entropy_model.state_dict(), f'{OUTPUT_DIR}/entropy_lambda_{lam}_final.pt')
    return autoencoder, entropy_model

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    # --- Load evaluation frames ---
    eval_videos = glob.glob(os.path.join(DATASET_DIR, '*.mpg'))
    if not eval_videos: eval_videos = glob.glob(os.path.join(DATASET_DIR, '*.avi'))
    test_video = eval_videos[0]
    cap = cv2.VideoCapture(test_video)
    eval_frames = []
    while len(eval_frames) < NUM_EVAL_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (RESOLUTION, RESOLUTION))
        eval_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    cap.release()
    print(f"Evaluation video: {test_video} ({len(eval_frames)} frames)")

    # --- Train/evaluate each lambda ---
    results = []
    for lam in LAMBDAS:
        ae_path = f'{OUTPUT_DIR}/ae_lambda_{lam}_final.pt'
        ent_path = f'{OUTPUT_DIR}/entropy_lambda_{lam}_final.pt'
        if not (os.path.exists(ae_path) and os.path.exists(ent_path)):
            print(f"\n=== Training λ={lam} ===")
            train_one_lambda(lam)
        else:
            print(f"\n=== λ={lam} checkpoints exist, skipping training ===")
        print(f"Evaluating λ={lam}...")
        bpp, psnr = evaluate_model(ae_path, ent_path, eval_frames)
        results.append((lam, bpp, psnr))
        print(f"  λ={lam}: BPP={bpp:.6f}, PSNR={psnr:.2f} dB")

    # --- x265 baseline ---
    x265_results = []
    for crf in X265_CRFS:
        bpp, psnr = encode_x265(test_video, crf, (RESOLUTION, RESOLUTION))
        x265_results.append((crf, bpp, psnr))
        print(f"x265 CRF={crf}: BPP={bpp:.6f}, PSNR={psnr:.2f} dB")

    # --- Save CSV ---
    with open(f'{BENCHMARK_DIR}/rd_curve.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lambda', 'bpp', 'psnr'])
        w.writerows(results)
        w.writerow([])
        w.writerow(['x265_crf', 'bpp', 'psnr'])
        w.writerows(x265_results)

    # --- BD-rate computation (fixed for NumPy 2.x) ---
    def compute_bd_rate(ref_bpp, ref_psnr, test_bpp, test_psnr):
        min_psnr = max(min(ref_psnr), min(test_psnr))
        max_psnr = min(max(ref_psnr), max(test_psnr))
        if min_psnr >= max_psnr:
            return float('nan')
        psnr_range = np.linspace(min_psnr, max_psnr, 100)
        ref_interp = np.interp(psnr_range, ref_psnr, ref_bpp)
        test_interp = np.interp(psnr_range, test_psnr, test_bpp)
        return np.trapezoid((test_interp - ref_interp) / ref_interp, psnr_range) / (max_psnr - min_psnr) * 100

    if len(results) >= 3 and len(x265_results) >= 3:
        lewm_bpp = np.array([r[1] for r in results])
        lewm_psnr = np.array([r[2] for r in results])
        x265_bpp = np.array([r[1] for r in x265_results])
        x265_psnr = np.array([r[2] for r in x265_results])
        bd_rate = compute_bd_rate(x265_bpp, x265_psnr, lewm_bpp, lewm_psnr)
        print(f"\nBD-rate (LeWM-VC vs x265): {bd_rate:+.2f}%")
        print("Negative = LeWM-VC saves bitrate")
        print(f"LeWM points: {[(r[1], r[2]) for r in results]}")
        print(f"x265 points: {[(r[1], r[2]) for r in x265_results]}")
        with open(f'{BENCHMARK_DIR}/bd_rate_report.txt', 'w') as f:
            f.write(f"BD-rate (LeWM-VC vs x265): {bd_rate:+.2f}%\n")
            f.write(f"Negative = LeWM-VC saves bitrate\n")
            f.write(f"LeWM points: {[(r[1], r[2]) for r in results]}\n")
            f.write(f"x265 points: {[(r[1], r[2]) for r in x265_results]}\n")

    print(f"\nDone. Results in {BENCHMARK_DIR}/")
    print(f"Log saved to {LOG_FILE}")
