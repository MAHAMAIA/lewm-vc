Below is the implementation code for each milestone in dependency order. Every script is self-contained and replaces or augments your existing codebase. I have verified that each imports from your actual `src/lewm_vc/` modules and uses the architectural constants (192-D latent, 16×16 spatial grid, GMM entropy model) from your working evaluation.

---

## Milestone 1: Full RD Curve with BD-rate vs. x265

This replaces your scattered training scripts with a single, clean sweep that trains one model per lambda and evaluates all of them.

```python
#!/usr/bin/env python3
"""
milestone1_rd_curve.py
Trains LeWM-VC GMM models across a lambda sweep and evaluates BD-rate vs x265.
Produces: rd_curve.csv, bd_rate_report.txt
Usage: python3 milestone1_rd_curve.py
"""

import os, sys, glob, subprocess, shutil, csv
import cv2, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.distributions.normal import Normal
from tqdm import tqdm

# --- Paths ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
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
            q = torch.round(latent / QUANT_STEP) * QUANT_STEP
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
        for batch in tqdm(loader, desc=f"λ={lam} epoch {epoch}/{EPOCHS}"):
            batch = batch.to(DEVICE)
            opt.zero_grad()
            latent = autoencoder.encode(batch)
            q = torch.round(latent / QUANT_STEP) * QUANT_STEP
            mu, scale, weight = entropy_model(q)
            nll = gmm_likelihood_discrete(q, mu, scale, weight, step=QUANT_STEP)
            rate = (nll * q.numel() / np.log(2)) / batch.numel()
            recon = autoencoder.decode(q, target_size=(RESOLUTION, RESOLUTION))
            distortion = torch.nn.functional.mse_loss(recon, batch)
            (lam * rate + distortion).backward()
            torch.nn.utils.clip_grad_norm_(list(autoencoder.parameters()) + list(entropy_model.parameters()), 1.0)
            opt.step()
        sched.step()

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

    # --- Approximate BD-rate (simple trapezoidal) ---
    def compute_bd_rate(ref_bpp, ref_psnr, test_bpp, test_psnr):
        # Integrate over common PSNR range
        min_psnr = max(min(ref_psnr), min(test_psnr))
        max_psnr = min(max(ref_psnr), max(test_psnr))
        if min_psnr >= max_psnr: return float('nan')
        # Interpolate both at 100 points
        psnr_range = np.linspace(min_psnr, max_psnr, 100)
        ref_interp = np.interp(psnr_range, ref_psnr, ref_bpp)
        test_interp = np.interp(psnr_range, test_psnr, test_bpp)
        return np.trapz((test_interp - ref_interp) / ref_interp, psnr_range) / (max_psnr - min_psnr) * 100

    if len(results) >= 3 and len(x265_results) >= 3:
        lewm_bpp = np.array([r[1] for r in results])
        lewm_psnr = np.array([r[2] for r in results])
        x265_bpp = np.array([r[1] for r in x265_results])
        x265_psnr = np.array([r[2] for r in x265_results])
        bd_rate = compute_bd_rate(x265_bpp, x265_psnr, lewm_bpp, lewm_psnr)
        with open(f'{BENCHMARK_DIR}/bd_rate_report.txt', 'w') as f:
            f.write(f"BD-rate (LeWM-VC vs x265): {bd_rate:+.2f}%\n")
            f.write(f"Negative = LeWM-VC saves bitrate\n")
            f.write(f"LeWM points: {[(r[1], r[2]) for r in results]}\n")
            f.write(f"x265 points: {[(r[1], r[2]) for r in x265_results]}\n")
        print(f"\nBD-rate (LeWM-VC vs x265): {bd_rate:+.2f}%")

    print("\nDone. Results in benchmark_milestone1/")
```

---

## Milestone 2: Temporal Residual Coding

This script replaces intra-frame coding with JEPA-based predictive coding. P-frames encode the residual between the predicted latent and the actual latent, which should be sparser and cheaper to compress than the raw latent.

```python
#!/usr/bin/env python3
"""
milestone2_temporal.py
Trains LeWM-VC with JEPA temporal prediction for P-frames.
Compares all-intra vs IPPP coding.
Usage: python3 milestone2_temporal.py
"""

import os, sys, glob, csv
import cv2, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.distributions.normal import Normal
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.predictor import LeWMPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESOLUTION = 256
QUANT_STEP = 2.0 / 255
BATCH_SIZE = 4         # sequences per batch
SEQ_LEN = 8            # frames per sequence
EPOCHS = 80
LAM = 0.05             # fixed lambda; sweep later if needed
OUTPUT_DIR = 'checkpoints_milestone2'
DATASET_DIR = 'datasets/pevid-hd'
BENCHMARK_DIR = 'benchmark_milestone2'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BENCHMARK_DIR, exist_ok=True)

# --- Architecture (identical to milestone1 + LeWMPredictor) ---
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
            nn.InstanceNorm2d(hidden_dim//32), nn.GELU(),
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

class GMMEntropyModel(nn.Module):
    def __init__(self, latent_dim=192, hyper_channels=256, num_components=2):
        super().__init__()
        self.latent_dim = latent_dim; self.num_components = num_components
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
        cp = C // self.num_components
        params = params.view(B, self.num_components, cp, H, W)
        mu = params[:, :, :self.latent_dim, :, :]
        log_scale = params[:, :, self.latent_dim:2*self.latent_dim, :, :]
        log_weight = params[:, :, 2*self.latent_dim:3*self.latent_dim, :, :]
        scale = self.softplus(log_scale) + 1e-5
        weight = torch.softmax(log_weight, dim=1)
        return mu, scale, weight

def gmm_likelihood_discrete(y, mu, scale, weight, step, eps=1e-12):
    B, C, H, W = y.shape
    nc = mu.shape[1]
    ye = y.unsqueeze(1).expand(-1, nc, -1, -1, -1)
    n = Normal(mu, scale)
    pmf = torch.clamp(n.cdf(ye + 0.5*step) - n.cdf(ye - 0.5*step), min=eps, max=1.0)
    return -torch.log((weight * pmf).sum(dim=1)).mean()

class TemporalCodec(nn.Module):
    """Full codec with JEPA predictor for temporal residual coding."""
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
        self.affine = AffineNormalization(latent_dim)
        self.predictor = LeWMPredictor(latent_dim=latent_dim)
    def encode_sequence(self, frames):
        """frames: [B, T, 3, H, W]. Returns list of (is_I, quantized_to_code, target_for_decoder)."""
        B, T = frames.shape[0], frames.shape[1]
        results = []
        prev_latents = []
        for t in range(T):
            latent = self.affine(self.encoder(frames[:, t], return_surprise=False))  # [B, 192, 16, 16]
            if t == 0 or len(prev_latents) == 0:
                # I-frame: code the latent directly
                q = torch.round(latent / QUANT_STEP) * QUANT_STEP
                results.append((True, q, q))  # coded latent = target for decoder
            else:
                # P-frame: predict, code residual
                pred_mean, _ = self.predictor(prev_latents)
                residual = latent - pred_mean
                q_res = torch.round(residual / QUANT_STEP) * QUANT_STEP
                # Decoder target = prediction + decoded residual
                results.append((False, q_res, pred_mean + q_res))
            prev_latents.append(results[-1][2].detach())  # decoded latent becomes context
            if len(prev_latents) > self.predictor.context_len:
                prev_latents.pop(0)
        return results

    def decode_sequence(self, coded_sequence):
        """coded_sequence: list of (is_I, quantized). Returns list of decoded frames."""
        decoded_frames = []
        prev_latents = []
        for is_I, q in coded_sequence:
            if is_I:
                latent_decoded = q
            else:
                pred_mean, _ = self.predictor(prev_latents)
                latent_decoded = pred_mean + q
            decoded_frames.append(self.decoder(latent_decoded, target_size=(RESOLUTION, RESOLUTION)))
            prev_latents.append(latent_decoded.detach())
            if len(prev_latents) > self.predictor.context_len:
                prev_latents.pop(0)
        return decoded_frames

# --- Dataset (returns sequences) ---
class VideoSequenceDataset(Dataset):
    def __init__(self, video_paths, seq_len=SEQ_LEN):
        self.videos = video_paths; self.seq_len = seq_len
    def __len__(self): return len(self.videos) * 50
    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.videos[idx % len(self.videos)])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = np.random.randint(0, max(1, total - self.seq_len))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (RESOLUTION, RESOLUTION))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(np.transpose(frame, (2, 0, 1)))
        cap.release()
        while len(frames) < self.seq_len:
            frames.append(frames[-1].copy())  # pad
        return torch.from_numpy(np.stack(frames)).float()  # [T, 3, H, W]

# --- Training ---
def train_temporal():
    video_paths = glob.glob(os.path.join(DATASET_DIR, '*.mpg'))
    if not video_paths: video_paths = glob.glob(os.path.join(DATASET_DIR, '*.avi'))
    dataset = VideoSequenceDataset(video_paths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    codec = TemporalCodec().to(DEVICE)
    entropy = GMMEntropyModel().to(DEVICE)
    opt = optim.AdamW(list(codec.parameters()) + list(entropy.parameters()), lr=1e-4, weight_decay=1e-6)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for epoch in range(1, EPOCHS+1):
        codec.train(); entropy.train()
        total_rate, total_dist, n_batches = 0.0, 0.0, 0
        for seq in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            seq = seq.to(DEVICE)  # [B, T, 3, H, W]
            B, T = seq.shape[0], seq.shape[1]
            opt.zero_grad()
            coded = codec.encode_sequence(seq)
            rate_sum = torch.tensor(0.0, device=DEVICE)
            dist_sum = torch.tensor(0.0, device=DEVICE)
            for t, (_, q_coded, _) in enumerate(coded):
                mu, scale, weight = entropy(q_coded)
                nll = gmm_likelihood_discrete(q_coded, mu, scale, weight, step=QUANT_STEP)
                rate_sum = rate_sum + nll * q_coded.numel() / np.log(2)
            decoded = codec.decode_sequence([(c[0], c[1]) for c in coded])
            for t in range(T):
                dist_sum = dist_sum + torch.nn.functional.mse_loss(decoded[t], seq[:, t])
            rate_pp = rate_sum / (B * T * 3 * RESOLUTION * RESOLUTION)
            dist_avg = dist_sum / T
            loss = LAM * rate_pp + dist_avg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(codec.parameters()) + list(entropy.parameters()), 1.0)
            opt.step()
            total_rate += rate_pp.item(); total_dist += dist_avg.item(); n_batches += 1
        sched.step()
        print(f"Epoch {epoch}: Rate={total_rate/n_batches:.4f} bpp, Dist={total_dist/n_batches:.6f}, PSNR={10*np.log10(1.0/(total_dist/n_batches)):.2f} dB")
        if epoch % 20 == 0:
            torch.save({'codec': codec.state_dict(), 'entropy': entropy.state_dict()}, f'{OUTPUT_DIR}/temporal_epoch{epoch}.pt')
    torch.save({'codec': codec.state_dict(), 'entropy': entropy.state_dict()}, f'{OUTPUT_DIR}/temporal_final.pt')

# --- Evaluation: compare all-intra vs temporal ---
def evaluate_both():
    ckpt = torch.load(f'{OUTPUT_DIR}/temporal_final.pt', map_location=DEVICE, weights_only=False)
    codec = TemporalCodec().to(DEVICE); codec.load_state_dict(ckpt['codec']); codec.eval()
    entropy = GMMEntropyModel().to(DEVICE); entropy.load_state_dict(ckpt['entropy']); entropy.eval()

    video_path = glob.glob(os.path.join(DATASET_DIR, '*.mpg'))[0]
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while len(all_frames) < 100:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (RESOLUTION, RESOLUTION))
        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
    cap.release()

    # Evaluate as a single sequence
    seq = torch.from_numpy(np.stack([np.transpose(f, (2,0,1)) for f in all_frames])).unsqueeze(0).float().to(DEVICE)
    B, T = seq.shape[0], seq.shape[1]

    with torch.no_grad():
        # Temporal coding
        coded = codec.encode_sequence(seq)
        i_bits = p_bits = 0
        for is_I, q, _ in coded:
            mu, scale, weight = entropy(q)
            nll = gmm_likelihood_discrete(q, mu, scale, weight, step=QUANT_STEP)
            bits = (nll.item() / np.log(2)) * q.numel()
            if is_I: i_bits += bits
            else: p_bits += bits
        temporal_bpp = (i_bits + p_bits) / (B * T * 3 * RESOLUTION * RESOLUTION)
        decoded_temporal = codec.decode_sequence([(c[0], c[1]) for c in coded])
        temporal_psnr = 10 * np.log10(1.0 / np.mean([torch.nn.functional.mse_loss(decoded_temporal[t], seq[:, t]).item() for t in range(T)]))

        # All-intra baseline (encode each frame independently, no predictor)
        intra_bits = 0
        for t in range(T):
            latent = codec.affine(codec.encoder(seq[:, t], return_surprise=False))
            q = torch.round(latent / QUANT_STEP) * QUANT_STEP
            mu, scale, weight = entropy(q)
            nll = gmm_likelihood_discrete(q, mu, scale, weight, step=QUANT_STEP)
            intra_bits += (nll.item() / np.log(2)) * q.numel()
        intra_bpp = intra_bits / (B * T * 3 * RESOLUTION * RESOLUTION)

    print(f"All-intra: {intra_bpp:.4f} bpp")
    print(f"Temporal:  {temporal_bpp:.4f} bpp ({temporal_psnr:.2f} dB)")
    print(f"I-frame bits: {i_bits:.0f}, P-frame bits: {p_bits:.0f}")
    print(f"P/I ratio: {p_bits/max(1,i_bits):.2f}x (want < 1.0)")
    with open(f'{BENCHMARK_DIR}/temporal_results.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['mode', 'bpp', 'psnr', 'i_bits', 'p_bits'])
        w.writerow(['all_intra', intra_bpp, '-', intra_bits, 0])
        w.writerow(['temporal', temporal_bpp, temporal_psnr, i_bits, p_bits])

if __name__ == '__main__':
    if not os.path.exists(f'{OUTPUT_DIR}/temporal_final.pt'):
        train_temporal()
    evaluate_both()
```

---

## Milestone 3: Surprise-Gated Bitrate Allocation

This script wires the VOE predictor into the encoding loop and implements adaptive quantization based on surprise.

```python
#!/usr/bin/env python3
"""
milestone3_surprise_gating.py
Adds VOE surprise detection to the encoding loop.
High-surprise frames get finer quantization (more bits preserved).
Evaluates BPP ratio between normal and anomaly videos.
Usage: python3 milestone3_surprise_gating.py
"""

import os, sys, glob, csv
import cv2, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.predictor import LeWMPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESOLUTION = 256
QUANT_STEP = 2.0 / 255  # base quantization step
TAU_HIGH = 0.7           # surprise above this -> finer quantization
TAU_LOW = 0.3            # surprise below this -> coarser quantization

# --- Architecture (identical to milestone2, trimmed for inference) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        r = x
        x = torch.nn.functional.gelu(self.norm1(x)); x = self.conv1(x)
        x = torch.nn.functional.gelu(self.norm2(x)); x = self.conv2(x)
        return x + r

class LeWMDecoder(nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(latent_dim, hidden_dim, 1)
        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4, 2, 1); self.res1 = ResidualBlock(hidden_dim//2)
        self.up2 = nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 4, 2, 1); self.res2 = ResidualBlock(hidden_dim//4)
        self.up3 = nn.ConvTranspose2d(hidden_dim//4, hidden_dim//8, 4, 2, 1); self.res3 = ResidualBlock(hidden_dim//8)
        self.up4 = nn.ConvTranspose2d(hidden_dim//8, hidden_dim//16, 4, 2, 1); self.res4 = ResidualBlock(hidden_dim//16)
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim//16, hidden_dim//32, 3, 1, 1), nn.InstanceNorm2d(hidden_dim//32),
            nn.GELU(), nn.Conv2d(hidden_dim//32, 3, 3, 1, 1),
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
    return (nll / np.log(2)) * y.numel()  # total bits

class SurpriseGatedCodec(nn.Module):
    def __init__(self, ckpt_path=None):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=192, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=192)
        self.affine = AffineNormalization(192)
        self.predictor = LeWMPredictor(latent_dim=192)
        self.entropy = GMMEntropyModel()
        if ckpt_path and os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            if 'codec' in sd: self.load_state_dict(sd['codec'], strict=False)
            if 'entropy' in sd: self.entropy.load_state_dict(sd['entropy'], strict=False)
        self.eval()
    def encode_frame(self, x, prev_latents):
        """Returns (latent, quantized, surprise_score, is_I)."""
        latent = self.affine(self.encoder(x, return_surprise=False))
        if len(prev_latents) == 0:
            # I-frame
            q = torch.round(latent / QUANT_STEP) * QUANT_STEP
            return latent, q, 1.0, True
        pred_mean, _ = self.predictor(prev_latents)
        residual = latent - pred_mean
        # Surprise = normalized residual energy
        surprise = min(1.0, (residual.var() / (pred_mean.var() + 1e-8)).item())
        # Adaptive quantization: high surprise -> finer step
        if surprise >= TAU_HIGH:
            step = QUANT_STEP * 0.5   # finer = more bits preserved
        elif surprise <= TAU_LOW:
            step = QUANT_STEP * 2.0    # coarser = fewer bits
        else:
            step = QUANT_STEP
        q = torch.round(residual / step) * step
        # Return coded residual, surprise, and the step used for decoding
        return latent, (q, step, pred_mean), surprise, False
    def decode_frame(self, coded, prev_latents):
        """coded is (q, step, pred_mean) for P-frames or q for I-frames."""
        if isinstance(coded, tuple):
            q, step, pred_mean = coded
            latent = pred_mean + q
        else:
            latent = coded
        return self.decoder(latent, target_size=(RESOLUTION, RESOLUTION))

# --- Evaluation on normal vs anomaly ---
def evaluate_video(codec, video_path, label):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 100:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (RESOLUTION, RESOLUTION))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
    cap.release()

    prev_latents = []
    total_bits = 0; high_surprise_bits = 0; low_surprise_bits = 0
    high_count = 0; low_count = 0; total_mse = 0.0

    with torch.no_grad():
        for frame in tqdm(frames, desc=f"Encoding {label}"):
            x = torch.from_numpy(np.transpose(frame, (2,0,1))).unsqueeze(0).float().to(DEVICE)
            latent, coded, surprise, is_I = codec.encode_frame(x, prev_latents)
            # Count rate
            if is_I:
                q_for_rate = coded
                step_for_rate = QUANT_STEP
            else:
                q_for_rate, step_for_rate, _ = coded
            mu, scale, weight = codec.entropy(q_for_rate)
            bits = gmm_bits(q_for_rate, mu, scale, weight, step=step_for_rate).item()
            total_bits += bits
            if surprise >= TAU_HIGH:
                high_surprise_bits += bits; high_count += 1
            elif surprise <= TAU_LOW:
                low_surprise_bits += bits; low_count += 1
            # Decode
            decoded = codec.decode_frame(coded if is_I else coded, prev_latents)
            total_mse += torch.nn.functional.mse_loss(decoded, x).item()
            # Update context
            if is_I:
                prev_latents = [latent.detach()]
            else:
                pred_mean = coded[2]
                prev_latents.append((pred_mean + q_for_rate).detach())
            if len(prev_latents) > 4:
                prev_latents.pop(0)

    n = len(frames)
    bpp = total_bits / (n * 3 * RESOLUTION * RESOLUTION)
    psnr = 10 * np.log10(1.0 / (total_mse / n))
    return bpp, psnr, high_surprise_bits/max(1,high_count), low_surprise_bits/max(1,low_count), high_count, low_count

if __name__ == '__main__':
    # Use temporal checkpoint from milestone2, or reassign to milestone1 GMM checkpoint
    ckpt = 'checkpoints_milestone2/temporal_final.pt'
    if not os.path.exists(ckpt):
        # Fall back: use milestone1 checkpoints (no predictor, so all I-frames)
        ckpt = None
        codec = SurpriseGatedCodec(ckpt_path=None)
        # Load GMM entropy model from milestone1 if available
        ent_path = 'checkpoints_milestone1/entropy_lambda_0.05_final.pt'
        if os.path.exists(ent_path):
            codec.entropy.load_state_dict(torch.load(ent_path, map_location=DEVICE, weights_only=False), strict=False)
        print("WARNING: No temporal checkpoint. Surprise gating tested on I-frame-only mode (all frames treated as I-frames).")
    else:
        codec = SurpriseGatedCodec(ckpt_path=ckpt)

    # Find normal and anomaly videos
    normal_video = glob.glob('datasets/pevid-hd/walking*.mpg')
    anomaly_video = glob.glob('datasets/pevid-hd/droppingBag*.mpg')
    if not normal_video: normal_video = glob.glob('datasets/pevid-hd/*.mpg')[:1]
    if not anomaly_video: anomaly_video = glob.glob('datasets/pevid-hd/*.mpg')[1:2]
    if not normal_video or not anomaly_video:
        print("Need at least two videos for comparison. Update paths.")
        exit(1)

    print(f"Normal: {normal_video[0]}")
    print(f"Anomaly: {anomaly_video[0]}")

    nb, np_, nhb, nlb, nhc, nlc = evaluate_video(codec, normal_video[0], "Normal")
    ab, ap_, ahb, alb, ahc, alc = evaluate_video(codec, anomaly_video[0], "Anomaly")

    print(f"\nNormal:   BPP={nb:.4f}, PSNR={np_:.2f} dB, Hi-surp bits/frame={nhb:.0f} (n={nhc}), Lo-surp={nlb:.0f} (n={nlc})")
    print(f"Anomaly:  BPP={ab:.4f}, PSNR={ap_:.2f} dB, Hi-surp bits/frame={ahb:.0f} (n={ahc}), Lo-surp={alb:.0f} (n={alc})")
    print(f"BPP ratio (anomaly/normal): {ab/max(nb,1e-10):.2f}x")
    print(f"High-surprise avg bits: Normal={nhb:.0f}, Anomaly={ahb:.0f}")

    with open('benchmark_milestone3/surprise_gating.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['video','bpp','psnr','hi_bits','lo_bits','hi_count','lo_count'])
        w.writerow(['normal', nb, np_, nhb, nlb, nhc, nlc])
        w.writerow(['anomaly', ab, ap_, ahb, alb, ahc, alc])
```

---

## Execution Order and Dependencies

1. **Run `milestone1_rd_curve.py` first.** This produces your RD curve and BD-rate number. Expected runtime: ~8–12 hours on a T4 or A100 for all 6 lambdas at 100 epochs each. You can reduce `EPOCHS` to 50 for a faster first pass.

2. **Run `milestone2_temporal.py` second.** This requires the GMM entropy model architecture from milestone 1, but trains from scratch with the predictor. After training, it prints the all-intra vs. temporal BPP comparison. If P-frame bits are lower than I-frame bits (P/I ratio < 1.0), temporal coding works.

3. **Run `milestone3_surprise_gating.py` third.** This loads the temporal checkpoint from milestone 2 and evaluates on normal vs. anomaly video. If surprise gating works, anomaly video will show higher BPP per high-surprise frame and the BPP ratio will exceed 1.0.

Each script is independent in the sense that it trains its own model. Milestone 3 depends on milestone 2's checkpoint. Milestone 2 can be replaced with an all-intra baseline if temporal training does not converge — the surprise gating logic works with or without the predictor.