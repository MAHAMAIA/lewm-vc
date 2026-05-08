#!/usr/bin/env python3
"""
milestone2_temporal.py
Trains LeWM-VC with JEPA temporal prediction for P-frames.
Compares all-intra vs IPPP coding.
Includes predictor pretraining phase and entropy warm start.
Usage: python3 milestone2_temporal.py
"""

import os, sys, glob, csv, datetime
import cv2, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.distributions.normal import Normal
from tqdm import tqdm

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.predictor import LeWMPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESOLUTION = 256
QUANT_STEP = 2.0 / 255
BATCH_SIZE = 4         # sequences per batch
SEQ_LEN = 8            # frames per sequence
EPOCHS = 80
PRETRAIN_EPOCHS = 20
LAM = 0.05             # fixed lambda; sweep later if needed
OUTPUT_DIR = 'checkpoints_milestone2'
DATASET_DIR = 'datasets/pevid-hd'
BENCHMARK_DIR = 'benchmark_milestone2'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BENCHMARK_DIR, exist_ok=True)

# --- Logging ---
LOG_FILE = os.path.join(BENCHMARK_DIR, f'temporal_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

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
print(f"Device: {DEVICE}")

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
            latent = self.affine(self.encoder(frames[:, t], return_surprise=False))
            if t == 0 or len(prev_latents) == 0:
                xq = torch.round(latent / QUANT_STEP) * QUANT_STEP
                q = xq + (latent - xq.detach()) * 0.5
                results.append((True, q, q))
            else:
                pred_mean, _ = self.predictor(prev_latents)
                residual = latent - pred_mean
                xq_res = torch.round(residual / QUANT_STEP) * QUANT_STEP
                q_res = xq_res + (residual - xq_res.detach()) * 0.5
                results.append((False, q_res, pred_mean + q_res))
            prev_latents.append(results[-1][2].detach())
            if len(prev_latents) > self.predictor.context_len:
                prev_latents.pop(0)
        return results

    def decode_sequence(self, coded_sequence):
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
            frames.append(frames[-1].copy())
        return torch.from_numpy(np.stack(frames)).float()

# --- Training ---
def train_temporal():
    video_paths = glob.glob(os.path.join(DATASET_DIR, '*.mpg'))
    if not video_paths: video_paths = glob.glob(os.path.join(DATASET_DIR, '*.avi'))
    if not video_paths: raise FileNotFoundError(f"No videos in {DATASET_DIR}")
    print(f"Training on {len(video_paths)} videos")
    dataset = VideoSequenceDataset(video_paths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    codec = TemporalCodec().to(DEVICE)
    entropy = GMMEntropyModel().to(DEVICE)

    # --- Warm start from milestone 1 ---
    ae_warm = 'checkpoints_milestone1/ae_lambda_0.05_final.pt'
    ent_warm = 'checkpoints_milestone1/entropy_lambda_0.05_final.pt'
    if os.path.exists(ae_warm):
        ae_state = torch.load(ae_warm, map_location=DEVICE, weights_only=False)
        codec.encoder.load_state_dict(
            {k.replace('encoder.', ''): v for k, v in ae_state.items() if k.startswith('encoder.')},
            strict=False)
        codec.decoder.load_state_dict(
            {k.replace('decoder.', ''): v for k, v in ae_state.items() if k.startswith('decoder.')},
            strict=False)
        codec.affine.load_state_dict(
            {k.replace('affine.', ''): v for k, v in ae_state.items() if k.startswith('affine.')},
            strict=False)
        print("Loaded milestone 1 autoencoder warm start")
    if os.path.exists(ent_warm):
        state = torch.load(ent_warm, map_location=DEVICE, weights_only=False)
        for k in list(state.keys()):
            if 'mask' in k: del state[k]
        entropy.load_state_dict(state, strict=False)
        print("Loaded milestone 1 entropy model warm start")

    # --- Phase 1: Pretrain predictor only ---
    print(f"\n=== Phase 1: Pretraining predictor ({PRETRAIN_EPOCHS} epochs) ===")
    for param in codec.encoder.parameters(): param.requires_grad = False
    for param in codec.decoder.parameters(): param.requires_grad = False
    for param in codec.affine.parameters(): param.requires_grad = False
    for param in entropy.parameters(): param.requires_grad = False

    opt_pred = optim.AdamW(codec.predictor.parameters(), lr=1e-3, weight_decay=1e-6)
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        codec.train()
        total_loss = 0.0; n_batches = 0
        for seq in tqdm(loader, desc=f"Pretrain epoch {epoch}/{PRETRAIN_EPOCHS}"):
            seq = seq.to(DEVICE)
            B, T = seq.shape[0], seq.shape[1]
            opt_pred.zero_grad()
            latents = []
            for t in range(T):
                with torch.no_grad():
                    latents.append(codec.affine(codec.encoder(seq[:, t], return_surprise=False)))
            pred_loss = torch.tensor(0.0, device=DEVICE)
            for t in range(1, T):
                context = latents[:t]
                if len(context) > codec.predictor.context_len:
                    context = context[-codec.predictor.context_len:]
                pred_mean, _ = codec.predictor(context)
                pred_loss = pred_loss + torch.nn.functional.mse_loss(pred_mean, latents[t])
            pred_loss.backward()
            opt_pred.step()
            total_loss += pred_loss.item(); n_batches += 1
        print(f"Pretrain epoch {epoch}: MSE={total_loss/n_batches:.6f}")

    # Unfreeze all
    for param in codec.parameters(): param.requires_grad = True
    for param in entropy.parameters(): param.requires_grad = True

    # --- Phase 2: Joint RD training ---
    print(f"\n=== Phase 2: Joint RD training ({EPOCHS} epochs) ===")
    opt = optim.AdamW(list(codec.parameters()) + list(entropy.parameters()), lr=5e-5, weight_decay=1e-6)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        codec.train(); entropy.train()
        total_rate, total_dist, n_batches = 0.0, 0.0, 0
        for seq in tqdm(loader, desc=f"RD epoch {epoch}/{EPOCHS}"):
            seq = seq.to(DEVICE)
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
        avg_rate = total_rate / n_batches
        avg_dist = total_dist / n_batches
        avg_psnr = 10 * np.log10(1.0 / avg_dist) if avg_dist > 0 else 100.0
        print(f"RD epoch {epoch}: Rate={avg_rate:.4f} bpp, Dist={avg_dist:.6f}, PSNR={avg_psnr:.2f} dB")
        if epoch % 20 == 0:
            torch.save({'codec': codec.state_dict(), 'entropy': entropy.state_dict()}, f'{OUTPUT_DIR}/temporal_epoch{epoch}.pt')
    torch.save({'codec': codec.state_dict(), 'entropy': entropy.state_dict()}, f'{OUTPUT_DIR}/temporal_final.pt')
    print(f"Training complete. Checkpoint saved to {OUTPUT_DIR}/temporal_final.pt")

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
    print(f"Evaluating on {len(all_frames)} frames from {video_path}")

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
        temporal_mse = np.mean([torch.nn.functional.mse_loss(decoded_temporal[t], seq[:, t]).item() for t in range(T)])
        temporal_psnr = 10 * np.log10(1.0 / temporal_mse) if temporal_mse > 0 else 100.0

        # All-intra baseline
        intra_bits = 0
        intra_mse = 0.0
        for t in range(T):
            latent = codec.affine(codec.encoder(seq[:, t], return_surprise=False))
            xq = torch.round(latent / QUANT_STEP) * QUANT_STEP
            q = xq + (latent - xq.detach()) * 0.5
            mu, scale, weight = entropy(q)
            nll = gmm_likelihood_discrete(q, mu, scale, weight, step=QUANT_STEP)
            intra_bits += (nll.item() / np.log(2)) * q.numel()
            recon = codec.decoder(q, target_size=(RESOLUTION, RESOLUTION))
            intra_mse += torch.nn.functional.mse_loss(recon, seq[:, t]).item()
        intra_bpp = intra_bits / (B * T * 3 * RESOLUTION * RESOLUTION)
        intra_psnr = 10 * np.log10(1.0 / (intra_mse / T)) if intra_mse > 0 else 100.0

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"All-intra: {intra_bpp:.4f} bpp, PSNR={intra_psnr:.2f} dB")
    print(f"Temporal:  {temporal_bpp:.4f} bpp, PSNR={temporal_psnr:.2f} dB")
    print(f"I-frame bits: {i_bits:.0f}, P-frame bits: {p_bits:.0f}")
    print(f"P-frames: {T-1} frames, avg {p_bits/max(1,T-1):.0f} bits/frame")
    print(f"I-frame: {i_bits:.0f} bits")
    print(f"P/I ratio: {p_bits/max(1,T-1)/max(1,i_bits):.2f}x per frame (want < 1.0)")
    savings = (intra_bpp - temporal_bpp) / intra_bpp * 100
    print(f"Temporal savings vs all-intra: {savings:+.2f}%")

    with open(f'{BENCHMARK_DIR}/temporal_results.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['mode', 'bpp', 'psnr', 'i_bits', 'p_bits'])
        w.writerow(['all_intra', intra_bpp, intra_psnr, intra_bits, 0])
        w.writerow(['temporal', temporal_bpp, temporal_psnr, i_bits, p_bits])
    print(f"Results saved to {BENCHMARK_DIR}/temporal_results.csv")

if __name__ == '__main__':
    if not os.path.exists(f'{OUTPUT_DIR}/temporal_final.pt'):
        train_temporal()
    evaluate_both()
