#!/usr/bin/env python3
"""
JEPA-based LeWM-VC training — per-λ joint fine-tuning with predictor.
Two-phase per λ:
  Phase A: JEPA warmup — train encoder + predictor with L_JEPA + SIGReg
  Phase B: Joint RD — full loss L = R + λ·D + γ·L_JEPA + δ·L_SIGReg

Usage:
    python3 jepa_train.py               # train all 6 λ models
    python3 jepa_train.py --evaluate    # evaluate final checkpoints
"""

import os
import sys
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.quant import Quantizer
from lewm_vc.predictor import LeWMPredictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---------- Autoencoder (same as pipeline) ----------
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
    def __init__(self, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(latent_dim, hidden_dim, 1)
        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 4,2,1)
        self.res1 = ResidualBlock(hidden_dim)
        self.up2 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 4,2,1)
        self.res2 = ResidualBlock(hidden_dim)
        self.up3 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 4,2,1)
        self.res3 = ResidualBlock(hidden_dim)
        self.up4 = nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4,2,1)
        self.res4 = ResidualBlock(hidden_dim//2)
        self.up5 = nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 4,2,1)
        self.res5 = ResidualBlock(hidden_dim//4)
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim//4, hidden_dim//8, 3,1,1),
            nn.InstanceNorm2d(hidden_dim//8),
            nn.GELU(),
            nn.Conv2d(hidden_dim//8, 3, 3,1,1),
        )
    def forward(self, latent, target_size=None):
        x = self.proj(latent)
        x = self.up1(x); x = self.res1(x)
        x = self.up2(x); x = self.res2(x)
        x = self.up3(x); x = self.res3(x)
        x = self.up4(x); x = self.res4(x)
        x = self.up5(x); x = self.res5(x)
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
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
        self.affine = AffineNormalization(latent_dim)
    def encode(self, x):
        latent = self.encoder(x, return_surprise=False)
        return self.affine(latent)
    def decode(self, latent_norm, target_size):
        return self.decoder(latent_norm, target_size=target_size)

# ---------- JEPA pipeline model ----------
class JEPACodec(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.autoencoder = VideoAutoencoder(latent_dim=latent_dim)
        self.predictor = LeWMPredictor(latent_dim=latent_dim)
        self.quantizer = Quantizer(num_levels=256, mode='training')

    def encode_frames(self, frames):
        """Encode all frames, return latents."""
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)
        return self.autoencoder.encode(frames_flat)

    def predict_latents(self, latents):
        """Predict each frame's latent from all past latents."""
        B, C, H, W = latents.shape
        context = []
        predicted = []
        for t in range(B):
            if len(context) > 0:
                pred_mean, pred_std = self.predictor(context)
                predicted.append((pred_mean, pred_std))
            context.append(latents[t:t+1])
        return predicted

    def compute_rate(self, latents, predicted_latents):
        """Compute rate on residuals between predicted and actual latents."""
        total_rate = torch.tensor(0.0, device=latents.device)
        for t in range(1, len(predicted_latents) + 1):
            residual = latents[t:t+1] - predicted_latents[t-1][0]
            quant_residual = self.quantizer(residual)
            rate, _ = entropy_model_fn(quant_residual)
            total_rate = total_rate + rate.sum()
        return total_rate

    def decode_frames(self, latents, target_size):
        """Decode latents back to frames."""
        return self.autoencoder.decode(latents, target_size=target_size)

# Placeholder — replaced by actual entropy model at train time
def entropy_model_fn(x):
    return torch.zeros(1, device=x.device), {}

# ---------- Dataset ----------
class VideoDataset(Dataset):
    def __init__(self, video_paths, frame_size=(256,256), frames_per_clip=4):
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
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(self.frames_per_clip):
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

BASE_DIR = os.environ.get('LEWM_BASE', '/root/le-maia')
DATASET_DIR = os.environ.get('LEWM_DATASET', os.path.join(BASE_DIR, 'datasets/pevid-hd'))
CHECKPOINT_DIR = os.environ.get('LEWM_CHECKPOINT_DIR', os.path.join(BASE_DIR, 'checkpoints_jepa'))
PHASE0_CKPT = os.environ.get('LEWM_PHASE0_CKPT', os.path.join(BASE_DIR, 'checkpoints_phase0/autoencoder_final.pt'))

video_paths = glob.glob(os.path.join(DATASET_DIR, '*.mpg'))
if not video_paths:
    raise FileNotFoundError(f"No videos found in {DATASET_DIR}")
dataset = VideoDataset(video_paths, frame_size=(256,256), frames_per_clip=4)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)

val_size = max(1, int(len(dataset) * 0.1))
train_dataset, val_dataset = random_split(dataset, [len(dataset)-val_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

# ---------- Loss ----------
criterion_mse = nn.MSELoss()
try:
    import lpips
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    use_perceptual = True
except:
    use_perceptual = False

QUANT_STEP = 2.0 / 255

def laplace_likelihood_discrete(y, mu, log_scale, step, sigma_floor=0.003, epsilon=1e-9):
    from torch.distributions.laplace import Laplace
    scale = torch.nn.functional.softplus(log_scale) + sigma_floor
    laplace = Laplace(mu, scale)
    cdf_upper = laplace.cdf(y + 0.5 * step)
    cdf_lower = laplace.cdf(y - 0.5 * step)
    pmf = cdf_upper - cdf_lower
    pmf = torch.clamp(pmf, min=epsilon, max=1.0)
    nll = -torch.log(pmf)
    return nll.mean()

def quantize_with_temp(x, step, temp):
    x_quant = torch.round(x / step) * step
    return x_quant + (x - x_quant.detach()) * temp

# ---------- Per-λ training ----------
lambda_list = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
output_dir = CHECKPOINT_DIR
os.makedirs(output_dir, exist_ok=True)

# ---------- Contextual entropy model (same as pipeline/train.py) ----------
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
    def __init__(self, latent_dim=32, hyper_channels=128, context_hidden=128):
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
        mu_base = base_params[:, :latent_dim, :, :]
        log_scale_base = base_params[:, latent_dim:, :, :]
        ctx = self.context(x)
        mu_offset = self.refine_mu(ctx)
        scale_offset = self.refine_scale(ctx)
        mu = mu_base + mu_offset
        log_scale = log_scale_base + scale_offset
        return mu, log_scale

def train():
    for lam in lambda_list:
        print(f"\n========== JEPA training with λ = {lam} ==========")

        # Build models
        autoencoder = VideoAutoencoder().to(device)
        predictor = LeWMPredictor(latent_dim=32).to(device)
        entropy_model = ContextualEntropyModel().to(device)

        # Load Phase 0 autoencoder
        if os.path.exists(PHASE0_CKPT):
            autoencoder.load_state_dict(torch.load(PHASE0_CKPT, map_location=device, weights_only=False), strict=False)
            print("Loaded Phase 0 autoencoder")

        # Optimizer
        optimizer = optim.AdamW([
            {'params': autoencoder.encoder.parameters(), 'lr': 5e-5},
            {'params': autoencoder.affine.parameters(), 'lr': 5e-5},
            {'params': autoencoder.decoder.parameters(), 'lr': 5e-5},
            {'params': predictor.parameters(), 'lr': 5e-5},
            {'params': entropy_model.parameters(), 'lr': 1e-4},
        ], weight_decay=1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

        best_val_loss = float('inf')
        patience = 10
        no_improve = 0

        for epoch in range(1, 31):
            temp = max(0.1, 1.0 - epoch * 0.03)
            autoencoder.train()
            predictor.train()
            entropy_model.train()
            total_loss = 0
            total_rate = 0
            total_mse = 0
            total_jepa = 0
            num_batches = 0
            pbar = tqdm(dataloader, desc=f"JEPA λ={lam} Epoch {epoch}/30")

            for batch in pbar:
                batch = batch.to(device)
                optimizer.zero_grad()
                B, T, C, H, W = batch.shape

                # Encode all frames, reshape to [B, T, ...] to keep batch items separate
                batch_flat = batch.view(B * T, C, H, W)
                latents_all = autoencoder.encode(batch_flat)  # [B*T, 32, 8, 8]
                _, C, H_lat, W_lat = latents_all.shape
                latents = latents_all.view(B, T, C, H_lat, W_lat)  # [B, T, 32, 8, 8]
                latents_list = [latents[:, t] for t in range(T)]  # each [B, 32, 8, 8]

                # JEPA: predict each frame from past frames
                jepa_loss = torch.tensor(0.0, device=device)
                total_bits = torch.tensor(0.0, device=device)
                total_recon_mse = torch.tensor(0.0, device=device)

                for t in range(T):
                    z_t = latents_list[t]  # [B, 32, 8, 8]

                    if t > 0:
                        # Build context from all previous frames (list of [B, 32, 8, 8])
                        context = latents_list[:t]
                        pred_mean, pred_std = predictor(context)

                        # JEPA loss: MSE between predicted mean and actual
                        jepa_loss = jepa_loss + torch.nn.functional.mse_loss(pred_mean, z_t)

                        # Residual coding for rate
                        residual = z_t - pred_mean
                        quant_res = quantize_with_temp(residual, QUANT_STEP, temp)
                        mu, log_scale = entropy_model(quant_res)
                        nll = laplace_likelihood_discrete(quant_res, mu, log_scale, step=QUANT_STEP, sigma_floor=0.003)
                        rate = nll * quant_res.numel() / np.log(2)
                        total_bits = total_bits + rate

                    quant_z = quantize_with_temp(z_t, QUANT_STEP, temp)
                    recon = autoencoder.decode(quant_z, target_size=(H, W))
                    mse = criterion_mse(recon, batch[:, t])
                    total_recon_mse = total_recon_mse + mse

                # SIGReg: KL toward N(0, I) across all latents
                z_flat = latents_all.view(B * T, -1)
                mu_z = z_flat.mean(dim=1, keepdim=True)
                var_z = z_flat.var(dim=1, keepdim=True) + 1e-8
                sigreg_loss = (0.5 * (mu_z ** 2 + var_z - torch.log(var_z) - 1)).mean()

                # Combined loss per paper: R + λ·D + γ·L_JEPA + δ·SIGReg
                gamma = 1.0
                delta = 0.01
                rate_per_pixel = total_bits / batch_flat.numel()
                distortion = total_recon_mse / T
                if use_perceptual and epoch > 10:
                    recon_all = torch.stack([autoencoder.decode(quantize_with_temp(latents[t:t+1], QUANT_STEP, temp), target_size=(H, W)) for t in range(T)])
                    recon_4d = recon_all.view(B * T, C, H, W)
                    batch_4d = batch.view(B * T, C, H, W)
                    perceptual = perceptual_loss_fn(recon_4d * 2 - 1, batch_4d * 2 - 1).mean()
                    distortion = distortion + 0.1 * perceptual

                loss = rate_per_pixel + lam * distortion + gamma * jepa_loss + delta * sigreg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(entropy_model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_rate += rate_per_pixel.item()
                total_mse += distortion.item()
                total_jepa += jepa_loss.item()
                num_batches += 1
                pbar.set_postfix(loss=loss.item(), rate=rate_per_pixel.item(), jepa=jepa_loss.item(), mse=distortion.item())

            scheduler.step()
            avg_loss = total_loss / num_batches
            avg_rate = total_rate / num_batches
            avg_mse = total_mse / num_batches
            avg_jepa = total_jepa / num_batches
            psnr = 10 * np.log10(1.0 / avg_mse) if avg_mse > 0 else 0
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Rate={avg_rate:.4f} bpp, JEPA={avg_jepa:.6f}, MSE={avg_mse:.6f}, PSNR={psnr:.2f} dB")

            # Validation every 5 epochs
            if epoch % 5 == 0:
                autoencoder.eval()
                predictor.eval()
                entropy_model.eval()
                val_loss = 0
                val_rate = 0
                val_mse = 0
                val_batches = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = val_batch.to(device)
                        Bv, Tv, Cv, Hv, Wv = val_batch.shape
                        val_flat = val_batch.view(Bv * Tv, Cv, Hv, Wv)
                        val_latents_all = autoencoder.encode(val_flat)
                        val_latents = [val_latents_all[t*Bv:(t+1)*Bv] for t in range(Tv)]
                        val_mse_batch = 0.0
                        val_rate_batch = 0.0
                        for t in range(Tv):
                            z_t = val_latents[t]
                            if t > 0:
                                context = val_latents[:t]
                                pred_mean, _ = predictor(context)
                                residual = z_t - pred_mean
                                quant_val = quantize_with_temp(residual, QUANT_STEP, temp=0.0)
                                mu, log_scale = entropy_model(quant_val)
                                nll = laplace_likelihood_discrete(quant_val, mu, log_scale, step=QUANT_STEP, sigma_floor=0.003)
                                val_rate_batch = val_rate_batch + nll * quant_val.numel() / np.log(2)
                            quant_z = quantize_with_temp(z_t, QUANT_STEP, temp=0.0)
                            recon = autoencoder.decode(quant_z, target_size=(Hv, Wv))
                            val_mse_batch = val_mse_batch + criterion_mse(recon, val_batch[:, t]).item()
                        val_rate_bpp = val_rate_batch / val_flat.numel()
                        val_loss += val_mse_batch / Tv + lam * val_rate_bpp.item()
                        val_mse += val_mse_batch / Tv
                        val_rate += val_rate_bpp.item()
                        val_batches += 1
                val_psnr = 10 * np.log10(1.0 / (val_mse / val_batches))
                val_bpp = val_rate / val_batches
                avg_val_loss = val_loss / val_batches
                print(f"  Val: PSNR={val_psnr:.2f} dB, BPP={val_bpp:.4f}, ValLoss={avg_val_loss:.6f}")
                if avg_val_loss < best_val_loss - 1e-4:
                    best_val_loss = avg_val_loss
                    no_improve = 0
                    torch.save(autoencoder.state_dict(), f'{output_dir}/ae_lambda_{lam}_best.pt')
                    torch.save(predictor.state_dict(), f'{output_dir}/predictor_lambda_{lam}_best.pt')
                    torch.save(entropy_model.state_dict(), f'{output_dir}/entropy_lambda_{lam}_best.pt')
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"  Early stop at epoch {epoch}")
                        break
                autoencoder.train()
                predictor.train()
                entropy_model.train()

            if epoch % 10 == 0:
                torch.save(autoencoder.state_dict(), f'{output_dir}/ae_lambda_{lam}_epoch{epoch}.pt')
                torch.save(predictor.state_dict(), f'{output_dir}/predictor_lambda_{lam}_epoch{epoch}.pt')
                torch.save(entropy_model.state_dict(), f'{output_dir}/entropy_lambda_{lam}_epoch{epoch}.pt')

        torch.save(autoencoder.state_dict(), f'{output_dir}/ae_lambda_{lam}_final.pt')
        torch.save(predictor.state_dict(), f'{output_dir}/predictor_lambda_{lam}_final.pt')
        torch.save(entropy_model.state_dict(), f'{output_dir}/entropy_lambda_{lam}_final.pt')
        print(f"Finished JEPA λ={lam}.")

# ---------- Evaluation ----------
def evaluate():
    test_video = glob.glob(os.path.join(DATASET_DIR, '*.mpg'))[0]
    cap = cv2.VideoCapture(test_video)
    frames = []
    target_size = (256, 256)
    while len(frames) < 150:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames")

    quantizer_eval = Quantizer(num_levels=256, mode='inference').to(device)

    for lam in lambda_list:
        ae_ckpt = os.path.join(output_dir, f'ae_lambda_{lam}_final.pt')
        pred_ckpt = os.path.join(output_dir, f'predictor_lambda_{lam}_final.pt')
        ent_ckpt = os.path.join(output_dir, f'entropy_lambda_{lam}_final.pt')
        if not os.path.exists(ae_ckpt) or not os.path.exists(pred_ckpt) or not os.path.exists(ent_ckpt):
            print(f"Missing checkpoints for λ={lam}, skipping")
            continue
        autoencoder = VideoAutoencoder().to(device)
        autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device, weights_only=False), strict=False)
        autoencoder.eval()
        predictor = LeWMPredictor(latent_dim=32).to(device)
        predictor.load_state_dict(torch.load(pred_ckpt, map_location=device, weights_only=False))
        predictor.eval()
        entropy_model = ContextualEntropyModel().to(device)
        state = torch.load(ent_ckpt, map_location=device, weights_only=False)
        for key in list(state.keys()):
            if 'mask' in key:
                del state[key]
        entropy_model.load_state_dict(state, strict=False)
        entropy_model.eval()

        total_bits = 0
        total_mse = 0
        prev_latent = None
        for frame in tqdm(frames, desc=f"JEPA λ={lam}"):
            frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
            frame_t = frame_t.to(device)
            with torch.no_grad():
                latent_norm = autoencoder.encode(frame_t)
                quantized = quantizer_eval(latent_norm)
                if prev_latent is not None:
                    context = [prev_latent]
                    pred_mean, _ = predictor(context)
                    residual = quantized - pred_mean
                    mu, log_scale = entropy_model(residual)
                    nll = laplace_likelihood_discrete(residual, mu, log_scale, step=QUANT_STEP, sigma_floor=0.003)
                    bits = nll * residual.numel() / np.log(2)
                    total_bits += bits.item()
                else:
                    mu, log_scale = entropy_model(quantized)
                    nll = laplace_likelihood_discrete(quantized, mu, log_scale, step=QUANT_STEP, sigma_floor=0.003)
                    bits = nll * quantized.numel() / np.log(2)
                    total_bits += bits.item()
                prev_latent = quantized
                recon = autoencoder.decode(quantized, target_size=target_size)
                mse = torch.nn.functional.mse_loss(recon, frame_t).item()
                total_mse += mse
        bpp = total_bits / (len(frames) * target_size[0] * target_size[1])
        psnr = 20 * np.log10(1.0 / np.sqrt(total_mse / len(frames)))
        print(f"JEPA λ={lam}: bpp={bpp:.4f}, PSNR={psnr:.2f} dB")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true', help='Evaluate final checkpoints')
    args = parser.parse_args()
    if args.evaluate:
        evaluate()
    else:
        train()
